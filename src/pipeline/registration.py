import os
import json
import numpy as np
import torch
from PIL import Image
from .pipeline import PipelineComponent, PipelineContext
from third_party.DiffusionReg.register_pointclouds import init_opts, get_model, register, resample_pcd, load_point_cloud
from third_party.DiffusionReg.utils.options import opts

def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    """A helper function to mimic the resizing in TraceAnything's infer.py"""
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)

class Registration(PipelineComponent):
    """Base class for registration components."""
    def __init__(self, context: PipelineContext):
        super().__init__(context)

    @property
    def short_name(self) -> str:
        return "registration"

class DiffusionReg(Registration):
    """
    A specific implementation of registration using the DiffusionReg model.
    """
    def __init__(self, context: PipelineContext):
        super().__init__(context)

    def run(self):
        print("--- Registering Point Clouds with DiffusionReg ---")
        
        scene_dir_name = f"{self.context.run_name}_scene"
        trace_output_file = os.path.join(self.context.paths["trace_output_dir"], scene_dir_name, "output.pt")
        # (File existence checks) ...

        device = "cuda" if torch.cuda.is_available() else "cpu"
        reg_opts = init_opts(opts)
        reg_model = get_model(reg_opts)
        if reg_model is None:
            raise RuntimeError("Failed to load DiffusionReg model.")

        trace_data = torch.load(trace_output_file, map_location=device)
        num_frames = len(trace_data['preds'])
        frame_paths = sorted([os.path.join(self.context.paths["frames_scene_dir"], f) for f in os.listdir(self.context.paths["frames_scene_dir"]) if f.endswith('.png')])

        if num_frames < 2:
            print("Need at least two frames for registration. Skipping.")
            return

        # Detect chunk boundaries: if chunking was used, boundaries occur at multiples of chunk size
        # Default max_frames_per_chunk is 41 from test_trace_anything.py
        max_frames_per_chunk = 41  # Default from test_trace_anything.py
        chunk_boundaries = set()
        if num_frames > max_frames_per_chunk:
            # Chunk boundaries occur at: max_frames_per_chunk, 2*max_frames_per_chunk, etc.
            # Skip registration when t_plus_1 equals these boundary values
            for chunk_idx in range(1, (num_frames + max_frames_per_chunk - 1) // max_frames_per_chunk):
                boundary_frame = chunk_idx * max_frames_per_chunk
                if boundary_frame < num_frames:
                    chunk_boundaries.add(boundary_frame)
            if chunk_boundaries:
                print(f"Detected chunk boundaries at frames: {sorted(chunk_boundaries)}")
                print("Skipping registration at chunk boundaries to avoid coordinate frame mismatches.")

        registration_errors = []
        skipped_boundaries = []
        n_points_resample = 1024

        for i in range(num_frames - 1):
            t, t_plus_1 = i, i + 1
            
            # Skip registration at chunk boundaries
            if t_plus_1 in chunk_boundaries:
                print(f"\nSkipping frames {t} and {t_plus_1} (chunk boundary)...")
                skipped_boundaries.append((t, t_plus_1))
                continue
            
            print(f"\nProcessing frames {t} and {t_plus_1}...")
            
            aligned_masks, full_points = self._get_aligned_data_for_pair(trace_data, frame_paths, t, t_plus_1, device)
            
            if aligned_masks.get(t) is None or aligned_masks.get(t_plus_1) is None: continue

            pcd_t_full = full_points.get(t, np.array([]))
            pcd_t_plus_1_full = full_points.get(t_plus_1, np.array([]))
            
            if pcd_t_full.shape[0] == 0 or pcd_t_plus_1_full.shape[0] == 0: continue

            pcd_t_reg = resample_pcd(pcd_t_full, n_points_resample)
            pcd_t_plus_1_reg = resample_pcd(pcd_t_plus_1_full, n_points_resample)
            
            transform_mat = register(reg_model, reg_opts, pcd_t_plus_1_reg, pcd_t_reg)
            
            transform_filename = os.path.join(self.context.paths["registration_output_dir"], f"transform_from_{t_plus_1}_to_{t}.npy")
            np.save(transform_filename, transform_mat)

            # Error calculation logic...
            self._calculate_and_log_error(trace_data, aligned_masks, transform_mat, t, t_plus_1, registration_errors, device)
        
        self._save_error_summary(registration_errors, skipped_boundaries)

    def _get_aligned_data_for_pair(self, trace_data, frame_paths, t, t_plus_1, device):
        aligned_masks, full_points = {}, {}
        for frame_idx in [t, t_plus_1]:
            mask_path = os.path.join(self.context.paths["masks_dir"], f"{frame_idx:05d}.png")
            if not os.path.exists(mask_path) or not os.path.exists(frame_paths[frame_idx]):
                aligned_masks[frame_idx] = None
                continue
            
            gtsam_mask_pil = Image.open(mask_path)
            original_frame_pil = Image.open(frame_paths[frame_idx])
            
            W0, H0 = original_frame_pil.size
            if H0 > W0: gtsam_mask_pil = gtsam_mask_pil.transpose(Image.Transpose.ROTATE_90)
            
            resized_mask_pil = _resize_long_side(gtsam_mask_pil, 512)
            H, W = resized_mask_pil.size[1], resized_mask_pil.size[0]
            Ht, Wt = (H - H % 16, W - W % 16)
            aligned_mask_pil = resized_mask_pil.crop((0, 0, Wt, Ht))
            aligned_masks[frame_idx] = np.array(aligned_mask_pil) > 0

            ctrl_pts_3d = trace_data['preds'][frame_idx]['ctrl_pts3d']
            if isinstance(ctrl_pts_3d, np.ndarray): ctrl_pts_3d = torch.from_numpy(ctrl_pts_3d).to(device)
            
            points_for_frame = ctrl_pts_3d.permute(1, 2, 0, 3)
            full_points[frame_idx] = points_for_frame[aligned_masks[frame_idx]][:, 0, :].cpu().numpy()
        return aligned_masks, full_points

    def _calculate_and_log_error(self, trace_data, aligned_masks, transform_mat, t, t_plus_1, registration_errors, device):
        common_mask = aligned_masks[t] & aligned_masks[t_plus_1]
        if not np.any(common_mask): return
        
        C_t = trace_data['preds'][t]['ctrl_pts3d']
        C_t_plus_1 = trace_data['preds'][t+1]['ctrl_pts3d']
        if isinstance(C_t, np.ndarray): C_t = torch.from_numpy(C_t).to(device)
        if isinstance(C_t_plus_1, np.ndarray): C_t_plus_1 = torch.from_numpy(C_t_plus_1).to(device)

        points_t_corr = C_t.permute(1, 2, 0, 3)[common_mask][:, 0, :].cpu().numpy()
        points_t_plus_1_corr = C_t_plus_1.permute(1, 2, 0, 3)[common_mask][:, 0, :].cpu().numpy()
        
        p_t_plus_1_homo = np.hstack((points_t_plus_1_corr, np.ones((points_t_plus_1_corr.shape[0], 1))))
        transformed_points = (transform_mat @ p_t_plus_1_homo.T).T[:, :3]

        distances = np.linalg.norm(points_t_corr - transformed_points, axis=1)
        absolute_error = np.mean(distances)

        min_coords, max_coords = np.min(points_t_corr, axis=0), np.max(points_t_corr, axis=0)
        bounding_box_diagonal = np.linalg.norm(max_coords - min_coords)
        normalized_error = absolute_error / bounding_box_diagonal if bounding_box_diagonal > 1e-6 else absolute_error

        registration_errors.append({
            "frame_pair": (t, t_plus_1),
            "absolute_error": float(absolute_error),
            "normalized_error": float(normalized_error),
            "bounding_box_diagonal": float(bounding_box_diagonal)
        })
        print(f"Absolute Registration Error E_{t}: {absolute_error:.6f}")
        print(f"Normalized Error (relative to diagonal): {normalized_error:.4f}")

    def _save_error_summary(self, registration_errors, skipped_boundaries=None):
        if registration_errors:
            avg_absolute_error = np.mean([e['absolute_error'] for e in registration_errors])
            avg_normalized_error = np.mean([e['normalized_error'] for e in registration_errors])
            error_summary = {
                "individual_errors": registration_errors,
                "average_absolute_error": float(avg_absolute_error),
                "average_normalized_error": float(avg_normalized_error),
            }
            if skipped_boundaries:
                error_summary["skipped_chunk_boundaries"] = skipped_boundaries
                error_summary["num_skipped_boundaries"] = len(skipped_boundaries)
        else:
             error_summary = {"note": "No valid point cloud pairs found for registration."}
             if skipped_boundaries:
                 error_summary["skipped_chunk_boundaries"] = skipped_boundaries
                 error_summary["num_skipped_boundaries"] = len(skipped_boundaries)
        
        summary_path = os.path.join(self.context.paths["registration_output_dir"], "error_summary.json")
        with open(summary_path, 'w') as f: json.dump(error_summary, f, indent=4)
        print(f"\nRegistration complete. Average Absolute Error: {error_summary.get('average_absolute_error', 0):.6f}")
        if skipped_boundaries:
            print(f"Skipped {len(skipped_boundaries)} frame pairs at chunk boundaries.")
        print(f"Error summary saved to {summary_path}")
