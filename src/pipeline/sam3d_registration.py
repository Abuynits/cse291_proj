import os
import json
import numpy as np
import torch
from PIL import Image
from .pipeline import PipelineComponent, PipelineContext
from third_party.DiffusionReg.register_pointclouds import init_opts, get_model, register, resample_pcd, load_point_cloud
from third_party.DiffusionReg.utils.options import opts

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    """A helper function to mimic the resizing in TraceAnything's infer.py"""
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)
from scipy.spatial import KDTree

def chamfer_distance(A: np.ndarray, B: np.ndarray, w_xyz=1.0, w_color=1.0) -> float:
    if A.shape[-1] == B.shape[-1] == 3:
        pass
    elif A.shape[-1] == B.shape[-1] == 6:
        pass
    else:
        raise ValueError("Point clouds must have either 3 (XYZ) or 6 (XYZRGB) dimensions.")
    if A.shape[1] == 6:
        A_xyz = A[:, :3]
        A_color = A[:, 3:]
        A = np.hstack((A_xyz * w_xyz, A_color * w_color))
    if B.shape[1] == 6:
        B_xyz = B[:, :3]
        B_color = B[:, 3:]
        B = np.hstack((B_xyz * w_xyz, B_color * w_color))
    tree_B = KDTree(B)
    dist_A_to_B = tree_B.query(A)[0]
    
    tree_A = KDTree(A)
    dist_B_to_A = tree_A.query(B)[0]
    
    chamfer_dist = np.mean(dist_A_to_B) + np.mean(dist_B_to_A)
    return chamfer_dist

def emd_distance(A: np.ndarray, B: np.ndarray) -> float:
    num_points = min(len(A), len(B))
    d = cdist(A, B)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / num_points

def get_point_cloud_for_timestep(pointclouds_dir, t):
    pcd_path = os.path.join(pointclouds_dir, f"{t:05d}.npy")
    if not os.path.exists(pcd_path):
        return np.array([])
    return np.load(pcd_path)

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
        # trace_output_files = os.path.join(self.context.paths["trace_output_dir"], scene_dir_name, "output.pt")
        trace_output_files = os.path.join(self.context.paths["trace_output_dir"])
        trace_output_files = os.path.join(trace_output_files, scene_dir_name)
        num_frames = len([f for f in os.listdir(trace_output_files) if f.endswith('.npy')])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        reg_opts = init_opts(opts)
        reg_model = get_model(reg_opts)
        if reg_model is None:
            raise RuntimeError("Failed to load DiffusionReg model.")

        # trace_data = torch.load(trace_output_file, map_location=device)
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
        n_points_resample = 4096

        for i in range(num_frames - 1):
            t, t_plus_1 = i, i + 1
            
            # Skip registration at chunk boundaries
            if t_plus_1 in chunk_boundaries:
                print(f"\nSkipping frames {t} and {t_plus_1} (chunk boundary)...")
                skipped_boundaries.append((t, t_plus_1))
                continue
            
            # print(f"\nProcessing frames {t} and {t_plus_1}...")
            
            # aligned_masks, full_points = self._get_aligned_data_for_pair(trace_data, frame_paths, t, t_plus_1, device)
            
            # if aligned_masks.get(t) is None or aligned_masks.get(t_plus_1) is None: continue

            # pcd_t_full = full_points.get(t, np.array([]))
            # pcd_t_plus_1_full = full_points.get(t_plus_1, np.array([]))
            pcd_t_reg = get_point_cloud_for_timestep(trace_output_files, t)
            pcd_t_plus_1_reg = get_point_cloud_for_timestep(trace_output_files, t_plus_1)
            
            if pcd_t_reg.shape[0] == 0 or pcd_t_plus_1_reg.shape[0] == 0: continue

            # pcd_t_reg = resample_pcd(pcd_t_full, n_points_resample)
            # pcd_t_plus_1_reg = resample_pcd(pcd_t_plus_1_full, n_points_resample)
            
            transform_mat = register(reg_model, reg_opts, pcd_t_plus_1_reg[:,:3], pcd_t_reg[:,:3])
            
            transform_filename = os.path.join(self.context.paths["registration_output_dir"], f"transform_from_{t_plus_1}_to_{t}.npy")
            np.save(transform_filename, transform_mat)

            # Error calculation logic...
            self._calculate_and_log_error(pcd_t_reg, pcd_t_plus_1_reg, transform_mat, t, t_plus_1, registration_errors, device)
        
        self._save_error_summary(registration_errors, skipped_boundaries)

    def _calculate_and_log_error(self, pcd_t_reg, pcd_t_plus_1_reg, transform_mat, t, t_plus_1, registration_errors, device):
        pcd_t_plus_1_reg_pts = pcd_t_plus_1_reg[:, :3]
        p_t_plus_1_homo = np.hstack((pcd_t_plus_1_reg_pts, np.ones((pcd_t_plus_1_reg_pts.shape[0], 1))))
        transformed_points = (transform_mat @ p_t_plus_1_homo.T).T[:, :3]
        pcd_t_plus_1_reg[:, :3] = transformed_points
        # p_t_average = np.mean(pcd_t_reg, axis=0)
        # p_t_plus_1_average = np.mean(pcd_t_plus_1_reg, axis=0)
        # delta = np.linalg.norm(p_t_average - p_t_plus_1_average)

        pcd_t_reg_pts = pcd_t_reg[:, :3]
        pcd_t_plus_1_reg_pts = pcd_t_plus_1_reg[:, :3]
        # extract min, max for each dimenion
        min = np.min(np.stack((np.min(pcd_t_plus_1_reg_pts, axis=0), np.min(pcd_t_reg_pts, axis=0))), axis=0)
        max = np.max(np.stack((np.max(pcd_t_plus_1_reg_pts, axis=0), np.max(pcd_t_reg_pts, axis=0))), axis=0)
        pcd_t_reg_pts = (pcd_t_reg_pts - min) / (max - min)
        pcd_t_plus_1_reg_pts = (pcd_t_plus_1_reg_pts - min) / (max - min)

        # normalize color to 255.
        pcd_t_reg_color = pcd_t_reg[:, 3:] / 255.
        pcd_t_plus_1_reg_color = pcd_t_plus_1_reg[:, 3:] / 255.

        pcd_t_reg = np.hstack((pcd_t_reg_pts, pcd_t_reg_color))
        pcd_t_plus_1_reg = np.hstack((pcd_t_plus_1_reg_pts, pcd_t_plus_1_reg_color))

        absolute_error = chamfer_distance(pcd_t_reg, pcd_t_plus_1_reg)# / delta
        # emd_error = emd_distance(pcd_t_reg, transformed_points)

        registration_errors.append({
            "frame_pair": (t, t_plus_1),
            "absolute_error": float(absolute_error),
            # "emd_error": float(emd_error),
        })
        print(f"Absolute Registration Error E_{t}: {absolute_error:.6f}")
        # print(f"EMD Registration Error E_{t}: {emd_error:.6f}")

    def _save_error_summary(self, registration_errors, skipped_boundaries=None):
        if registration_errors:
            abs_errors = [e['absolute_error'] for e in registration_errors]
            # now remove outliers beyond 1.5 * IQR
            if False:
                Q1 = np.percentile(abs_errors, 25)
                Q3 = np.percentile(abs_errors, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                filtered_errors = [e for e in registration_errors if lower_bound <= e['absolute_error'] <= upper_bound]
                abs_errors = [e['absolute_error'] for e in filtered_errors]
            avg_absolute_error = np.mean(abs_errors)
            # avg_emd_error = np.mean([e['emd_error'] for e in registration_errors])
            error_summary = {
                "individual_errors": registration_errors,
                "average_absolute_error": float(avg_absolute_error),
#                 "average_emd_error": float(avg_emd_error),
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
