import os
import json
import numpy as np
import torch
from .pipeline import PipelineComponent, PipelineContext
from third_party.DiffusionReg.register_pointclouds import init_opts, get_model, register, resample_pcd
from third_party.DiffusionReg.utils.options import opts
from scipy.spatial import cKDTree

def apply_transformation(points, transform):
    """Helper to apply a 4x4 transformation matrix to a point cloud."""
    p_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    p_transformed = (transform @ p_homo.T).T[:, :3]
    return p_transformed

def chamfer_distance(A, B):
    """
    Calculates the Chamfer distance between two point clouds.
    The distance is symmetrical and handles unordered point sets of different sizes.

    Args:
        A: First point cloud (N, 3).
        B: Second point cloud (M, 3).

    Returns:
        float: chamfer distance
    """
    # Create KD-Trees for fast nearest-neighbor lookup
    kdtree_B = cKDTree(B)
    kdtree_A = cKDTree(A)
    
    # Find the nearest neighbor in B for each point in A
    d1, _ = kdtree_B.query(A, k=1)
    
    # Find the nearest neighbor in A for each point in B
    d2, _ = kdtree_A.query(B, k=1)
    
    # The Chamfer distance is the sum of the mean squared distances
    return np.mean(d1**2) + np.mean(d2**2)


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
        
        # Check that pointclouds exist
        if not os.path.exists(self.context.paths["pointclouds_dir"]) or not os.listdir(self.context.paths["pointclouds_dir"]):
            raise FileNotFoundError(f"Pointclouds directory not found or is empty: {self.context.paths['pointclouds_dir']}")

        # Load trace data to get chunk boundaries
        scene_dir_name = f"{os.path.basename(self.context.run_name)}_scene"
        trace_output_file = os.path.join(self.context.paths["trace_output_dir"], scene_dir_name, "output.pt")
        if not os.path.exists(trace_output_file):
            raise FileNotFoundError(f"Trace Anything output file not found: {trace_output_file}")
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trace_data = torch.load(trace_output_file, map_location=device)

        reg_opts = init_opts(opts)
        reg_model = get_model(reg_opts)
        if reg_model is None:
            raise RuntimeError("Failed to load DiffusionReg model.")

        pointcloud_files = sorted([f for f in os.listdir(self.context.paths["pointclouds_dir"]) if f.endswith('.npy')])
        num_frames = len(pointcloud_files)
        
        if num_frames < 2:
            print("Need at least two frames for registration. Skipping.")
            return

        # Infer chunk boundaries from trace data to avoid registering across discontinuous coordinate frames
        chunk_boundaries = set()
        for i in range(1, len(trace_data['preds'])):
            # based on how the pointcloud extraction step is done, this should never happen
            # cross-chunk-boundary pointclouds are removed.
            if trace_data['preds'][i].get('chunk_idx', 0) > trace_data['preds'][i-1].get('chunk_idx', 0):
                chunk_boundaries.add(i)
        
        if chunk_boundaries:
            print(f"Detected chunk boundaries at frames: {sorted(chunk_boundaries)}. Skipping registration for these pairs.")

        registration_errors = []
        skipped_boundaries = []
        n_points_resample = 1024

        for i in range(num_frames - 1):
            t, t_plus_1 = i, i + 1

            if t_plus_1 in chunk_boundaries:
                print(f"\nSkipping frames {t} and {t_plus_1} (chunk boundary)...")
                skipped_boundaries.append((t, t_plus_1))
                continue
            
            print(f"\nProcessing frames {t} and {t_plus_1}...")

            # Load the saved pointclouds
            pcd_t_path = os.path.join(self.context.paths["pointclouds_dir"], f"pointcloud_{t:05d}.npy")
            pcd_t1_path = os.path.join(self.context.paths["pointclouds_dir"], f"pointcloud_{t_plus_1:05d}.npy")
            
            if not os.path.exists(pcd_t_path) or not os.path.exists(pcd_t1_path):
                print(f"  - Warning: Pointcloud files not found for frames {t} and/or {t_plus_1}. Skipping.")
                continue

            pcd_t_full = np.load(pcd_t_path)
            pcd_t1_full = np.load(pcd_t1_path)

            if pcd_t_full.shape[0] < n_points_resample or pcd_t1_full.shape[0] < n_points_resample:
                print(f"  - Skipping due to insufficient points for resampling ({pcd_t_full.shape[0]} or {pcd_t1_full.shape[0]} < {n_points_resample}).")
                continue
            
            # Resample pointclouds ONLY for registration (DiffusionReg requires fixed size inputs)
            pcd_t_reg = resample_pcd(pcd_t_full, n_points_resample)
            pcd_t1_reg = resample_pcd(pcd_t1_full, n_points_resample)
            
            # Register: find transformation from t+1 to t using resampled clouds
            transform_mat = register(reg_model, reg_opts, pcd_t1_reg, pcd_t_reg)
            
            transform_filename = os.path.join(self.context.paths["registration_output_dir"], f"transform_from_{t_plus_1}_to_{t}.npy")
            np.save(transform_filename, transform_mat)

            # --- Error Calculation using FULL point clouds ---
            # Transform t+1 to t's coordinate frame
            pcd_t1_full_transformed = apply_transformation(pcd_t1_full, transform_mat)
            
            # compute Chamfer distance between original t and transformed t+1 (full clouds)
            chamfer_dist = chamfer_distance(pcd_t_full, pcd_t1_full_transformed)
            
            # compute L2 error
            if pcd_t_full.shape[0] == pcd_t1_full_transformed.shape[0]:
                distances = np.linalg.norm(pcd_t_full - pcd_t1_full_transformed, axis=1)
                absolute_error = np.mean(distances)
                
                min_coords = np.min(pcd_t_full, axis=0)
                max_coords = np.max(pcd_t_full, axis=0)
                bounding_box_diagonal = np.linalg.norm(max_coords - min_coords)
                normalized_error = absolute_error / bounding_box_diagonal if bounding_box_diagonal > 1e-6 else absolute_error
            else:
                # If sizes don't match, we can't compute point-to-point L2
                # since all trajectories derive from frame0, should not reach this part
                print(f"  - Note: Point counts differ ({pcd_t_full.shape[0]} vs {pcd_t1_full_transformed.shape[0]}), skipping L2 correspondence error.")
                absolute_error = -1.0
                normalized_error = -1.0
                bounding_box_diagonal = -1.0

            error_metrics = {
                "frame_pair": (t, t_plus_1),
                "absolute_error": float(absolute_error),
                "normalized_error": float(normalized_error),
                "bounding_box_diagonal": float(bounding_box_diagonal),
                "chamfer_distance": float(chamfer_dist)
            }
            
            print(f"  - Chamfer Distance: {chamfer_dist:.6f}")
            if absolute_error != -1.0:
                print(f"  - L2 Correspondence Error: {absolute_error:.6f} (Normalized: {normalized_error:.4f})")
            
            registration_errors.append(error_metrics)
        
        self._save_error_summary(registration_errors, skipped_boundaries)


    def _save_error_summary(self, registration_errors, skipped_boundaries=None):
        if registration_errors:
            # Filter out unsuccessful L2 error calculations if they occurred
            valid_l2_errors = [e for e in registration_errors if e['absolute_error'] != -1.0]
            avg_absolute_error = np.mean([e['absolute_error'] for e in valid_l2_errors]) if valid_l2_errors else -1.0
            avg_normalized_error = np.mean([e['normalized_error'] for e in valid_l2_errors]) if valid_l2_errors else -1.0
            
            avg_chamfer_dist = np.mean([e['chamfer_distance'] for e in registration_errors])
            
            error_summary = {
                "individual_errors": registration_errors,
                "average_absolute_error": float(avg_absolute_error),
                "average_normalized_error": float(avg_normalized_error),
                "average_chamfer_distance": float(avg_chamfer_dist),
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
        print(f"\nRegistration complete.")
        print(f"  - Average L2 Error: {error_summary.get('average_absolute_error', 'N/A'):.6f}")
        print(f"  - Average Chamfer Distance: {error_summary.get('average_chamfer_distance', 'N/A'):.6f}")
        if skipped_boundaries:
            print(f"Skipped {len(skipped_boundaries)} frame pairs at inferred chunk boundaries.")
        print(f"Error summary saved to {summary_path}")
