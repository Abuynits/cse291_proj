import os
import numpy as np
import torch
from PIL import Image
from .pipeline import PipelineComponent, PipelineContext
from TraceAnything.trace_anything.trace_anything import evaluate_bspline_conf

# INPUT: output.py (from 4) and frames from (2)
# OUTPUT: pointclouds (npy) saved to self.context.paths["pointclouds_dir"]

def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    """A helper function to mimic the resizing in TraceAnything's infer.py"""
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)

# several methods to predict pointclouds
# using the simplest one for now: take the B-spline defined at frame 0
# and sample it at time t, optionally applying the frame-0 mask.
def predict_pointcloud_from_frame_0_to_t(frame0_preds, t, device, mask):
    """
    frame0_preds: Prediction for frame 0 (first frame of the sequence).
    t:            Scalar time or 0D/1D tensor indicating the target time along the spline.
    device:       CUDA/CPU.
    mask:         Mask at frame 0 that keeps only the object points.
    """
    # Prepare time tensor
    if isinstance(t, torch.Tensor):
        time_t = t.to(device).view(-1)
    else:
        time_t = torch.tensor([float(t)], device=device, dtype=torch.float32)

    # Control points & confidences (usually already torch tensors on CPU)
    ctrl_pts = frame0_preds["ctrl_pts3d"]
    ctrl_conf = frame0_preds["ctrl_conf"]
    if isinstance(ctrl_pts, np.ndarray):
        ctrl_pts = torch.from_numpy(ctrl_pts)
    if isinstance(ctrl_conf, np.ndarray):
        ctrl_conf = torch.from_numpy(ctrl_conf)
    ctrl_pts = ctrl_pts.to(device)
    ctrl_conf = ctrl_conf.to(device)

    # Evaluate B-spline at time t
    pts_full, _ = evaluate_bspline_conf(ctrl_pts, ctrl_conf, time_t)  # [1,H,W,3]
    pts_full = pts_full.squeeze(0)  # [H,W,3]

    # mask for object trajectories only
    pts_full_masked = pts_full[mask]
    if pts_full_masked.shape[0] > 0:
        return pts_full_masked
    else:
        return pts_full.reshape(-1, 3) # return all points, flattened

class PointCloudExtraction(PipelineComponent):
    def __init__(self, context: PipelineContext):
        super().__init__(context)

    @property
    def short_name(self) -> str:
        return "pointclouds"

    def run(self):
        scene_dir_name = f"{os.path.basename(self.context.run_name)}_scene"
        trace_output_file = os.path.join(self.context.paths["trace_output_dir"], scene_dir_name, "output.pt")
        
        if not os.path.exists(trace_output_file):
            raise FileNotFoundError(f"Trace Anything output file not found: {trace_output_file}.")
        if not os.path.exists(self.context.paths["masks_dir"]) or not os.listdir(self.context.paths["masks_dir"]):
             raise FileNotFoundError(f"Masks directory not found or is empty: {self.context.paths['masks_dir']}.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trace_data = torch.load(trace_output_file, map_location=device)
        
        frame_paths = sorted(
            [
                os.path.join(self.context.paths["frames_scene_dir"], f)
                for f in os.listdir(self.context.paths["frames_scene_dir"])
                if f.endswith('.png')
            ]
        )
        num_frames = len(trace_data['preds'])

        # build frame-0 mask using same logic as TraceAnything
        def _load_aligned_mask(frame_idx: int):
            """Load segmentation mask for frame_idx and resize/crop it like TraceAnything does."""
            mask_path = os.path.join(self.context.paths["masks_dir"], f"{frame_idx:05d}.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for frame {frame_idx}: {mask_path}.")

            mask_pil = Image.open(mask_path)
            original_frame_pil = Image.open(frame_paths[frame_idx])
            W0, H0 = original_frame_pil.size
            if H0 > W0:
                mask_pil = mask_pil.transpose(Image.Transpose.ROTATE_90)

            resized_mask_pil = _resize_long_side(mask_pil, 512)
            H, W = resized_mask_pil.size[1], resized_mask_pil.size[0]
            Ht, Wt = (H - H % 16, W - W % 16)
            aligned_mask_pil = resized_mask_pil.crop((0, 0, Wt, Ht))
            return np.array(aligned_mask_pil) > 0

        # NOTE: when chunking, this needs to be updated to the first frame of the chunk
        # a rough way to do this (as used here) is to check when time is close to 0.
        current_chunk = -1

        for i in range(num_frames):
            if trace_data['preds'][i]['chunk_idx'] > current_chunk:
                frame0_preds = trace_data['preds'][i] # update frame0 of the current chunk
                frame0_mask = _load_aligned_mask(i)
                current_chunk = trace_data['preds'][i]['chunk_idx']

            frame_i_time = trace_data['preds'][i]['time'] # get current frame's time

            # get pointcloud of time@frame_i w.r.t frame0's B-spline
            pts_predicted = predict_pointcloud_from_frame_0_to_t(frame0_preds, frame_i_time, device, mask=frame0_mask)
            if pts_predicted.shape[0] > 0:
                predicted_filename = os.path.join(self.context.paths["pointclouds_dir"], f"pointcloud_{i:05d}.npy")
                np.save(predicted_filename, pts_predicted.cpu().numpy())
                print(f"  - Saved point cloud for frame {i}")

        print(f"\nExtracted actual and predicted point clouds to {self.context.paths['pointclouds_dir']}")
