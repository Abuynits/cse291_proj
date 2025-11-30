import os
import numpy as np
import torch
from PIL import Image
from .pipeline import PipelineComponent, PipelineContext

# INPUT: output.py (from 4) and frames from (2)
# OUTPUT: pointclouds (npy) saved to self.context.paths["pointclouds_dir"]

def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    """A helper function to mimic the resizing in TraceAnything's infer.py"""
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)

class PointCloudExtraction(PipelineComponent):
    def __init__(self, context: PipelineContext):
        super().__init__(context)

    @property
    def short_name(self) -> str:
        return "pointclouds"

    def run(self):
        if False:
            scene_dir_name = f"{self.context.run_name}_scene"
            trace_output_file = os.path.join(self.context.paths["trace_output_dir"], scene_dir_name, "output.pt")
        trace_output_file = os.path.join(self.context.paths["trace_output_dir"], 'output.pt')
        
        if not os.path.exists(trace_output_file):
            raise FileNotFoundError(f"Trace Anything output file not found: {trace_output_file}.")
        if not os.path.exists(self.context.paths["masks_dir"]) or not os.listdir(self.context.paths["masks_dir"]):
             raise FileNotFoundError(f"Masks directory not found or is empty: {self.context.paths['masks_dir']}.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trace_data = torch.load(trace_output_file, map_location=device)
        
        frame_paths = sorted([os.path.join(self.context.paths["frames_scene_dir"], f) for f in os.listdir(self.context.paths["frames_scene_dir"]) if f.endswith('.png')])
        
        num_frames = len(trace_data['preds'])
        for i in range(num_frames):
            ctrl_pts_3d = trace_data['preds'][i]['ctrl_pts3d']
            
            mask_path = os.path.join(self.context.paths["masks_dir"], f"{i:05d}.png")
            if not os.path.exists(mask_path):
                continue
            
            # use our own masks from SAM2 rather than TraceAnything Otsu
            gtsam_mask_pil = Image.open(mask_path)
            original_frame_pil = Image.open(frame_paths[i])

            W0, H0 = original_frame_pil.size
            if H0 > W0:
                gtsam_mask_pil = gtsam_mask_pil.transpose(Image.Transpose.ROTATE_90)
                original_frame_pil = original_frame_pil.transpose(Image.Transpose.ROTATE_90)

            resized_mask_pil = _resize_long_side(gtsam_mask_pil, 512)
            resized_frame_pil = _resize_long_side(original_frame_pil, 512)

            H, W = resized_frame_pil.size[1], resized_frame_pil.size[0]
            Ht, Wt = (H - H % 16, W - W % 16)
            
            aligned_mask_pil = resized_mask_pil.crop((0, 0, Wt, Ht))
            
            mask_np = np.array(aligned_mask_pil) > 0
            
            if isinstance(ctrl_pts_3d, np.ndarray):
                ctrl_pts_3d = torch.from_numpy(ctrl_pts_3d).to(device)

            points_for_frame = ctrl_pts_3d.permute(1, 2, 0, 3) # H, W, K, 3
            object_points = points_for_frame[mask_np]
            
            if object_points.shape[0] > 0:
                object_points_k0 = object_points[:, 0, :]
                pc_filename = os.path.join(self.context.paths["pointclouds_dir"], f"pointcloud_{i:05d}.npy")
                np.save(pc_filename, object_points_k0.cpu().numpy())

        print(f"Extracted point clouds to {self.context.paths['pointclouds_dir']}")
