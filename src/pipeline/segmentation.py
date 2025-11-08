import os
import cv2
import numpy as np
from PIL import Image
import torch

from .pipeline import PipelineComponent, PipelineContext
from src.trajectory_estimation.grounded_sam import GroundedSamModel

# A simple config class to hold model IDs, to avoid hydra conflicts
class GroundedSamConfig:
    sam2_model_id = "facebook/sam2-hiera-large"
    model_id = "IDEA-Research/grounding-dino-base"

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(output_folder, f"{frame_count:05d}.png"), frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")
    return [os.path.join(output_folder, f) for f in sorted(os.listdir(output_folder))]

def save_masks(mask_results, output_folder):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    mask_paths = []
    for i, result in enumerate(mask_results):
        if result["found_object"]:
            mask_img = Image.fromarray(result["masks"][0].astype(np.uint8) * 255)
            mask_img.save(os.path.join(output_folder, f"{i:05d}.png"))
            mask_paths.append(os.path.join(output_folder, f"{i:05d}.png"))
    print(f"Saved {len(mask_paths)} masks to {output_folder}")
    return mask_paths

class Segmentation(PipelineComponent):
    def __init__(self, context: PipelineContext):
        super().__init__(context)

    @property
    def short_name(self) -> str:
        return "segmentation"


class SAM2Segmenter(Segmentation):
    def run(self):
        video_path = self.context.paths["generated_video_path"]
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for segmentation: {video_path}.")

        frame_paths = video_to_frames(video_path, self.context.paths["frames_scene_dir"])
        pil_images = [Image.open(fp) for fp in frame_paths]

        sam_config = GroundedSamConfig()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        grounded_sam = GroundedSamModel(sam_config, device)

        mask_results = grounded_sam.get_grounded_dino_masks_for_views(pil_images, self.context.args.target_object)
        
        save_masks(mask_results, self.context.paths["masks_dir"])

        # Also save the annotated frames for visualization
        for i, result in enumerate(mask_results):
            if result["found_object"]:
                annotated_img_bgr = cv2.cvtColor(np.array(result["annotated_frame"]), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.context.paths["overlaid_masks_dir"], f"{i:05d}.png"), annotated_img_bgr)
        print(f"Saved overlaid masks to {self.context.paths['overlaid_masks_dir']}")

        # Store data for subsequent steps
        self.context.data['pil_images'] = pil_images
        self.context.data['grounded_sam'] = grounded_sam
        self.context.data['video_generator'] = self.context.data.get('video_generator')
