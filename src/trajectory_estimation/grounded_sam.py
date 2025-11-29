# give a prompt and a set of images, generates masks for the target object in each image
import os
import cv2
import numpy as np
import torch
from PIL import Image
import supervision as sv
from supervision.draw.color import ColorPalette
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class GroundedSamModel:
    def __init__(self, cfg, device, box_threshold=0.3, text_threshold=0.2):
        """
        Initialize Grounded SAM model.
        
        Args:
            cfg: Config object with sam2_model_id and model_id
            device: torch device
            box_threshold: Confidence threshold for bounding box detection (default: 0.4)
                          Lower values (e.g., 0.2-0.3) detect more objects but may include false positives
            text_threshold: Confidence threshold for text matching (default: 0.3)
                           Lower values make text matching more lenient
        """
        print(f"Initializing GroundedSamModel with cfg: {cfg.sam2_model_id}")
        print(f"Detection thresholds - box: {box_threshold}, text: {text_threshold}")
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(cfg.sam2_model_id)
        # build grounding dino from huggingface
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.device = device
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_id).to(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def get_grounded_dino_masks_for_views(self, views, target_object):
        mask_results = []
        for pil_image in views:
            self.sam2_predictor.set_image(pil_image)

            inputs = self.processor(images=pil_image, text=target_object, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[pil_image.size[::-1]]
            )
            if len(results[0]["labels"]) == 0:
                # no detections
                mask_results.append({
                    "masks": np.zeros((1, pil_image.height, pil_image.width), dtype=bool),
                    "scores": np.array([]),
                    "logits": np.array([]),
                    "org_image": pil_image.copy(),
                    "annotated_frame": pil_image.copy(),
                    "found_object": False
                })
                continue
            # get the box prompt for SAM 2
            input_boxes = results[0]["boxes"].cpu().numpy()

            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            class_names = results[0]["labels"]
            class_ids = np.array(list(range(len(class_names))))

            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=class_ids
            )

            mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)

            annotated_frame = mask_annotator.annotate(scene=pil_image.copy(), detections=detections)

            mask_results.append({
                "masks": masks,
                "scores": scores,
                "logits": logits,
                "annotated_frame": annotated_frame,
                "org_image": pil_image.copy(),
                "found_object": True
            })
        return mask_results
