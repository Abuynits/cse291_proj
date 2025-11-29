import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import argparse


def generate_and_save_mask(image_path: str, output_path: str, text_prompt: str, mask_index: int = 0):
    """
    Generate a mask for the given image using Grounding DINO + SAM2 with a text prompt.
    
    Args:
        image_path: Path to the input image
        output_path: Path where the mask will be saved (should end in .png)
        text_prompt: Text description of the object to segment (e.g., "bus", "person", "dog")
        mask_index: Which mask to save if multiple objects are detected (default: 0, the highest scoring)
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Ensure output has .png extension
    if output_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        output_path = output_path.with_suffix('.png')
        print(f"Added .png extension to output path: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = Image.open(image_path)
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Grounding DINO and SAM2 models on {device}...")
    
    sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    grounding_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    
    # Use Grounding DINO to detect object based on text prompt
    print(f"Detecting '{text_prompt}' in image...")
    sam2_predictor.set_image(image)
    
    inputs = grounding_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    
    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    if len(results[0]["labels"]) == 0:
        print(f"No objects matching '{text_prompt}' were detected in the image.")
        # Save an empty mask
        empty_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        cv2.imwrite(str(output_path), empty_mask)
        print(f"Saved empty mask to {output_path}")
        return None, 0.0
    
    # Get bounding boxes for detected objects
    input_boxes = results[0]["boxes"].cpu().numpy()
    labels = results[0]["labels"]
    print(f"Detected {len(labels)} object(s): {labels}")
    
    # Use SAM2 to generate masks from the bounding boxes
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    # Convert shape to (n, H, W) if needed
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    print(f"Generated {len(masks)} mask(s) with scores: {scores}")
    
    # Save the specified mask (or highest scoring one)
    if mask_index >= len(masks):
        print(f"Warning: mask_index {mask_index} out of range, using mask 0")
        mask_index = 0
    
    mask_to_save = masks[mask_index]
    mask_uint8 = (mask_to_save * 255).astype(np.uint8)
    
    cv2.imwrite(str(output_path), mask_uint8)
    print(f"Saved mask {mask_index} (label: '{labels[mask_index]}', score: {scores[mask_index]:.3f}) to {output_path}")
    
    return mask_to_save, scores[mask_index]


def main():
    parser = argparse.ArgumentParser(description="Generate and save a mask for an image using Grounding DINO + SAM2")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("output_path", type=str, help="Path where the mask will be saved")
    parser.add_argument("text_prompt", type=str, help="Text description of the object to segment (e.g., 'bus', 'person')")
    parser.add_argument("--mask_index", type=int, default=0, 
                        help="Which mask to save if multiple objects are detected (default: 0)")
    args = parser.parse_args()
    
    generate_and_save_mask(args.image_path, args.output_path, args.text_prompt, args.mask_index)


if __name__ == "__main__":
    main()