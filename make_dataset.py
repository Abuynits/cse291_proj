import os
import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from enum import Enum
import json
from tqdm import tqdm

class VideoQuality(Enum):
    P480 = (480, 848)
    P720 = (720, 1280)

# Configuration
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
video_quality = VideoQuality.P480

# Number of videos per object
NUM_VIDEOS_PER_OBJECT = 10

# Batch size for video generation (adjust based on GPU memory)
# Larger batches = faster but more memory usage
BATCH_SIZE = 2  # Start with 4, increase if you have more GPU memory

# Read prompts from a text file with structure: <object>:<prompt w/ object>
PROMPT_TXT_PATH = "prompts/prompts4data.txt"
if not os.path.exists(PROMPT_TXT_PATH):
    raise FileNotFoundError(f"{PROMPT_TXT_PATH} not found. Please provide a prompt file.")

def read_object_prompts(txt_path):
    object_prompt_list = []
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip() or line.strip().startswith("#"):
                continue
            if ":" not in line:
                print(f"Line does not contain ':', skipping: {line.strip()}")
                continue
            obj, prompt = line.strip().split(":", 1)
            obj = obj.strip()
            prompt = prompt.strip()
            if not obj or not prompt:
                print(f"Invalid <object>:<prompt>, skipping: {line.strip()}")
                continue
            object_prompt_list.append((obj, prompt))
    return object_prompt_list

object_prompts = read_object_prompts(PROMPT_TXT_PATH)
if len(object_prompts) == 0:
    raise ValueError("No valid prompts found in the prompt file.")

negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, moving camera, focused,"
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)

# Output dataset folder
OUTPUT_DIR = "video_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize model components only once
vae = AutoencoderKLWan.from_pretrained(
    model_id, 
    subfolder="vae", 
    torch_dtype=torch.float32
)

if video_quality == VideoQuality.P480:
    flow_shift = 3.0  # 3.0 for 480P
else: 
    flow_shift = 5.0  # 5.0 for 720P

scheduler = UniPCMultistepScheduler(
    prediction_type='flow_prediction', 
    use_flow_sigmas=True, 
    num_train_timesteps=1000, 
    flow_shift=flow_shift
)

pipe = WanPipeline.from_pretrained(
    model_id, 
    vae=vae, 
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = scheduler
pipe.enable_sequential_cpu_offload()

dataset_metadata = []

for obj_idx, (obj_name, prompt) in enumerate(object_prompts):
    object_dir = os.path.join(OUTPUT_DIR, obj_name)
    os.makedirs(object_dir, exist_ok=True)
    
    # Generate videos in batches
    total_batches = (NUM_VIDEOS_PER_OBJECT + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    pbar = tqdm(range(total_batches), desc=f"Object '{obj_name}' ({obj_idx+1}/{len(object_prompts)})")
    
    for batch_idx in pbar:
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, NUM_VIDEOS_PER_OBJECT)
        batch_size_actual = batch_end - batch_start
        
        # Create batch of prompts (same prompt repeated for this object)
        batch_prompts = [prompt] * batch_size_actual
        
        try:
            # Generate batch of videos
            output = pipe(
                prompt=batch_prompts,
                negative_prompt=negative_prompt,  # Single negative prompt applies to all
                height=video_quality.value[0],
                width=video_quality.value[1],
                num_frames=61,  # num_frames -1 needs to be divisible by 4
                guidance_scale=5.0,
            ).frames

            # Process each video in the batch
            for i, video_tensor in enumerate(output):
                vid_num = batch_start + i
                video_filename = f"video_{vid_num:02d}.mp4"
                video_path = os.path.join(object_dir, video_filename)
                export_to_video(video_tensor, video_path, fps=16)

                # Save metadata for each video
                metadata = {
                    'object_id': obj_idx,
                    'object': obj_name,
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'video_filename': video_filename,
                    'video_path': os.path.relpath(video_path, OUTPUT_DIR),
                }
                dataset_metadata.append(metadata)
                
            pbar.set_postfix({'generated': f'{batch_end}/{NUM_VIDEOS_PER_OBJECT}'})

        except Exception as e:
            print(f"Failed to generate batch {batch_idx} for object '{obj_name}' (index {obj_idx}): {e}")
            # Try generating videos one by one as fallback
            print(f"Falling back to individual generation for remaining videos...")
            for vid_num in range(batch_start, batch_end):
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=video_quality.value[0],
                        width=video_quality.value[1],
                        num_frames=61,
                        guidance_scale=5.0,
                    ).frames
                    
                    video_tensor = output[0]
                    video_filename = f"video_{vid_num:02d}.mp4"
                    video_path = os.path.join(object_dir, video_filename)
                    export_to_video(video_tensor, video_path, fps=16)
                    
                    metadata = {
                        'object_id': obj_idx,
                        'object': obj_name,
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'video_filename': video_filename,
                        'video_path': os.path.relpath(video_path, OUTPUT_DIR),
                    }
                    dataset_metadata.append(metadata)
                except Exception as e2:
                    print(f"Failed to generate video {vid_num} for object '{obj_name}': {e2}")

# Write global metadata
metadata_json = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_json, "w") as f:
    json.dump(dataset_metadata, f, indent=2)

print(f"Dataset generation complete. Videos and metadata available in '{OUTPUT_DIR}'.")

