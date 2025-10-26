import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from enum import Enum

class VideoQuality(Enum):
    P480 = (480, 848)
    P720 = (720, 1280)

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
video_quality = VideoQuality.P480

prompt = "A sleek silver sports car driving smoothly through a minimal, softly lit tunnel environment. Static camera with cinematic composition. The car leaves subtle glowing trajectory trails showing its motion path. Photorealistic lighting, reflections on the car's surface, volumetric light rays, and 8K high detail."

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

# Removed device_map="balanced"
pipe = WanPipeline.from_pretrained(
    model_id, 
    vae=vae, 
    torch_dtype=torch.bfloat16,
)
pipe.scheduler = scheduler

# Enable sequential CPU offload for memory management
pipe.enable_sequential_cpu_offload()

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=video_quality.value[0],
    width=video_quality.value[1],
    num_frames=61,  # num_frames -1 needs to be divisible by 4
    guidance_scale=5.0,
).frames

breakpoint()
generated_video = output[0]
export_to_video(generated_video, "output.mp4", fps=16)