import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import hydra
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


class VideoGenerator:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.setup_model()

    def setup_model(self):
        self.vae = AutoencoderKLWan.from_pretrained(
            self.cfg.model.model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )

        resolution = self.cfg.video.quality.resolution
        flow_shift = self.cfg.video.quality.flow_shift[resolution]

        self.scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=flow_shift
        )

        self.pipe = WanPipeline.from_pretrained(
            self.cfg.model.model_id,
            vae=self.vae,
            torch_dtype=getattr(torch, self.cfg.model.torch_dtype),
        )
        self.pipe.scheduler = self.scheduler
        self.pipe.enable_sequential_cpu_offload()

    def load_prompts(self) -> Dict[str, str]:
        prompts_path = Path(self.cfg.paths.prompts_path)
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        return prompts
    
    def load_negative_prompt(self, negative_prompt_path: str) -> str:
        with open(negative_prompt_path, 'r') as f:
            negative_prompt = f.read().strip()
        return negative_prompt

    def generate_video(self, prompt: str, output_folder: Path) -> None:
        resolution = self.cfg.video.quality.resolution
        if resolution not in self.cfg.video.quality.dimensions:
            raise ValueError(f"Resolution '{resolution}' not found in configuration dimensions.")
        dimensions = self.cfg.video.quality.dimensions[resolution]

        negative_prompt = self.load_negative_prompt(self.cfg.model.negative_prompt_path)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=dimensions.height,
            width=dimensions.width,
            num_frames=self.cfg.video.num_frames,
            guidance_scale=self.cfg.video.guidance_scale,
        ).frames

        generated_video = output[0]

        output_folder.mkdir(parents=True, exist_ok=True)

        video_path = output_folder / "video.mp4"
        export_to_video(generated_video, str(video_path), fps=self.cfg.video.fps)

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_config": self.cfg
        }
        metadata_path = output_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Video and metadata saved to: {output_folder}")

    def generate_all_videos(self) -> None:
        prompts = self.load_prompts()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.cfg.paths.output_dir) / f"{self.cfg.paths.output_prefix}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for prompt_name, prompt in prompts.items():
            print(f"Generating video for prompt: {prompt_name}")
            video_folder = output_dir / prompt_name
            self.generate_video(prompt, video_folder)


@hydra.main()
def main(cfg: OmegaConf) -> None:
    OmegaConf.resolve(cfg)
    generator = VideoGenerator(cfg)
    generator.generate_all_videos()


if __name__ == "__main__":
    main()