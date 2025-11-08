import os
from omegaconf import OmegaConf
from .pipeline import PipelineComponent, PipelineContext
from src.video_generation.generator import VideoGenerator

class VideoGeneration(PipelineComponent):
    def __init__(self, context: PipelineContext):
        super().__init__(context)

    @property
    def short_name(self) -> str:
        return "videogeneration"
        
class Wan2_1VideoGenerator(VideoGeneration):
    def run(self):
        print("--- Part 1: Generating Video ---")
        
        video_cfg = OmegaConf.load("config/video_generation/config.yaml")

        project_root = os.getcwd()
        video_cfg.model.negative_prompt_path = os.path.join(project_root, "prompts/video_generation/negative_prompt.txt")
        video_cfg.paths.prompts_path = None
        video_cfg.paths.output_dir = None

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        
        OmegaConf.resolve(video_cfg)

        prompt_data = {
            "prompt": self.context.args.video_prompt,
            "target_object": self.context.args.target_object
        }
        
        video_generator = VideoGenerator(video_cfg)
        video_generator.generate_video(prompt_data, self.context.paths["video_output_folder"])
        
        print(f"Video generated at: {self.context.paths['generated_video_path']}")
