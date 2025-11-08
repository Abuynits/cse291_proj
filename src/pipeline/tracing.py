import os
import subprocess
import gc
import torch
from .pipeline import PipelineComponent, PipelineContext

class PointTracer(PipelineComponent):
    def __init__(self, context: PipelineContext):
        super().__init__(context)

    @property
    def short_name(self) -> str:
        return "tracer"

class TraceAnythingTracer(PointTracer):
    def run(self):
        print("--- Preparing for Tracing: Releasing VRAM from previous steps ---")

        # Offload and delete models from previous steps
        if self.context.data.get('video_generator'):
            print("Deleting video generation model...")
            del self.context.data['video_generator']
        
        if self.context.data.get('grounded_sam'):
            print("Offloading segmentation model...")
            del self.context.data['grounded_sam']

        # Explicitly clear the cache
        print("Clearing CUDA cache...")
        gc.collect()
        torch.cuda.empty_cache()

        print("\n--- Running Trace Anything ---")
        frames_base_dir = os.path.dirname(self.context.paths["frames_scene_dir"])
        if not os.path.exists(frames_base_dir) or not os.listdir(frames_base_dir):
            raise FileNotFoundError(f"Frames directory not found or is empty: {frames_base_dir}.")

        trace_anything_script = "tests/test_trace_anything.py"
        
        my_env = os.environ.copy()
        project_root = os.getcwd()
        python_path = my_env.get("PYTHONPATH", "")
        my_env["PYTHONPATH"] = f"{project_root}:{python_path}"

        cmd = [
            "python", trace_anything_script,
            "--input_dir", frames_base_dir,
            "--output_dir", self.context.paths["trace_output_dir"],
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=my_env)
