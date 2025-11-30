import argparse
import sys
import os
import json

# Add third-party directories to Python's path
sys.path.append(os.path.join(os.getcwd(), "third_party/DiffusionReg"))

# === Add imports here === # 
# To add a new component (e.g., a new registration method), just import it here.
# Most likely, only need to change out video model or point registration method.
from src.pipeline import (
    Pipeline, PipelineContext,
    Wan2_1VideoGenerator,
    SAM2Segmenter,
    TraceAnythingTracer,
    PointCloudExtraction,
    DiffusionReg
)


def create_pipeline(run_name: str, args: argparse.Namespace) -> Pipeline:
    """
    Creates and configures the pipeline by adding components in order.
    To swap out a component, change the class passed to `add_component`.
    """
    context = PipelineContext(run_name=run_name, args=args)
    pipeline = Pipeline(context)

    # --- Assemble the pipeline ---
    # To use a different implementation for a step, simply:
    # - create class in corresponding pipeline/*.py file abstracting from parent
    # - change the class here
    # - import at the top, at src.pipeline/__init__.py
    pipeline.add_component(Wan2_1VideoGenerator)
    pipeline.add_component(SAM2Segmenter)
    pipeline.add_component(TraceAnythingTracer)
    pipeline.add_component(PointCloudExtraction)
    pipeline.add_component(DiffusionReg)
    
    return pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the full 3D reconstruction pipeline.")
    parser.add_argument("-n", "--run_name", type=str, default="", help="A base name for this pipeline run. If a prompt config is used, this is a prefix. If the run exists, it will be reused.")
    parser.add_argument("--video_prompt", type=str, default=None, help="The prompt to generate the video. This overrides the prompt_path if provided.")
    parser.add_argument("--target_object", type=str, default=None, help="A short description of the object to be tracked.")
    parser.add_argument("--prompt_path", type=str, default="prompts/video_generation/prompts/sample_prompts.json", help="The path to the video generation config file to use (if not using CLI)")

    # for debugging: control which contiguous sequence of steps to run
    # these are: ["videogeneration", "segmentation", "tracing", "pointclouds", "registration"]
    parser.add_argument("--start-at", type=str, default="videogeneration", choices=["videogeneration", "segmentation", "tracing", "pointclouds", "registration"], help="The step to start from.")
    parser.add_argument("--end-at", type=str, default="registration", choices=["videogeneration", "segmentation", "tracing", "pointclouds", "registration"], help="The step to end at.")
    
    parser.add_argument("--box_threshold", type=float, default=0.4, help="The confidence threshold for bounding box detection.")
    parser.add_argument("--text_threshold", type=float, default=0.3, help="The confidence threshold for text matching.")
    parser.add_argument("--step_size", type=int, default=1, help="The number of frames between registered point clouds (point registrations).")
    
    args = parser.parse_args()

    runs = {}

    # Priority 1: A specific run name is given and it already exists.
    if args.run_name and os.path.exists(os.path.join("results", args.run_name)):
        print(f"Found existing run '{args.run_name}'. Loading its metadata to re-run parts of the pipeline.")
        metadata_path = os.path.join("results", args.run_name, "1_video", "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            runs = {
                args.run_name: {
                    "prompt": meta.get('prompt', ''),
                    "target_object": meta.get('target_object', '')
                }
            }
            print(f"  - Loaded target object: '{meta.get('target_object', '')}'")
        except FileNotFoundError:
            print(f"Warning: Could not find metadata.json at '{metadata_path}'. Proceeding without prompt/object info.")
            runs = { args.run_name: { "prompt": "", "target_object": "" } }

    # Priority 2: A specific prompt is given via CLI for a new run.
    elif args.video_prompt:
        if not args.run_name:
            print("Error: --run_name is required when providing --video_prompt for a new run.")
            return
        if not args.target_object:
            print("Error: --target_object is required when providing --video_prompt for a new run.")
            return
        runs = { args.run_name: { "prompt": args.video_prompt, "target_object": args.target_object } }

    # Priority 3: Use a prompt file to generate (potentially multiple) new runs.
    elif args.prompt_path:
        print(f"Loading runs from {args.prompt_path}...")
        try:
            with open(args.prompt_path, 'r') as f:
                prompt_runs = json.load(f)
            print(f"Found {len(prompt_runs)} runs in {args.prompt_path}:")
            for run_key in prompt_runs.keys():
                print(f"- {run_key}")
            
            if args.run_name: # Use as a prefix for all runs from the file
                for run_key, run_data in prompt_runs.items():
                    new_key = f"{args.run_name}_{run_key}"
                    runs[new_key] = run_data
            else: # Use keys from file directly
                runs = prompt_runs

        except Exception as e:
            print(f"Error: Could not load or parse prompt file at {args.prompt_path}: {e}")
            return
    
    if not runs:
        if args.run_name:
            # This case happens if a run_name was given, but it didn't exist and no other instructions were provided.
            # Assume it's a new run with no prompt (e.g., for custom video).
            print(f"Starting new run '{args.run_name}' with no prompt. Assumes video/frames exist.")
            runs = { args.run_name: { "prompt": "", "target_object": args.target_object or "" } }
        else:
            print("Error: No runs to process. Provide a --run_name, --video_prompt, or valid --prompt_path.")
            return

    # Create and run pipeline(s)
    for run_key, run_data in runs.items():
        print(f"\n--- Processing run: {run_key} ---")
        if run_data.get('prompt'):
            print(f"  - Prompt: {run_data['prompt']}")
        if run_data.get('target_object'):
            print(f"  - Target Object: {run_data['target_object']}")

        # Pass the specific run's data to the context
        args.video_prompt = run_data.get("prompt", "")
        args.target_object = run_data.get("target_object", "")

        # The pipeline is ALWAYS created with the final, resolved run_key
        pipeline = create_pipeline(run_key, args)
        
        start_component, end_component = get_partial_run_range_for_debug(pipeline, args)
        if start_component is None or end_component is None:
            continue
        
        print(f"Executing components: {pipeline.get_components_in_range(start_component, end_component)}")
        pipeline.run(start_at=start_component, end_at=end_component)


# === Helpers ==== #
def get_partial_run_range_for_debug(pipeline: Pipeline, args: argparse.Namespace) -> tuple[str, str]:
    all_component_short_names = [c.short_name for c in pipeline.components]
    
    start_step_arg = args.start_at.lower()
    end_step_arg = args.end_at.lower()

    start_component = next((name for name in all_component_short_names if name.startswith(start_step_arg)), None)
    end_component = next((name for name in all_component_short_names if name.startswith(end_step_arg)), None)
    
    if not start_component or not end_component:
        print(f"Error: Could not match provided start/end components ('{args.start_at}', '{args.end_at}') to available components.")
        print(f"Available names: {all_component_short_names}")
        return None, None
    return start_component, end_component


if __name__ == '__main__':
    main()

