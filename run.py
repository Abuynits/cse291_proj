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
    parser.add_argument("-n", "--run_name", type=str, default="", help="A base name for this pipeline run. If prompt config defines multiple runs, this will be used as a prefix.")
    parser.add_argument("--video_prompt", type=str, default=None, help="The prompt to generate the video. This overrides the prompt_path if provided.")
    parser.add_argument("--target_object", type=str, default=None, help="A short description of the object to be tracked.")
    parser.add_argument("--prompt_path", type=str, default="prompts/video_generation/prompts/sample_prompts.json", help="The path to the video generation config file to use (if not using CLI)")

    # for debugging: control which contiguous sequence of steps to run
    # these are: ["videogeneration", "segmentation", "tracing", "pointclouds", "registration"]
    parser.add_argument("--start-at", type=str, default="videogeneration", choices=["videogeneration", "segmentation", "tracing", "pointclouds", "registration"], help="The step to start from.")
    parser.add_argument("--end-at", type=str, default="registration", choices=["videogeneration", "segmentation", "tracing", "pointclouds", "registration"], help="The step to end at.")
    
    args = parser.parse_args()

    if args.video_prompt is not None:
        assert args.target_object is not None, "Error: Target object is required when video prompt is provided."

    runs = None
    if args.video_prompt is None and args.prompt_path:
        # if empty prompt, check the prompt path json file
        try:
            print(f"Loading runs from {args.prompt_path}...")
            runs = json.load(open(args.prompt_path))
            print(f"Found {len(runs)} runs in {args.prompt_path}:")
            for run_key in runs.keys():
                print(f"- {run_key}")
        except Exception as e:
            # if doesn't exist, try to see if the args.run_name output exists previously
            if os.path.exists(os.path.join("results", args.run_name)):
                metadata = os.path.join("results", args.run_name, "1_video", "metadata.json")
                with open(metadata, 'r') as f:
                    meta = json.load(f)
                # read prompt information from existing metadata
                runs = { args.run_name: { "prompt": meta['prompt'], "target_object": meta['target_object'] } }
    else:
        runs = { args.run_name: { "prompt": args.video_prompt, "target_object": args.target_object } }

    # create and run pipeline(s)
    for run_key, run_data in runs.items():
        print(f"\nGenerating run: {run_key}")
        print(f"  - Prompt: {run_data['prompt']}")
        print(f"  - Target Object: {run_data['target_object']}")

        args.video_prompt = run_data["prompt"]
        args.target_object = run_data["target_object"]

        pipeline = create_pipeline(run_key, args)
        start_component, end_component = get_partial_run_range_for_debug(pipeline, args)
        start_component = start_component if start_component is not None else "videogeneration"
        end_component = end_component if end_component is not None else "registration"
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

