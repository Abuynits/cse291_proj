import argparse
import sys
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add third-party directories to Python's path
sys.path.append(os.path.join(os.getcwd(), "third_party/DiffusionReg"))

# Import pipeline components
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
    """
    context = PipelineContext(run_name=run_name, args=args)
    pipeline = Pipeline(context)

    pipeline.add_component(Wan2_1VideoGenerator)
    pipeline.add_component(SAM2Segmenter)
    pipeline.add_component(TraceAnythingTracer)
    pipeline.add_component(PointCloudExtraction)
    pipeline.add_component(DiffusionReg)
    
    return pipeline


def load_full_error_summary(error_summary_path: Path) -> Dict[str, Any]:
    """
    Loads the full error_summary.json from registration output.
    Returns None on error or missing file.
    """
    if not error_summary_path.exists():
        return None
    try:
        with open(error_summary_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to parse {error_summary_path}: {e}")
        return None


def extract_error_metrics(error_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract high-level error metrics for eval reporting (not the full error_summary, just summary numbers).
    """
    if error_summary is None:
        return None
    return {
        'average_normalized_error': error_summary.get('average_normalized_error'),
        'average_absolute_error': error_summary.get('average_absolute_error'),
        'num_frame_pairs': len(error_summary.get('individual_errors', []))
    }


def run_single_resample(
    subject_name: str,
    prompt: str,
    target_object: str,
    resample_idx: int,
    results_dir: Path,
    pipeline_args: argparse.Namespace
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run a single resample of the pipeline for a subject.
    Cleans up the entire run directory once finished but returns the error_summary.
    
    Returns:
        Tuple: (metrics, full_error_summary_json_dict)
    """
    run_name = f"{subject_name}_resample_{resample_idx}"
    print(f"\n{'='*60}")
    print(f"Running resample {resample_idx} for {subject_name}")
    print(f"{'='*60}")

    run_args = argparse.Namespace()
    run_args.video_prompt = prompt
    run_args.target_object = target_object
    run_args.start_at = pipeline_args.start_at
    run_args.end_at = pipeline_args.end_at

    run_dir = results_dir / run_name

    try:
        pipeline = create_pipeline(run_name, run_args)
        start_component = "videogeneration"
        end_component = "registration"
        pipeline.run(start_at=start_component, end_at=end_component)
        
        # Extract full error summary
        error_summary_path = run_dir / "6_registration" / "error_summary.json"
        error_summary = load_full_error_summary(error_summary_path)
        metrics = extract_error_metrics(error_summary)
        if metrics is None or error_summary is None:
            print(f"Warning: No valid error summary found for {run_name}")
            # Remove the run dir utterly
            if run_dir.exists():
                shutil.rmtree(run_dir)
            return None, None
        
        print(f"\nRemoving temporary results for {run_name}...")
        # Remove the whole run directory; everything is stored in error_summary now
        try:
            shutil.rmtree(run_dir)
        except Exception as e:
            print(f"Warning: Could not cleanup {run_dir}: {e}")
        
        print(f"✓ Resample {resample_idx} completed and cleaned up successfully")
        return metrics, error_summary
    except Exception as e:
        print(f"Error running resample {resample_idx} for {subject_name}: {e}")
        import traceback
        traceback.print_exc()
        # Remove possible run_dir on failure
        if run_dir.exists():
            shutil.rmtree(run_dir)
        return None, None


def aggregate_results(all_runs_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across all runs for a subject (metrics part only).
    """
    if not all_runs_metrics:
        return None
    valid_runs = [r for r in all_runs_metrics if r is not None]
    if not valid_runs:
        return None

    normalized_errors = [r['average_normalized_error'] for r in valid_runs if r.get('average_normalized_error') is not None]
    absolute_errors = [r['average_absolute_error'] for r in valid_runs if r.get('average_absolute_error') is not None]
    aggregated = {
        'num_runs': len(valid_runs),
        'num_successful_runs': len(valid_runs),
        'runs': []
    }
    for i, run in enumerate(valid_runs):
        aggregated['runs'].append({
            'run_index': i,
            'average_normalized_error': run.get('average_normalized_error'),
            'average_absolute_error': run.get('average_absolute_error'),
            'num_frame_pairs': run.get('num_frame_pairs')
        })
    if normalized_errors:
        mean_norm = sum(normalized_errors) / len(normalized_errors)
        aggregated['average_normalized_error'] = {
            'mean': mean_norm,
            'min': min(normalized_errors),
            'max': max(normalized_errors),
            'std': (sum((x - mean_norm)**2 for x in normalized_errors) / len(normalized_errors))**0.5 if len(normalized_errors) > 1 else 0.0
        }
    if absolute_errors:
        mean_abs = sum(absolute_errors) / len(absolute_errors)
        aggregated['average_absolute_error'] = {
            'mean': mean_abs,
            'min': min(absolute_errors),
            'max': max(absolute_errors),
            'std': (sum((x - mean_abs)**2 for x in absolute_errors) / len(absolute_errors))**0.5 if len(absolute_errors) > 1 else 0.0
        }
    return aggregated


def combine_error_summaries(error_summary_dicts: List[Dict[str, Any]], scene_names: List[str]) -> Dict[str, Any]:
    """
    Combine error_summary.jsons across samples into a single error_summary.json.
    Adds a 'samples' key, which is a list of error_summary.json as produced for each run.
    Renames 'individual_errors' to 'errors' in each sample.
    Computes aggregate statistics across all samples.
    """
    combined = {}
    samples_key = []
    all_errors_for_stats = []
    bounding_box_diagonals = []
    
    # Process each error summary
    for i, es in enumerate(error_summary_dicts):
        if es is not None:
            # Create a copy and rename 'individual_errors' to 'errors'
            sample_record = dict(es)  # shallow copy
            if 'individual_errors' in sample_record:
                sample_record['errors'] = sample_record.pop('individual_errors')
            samples_key.append(sample_record)
            
            # Collect errors for aggregate statistics
            errors = sample_record.get('errors', [])
            if isinstance(errors, list):
                all_errors_for_stats.extend(errors)
                bounding_box_diagonals.extend([
                    err['bounding_box_diagonal']
                    for err in errors if 'bounding_box_diagonal' in err
                ])

    # If nothing to report:
    if not samples_key:
        combined["samples"] = []
        combined["num_samples"] = 0
        combined["scene_names"] = []
        return combined

    # Compute aggregate statistics across all samples
    if all_errors_for_stats:
        total_abs = [err['absolute_error'] for err in all_errors_for_stats if 'absolute_error' in err]
        total_norm = [err['normalized_error'] for err in all_errors_for_stats if 'normalized_error' in err]

        if total_abs:
            combined['average_absolute_error'] = sum(total_abs) / len(total_abs)
        if total_norm:
            combined['average_normalized_error'] = sum(total_norm) / len(total_norm)
        if bounding_box_diagonals:
            combined['average_bounding_box_diagonal'] = sum(bounding_box_diagonals) / len(bounding_box_diagonals)

    combined['num_samples'] = len(samples_key)
    combined['samples'] = samples_key
    combined['scene_names'] = scene_names[:len(samples_key)]  # Match length to actual samples
    return combined


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with multiple resamples for subjects.")
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="List of subject names to evaluate. If not provided, uses subjects from prompt file."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="Number of times to sample (run) each subject (default: 3)"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/video_generation/prompts/sample_prompts.json",
        help="Path to the prompt configuration file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="reporting/eval_results.json",
        help="Path to save aggregated evaluation results"
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default="videogeneration",
        choices=["videogeneration", "segmentation", "tracing", "pointclouds", "registration"],
        help="Pipeline step to start from"
    )
    parser.add_argument(
        "--end-at",
        type=str,
        default="registration",
        choices=["videogeneration", "segmentation", "tracing", "pointclouds", "registration"],
        help="Pipeline step to end at"
    )
    
    args = parser.parse_args()
    
    # Load prompts
    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        print(f"Error: Prompt file not found at {prompt_path}")
        return
    
    with open(prompt_path, 'r') as f:
        all_prompts = json.load(f)
    
    # Determine subjects to evaluate
    if args.subjects:
        subjects_to_evaluate = args.subjects
        # Verify all subjects exist in prompt file
        missing = [s for s in subjects_to_evaluate if s not in all_prompts]
        if missing:
            print(f"Warning: Subjects not found in prompt file: {missing}")
            subjects_to_evaluate = [s for s in subjects_to_evaluate if s in all_prompts]
    else:
        subjects_to_evaluate = list(all_prompts.keys())
    
    if not subjects_to_evaluate:
        print("No valid subjects to evaluate.")
        return
    
    print(f"Evaluating {len(subjects_to_evaluate)} subject(s) with {args.n_samples} sample(s) each")
    print(f"Subjects: {', '.join(subjects_to_evaluate)}")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create a datetime-based run directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_timestamp}"
    reports_dir = Path("reporting/reports") / run_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Run directory: {reports_dir}")
    
    # Run evaluation for each subject
    all_results = {}
    
    for subject_name in subjects_to_evaluate:
        if subject_name not in all_prompts:
            print(f"Skipping {subject_name}: not found in prompt file")
            continue

        prompt_data = all_prompts[subject_name]
        prompt = prompt_data['prompt']
        target_object = prompt_data['target_object']
        
        print(f"\n{'#'*60}")
        print(f"Evaluating subject: {subject_name}")
        print(f"  Prompt: {prompt}")
        print(f"  Target Object: {target_object}")
        print(f"  Samples: {args.n_samples}")
        print(f"{'#'*60}")
        
        # Run all resamples for this subject; collect both run metrics and full error_summary.jsons
        subject_runs_metrics = []
        subject_error_summaries = []
        scene_names = []
        for sample_idx in range(args.n_samples):
            run_name = f"{subject_name}_resample_{sample_idx}"
            metrics, error_summary_json = run_single_resample(
                subject_name=subject_name,
                prompt=prompt,
                target_object=target_object,
                resample_idx=sample_idx,
                results_dir=results_dir,
                pipeline_args=args
            )
            subject_runs_metrics.append(metrics)
            subject_error_summaries.append(error_summary_json)
            if error_summary_json is not None:
                scene_names.append(run_name)
            else:
                scene_names.append(None)  # Keep alignment even if sample failed
        
        # Aggregate results for this subject (for eval_results)
        aggregated = aggregate_results(subject_runs_metrics)
        if aggregated:
            all_results[subject_name] = aggregated
            print(f"\n{subject_name} - Summary:")
            print(f"  Successful runs: {aggregated['num_successful_runs']}/{args.n_samples}")
            if 'average_normalized_error' in aggregated:
                err = aggregated['average_normalized_error']
                print(f"  Average Normalized Error: {err['mean']:.4f} (std: {err['std']:.4f}, min: {err['min']:.4f}, max: {err['max']:.4f})")
        else:
            print(f"Warning: No valid results for {subject_name}")

        # --- Save Combined error_summary.json under reporting/reports/<run_name>/<subject_name>/error_summary.json
        subject_report_dir = reports_dir / subject_name
        subject_report_dir.mkdir(parents=True, exist_ok=True)
        # Filter scene names to match only successful samples (where error_summary is not None)
        valid_scene_names = [scene_names[i] for i, es in enumerate(subject_error_summaries) if es is not None and scene_names[i] is not None]
        combined_error_summary = combine_error_summaries(subject_error_summaries, valid_scene_names)
        error_summary_save_path = subject_report_dir / "error_summary.json"
        with open(error_summary_save_path, "w") as f:
            json.dump(combined_error_summary, f, indent=2)
        print(f"Saved combined error summary for {subject_name} to {error_summary_save_path}")
    
    # Save aggregated results inside the run directory
    # Note: For this run, we don't merge with existing - each run is independent
    output_path = reports_dir / "eval_results.json"
    
    # Add metadata to the results
    results_with_metadata = {
        "run_timestamp": run_timestamp,
        "num_subjects": len(all_results),
        "subjects": list(all_results.keys()),
        "results": all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Run directory: {reports_dir}")
    print(f"Results saved to: {output_path}")
    print(f"  Total subjects: {len(all_results)}")
    print(f"  Subjects: {', '.join(all_results.keys())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
