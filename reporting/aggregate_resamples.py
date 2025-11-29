import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

## use this to merge results into eval_results.json (using eval.py) 
## when you run samples and eval.py doesn't save them (or you stop them mid-run.)

# usage:
# # Aggregate a single subject
# python3 reporting/aggregate_resamples.py --subjects walking_robot_dog

# # Aggregate multiple subjects
# python3 reporting/aggregate_resamples.py --subjects walking_robot_dog hovering_drone

# # Create a new file instead of updating existing
# python3 reporting/aggregate_resamples.py --subjects walking_robot_dog --no-update

# # Custom output path
# python3 reporting/aggregate_resamples.py --subjects walking_robot_dog --output my_results.json


def find_resample_directories(results_dir: Path, subject_name: str) -> List[Path]:
    """
    Find all resample directories for a given subject.
    
    Args:
        results_dir: Path to results directory
        subject_name: Name of the subject (e.g., "walking_robot_dog")
        
    Returns:
        List of paths to resample directories, sorted by resample index
    """
    resample_dirs = []
    
    # Pattern: {subject_name}_resample_{index}
    pattern = f"{subject_name}_resample_"
    
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith(pattern):
            # Extract resample index
            suffix = item.name[len(pattern):]
            try:
                resample_idx = int(suffix)
                resample_dirs.append((resample_idx, item))
            except ValueError:
                continue
    
    # Sort by resample index
    resample_dirs.sort(key=lambda x: x[0])
    return [path for _, path in resample_dirs]


def extract_error_metrics(error_summary_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract error metrics from error_summary.json.
    
    Args:
        error_summary_path: Path to error_summary.json
        
    Returns:
        Dictionary with metrics, or None if file doesn't exist or is invalid
    """
    if not error_summary_path.exists():
        return None
    
    try:
        with open(error_summary_path, 'r') as f:
            data = json.load(f)
        
        metrics = {
            'average_normalized_error': data.get('average_normalized_error'),
            'average_absolute_error': data.get('average_absolute_error'),
            'num_frame_pairs': len(data.get('individual_errors', []))
        }
        
        return metrics
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse {error_summary_path}: {e}")
        return None


def aggregate_results(all_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across all runs for a subject.
    
    Args:
        all_runs: List of metrics dictionaries from each run
        
    Returns:
        dict: Aggregated statistics
    """
    if not all_runs:
        return None
    
    # Filter out None values
    valid_runs = [r for r in all_runs if r is not None]
    
    if not valid_runs:
        return None
    
    # Extract metrics
    normalized_errors = [r['average_normalized_error'] for r in valid_runs 
                        if r.get('average_normalized_error') is not None]
    absolute_errors = [r['average_absolute_error'] for r in valid_runs 
                      if r.get('average_absolute_error') is not None]
    
    aggregated = {
        'num_runs': len(valid_runs),
        'num_successful_runs': len(valid_runs),
        'runs': []
    }
    
    # Add individual run data
    for i, run in enumerate(valid_runs):
        aggregated['runs'].append({
            'run_index': i,
            'average_normalized_error': run.get('average_normalized_error'),
            'average_absolute_error': run.get('average_absolute_error'),
            'num_frame_pairs': run.get('num_frame_pairs')
        })
    
    # Calculate averages
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


def aggregate_subject_resamples(
    subject_name: str,
    results_dir: Path,
    eval_results_path: Path,
    update_existing: bool = True
) -> bool:
    """
    Aggregate resample results for a subject and update eval_results.json.
    
    Args:
        subject_name: Name of the subject
        results_dir: Path to results directory
        eval_results_path: Path to eval_results.json
        update_existing: If True, update existing eval_results.json; otherwise create new
        
    Returns:
        True if successful, False otherwise
    """
    # Find all resample directories
    resample_dirs = find_resample_directories(results_dir, subject_name)
    
    if not resample_dirs:
        print(f"No resample directories found for {subject_name}")
        return False
    
    print(f"Found {len(resample_dirs)} resample directories for {subject_name}")
    
    # Extract metrics from each resample
    all_runs = []
    for resample_dir in resample_dirs:
        error_summary_path = resample_dir / "6_registration" / "error_summary.json"
        metrics = extract_error_metrics(error_summary_path)
        if metrics:
            all_runs.append(metrics)
            print(f"  ✓ {resample_dir.name}: normalized_error = {metrics.get('average_normalized_error', 'N/A'):.4f}")
        else:
            print(f"  ✗ {resample_dir.name}: no valid error_summary.json found")
    
    if not all_runs:
        print(f"No valid metrics found for {subject_name}")
        return False
    
    # Aggregate results
    aggregated = aggregate_results(all_runs)
    if not aggregated:
        print(f"Failed to aggregate results for {subject_name}")
        return False
    
    # Load existing eval_results.json if it exists
    eval_data = {}
    if update_existing and eval_results_path.exists():
        try:
            with open(eval_results_path, 'r') as f:
                eval_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing {eval_results_path}: {e}")
            eval_data = {}
    
    # Update or add subject data
    eval_data[subject_name] = aggregated
    
    # Save updated eval_results.json
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_results_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"\n✓ Aggregated results for {subject_name}:")
    if 'average_normalized_error' in aggregated:
        err = aggregated['average_normalized_error']
        print(f"  Normalized Error: {err['mean']:.4f} ± {err['std']:.4f} (min: {err['min']:.4f}, max: {err['max']:.4f})")
    if 'average_absolute_error' in aggregated:
        err = aggregated['average_absolute_error']
        print(f"  Absolute Error: {err['mean']:.4f} ± {err['std']:.4f} (min: {err['min']:.4f}, max: {err['max']:.4f})")
    print(f"  Results saved to {eval_results_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate resample results from existing error_summary.json files"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        required=True,
        help="List of subject names to aggregate (e.g., walking_robot_dog)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reporting/eval_results.json",
        help="Path to output eval_results.json (default: reporting/eval_results.json)"
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Create new file instead of updating existing eval_results.json"
    )
    
    args = parser.parse_args()
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    
    eval_results_path = Path(args.output)
    if not eval_results_path.is_absolute():
        eval_results_path = project_root / eval_results_path
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return
    
    print(f"Results directory: {results_dir}")
    print(f"Output file: {eval_results_path}")
    print(f"Subjects to aggregate: {', '.join(args.subjects)}")
    print()
    
    # Aggregate each subject
    success_count = 0
    for subject_name in args.subjects:
        print(f"{'='*60}")
        success = aggregate_subject_resamples(
            subject_name=subject_name,
            results_dir=results_dir,
            eval_results_path=eval_results_path,
            update_existing=not args.no_update
        )
        if success:
            success_count += 1
        print()
    
    print(f"{'='*60}")
    print(f"Aggregation complete: {success_count}/{len(args.subjects)} subjects processed successfully")


if __name__ == "__main__":
    main()

