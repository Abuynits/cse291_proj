import os
import json
from pathlib import Path
import argparse

# All subjects found in the JSON will be processed automatically


def extract_metrics_from_eval_results(eval_results_path, subject_name):
    """
    Extract metrics from eval_results.json file (new aggregated format).
    Supports both old format (subjects at top level) and new format (subjects under 'results' key).
    
    Args:
        eval_results_path: Path to eval_results.json (str or Path)
        subject_name: Name of the subject
        
    Returns:
        dict: Dictionary of metric names to statistics (mean, std, min, max), or None if not found
    """
    eval_results_path = Path(eval_results_path)
    if not eval_results_path.exists():
        return None
    
    try:
        with open(eval_results_path, 'r') as f:
            data = json.load(f)
        
        # Check if new format (has 'results' key) or old format
        if 'results' in data:
            # New format: subjects are under 'results' key
            results_data = data['results']
            if subject_name not in results_data:
                return None
            subject_data = results_data[subject_name]
        else:
            # Old format: subjects are at top level
            if subject_name not in data:
                return None
            subject_data = data[subject_name]
        
        metrics = {}
        
        # Extract normalized error statistics
        if 'average_normalized_error' in subject_data and isinstance(subject_data['average_normalized_error'], dict):
            norm_err = subject_data['average_normalized_error']
            metrics['normalized_error_mean'] = norm_err.get('mean')
            metrics['normalized_error_std'] = norm_err.get('std')
            metrics['normalized_error_min'] = norm_err.get('min')
            metrics['normalized_error_max'] = norm_err.get('max')
        
        # Extract absolute error statistics
        if 'average_absolute_error' in subject_data and isinstance(subject_data['average_absolute_error'], dict):
            abs_err = subject_data['average_absolute_error']
            metrics['absolute_error_mean'] = abs_err.get('mean')
            metrics['absolute_error_std'] = abs_err.get('std')
            metrics['absolute_error_min'] = abs_err.get('min')
            metrics['absolute_error_max'] = abs_err.get('max')
        
        return metrics if metrics else None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse {eval_results_path}: {e}")
        return None


def extract_metrics(error_summary_path):
    """
    Extract metrics from error_summary.json file.
    
    Args:
        error_summary_path: Path to error_summary.json
        
    Returns:
        dict: Dictionary of metric names to values, or None if file doesn't exist or is invalid
    """
    if not os.path.exists(error_summary_path):
        return None
    
    try:
        with open(error_summary_path, 'r') as f:
            data = json.load(f)
        
        metrics = {}
        
        # Extract average_normalized_error
        if 'average_normalized_error' in data:
            metrics['average_normalized_error'] = data['average_normalized_error']
        
        # TODO: Add more metrics here as needed
        # Example:
        # if 'average_absolute_error' in data:
        #     metrics['average_absolute_error'] = data['average_absolute_error']
        
        return metrics
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse {error_summary_path}: {e}")
        return None


def get_subject_directories(results_dir):
    """
    Get list of subject directories to process.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        list: Sorted list of directory names
    """
    # Include all directories
    return sorted([d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))])


def generate_latex_table(results_data, output_path=None, use_eval_format=False):
    """
    Generate LaTeX table from results data.
    
    Args:
        results_data: List of dicts with 'subject' and metrics
        output_path: Optional path to save LaTeX file
        use_eval_format: If True, format for eval_results.json (with mean/std/min/max)
        
    Returns:
        str: LaTeX table code
    """
    if not results_data:
        return "% No data to generate table"
    
    if use_eval_format:
        # Special formatting for eval_results.json with statistics
        # Group metrics by type (normalized_error, absolute_error) and show mean, std, min, max
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\begin{tabular}{lcccccccc}")
        latex_lines.append("\\toprule")
        
        # Header row with subheaders
        latex_lines.append("Subject & \\multicolumn{4}{c}{Normalized Error} & \\multicolumn{4}{c}{Absolute Error} \\\\")
        latex_lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
        latex_lines.append(" & Mean & Std & Min & Max & Mean & Std & Min & Max \\\\")
        latex_lines.append("\\midrule")
        
        # Data rows
        for entry in results_data:
            row = entry['subject'].replace('_', ' ').title()
            
            # Normalized error stats
            for stat in ['normalized_error_mean', 'normalized_error_std', 'normalized_error_min', 'normalized_error_max']:
                value = entry.get(stat)
                if isinstance(value, float):
                    row += f" & {value:.4f}"
                else:
                    row += " & N/A"
            
            # Absolute error stats
            for stat in ['absolute_error_mean', 'absolute_error_std', 'absolute_error_min', 'absolute_error_max']:
                value = entry.get(stat)
                if isinstance(value, float):
                    row += f" & {value:.4f}"
                else:
                    row += " & N/A"
            
            row += " \\\\"
            latex_lines.append(row)
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\caption{Registration Error Comparison (Aggregated Statistics)}")
        latex_lines.append("\\label{tab:registration_errors}")
        latex_lines.append("\\end{table}")
        
        latex_code = "\n".join(latex_lines)
    else:
        # Original format for single error_summary.json files
        # Get all metric names from first entry (assuming all have same metrics)
        metric_names = [k for k in results_data[0].keys() if k != 'subject']
        
        # Generate LaTeX table
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\begin{tabular}{l" + "c" * len(metric_names) + "}")
        latex_lines.append("\\toprule")
        
        # Header row
        header = "Subject"
        for metric in metric_names:
            # Format metric name for LaTeX (replace underscores, capitalize)
            metric_label = metric.replace('_', ' ').title()
            header += f" & {metric_label}"
        header += " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\midrule")
        
        # Data rows
        for entry in results_data:
            row = entry['subject'].replace('_', ' ').title()
            for metric in metric_names:
                value = entry.get(metric, 'N/A')
                if isinstance(value, float):
                    # Format float to 4 decimal places
                    row += f" & {value:.4f}"
                else:
                    row += f" & {value}"
            row += " \\\\"
            latex_lines.append(row)
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\caption{Registration Error Comparison}")
        latex_lines.append("\\label{tab:registration_errors}")
        latex_lines.append("\\end{table}")
        
        latex_code = "\n".join(latex_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX table saved to {output_path}")
    
    return latex_code


def main(eval_results_path):
    """Main function to generate report table."""
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "results"
    
    # Check if eval_results.json exists (new format)
    use_eval_format = Path(eval_results_path).exists()
    
    if use_eval_format:
        print(f"Using eval_results.json format from {eval_results_path}")
        
        # Load eval_results.json
        try:
            with open(eval_results_path, 'r') as f:
                eval_data = json.load(f)
        except Exception as e:
            print(f"Error: Failed to load {eval_results_path}: {e}")
            return
        
        # Check if new format (has 'results' key) or old format
        if 'results' in eval_data:
            # New format: subjects are under 'results' key
            results_data = eval_data['results']
            subjects_to_process = sorted(results_data.keys())
        else:
            # Old format: subjects are at top level
            subjects_to_process = sorted(eval_data.keys())
        
        if not subjects_to_process:
            print("No valid subjects found in eval_results.json.")
            return
        
        print(f"Processing {len(subjects_to_process)} subject(s)...")
        
        # Collect results data
        results_data = []
        for subject_name in subjects_to_process:
            metrics = extract_metrics_from_eval_results(eval_results_path, subject_name)
            if metrics is None:
                print(f"Warning: Skipping {subject_name} - no valid metrics found")
                continue
            
            entry = {'subject': subject_name, **metrics}
            results_data.append(entry)
            
            # Print summary
            norm_mean = metrics.get('normalized_error_mean', 'N/A')
            norm_std = metrics.get('normalized_error_std', 'N/A')
            norm_min = metrics.get('normalized_error_min', 'N/A')
            norm_max = metrics.get('normalized_error_max', 'N/A')
            if isinstance(norm_mean, float):
                print(f"  ✓ {subject_name}: normalized_error = {norm_mean:.4f} ± {norm_std:.4f} (min: {norm_min:.4f}, max: {norm_max:.4f})")
            else:
                print(f"  ✓ {subject_name}: normalized_error = N/A")
        
    else:
        # Original format: read from individual error_summary.json files
        print("Using individual error_summary.json files")
        
        if not results_dir.exists():
            print(f"Error: Results directory not found at {results_dir}")
            return
        
        # Get subject directories to process
        subject_dirs = get_subject_directories(results_dir)
        
        if not subject_dirs:
            print("No subject directories found to process.")
            return
        
        print(f"Processing {len(subject_dirs)} subject(s)...")
        
        # Collect results data
        results_data = []
        for subject_dir in subject_dirs:
            error_summary_path = results_dir / subject_dir / "6_registration" / "error_summary.json"
            
            metrics = extract_metrics(error_summary_path)
            if metrics is None:
                print(f"Warning: Skipping {subject_dir} - no valid error_summary.json found")
                continue
            
            entry = {'subject': subject_dir, **metrics}
            results_data.append(entry)
            print(f"  ✓ {subject_dir}: average_normalized_error = {metrics.get('average_normalized_error', 'N/A'):.4f}")
    
    if not results_data:
        print("No valid data found to generate table.")
        return
    
    # Generate LaTeX table
    output_path = script_dir / "registration_errors_table.tex"
    latex_code = generate_latex_table(results_data, output_path, use_eval_format=use_eval_format)
    
    print(f"\nGenerated LaTeX table with {len(results_data)} entries.")
    print(f"\nLaTeX code:\n")
    print(latex_code)


if __name__ == "__main__":
    # usage: python results2table.py --json_path reporting/reports/run123/eval_results.json
    parser = argparse.ArgumentParser(description="Generate LaTeX table from results data.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="reporting/eval_results.json",
        help="Path to eval_results.json"
    )
    args = parser.parse_args()
    main(args.json_path)

