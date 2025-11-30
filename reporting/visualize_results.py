#!/usr/bin/env python3
"""
One-click visualization of registration results from all runs in results/ directory.

Usage:
    python reporting/visualize_results.py
    python reporting/visualize_results.py --results_dir /path/to/results
    python reporting/visualize_results.py --output_dir reporting/plots
    python reporting/visualize_results.py --no-plots  # Only generate tables
    python reporting/visualize_results.py --prefixes bus car-roundabout  # Filter by prefixes
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


def extract_prefix(run_name: str) -> str:
    """
    Extract the prefix from a run name (everything before the last underscore and number).
    
    Examples:
        'bus_00' -> 'bus'
        'car-roundabout_03' -> 'car-roundabout'
        'tram_07' -> 'tram'
    """
    # Find the last underscore followed by digits
    parts = run_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return run_name  # Fallback to full name if pattern doesn't match


def group_metrics_by_prefix(all_metrics: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    """
    Group metrics by their prefix.
    
    Returns:
        Dict mapping prefix -> {run_name -> metrics}
    """
    grouped = {}
    for run_name, metrics in all_metrics.items():
        prefix = extract_prefix(run_name)
        if prefix not in grouped:
            grouped[prefix] = {}
        grouped[prefix][run_name] = metrics
    return grouped


def find_error_summary_files(results_dir: Path, prefixes: List[str] = None) -> Dict[str, Path]:
    """
    Scan results directory for all error_summary.json files.
    
    Args:
        results_dir: Path to results directory
        prefixes: Optional list of prefixes to filter by (e.g., ['bus', 'car-roundabout'])
                  If None, includes all runs.
    
    Returns:
        Dict mapping run_name to error_summary.json path
    """
    error_summaries = {}
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return error_summaries
    
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Filter by prefixes if specified
        if prefixes is not None:
            if not any(run_dir.name.startswith(prefix) for prefix in prefixes):
                continue
        
        # Check for standard pipeline output structure
        error_summary_path = run_dir / "6_registration" / "error_summary.json"
        if error_summary_path.exists():
            error_summaries[run_dir.name] = error_summary_path
        else:
            # Check for saved report structure (from --save-reports-to)
            error_summary_alt = run_dir / "error_summary.json"
            if error_summary_alt.exists():
                error_summaries[run_dir.name] = error_summary_alt
    
    return error_summaries


def extract_metrics(error_summary_path: Path) -> Dict:
    """Extract key metrics from error_summary.json."""
    try:
        with open(error_summary_path, 'r') as f:
            data = json.load(f)
        
        metrics = {
            'average_chamfer_distance': data.get('average_chamfer_distance', None),
            'average_absolute_error': data.get('average_absolute_error', None),
            'average_normalized_error': data.get('average_normalized_error', None),
            'average_velocity_normalized_l2': data.get('average_velocity_normalized_l2', None),
            'average_velocity_normalized_chamfer': data.get('average_velocity_normalized_chamfer', None),
            'average_displacement': data.get('average_displacement', None),
            'step_size': data.get('step_size', 1),
            'num_pairs': len(data.get('individual_errors', [])),
            'num_skipped': data.get('num_skipped_boundaries', 0),
        }
        
        # Extract per-frame errors for detailed plots
        individual_errors = data.get('individual_errors', [])
        metrics['chamfer_distances'] = [e['chamfer_distance'] for e in individual_errors]
        metrics['l2_absolute_errors'] = [e['absolute_error'] for e in individual_errors if e['absolute_error'] is not None]
        metrics['l2_normalized_errors'] = [e['normalized_error'] for e in individual_errors if e['normalized_error'] is not None]
        metrics['velocity_normalized_l2_errors'] = [e.get('velocity_normalized_l2') for e in individual_errors if e.get('velocity_normalized_l2') is not None]
        metrics['velocity_normalized_chamfers'] = [e.get('velocity_normalized_chamfer') for e in individual_errors if e.get('velocity_normalized_chamfer') is not None]
        metrics['displacements'] = [e.get('average_displacement') for e in individual_errors if e.get('average_displacement') is not None]
        
        return metrics
    except Exception as e:
        print(f"Warning: Failed to parse {error_summary_path}: {e}")
        return None


def create_summary_table(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Create a summary table DataFrame."""
    rows = []
    for run_name, metrics in sorted(all_metrics.items()):
        if metrics is None:
            continue
        
        row = {
            'Run': run_name,
            'Step Size': metrics.get('step_size', 1),
            'Avg Chamfer': f"{metrics['average_chamfer_distance']:.6f}" if metrics['average_chamfer_distance'] is not None else "N/A",
            'Vel Norm Chamfer': f"{metrics['average_velocity_normalized_chamfer']:.4f}" if metrics.get('average_velocity_normalized_chamfer') is not None else "N/A",
            'Avg L2 (Abs)': f"{metrics['average_absolute_error']:.6f}" if metrics['average_absolute_error'] is not None else "N/A",
            'Avg L2 (Norm)': f"{metrics['average_normalized_error']:.4f}" if metrics['average_normalized_error'] is not None else "N/A",
            'Vel Norm L2': f"{metrics['average_velocity_normalized_l2']:.4f}" if metrics.get('average_velocity_normalized_l2') is not None else "N/A",
            'Avg Displacement': f"{metrics['average_displacement']:.6f}" if metrics.get('average_displacement') is not None else "N/A",
            'Frame Pairs': metrics['num_pairs'],
            'Skipped': metrics['num_skipped']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_latex_table_by_prefix(all_metrics: Dict[str, Dict], output_path: Path = None) -> str:
    """
    Generate LaTeX tables grouped by prefix with separate rankings for each group.
    
    Args:
        all_metrics: Dictionary of run_name -> metrics
        output_path: Optional path to save the LaTeX file
        
    Returns:
        str: LaTeX table code
    """
    # Group metrics by prefix
    grouped_metrics = group_metrics_by_prefix(all_metrics)
    
    latex_lines = []
    
    # Generate a table for each prefix group
    for prefix_idx, (prefix, prefix_metrics) in enumerate(sorted(grouped_metrics.items())):
        # Filter valid metrics for this prefix
        valid_data = []
        for run_name, metrics in sorted(prefix_metrics.items()):
            if metrics is None:
                continue
            
            if (metrics['average_chamfer_distance'] is not None and 
                metrics['average_absolute_error'] is not None and 
                metrics['average_normalized_error'] is not None):
                valid_data.append({
                    'run': run_name,
                    'chamfer': metrics['average_chamfer_distance'],
                    'vel_norm_chamfer': metrics.get('average_velocity_normalized_chamfer'),
                    'l2_abs': metrics['average_absolute_error'],
                    'l2_norm': metrics['average_normalized_error'],
                    'vel_norm_l2': metrics.get('average_velocity_normalized_l2'),
                    'num_pairs': metrics['num_pairs'],
                    'step_size': metrics.get('step_size', 1)
                })
        
        if not valid_data:
            continue
        
        # Sort by different metrics to get rankings (within this prefix group)
        sorted_by_chamfer = sorted(valid_data, key=lambda x: x['chamfer'])
        sorted_by_vel_chamfer = sorted(
            [d for d in valid_data if d['vel_norm_chamfer'] is not None],
            key=lambda x: x['vel_norm_chamfer']
        )
        sorted_by_l2_norm = sorted(valid_data, key=lambda x: x['l2_norm'])
        sorted_by_vel_l2 = sorted(
            [d for d in valid_data if d['vel_norm_l2'] is not None],
            key=lambda x: x['vel_norm_l2']
        )
        
        # Create ranking dictionaries
        chamfer_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_chamfer)}
        vel_chamfer_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_vel_chamfer)}
        l2_norm_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_l2_norm)}
        vel_l2_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_vel_l2)}
        
        # Start table for this prefix
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append(f"\\caption{{Registration Error Comparison for {prefix.replace('_', ' ').replace('-', ' ').title()} Scenes (Lower is Better)}}")
        latex_lines.append(f"\\label{{tab:registration_errors_{prefix.replace('-', '_')}}}")
        latex_lines.append("\\begin{tabular}{lccccccccc}")
        latex_lines.append("\\toprule")
        latex_lines.append("\\textbf{Run} & \\textbf{Step} & \\multicolumn{2}{c}{\\textbf{Chamfer Distance}} & \\multicolumn{2}{c}{\\textbf{Vel. Norm. Chamfer}} & \\multicolumn{2}{c}{\\textbf{L2 Norm. Error}} & \\multicolumn{2}{c}{\\textbf{Vel. Norm. L2}} \\\\")
        latex_lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}")
        latex_lines.append(" &  & Value & Rank & Value & Rank & Value & Rank & Value & Rank \\\\")
        latex_lines.append("\\midrule")
        
        # Data rows (sorted alphabetically by run name)
        for item in sorted(valid_data, key=lambda x: x['run']):
            run_name = item['run']
            # Escape underscores for LaTeX
            latex_run_name = run_name.replace('_', '\\_')
            
            chamfer_val = item['chamfer']
            vel_chamfer_val = item['vel_norm_chamfer']
            l2_norm_val = item['l2_norm']
            vel_l2_val = item['vel_norm_l2']
            step_size = item['step_size']
            
            chamfer_rank = chamfer_ranks[run_name]
            l2_norm_rank = l2_norm_ranks[run_name]
            
            # Highlight best (rank 1) in bold
            chamfer_str = f"\\textbf{{{chamfer_val:.6f}}}" if chamfer_rank == 1 else f"{chamfer_val:.6f}"
            chamfer_rank_str = f"\\textbf{{{chamfer_rank}}}" if chamfer_rank == 1 else f"{chamfer_rank}"
            
            # Velocity normalized chamfer
            if vel_chamfer_val is not None and run_name in vel_chamfer_ranks:
                vel_chamfer_rank = vel_chamfer_ranks[run_name]
                vel_chamfer_str = f"\\textbf{{{vel_chamfer_val:.4f}}}" if vel_chamfer_rank == 1 else f"{vel_chamfer_val:.4f}"
                vel_chamfer_rank_str = f"\\textbf{{{vel_chamfer_rank}}}" if vel_chamfer_rank == 1 else f"{vel_chamfer_rank}"
            else:
                vel_chamfer_str = "N/A"
                vel_chamfer_rank_str = "-"
            
            l2_norm_str = f"\\textbf{{{l2_norm_val:.4f}}}" if l2_norm_rank == 1 else f"{l2_norm_val:.4f}"
            l2_norm_rank_str = f"\\textbf{{{l2_norm_rank}}}" if l2_norm_rank == 1 else f"{l2_norm_rank}"
            
            # Velocity normalized L2
            if vel_l2_val is not None and run_name in vel_l2_ranks:
                vel_l2_rank = vel_l2_ranks[run_name]
                vel_l2_str = f"\\textbf{{{vel_l2_val:.4f}}}" if vel_l2_rank == 1 else f"{vel_l2_val:.4f}"
                vel_l2_rank_str = f"\\textbf{{{vel_l2_rank}}}" if vel_l2_rank == 1 else f"{vel_l2_rank}"
            else:
                vel_l2_str = "N/A"
                vel_l2_rank_str = "-"
            
            row = f"{latex_run_name} & {step_size} & {chamfer_str} & {chamfer_rank_str} & {vel_chamfer_str} & {vel_chamfer_rank_str} & {l2_norm_str} & {l2_norm_rank_str} & {vel_l2_str} & {vel_l2_rank_str} \\\\"
            latex_lines.append(row)
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Add spacing between tables
        if prefix_idx < len(grouped_metrics) - 1:
            latex_lines.append("")
            latex_lines.append("")
    
    latex_code = "\n".join(latex_lines)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX tables (grouped by prefix) saved to: {output_path}")
    
    return latex_code


def generate_latex_table(all_metrics: Dict[str, Dict], output_path: Path = None) -> str:
    """
    Generate a single LaTeX table with global rankings across all runs.
    
    Args:
        all_metrics: Dictionary of run_name -> metrics
        output_path: Optional path to save the LaTeX file
        
    Returns:
        str: LaTeX table code
    """
    # Filter valid metrics
    valid_data = []
    for run_name, metrics in sorted(all_metrics.items()):
        if metrics is None:
            continue
        
        if (metrics['average_chamfer_distance'] is not None and 
            metrics['average_absolute_error'] is not None and 
            metrics['average_normalized_error'] is not None):
            valid_data.append({
                'run': run_name,
                'chamfer': metrics['average_chamfer_distance'],
                    'vel_norm_chamfer': metrics.get('average_velocity_normalized_chamfer'),
                'l2_abs': metrics['average_absolute_error'],
                'l2_norm': metrics['average_normalized_error'],
                'vel_norm_l2': metrics.get('average_velocity_normalized_l2'),
                'num_pairs': metrics['num_pairs'],
                'step_size': metrics.get('step_size', 1)
            })
    
    if not valid_data:
        return "% No valid data to generate table"
    
    # Sort by different metrics to get rankings
    sorted_by_chamfer = sorted(valid_data, key=lambda x: x['chamfer'])
    sorted_by_vel_chamfer = sorted(
        [d for d in valid_data if d['vel_norm_chamfer'] is not None],
        key=lambda x: x['vel_norm_chamfer']
    )
    sorted_by_l2_norm = sorted(valid_data, key=lambda x: x['l2_norm'])
    sorted_by_vel_l2 = sorted(
        [d for d in valid_data if d['vel_norm_l2'] is not None],
        key=lambda x: x['vel_norm_l2']
    )
    
    # Create ranking dictionaries
    chamfer_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_chamfer)}
    vel_chamfer_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_vel_chamfer)}
    l2_norm_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_l2_norm)}
    vel_l2_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_vel_l2)}
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Registration Error Comparison with Global Rankings (Lower is Better)}")
    latex_lines.append("\\label{tab:registration_errors_global}")
    latex_lines.append("\\begin{tabular}{lccccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Run} & \\textbf{Step} & \\multicolumn{2}{c}{\\textbf{Chamfer Distance}} & \\multicolumn{2}{c}{\\textbf{Vel. Norm. Chamfer}} & \\multicolumn{2}{c}{\\textbf{L2 Norm. Error}} & \\multicolumn{2}{c}{\\textbf{Vel. Norm. L2}} \\\\")
    latex_lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}")
    latex_lines.append(" &  & Value & Rank & Value & Rank & Value & Rank & Value & Rank \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows (sorted alphabetically by run name)
    for item in sorted(valid_data, key=lambda x: x['run']):
        run_name = item['run']
        # Escape underscores for LaTeX
        latex_run_name = run_name.replace('_', '\\_')
        
        chamfer_val = item['chamfer']
        vel_chamfer_val = item['vel_norm_chamfer']
        l2_norm_val = item['l2_norm']
        vel_l2_val = item['vel_norm_l2']
        step_size = item['step_size']
        
        chamfer_rank = chamfer_ranks[run_name]
        l2_norm_rank = l2_norm_ranks[run_name]
        
        # Highlight best (rank 1) in bold
        chamfer_str = f"\\textbf{{{chamfer_val:.6f}}}" if chamfer_rank == 1 else f"{chamfer_val:.6f}"
        chamfer_rank_str = f"\\textbf{{{chamfer_rank}}}" if chamfer_rank == 1 else f"{chamfer_rank}"
        
        # Velocity normalized chamfer
        if vel_chamfer_val is not None and run_name in vel_chamfer_ranks:
            vel_chamfer_rank = vel_chamfer_ranks[run_name]
            vel_chamfer_str = f"\\textbf{{{vel_chamfer_val:.4f}}}" if vel_chamfer_rank == 1 else f"{vel_chamfer_val:.4f}"
            vel_chamfer_rank_str = f"\\textbf{{{vel_chamfer_rank}}}" if vel_chamfer_rank == 1 else f"{vel_chamfer_rank}"
        else:
            vel_chamfer_str = "N/A"
            vel_chamfer_rank_str = "-"
        
        l2_norm_str = f"\\textbf{{{l2_norm_val:.4f}}}" if l2_norm_rank == 1 else f"{l2_norm_val:.4f}"
        l2_norm_rank_str = f"\\textbf{{{l2_norm_rank}}}" if l2_norm_rank == 1 else f"{l2_norm_rank}"
        
        # Velocity normalized L2
        if vel_l2_val is not None and run_name in vel_l2_ranks:
            vel_l2_rank = vel_l2_ranks[run_name]
            vel_l2_str = f"\\textbf{{{vel_l2_val:.4f}}}" if vel_l2_rank == 1 else f"{vel_l2_val:.4f}"
            vel_l2_rank_str = f"\\textbf{{{vel_l2_rank}}}" if vel_l2_rank == 1 else f"{vel_l2_rank}"
        else:
            vel_l2_str = "N/A"
            vel_l2_rank_str = "-"
        
        row = f"{latex_run_name} & {step_size} & {chamfer_str} & {chamfer_rank_str} & {vel_chamfer_str} & {vel_chamfer_rank_str} & {l2_norm_str} & {l2_norm_rank_str} & {vel_l2_str} & {vel_l2_rank_str} \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_code = "\n".join(latex_lines)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_code
    """
    Generate a LaTeX table with rankings for each metric.
    
    Args:
        all_metrics: Dictionary of run_name -> metrics
        output_path: Optional path to save the LaTeX file
        
    Returns:
        str: LaTeX table code
    """
    # Filter valid metrics
    valid_data = []
    for run_name, metrics in sorted(all_metrics.items()):
        if metrics is None:
            continue
        
        if (metrics['average_chamfer_distance'] is not None and 
            metrics['average_absolute_error'] is not None and 
            metrics['average_normalized_error'] is not None):
            valid_data.append({
                'run': run_name,
                'chamfer': metrics['average_chamfer_distance'],
                'l2_abs': metrics['average_absolute_error'],
                'l2_norm': metrics['average_normalized_error'],
                'num_pairs': metrics['num_pairs']
            })
    
    if not valid_data:
        return "% No valid data to generate table"
    
    # Sort by different metrics to get rankings
    # Lower is better for all metrics
    sorted_by_chamfer = sorted(valid_data, key=lambda x: x['chamfer'])
    sorted_by_l2_abs = sorted(valid_data, key=lambda x: x['l2_abs'])
    sorted_by_l2_norm = sorted(valid_data, key=lambda x: x['l2_norm'])
    
    # Create ranking dictionaries
    chamfer_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_chamfer)}
    l2_abs_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_l2_abs)}
    l2_norm_ranks = {item['run']: idx + 1 for idx, item in enumerate(sorted_by_l2_norm)}
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Registration Error Comparison with Rankings (Lower is Better)}")
    latex_lines.append("\\label{tab:registration_errors}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Run} & \\multicolumn{2}{c}{\\textbf{Chamfer Distance}} & \\multicolumn{2}{c}{\\textbf{L2 Absolute Error}} & \\multicolumn{2}{c}{\\textbf{L2 Normalized Error}} \\\\")
    latex_lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
    latex_lines.append(" & Value & Rank & Value & Rank & Value & Rank \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows (sorted alphabetically by run name)
    for item in sorted(valid_data, key=lambda x: x['run']):
        run_name = item['run']
        # Escape underscores for LaTeX
        latex_run_name = run_name.replace('_', '\\_')
        
        chamfer_val = item['chamfer']
        l2_abs_val = item['l2_abs']
        l2_norm_val = item['l2_norm']
        
        chamfer_rank = chamfer_ranks[run_name]
        l2_abs_rank = l2_abs_ranks[run_name]
        l2_norm_rank = l2_norm_ranks[run_name]
        
        # Highlight best (rank 1) in bold
        chamfer_str = f"\\textbf{{{chamfer_val:.6f}}}" if chamfer_rank == 1 else f"{chamfer_val:.6f}"
        chamfer_rank_str = f"\\textbf{{{chamfer_rank}}}" if chamfer_rank == 1 else f"{chamfer_rank}"
        
        l2_abs_str = f"\\textbf{{{l2_abs_val:.6f}}}" if l2_abs_rank == 1 else f"{l2_abs_val:.6f}"
        l2_abs_rank_str = f"\\textbf{{{l2_abs_rank}}}" if l2_abs_rank == 1 else f"{l2_abs_rank}"
        
        l2_norm_str = f"\\textbf{{{l2_norm_val:.4f}}}" if l2_norm_rank == 1 else f"{l2_norm_val:.4f}"
        l2_norm_rank_str = f"\\textbf{{{l2_norm_rank}}}" if l2_norm_rank == 1 else f"{l2_norm_rank}"
        
        row = f"{latex_run_name} & {chamfer_str} & {chamfer_rank_str} & {l2_abs_str} & {l2_abs_rank_str} & {l2_norm_str} & {l2_norm_rank_str} \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_code = "\n".join(latex_lines)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_code


def generate_ranked_summary_latex(all_metrics: Dict[str, Dict], output_path: Path = None) -> str:
    """
    Generate a compact LaTeX table showing top-3 rankings for each metric.
    """
    # Filter valid metrics
    valid_data = []
    for run_name, metrics in sorted(all_metrics.items()):
        if metrics is None:
            continue
        
        if (metrics['average_chamfer_distance'] is not None and 
            metrics['average_absolute_error'] is not None and 
            metrics['average_normalized_error'] is not None):
            valid_data.append({
                'run': run_name,
                'chamfer': metrics['average_chamfer_distance'],
                'l2_abs': metrics['average_absolute_error'],
                'l2_norm': metrics['average_normalized_error'],
            })
    
    if not valid_data:
        return "% No valid data to generate table"
    
    # Sort by different metrics
    sorted_by_chamfer = sorted(valid_data, key=lambda x: x['chamfer'])[:3]
    sorted_by_l2_abs = sorted(valid_data, key=lambda x: x['l2_abs'])[:3]
    sorted_by_l2_norm = sorted(valid_data, key=lambda x: x['l2_norm'])[:3]
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Top-3 Rankings by Metric (Lower is Better)}")
    latex_lines.append("\\label{tab:registration_top3}")
    latex_lines.append("\\begin{tabular}{clcclccl}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multicolumn{2}{c}{\\textbf{Chamfer Distance}} & \\phantom{a} & \\multicolumn{2}{c}{\\textbf{L2 Absolute Error}} & \\phantom{a} & \\multicolumn{2}{c}{\\textbf{L2 Normalized Error}} \\\\")
    latex_lines.append("\\cmidrule{1-2} \\cmidrule{4-5} \\cmidrule{7-8}")
    latex_lines.append("Rank & Run & & Rank & Run & & Rank & Run \\\\")
    latex_lines.append("\\midrule")
    
    for i in range(3):
        chamfer_run = sorted_by_chamfer[i]['run'].replace('_', '\\_') if i < len(sorted_by_chamfer) else "-"
        l2_abs_run = sorted_by_l2_abs[i]['run'].replace('_', '\\_') if i < len(sorted_by_l2_abs) else "-"
        l2_norm_run = sorted_by_l2_norm[i]['run'].replace('_', '\\_') if i < len(sorted_by_l2_norm) else "-"
        
        latex_lines.append(f"{i+1} & {chamfer_run} & & {i+1} & {l2_abs_run} & & {i+1} & {l2_norm_run} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_code = "\n".join(latex_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"Top-3 rankings table saved to: {output_path}")
    
    return latex_code


def plot_summary_bar_charts(all_metrics: Dict[str, Dict], output_dir: Path):
    """Create bar charts comparing metrics across all runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out None metrics
    valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}
    
    if not valid_metrics:
        print("No valid metrics to plot.")
        return
    
    run_names = list(valid_metrics.keys())
    
    # Extract metrics
    chamfer_dists = [m['average_chamfer_distance'] for m in valid_metrics.values() if m['average_chamfer_distance'] is not None]
    l2_abs = [m['average_absolute_error'] for m in valid_metrics.values() if m['average_absolute_error'] is not None]
    l2_norm = [m['average_normalized_error'] for m in valid_metrics.values() if m['average_normalized_error'] is not None]
    vel_norm_chamfer = [m.get('average_velocity_normalized_chamfer') for m in valid_metrics.values() if m.get('average_velocity_normalized_chamfer') is not None]
    vel_norm_l2 = [m.get('average_velocity_normalized_l2') for m in valid_metrics.values() if m.get('average_velocity_normalized_l2') is not None]
    
    # 1. Chamfer Distance Bar Chart
    if chamfer_dists:
        fig, ax = plt.subplots(figsize=(12, 6))
        valid_runs = [name for name, m in valid_metrics.items() if m['average_chamfer_distance'] is not None]
        ax.bar(range(len(valid_runs)), chamfer_dists, color='steelblue')
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('Average Chamfer Distance', fontsize=12)
        ax.set_title('Average Chamfer Distance Across Runs', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(valid_runs)))
        ax.set_xticklabels(valid_runs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'chamfer_distance_comparison.png', dpi=150)
        print(f"Saved: {output_dir / 'chamfer_distance_comparison.png'}")
        plt.close()
    
    # 2. Velocity Normalized Chamfer Bar Chart
    if vel_norm_chamfer:
        fig, ax = plt.subplots(figsize=(12, 6))
        valid_runs = [name for name, m in valid_metrics.items() if m.get('average_velocity_normalized_chamfer') is not None]
        ax.bar(range(len(valid_runs)), vel_norm_chamfer, color='darkblue')
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('Average Velocity Normalized Chamfer', fontsize=12)
        ax.set_title('Average Velocity Normalized Chamfer Distance Across Runs', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(valid_runs)))
        ax.set_xticklabels(valid_runs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_normalized_chamfer_comparison.png', dpi=150)
        print(f"Saved: {output_dir / 'velocity_normalized_chamfer_comparison.png'}")
        plt.close()
    
    # 3. L2 Absolute Error Bar Chart
    if l2_abs:
        fig, ax = plt.subplots(figsize=(12, 6))
        valid_runs = [name for name, m in valid_metrics.items() if m['average_absolute_error'] is not None]
        ax.bar(range(len(valid_runs)), l2_abs, color='coral')
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('Average L2 Absolute Error', fontsize=12)
        ax.set_title('Average L2 Absolute Error Across Runs', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(valid_runs)))
        ax.set_xticklabels(valid_runs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'l2_absolute_error_comparison.png', dpi=150)
        print(f"Saved: {output_dir / 'l2_absolute_error_comparison.png'}")
        plt.close()
    
    # 4. L2 Normalized Error Bar Chart
    if l2_norm:
        fig, ax = plt.subplots(figsize=(12, 6))
        valid_runs = [name for name, m in valid_metrics.items() if m['average_normalized_error'] is not None]
        ax.bar(range(len(valid_runs)), l2_norm, color='mediumseagreen')
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('Average L2 Normalized Error', fontsize=12)
        ax.set_title('Average L2 Normalized Error Across Runs', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(valid_runs)))
        ax.set_xticklabels(valid_runs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'l2_normalized_error_comparison.png', dpi=150)
        print(f"Saved: {output_dir / 'l2_normalized_error_comparison.png'}")
        plt.close()
    
    # 5. Velocity Normalized L2 Bar Chart
    if vel_norm_l2:
        fig, ax = plt.subplots(figsize=(12, 6))
        valid_runs = [name for name, m in valid_metrics.items() if m.get('average_velocity_normalized_l2') is not None]
        ax.bar(range(len(valid_runs)), vel_norm_l2, color='darkgreen')
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('Average Velocity Normalized L2 Error', fontsize=12)
        ax.set_title('Average Velocity Normalized L2 Error Across Runs', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(valid_runs)))
        ax.set_xticklabels(valid_runs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_normalized_l2_comparison.png', dpi=150)
        print(f"Saved: {output_dir / 'velocity_normalized_l2_comparison.png'}")
        plt.close()


def plot_per_run_distributions(all_metrics: Dict[str, Dict], output_dir: Path):
    """Create box plots showing distribution of errors across frame pairs for each run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_metrics = {k: v for k, v in all_metrics.items() if v is not None and len(v.get('chamfer_distances', [])) > 0}
    
    if not valid_metrics:
        return
    
    # Chamfer Distance Distribution
    fig, ax = plt.subplots(figsize=(14, 7))
    chamfer_data = [m['chamfer_distances'] for m in valid_metrics.values()]
    run_names = list(valid_metrics.keys())
    
    bp = ax.boxplot(chamfer_data, labels=run_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Chamfer Distance Distribution (Per Frame Pair)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(run_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'chamfer_distance_distribution.png', dpi=150)
    print(f"Saved: {output_dir / 'chamfer_distance_distribution.png'}")
    plt.close()
    
    # Velocity Normalized Chamfer Distribution
    valid_vel_chamfer = {k: v for k, v in valid_metrics.items() if len(v.get('velocity_normalized_chamfers', [])) > 0}
    if valid_vel_chamfer:
        fig, ax = plt.subplots(figsize=(14, 7))
        vel_chamfer_data = [m['velocity_normalized_chamfers'] for m in valid_vel_chamfer.values()]
        vel_chamfer_names = list(valid_vel_chamfer.keys())
        
        bp = ax.boxplot(vel_chamfer_data, labels=vel_chamfer_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightsteelblue')
        
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('Velocity Normalized Chamfer', fontsize=12)
        ax.set_title('Velocity Normalized Chamfer Distribution (Per Frame Pair)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(vel_chamfer_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_normalized_chamfer_distribution.png', dpi=150)
        print(f"Saved: {output_dir / 'velocity_normalized_chamfer_distribution.png'}")
        plt.close()
    
    # L2 Normalized Error Distribution
    valid_l2 = {k: v for k, v in valid_metrics.items() if len(v.get('l2_normalized_errors', [])) > 0}
    if valid_l2:
        fig, ax = plt.subplots(figsize=(14, 7))
        l2_data = [m['l2_normalized_errors'] for m in valid_l2.values()]
        l2_names = list(valid_l2.keys())
        
        bp = ax.boxplot(l2_data, labels=l2_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('L2 Normalized Error', fontsize=12)
        ax.set_title('L2 Normalized Error Distribution (Per Frame Pair)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(l2_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'l2_normalized_error_distribution.png', dpi=150)
        print(f"Saved: {output_dir / 'l2_normalized_error_distribution.png'}")
        plt.close()
    
    # Velocity Normalized L2 Distribution
    valid_vel_l2 = {k: v for k, v in valid_metrics.items() if len(v.get('velocity_normalized_l2_errors', [])) > 0}
    if valid_vel_l2:
        fig, ax = plt.subplots(figsize=(14, 7))
        vel_l2_data = [m['velocity_normalized_l2_errors'] for m in valid_vel_l2.values()]
        vel_l2_names = list(valid_vel_l2.keys())
        
        bp = ax.boxplot(vel_l2_data, labels=vel_l2_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('palegreen')
        
        ax.set_xlabel('Run', fontsize=12)
        ax.set_ylabel('Velocity Normalized L2 Error', fontsize=12)
        ax.set_title('Velocity Normalized L2 Error Distribution (Per Frame Pair)', fontsize=14, fontweight='bold')
        ax.set_xticklabels(vel_l2_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_normalized_l2_distribution.png', dpi=150)
        print(f"Saved: {output_dir / 'velocity_normalized_l2_distribution.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize registration results from all runs")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to results directory (default: results/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reporting/plots",
        help="Path to save plots and tables (default: reporting/plots/)"
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="+",
        default=None,
        help="Filter runs by prefix(es). Example: --prefixes bus car-roundabout"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only generate tables)"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("Registration Results Visualization")
    print("=" * 70)
    
    # Find all error summary files
    if args.prefixes:
        print(f"\nScanning {results_dir} for error_summary.json files with prefixes: {', '.join(args.prefixes)}...")
    else:
        print(f"\nScanning {results_dir} for error_summary.json files...")
    
    error_summaries = find_error_summary_files(results_dir, prefixes=args.prefixes)
    
    if not error_summaries:
        print("No error_summary.json files found. Make sure you've run registration.")
        return
    
    print(f"Found {len(error_summaries)} run(s) with registration results.\n")
    
    # Extract metrics from all runs
    all_metrics = {}
    # Extract metrics from all runs
    all_metrics = {}
    for run_name, error_path in error_summaries.items():
        metrics = extract_metrics(error_path)
        if metrics:
            all_metrics[run_name] = metrics
            print(f"✓ {run_name} (step_size={metrics.get('step_size', 1)})")
            print(f"    Chamfer: {metrics['average_chamfer_distance']:.6f}" if metrics['average_chamfer_distance'] else "    Chamfer: N/A")
            if metrics.get('average_velocity_normalized_chamfer') is not None:
                print(f"    Vel. Norm. Chamfer: {metrics['average_velocity_normalized_chamfer']:.4f}")
            print(f"    L2 (Abs): {metrics['average_absolute_error']:.6f}" if metrics['average_absolute_error'] else "    L2 (Abs): N/A")
            print(f"    L2 (Norm): {metrics['average_normalized_error']:.4f}" if metrics['average_normalized_error'] else "    L2 (Norm): N/A")
            if metrics.get('average_velocity_normalized_l2') is not None:
                print(f"    Vel. Norm. L2: {metrics['average_velocity_normalized_l2']:.4f}")
        else:
            print(f"✗ {run_name} - failed to parse")
    
    if not all_metrics:
        print("\nNo valid metrics extracted. Exiting.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary table
    print("\n" + "=" * 70)
    print("Summary Table (Grouped by Scene Type)")
    print("=" * 70)
    
    grouped_metrics = group_metrics_by_prefix(all_metrics)
    
    all_rows = []
    for prefix in sorted(grouped_metrics.keys()):
        print(f"\n--- {prefix.upper().replace('-', ' ').replace('_', ' ')} ---")
        prefix_metrics = grouped_metrics[prefix]
        
        # Create DataFrame for this prefix group
        rows = []
        for run_name, metrics in sorted(prefix_metrics.items()):
            if metrics is None:
                continue
            
            row = {
                'Run': run_name,
                'Step': metrics.get('step_size', 1),
                'Avg Chamfer': f"{metrics['average_chamfer_distance']:.6f}" if metrics['average_chamfer_distance'] is not None else "N/A",
                'Vel Norm Chamfer': f"{metrics.get('average_velocity_normalized_chamfer', 0):.4f}" if metrics.get('average_velocity_normalized_chamfer') is not None else "N/A",
                'Avg L2 (Abs)': f"{metrics['average_absolute_error']:.6f}" if metrics['average_absolute_error'] is not None else "N/A",
                'Avg L2 (Norm)': f"{metrics['average_normalized_error']:.4f}" if metrics['average_normalized_error'] is not None else "N/A",
                'Vel Norm L2': f"{metrics.get('average_velocity_normalized_l2', 0):.4f}" if metrics.get('average_velocity_normalized_l2') is not None else "N/A",
                'Frame Pairs': metrics['num_pairs'],
                'Skipped': metrics['num_skipped']
            }
            rows.append(row)
            all_rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
    
    # Save combined table to CSV
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        csv_path = output_dir / "registration_summary.csv"
        df_all.to_csv(csv_path, index=False)
        print(f"\nCombined summary table saved to: {csv_path}")
    
    # Generate LaTeX tables
    print("\n" + "=" * 70)
    print("Generating LaTeX Tables")
    print("=" * 70)
    
    # Grouped tables by prefix (separate rankings per scene type)
    latex_grouped_path = output_dir / "registration_errors_by_prefix.tex"
    latex_grouped = generate_latex_table_by_prefix(all_metrics, latex_grouped_path)
    print("\nLaTeX tables grouped by prefix (separate rankings):\n")
    print(latex_grouped)
    
    # Full table with global rankings (all runs combined)
    latex_full_path = output_dir / "registration_errors_ranked_global.tex"
    latex_full = generate_latex_table(all_metrics, latex_full_path)
    print("\nFull LaTeX table with global rankings:\n")
    print(latex_full)
    
    # Top-3 summary table
    latex_top3_path = output_dir / "registration_top3.tex"
    latex_top3 = generate_ranked_summary_latex(all_metrics, latex_top3_path)
    print("\nTop-3 Rankings LaTeX table:\n")
    print(latex_top3)
    
    # Generate plots (unless --no-plots flag is set)
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("Generating Plots")
        print("=" * 70)
        
        plot_summary_bar_charts(all_metrics, output_dir)
        plot_per_run_distributions(all_metrics, output_dir)
        
        print("\n" + "=" * 70)
        print(f"✓ All visualizations saved to: {output_dir}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"✓ Tables saved to: {output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()

