import os
import sys
import argparse
import shutil
import json
from pathlib import Path

# Add project root to sys.path to allow importing from src
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)
# Add third-party directories for imports within the pipeline
sys.path.append(os.path.join(project_root, "third_party/DiffusionReg"))

from run import create_pipeline, get_partial_run_range_for_debug

def process_video(video_path: Path, target_object: str, start_at: str, end_at: str, step_size: int = 1, 
                  report_dir: Path = None):
    """
    Sets up the directory for an existing video and runs the pipeline on it.
    
    Args:
        video_path: Path to the video file
        target_object: Object to track in the video
        start_at: Pipeline component to start from
        end_at: Pipeline component to end at
        step_size: Frame step size for registration (default=1)
        report_dir: Optional directory to save error summaries for permanent storage
    
    Returns:
        Tuple[str, bool]: (run_name, success) - run name and whether processing succeeded
    """
    try:
        # 1. Determine run_name and target_object
        video_stem = video_path.stem
        subject_name = video_path.parent.name
        run_name = f"{subject_name}_{video_stem}"
        
        if not target_object:
            # Infer from the folder structure or prompt file
            prompt_path = video_path.parent / "prompt.txt"
            if prompt_path.exists():
                # Simple heuristic: take the first noun phrase from the prompt
                prompt_text = prompt_path.read_text().strip()
                # This is a placeholder; more advanced NLP could be used.
                # For now, using the parent directory name is more reliable.
                target_object = subject_name
            else:
                target_object = subject_name
        
        print(f"\n--- Preparing to process video: {video_path.name} ---")
        print(f"  - Run Name: {run_name}")
        print(f"  - Target Object: {target_object}")

        # 2. Create the necessary directory structure
        run_dir = Path("results") / run_name
        video_dir = run_dir / "1_video"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Copy the video file
        dest_video_path = video_dir / "video.mp4"
        print(f"  - Copying video to: {dest_video_path}")
        shutil.copy2(video_path, dest_video_path)

        # 4. Create metadata.json for consistency
        metadata = {
            "source_video": str(video_path.resolve()),
            "prompt": "",
            "target_object": target_object
        }
        with open(video_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 5. Create mock args and run the pipeline
        args = argparse.Namespace(
            run_name=run_name,
            video_prompt="",
            target_object=target_object,
            start_at=start_at,
            end_at=end_at,
            box_threshold=0.1,  # Default threshold, can be overridden
            text_threshold=0.1,  # Default threshold, can be overridden
            step_size=step_size
        )
        
        pipeline = create_pipeline(run_name, args)
        start_component, end_component = get_partial_run_range_for_debug(pipeline, args)
        if start_component is None or end_component is None:
            print(f"Error determining run range for {run_name}. Skipping.")
            return
            
        print(f"  - Starting pipeline from '{start_component}' to '{end_component}'...")
        pipeline.run(start_at=start_component, end_at=end_component)
        print(f"--- Finished processing {run_name} ---")
        
        # Save error summary to report directory if specified
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)
            error_summary_src = run_dir / "6_registration" / "error_summary.json"
            
            if error_summary_src.exists():
                # Create a subdirectory for this run's report
                run_report_dir = report_dir / run_name
                run_report_dir.mkdir(parents=True, exist_ok=True)
                
                error_summary_dst = run_report_dir / "error_summary.json"
                shutil.copy2(error_summary_src, error_summary_dst)
                print(f"  - Saved error summary to: {error_summary_dst}")
                
                # Also save metadata for context
                metadata_dst = run_report_dir / "metadata.json"
                run_metadata = {
                    "source_video": str(video_path.resolve()),
                    "target_object": target_object,
                    "step_size": step_size,
                    "run_name": run_name,
                    "subject_name": subject_name
                }
                with open(metadata_dst, 'w') as f:
                    json.dump(run_metadata, f, indent=2)
            else:
                print(f"  - Warning: No error summary found at {error_summary_src}")
        
        return run_name, True
        
    except Exception as e:
        print(f"\n!!! ERROR processing {video_path.name}: {str(e)} !!!")
        print(f"Continuing with remaining videos...\n")
        return None, False


def main():
    parser = argparse.ArgumentParser(description="Process existing videos through the pipeline.")
    parser.add_argument("input_path", type=str, help="Path to a video file or a directory of videos.")
    parser.add_argument("--target_object", type=str, default=None, help="The object to track. If not provided, it's inferred from the directory structure.")
    parser.add_argument("--recursive", action="store_true", help="Search for videos recursively in the input directory.")
    parser.add_argument("--start-at", type=str, default="segmentation", help="The pipeline step to start from.")
    parser.add_argument("--end-at", type=str, default="registration", help="The pipeline step to end at.")
    parser.add_argument("--step-size", type=int, default=1, help="Frame step size for registration (default=1)")
    parser.add_argument("--include", type=str, nargs="+", help="Only process videos whose path contains ANY of these substrings (e.g., --include scene1 scene3)")
    parser.add_argument("--exclude", type=str, nargs="+", help="Exclude videos whose path contains ANY of these substrings (e.g., --exclude test debug)")
    parser.add_argument("--pattern", type=str, help="Only process videos whose filename matches this pattern (supports wildcards, e.g., 'video_0*.mp4')")
    parser.add_argument("--list-only", action="store_true", help="List videos that would be processed without actually processing them")
    parser.add_argument("--save-reports-to", type=str, default=None, help="Directory to save error summaries for permanent storage (e.g., 'reporting/batch_results')")
    parser.add_argument("--cleanup", action="store_true", help="Remove run directories after saving error summaries (saves disk space)")
    parser.add_argument("--batch-by-subject", action="store_true", help="Process videos by subject, cleaning up after each subject (recommended for disk space)")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    video_files = []

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    if input_path.is_dir():
        print(f"Searching for videos in directory: {input_path}")
        glob_pattern = "**/*.mp4" if args.recursive else "*.mp4"
        
        # Apply pattern matching if specified
        if args.pattern:
            glob_pattern = f"**/{args.pattern}" if args.recursive else args.pattern
            
        video_files = sorted(list(input_path.glob(glob_pattern)))
        print(f"Found {len(video_files)} videos matching pattern.")
        
        # Apply include filter
        if args.include:
            original_count = len(video_files)
            video_files = [v for v in video_files if any(inc in str(v) for inc in args.include)]
            print(f"After include filter: {len(video_files)} videos (filtered out {original_count - len(video_files)})")
        
        # Apply exclude filter
        if args.exclude:
            original_count = len(video_files)
            video_files = [v for v in video_files if not any(exc in str(v) for exc in args.exclude)]
            print(f"After exclude filter: {len(video_files)} videos (filtered out {original_count - len(video_files)})")
            
    elif input_path.is_file() and input_path.suffix.lower() == '.mp4':
        video_files.append(input_path)
    else:
        print(f"Error: Input path is not a valid video file or directory: {input_path}")
        return

    if not video_files:
        print("No video files found to process.")
        return
    
    # Parse report directory
    report_dir = Path(args.save_reports_to) if args.save_reports_to else None
    
    # Validate arguments
    if args.cleanup and not report_dir:
        print("Warning: --cleanup specified without --save-reports-to.")
        print("Error summaries will be lost if you cleanup without saving them!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Group videos by subject if batch-by-subject is enabled
    if args.batch_by_subject:
        from collections import defaultdict
        videos_by_subject = defaultdict(list)
        for video_file in video_files:
            subject_name = video_file.parent.name
            videos_by_subject[subject_name].append(video_file)
        
        subject_groups = list(videos_by_subject.items())
    else:
        # Process all videos as one batch
        subject_groups = [("all", video_files)]
    
    # List videos if --list-only flag is set
    if args.list_only:
        print(f"\n{'='*60}")
        print(f"Videos to be processed ({len(video_files)} total):")
        if args.batch_by_subject:
            print(f"Grouped into {len(subject_groups)} subject(s)")
        print(f"{'='*60}")
        
        if args.batch_by_subject:
            for subject_name, subject_videos in subject_groups:
                print(f"\n--- {subject_name.upper()} ({len(subject_videos)} videos) ---")
                for i, video_file in enumerate(subject_videos, 1):
                    print(f"  {i:2d}. {video_file}")
        else:
            for i, video_file in enumerate(video_files, 1):
                print(f"{i:3d}. {video_file}")
        
        print(f"\n{'='*60}")
        if report_dir:
            print(f"Reports will be saved to: {report_dir}")
        if args.cleanup:
            if args.batch_by_subject:
                print("Run directories will be cleaned up AFTER EACH SUBJECT")
            else:
                print("Run directories will be cleaned up after all videos")
        print(f"{'='*60}")
        return
        
    # Print summary of what will happen
    print(f"\n{'='*60}")
    print(f"Processing {len(video_files)} video(s)")
    if args.batch_by_subject:
        print(f"Batched by subject: {len(subject_groups)} subject(s)")
    if report_dir:
        print(f"Reports will be saved to: {report_dir}")
    if args.cleanup:
        if args.batch_by_subject:
            print("⚠️  Runs will be DELETED after EACH SUBJECT (disk space managed)")
        else:
            print("⚠️  Runs will be DELETED after ALL videos")
        print("    (existing runs in results/ will remain untouched)")
    print(f"{'='*60}\n")
    
    # Process videos (batched by subject if enabled)
    results_dir = Path("results")
    
    for subject_idx, (subject_name, subject_videos) in enumerate(subject_groups, 1):
        if args.batch_by_subject:
            print(f"\n{'#'*60}")
            print(f"SUBJECT {subject_idx}/{len(subject_groups)}: {subject_name.upper()}")
            print(f"Processing {len(subject_videos)} video(s)")
            print(f"{'#'*60}\n")
        
        # Track runs for this batch
        batch_runs = []
        
        for video_file in subject_videos:
            run_name, success = process_video(video_file, args.target_object, args.start_at, args.end_at, 
                                             args.step_size, report_dir)
            if success and run_name:
                batch_runs.append(run_name)
        
        # Cleanup runs for this batch if cleanup and batch-by-subject are enabled
        if args.cleanup and args.batch_by_subject and batch_runs:
            print(f"\n{'='*60}")
            print(f"Cleanup: Removing {len(batch_runs)} run(s) for {subject_name}")
            print(f"{'='*60}")
            
            for run_name in batch_runs:
                run_dir = results_dir / run_name
                if run_dir.exists():
                    print(f"  - Deleting: {run_dir}")
                    try:
                        shutil.rmtree(run_dir)
                        print(f"    ✓ Removed")
                    except Exception as e:
                        print(f"    ✗ Failed: {e}")
            
            print(f"  ✓ Subject '{subject_name}' cleanup complete, disk space freed\n")
    
    # Final cleanup if not batching by subject
    if args.cleanup and not args.batch_by_subject:
        # Collect all processed runs
        all_runs = []
        for _, subject_videos in subject_groups:
            for video_file in subject_videos:
                video_stem = video_file.stem
                subject_name = video_file.parent.name
                run_name = f"{subject_name}_{video_stem}"
                if (results_dir / run_name).exists():
                    all_runs.append(run_name)
        
        if all_runs:
            print(f"\n{'='*60}")
            print(f"Cleanup: Removing {len(all_runs)} run(s)")
            print(f"{'='*60}")
            
            for run_name in all_runs:
                run_dir = results_dir / run_name
                if run_dir.exists():
                    print(f"  - Deleting: {run_dir}")
                    try:
                        shutil.rmtree(run_dir)
                        print(f"    ✓ Removed")
                    except Exception as e:
                        print(f"    ✗ Failed: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✓ ALL PROCESSING COMPLETE")
    if report_dir:
        print(f"  Reports saved in: {report_dir}")
    if args.cleanup:
        print(f"  Disk space freed by cleanup")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()