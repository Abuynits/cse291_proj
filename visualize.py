import os
import subprocess
import argparse
import torch
from PIL import Image
import numpy as np
import tempfile

def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    """A helper function to mimic the resizing in TraceAnything's infer.py"""
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)

def main():
    parser = argparse.ArgumentParser(description="Visualize the output of a pipeline run using the TraceAnything viewer.")
    parser.add_argument("run_name", type=str, help="The name of the pipeline run to visualize.")
    args = parser.parse_args()

    run_name = args.run_name
    
    # --- Construct paths ---
    scene_dir_name = f"{run_name}_scene"
    results_base_dir = os.path.join("results", run_name)
    output_pt_path = os.path.join(results_base_dir, "4_trace_anything_output", scene_dir_name, "output.pt")
    masks_dir = os.path.join(results_base_dir, "3_masks")
    frames_scene_dir = os.path.join(results_base_dir, "2_frames", scene_dir_name)

    # --- Check for necessary files ---
    if not os.path.exists(output_pt_path):
        print(f"Error: Could not find output file for run '{run_name}'. Expected path: {output_pt_path}")
        return
    if not os.path.exists(masks_dir):
        print(f"Error: Could not find masks directory for run '{run_name}'. Expected path: {masks_dir}")
        return
    if not os.path.exists(frames_scene_dir):
        print(f"Error: Could not find frames directory for run '{run_name}'. Expected path: {frames_scene_dir}")
        return

    print("Found all necessary files. Filtering point cloud with masks...")

    # --- Load and filter the data ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trace_data = torch.load(output_pt_path, map_location=device)
    
    # Create a deep copy to modify
    filtered_trace_data = {
        'preds': [p.copy() for p in trace_data['preds']],
        'views': [v.copy() for v in trace_data['views']],
    }

    frame_paths = sorted([os.path.join(frames_scene_dir, f) for f in os.listdir(frames_scene_dir) if f.endswith('.png')])
    num_frames = len(filtered_trace_data['preds'])

    for i in range(num_frames):
        ctrl_pts_3d = filtered_trace_data['preds'][i]['ctrl_pts3d']
        mask_path = os.path.join(masks_dir, f"{i:05d}.png")

        if not os.path.exists(mask_path):
            # If no mask exists, nullify all points for this frame
            filtered_trace_data['preds'][i]['ctrl_pts3d'] = torch.zeros_like(ctrl_pts_3d)
            continue
        
        # --- Align the mask with the control points, same logic as pipeline ---
        mask_pil = Image.open(mask_path)
        original_frame_pil = Image.open(frame_paths[i])

        W0, H0 = original_frame_pil.size
        if H0 > W0:
            mask_pil = mask_pil.transpose(Image.Transpose.ROTATE_90)
        
        resized_mask_pil = _resize_long_side(mask_pil, 512)
        H, W = resized_mask_pil.size[1], resized_mask_pil.size[0]
        Ht, Wt = (H - H % 16, W - W % 16)
        aligned_mask_pil = resized_mask_pil.crop((0, 0, Wt, Ht))
        
        aligned_mask_np = np.array(aligned_mask_pil) == 0 # Invert mask: True for background

        # Use the inverted mask to zero out background control points
        ctrl_pts_3d[:, aligned_mask_np] = 0.0
        filtered_trace_data['preds'][i]['ctrl_pts3d'] = ctrl_pts_3d

    # --- Save filtered data to a temporary file ---
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        torch.save(filtered_trace_data, tmp_file.name)
        filtered_output_path = tmp_file.name
    
    print(f"Filtered point cloud saved to temporary file: {filtered_output_path}")
    print("Launching TraceAnything viewer...")

    # --- Launch the viewer with the filtered data ---
    viewer_script = "TraceAnything/scripts/view.py"
    my_env = os.environ.copy()
    project_root = os.getcwd()
    python_path = my_env.get("PYTHONPATH", "")
    my_env["PYTHONPATH"] = f"{project_root}:{python_path}"

    cmd = [
        "python", viewer_script,
        "--output", filtered_output_path,
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("Open your browser to the URL printed by the viewer.")
    print("Press Ctrl+C in this terminal to stop the viewer.")

    try:
        subprocess.run(cmd, check=True, env=my_env)
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while running the viewer: {e}")
    finally:
        # --- Clean up the temporary file ---
        os.remove(filtered_output_path)
        print(f"Cleaned up temporary file: {filtered_output_path}")

if __name__ == '__main__':
    main()
