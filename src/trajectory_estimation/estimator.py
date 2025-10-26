import os
import time
from pathlib import Path
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

from model_utils import *
from image_utils import *

MAX_FRAMES = 40

def process_single_video(video_path: str, output_dir: str, model, cfg: DictConfig, device: torch.device):
    breakpoint()
    video_frames = load_video(video_path)
    views = process_video(video_frames, device)

    # Downsample if requested
    if cfg.processing.downsample_factor > 1:
        views = views[::cfg.processing.downsample_factor]

    # only support up to 40 frames (TraceAnything limitation)
    while len(views) > MAX_FRAMES:
        views = views[::2]

    print(f"{len(views)} views loaded for {video_path}")

    print("inference ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model.forward(views)
    dt = time.perf_counter() - t0
    ms_per_view = (dt / max(1, len(views))) * 1000.0
    print(f"done | {dt:.2f}s total | {ms_per_view:.1f} ms/view")

    masks_dir = os.path.join(output_dir, "masks")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # ---- compute + save FG masks and images ----
    print("computing FG masks + saving frames ...")
    for i, pred in enumerate(preds):
        # variance map over control points (K), mean over xyz -> [H,W]
        ctrl_pts3d = pred["ctrl_pts3d"]
        ctrl_pts3d_t = torch.from_numpy(ctrl_pts3d) if isinstance(ctrl_pts3d, np.ndarray) else ctrl_pts3d
        var_map = torch.var(ctrl_pts3d_t, dim=0, unbiased=False).mean(-1)  # [H,W]
        thr = smart_var_threshold(var_map)
        fg_mask = (~(var_map <= thr)).detach().cpu().numpy().astype(bool)

        # save mask as binary PNG and stash in preds
        cv2.imwrite(os.path.join(masks_dir, f"{i:03d}.png"), (fg_mask.astype(np.uint8) * 255))
        pred["fg_mask"] = torch.from_numpy(fg_mask)  # CPU bool tensor

        # also save the RGB image we actually ran on
        img = views[i]["img"].detach().cpu().squeeze(0)  # [3,H,W] in [-1,1]
        img_np = (img.permute(1, 2, 0).numpy() + 1.0) * 127.5
        img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(images_dir, f"{i:03d}.png"), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

        # trim heavy intermediates just in case
        pred.pop("track_pts3d", None)
        pred.pop("track_conf", None)

    return preds


@hydra.main(version_base=None, config_path="../../config/trajectory_estimation", config_name="base")
@hydra.main()
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading model ...")
    model = build_model_from_cfg(cfg.model, ckpt_path=cfg.paths.trace_anything, device=device)
    print("model ready")

    input_dir = Path(cfg.paths.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in input_dir.glob('**/*') if f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    for video_file in video_files:
        print(f"\nProcessing video: {video_file}")
        
        output_dir = video_file.parent / cfg.output.subdir
        
        if output_dir.exists():
            if cfg.output.overwrite:
                shutil.rmtree(output_dir)
            else:
                print(f"Output directory exists and overwrite=False, skipping: {output_dir}")
                continue
        
        os.makedirs(output_dir, exist_ok=True)
        process_single_video(str(video_file), str(output_dir), model, cfg, device)


if __name__ == "__main__":
    main()