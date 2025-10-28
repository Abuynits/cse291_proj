import os
import time
from pathlib import Path
import shutil
import trimesh
import hydra
from omegaconf import DictConfig, OmegaConf

# Register OmegaConf resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

from model_utils import *
from image_utils import *
from grounded_sam import GroundedSAMModel

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
from TraceAnything.trace_anything.trace_anything import evaluate_bspline_conf

MAX_FRAMES = 40

def fetch_files_in_dir(cfg) -> list:
    input_dir = Path(cfg.paths.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in input_dir.glob('**/*') if f.suffix.lower() in video_extensions]
    if not video_files:
        raise FileNotFoundError(f"No video files found in directory: {input_dir}")
    return video_files


def compute_pointclouds_from_preds(preds, masks, t_step=0.025, conf_percentile = 10.0):
    frames_data = []
    t_vals = [preds[i]['time'].item() for i in range(len(preds))]
    fg_conf_pool_per_t = [[] for _ in range(len(t_vals))]
    t_tensor = torch.from_numpy(np.array(t_vals))
    for (pred, mask) in zip(preds, masks):

        fg_mask = mask['masks'].squeeze().astype(bool)
        fg_mask_flat = fg_mask.reshape(-1)
        
        ctrl_pts3d = pred["ctrl_pts3d"].cpu()  # [K, H, W, 3]
        ctrl_conf = pred["ctrl_conf"].cpu()  # [K, H, W]
        
        # Evaluate B-spline over all t
        pts3d_t, conf_t = evaluate_bspline_conf(ctrl_pts3d, ctrl_conf, t_tensor)  # [T, H, W, 3], [T, H, W]
        pts3d_t = pts3d_t.detach().cpu().numpy()  # [T, H, W, 3]
        conf_t = conf_t.detach().cpu().numpy()  # [T, H, W]
        H, W = fg_mask.shape
        pts_fg_per_t = [pts3d_t[t].reshape(-1, 3)[fg_mask_flat] for t in range(len(t_vals))]
        conf_fg_per_t = [conf_t[t].reshape(-1)[fg_mask_flat] for t in range(len(t_vals))]
        
        # Pool conf for percentile filtering (per t, across frames)
        for t in range(len(t_vals)):
            if conf_fg_per_t[t].size > 0:
                fg_conf_pool_per_t[t].append(conf_fg_per_t[t])
        
        img = np.asarray(mask['org_image'])
        img_flat = img.reshape(H * W, 3)
        img_fg = img_flat[fg_mask_flat]
        img_fg_per_t = [img_fg for _ in range(len(t_vals))]
        
        frames_data.append({
            'pts_fg_per_t': pts_fg_per_t,
            'conf_fg_per_t': conf_fg_per_t,
            'img_fg_per_t': img_fg_per_t,
        })

    # Compute global conf thresholds per t
    fg_conf_thr_per_t = []
    for pool in fg_conf_pool_per_t:
        if pool:
            all_conf = np.concatenate(pool)
            thr = np.percentile(all_conf, conf_percentile) if all_conf.size > 0 else np.inf
        else:
            thr = np.inf
        fg_conf_thr_per_t.append(thr)

    # Second pass: Generate point clouds per timestep t
    pointclouds_per_t = []
    for t, t_val in enumerate(t_vals):
        # Select nearest frame's data
        fr = frames_data[t]
        points = fr['pts_fg_per_t'][t]
        conf = fr['conf_fg_per_t'][t]
        colors = fr['img_fg_per_t'][t] if fr['img_fg_per_t'][t] is not None else None
        
        # Filter by conf
        thr = fg_conf_thr_per_t[t]
        keep = conf >= thr
        points = points[keep]
        conf = conf[keep]
        if colors is not None:
            colors = colors[keep]
        
        pointclouds_per_t.append({
            'points': points,
            'colors': colors,
            'conf': conf,
        })
        print(f"Timestep t={t_val:.2f} (from frame {t}): {points.shape[0]} points after filtering")
    return pointclouds_per_t


def process_single_video(video_path: str, output_dir: str, trace_anything, grounded_sam:GroundedSAMModel, cfg: DictConfig, device: torch.device):
    video_frames = load_video(video_path)
    raw_views, views = process_video(video_frames, device)

    # Downsample if requested
    if cfg.processing.downsample_factor > 1:
        views = views[::cfg.processing.downsample_factor]
        raw_views = raw_views[::cfg.processing.downsample_factor]

    # only support up to 40 frames (TraceAnything limitation)
    while len(views) > 20:
        views = views[::2]
        raw_views = raw_views[::2]

    masks = grounded_sam.get_grounded_dino_masks_for_views(raw_views, 'racecar.')
    indices_with_object = [i for i, res in enumerate(masks) if res['found_object']]
    print(f"found {len(indices_with_object)}/{len(masks)} frames with the target object")

    print(f"{len(views)} views loaded for {video_path}")

    print("inference ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = trace_anything.forward(views)
    dt = time.perf_counter() - t0
    ms_per_view = (dt / max(1, len(views))) * 1000.0
    print(f"done | {dt:.2f}s total | {ms_per_view:.1f} ms/view")

    masks_dir = os.path.join(output_dir, "masks")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # ---- compute + save FG masks and images ----
    print("computing FG masks + saving frames ...")
    # set to only use valid scenes / masks 
    preds = [preds[i] for i in range(len(preds)) if masks[i]['found_object']]
    views = [views[i] for i in range(len(views)) if masks[i]['found_object']]
    raw_views = [raw_views[i] for i in range(len(raw_views)) if masks[i]['found_object']]
    masks = [masks[i] for i in range(len(masks)) if masks[i]['found_object']]
    pcls = compute_pointclouds_from_preds(preds, masks)
    for i, (_, mask, pcl, view) in enumerate(zip(preds, masks, pcls, raw_views)):
        cv2.imwrite(os.path.join(images_dir, f"{i:03d}_mask.png"), np.asarray(mask['annotated_frame']))
        cv2.imwrite(os.path.join(images_dir, f"{i:03d}.png"), np.asarray(view))
        mesh = trimesh.points.PointCloud(pcl['points'], colors=pcl['colors'])
        mesh.export(os.path.join(images_dir, f"{i:03d}_pcl.glb"))


@hydra.main(version_base=None, config_path="../../config/trajectory_estimation", config_name="base")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounded_sam = GroundedSAMModel(cfg=cfg.grounding_sam, device=device)
    trace_anything = build_model_from_cfg(cfg.model, ckpt_path=cfg.paths.trace_anything, device=device)
    video_files = fetch_files_in_dir(cfg)

    for video_file in video_files:
        output_dir = video_file.parent / cfg.output.subdir
        if output_dir.exists():
            if cfg.output.overwrite:
                shutil.rmtree(output_dir)
            else:
                print(f"Output directory exists and overwrite=False, skipping: {output_dir}")
                continue
        
        os.makedirs(output_dir, exist_ok=True)
        process_single_video(str(video_file), str(output_dir), trace_anything, grounded_sam, cfg, device)


if __name__ == "__main__":
    main()