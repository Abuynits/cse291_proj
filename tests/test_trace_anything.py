# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# scripts/infer.py
"""
Run inference on all scenes and save:
  - <scene>/output.pt with {'preds','views'}
  - <scene>/masks/{i:03d}.png   (binary FG masks)
  - <scene>/images/{i:03d}.png  (RGB frames used for inference)

Masks are computed from ctrl-pt variance + smart Otsu.
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
import cv2
import time
import argparse
import gc
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as tvf
from omegaconf import OmegaConf

from TraceAnything.trace_anything.trace_anything import TraceAnything


def _pretty(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# allow ${python_eval: ...} in YAML if used
OmegaConf.register_new_resolver("python_eval", lambda code: eval(code))


# ---------------- image I/O ----------------
def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)


def _load_images_with_original_indices(input_dir: str):
    """Load images and return both views and original frame indices on CPU."""
    tfm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5,)*3, (0.5,)*3)])

    fnames = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_vis" not in f
    )
    if not fnames:
        raise FileNotFoundError(f"No images in {input_dir}")

    views, target = [], None
    for i, f in enumerate(fnames):
        arr = cv2.imread(os.path.join(input_dir, f))
        if arr is None:
            raise FileNotFoundError(f"Failed to read {f}")
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

        W0, H0 = pil.size
        if H0 > W0:  # portrait -> landscape
            pil = pil.transpose(Image.Transpose.ROTATE_90)

        pil = _resize_long_side(pil, 512)
        if target is None:
            H, W = pil.size[1], pil.size[0]
            target = (H - H % 16, W - W % 16)
        Ht, Wt = target
        pil = pil.crop((0, 0, Wt, Ht))

        # Keep on CPU initially
        tensor = tfm(pil).unsqueeze(0)
        
        views.append({
            "img": tensor, 
            "time_step": i / (len(fnames) - 1) if len(fnames) > 1 else 0.0,
            "original_idx": i,
            "filename": f
        })
    return views, len(fnames)


# ---------------- ckpt + model ----------------
def _get_state_dict(ckpt: dict) -> dict:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def _load_cfg(cfg_path: str):
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    return OmegaConf.load(cfg_path)


def _to_dict(x):
    return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x


def _build_model_from_cfg(cfg, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    net_cfg = cfg.get("model", {}).get("net", None) or cfg.get("net", None)
    if net_cfg is None:
        raise KeyError("expect cfg.model.net or cfg.net in YAML")

    model = TraceAnything(
        encoder_args=_to_dict(net_cfg["encoder_args"]),
        decoder_args=_to_dict(net_cfg["decoder_args"]),
        head_args=_to_dict(net_cfg["head_args"]),
        targeting_mechanism=net_cfg.get("targeting_mechanism", "bspline_conf"),
        poly_degree=net_cfg.get("poly_degree", 10),
        whether_local=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _get_state_dict(ckpt)

    if all(k.startswith("net.") for k in sd.keys()):
        sd = {k[4:]: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


# ---------------- smart var threshold ----------------
def _otsu_threshold_from_hist(hist: np.ndarray, bin_edges: np.ndarray) -> float | None:
    total = hist.sum()
    if total <= 0:
        return None
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w1 = np.cumsum(hist)
    w2 = total - w1
    sum_total = (hist * bin_centers).sum()
    sumB = np.cumsum(hist * bin_centers)
    valid = (w1 > 0) & (w2 > 0)
    if not np.any(valid):
        return None
    m1 = sumB[valid] / w1[valid]
    m2 = (sum_total - sumB[valid]) / w2[valid]
    between = w1[valid] * w2[valid] * (m1 - m2) ** 2
    idx = np.argmax(between)
    return float(bin_centers[valid][idx])


def _smart_var_threshold(var_map_t: torch.Tensor) -> float:
    var_np = var_map_t.detach().float().cpu().numpy()
    v = np.log(var_np + 1e-9)
    hist, bin_edges = np.histogram(v, bins=256)
    thr_log = _otsu_threshold_from_hist(hist, bin_edges)
    if thr_log is None or not np.isfinite(thr_log):
        q65 = float(np.quantile(var_np, 0.65))
        q80 = float(np.quantile(var_np, 0.80))
        return 0.5 * (q65 + q80)
    thr_var = float(np.exp(thr_log))
    q40 = float(np.quantile(var_np, 0.40))
    q95 = float(np.quantile(var_np, 0.95))
    return max(q40, min(q95, thr_var))


# ---------------- helper: deep conversion ----------------
def _recursive_to_cpu(data: Any) -> Any:
    """Recursively move tensors to CPU/NumPy to ensure no GPU references remain."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, dict):
        return {k: _recursive_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_recursive_to_cpu(x) for x in data]
    return data

def _convert_pred_format(pred_cpu: dict) -> dict:
    """Convert specific keys to numpy for final storage, keep others as CPU tensors."""
    final_pred = {}
    for key, value in pred_cpu.items():
        if isinstance(value, torch.Tensor):
            if key in ["ctrl_pts3d", "ctrl_conf", "time", "fg_mask"]:
                final_pred[key] = value # Keep as CPU tensor
            else:
                final_pred[key] = value.detach().cpu() if value.numel() > 0 else value
        else:
            final_pred[key] = value
    return final_pred


# ---------------- chunked processing ----------------
def _process_chunk(model, views_chunk, chunk_idx, total_chunks):
    """Process a single chunk of views through TraceAnything."""
    num_views = len(views_chunk)
    _pretty(f"  Chunk {chunk_idx + 1}/{total_chunks}: processing {num_views} frames")
    
    t0 = time.perf_counter()
    with torch.no_grad():
        preds_chunk = model.forward(views_chunk)
    dt = time.perf_counter() - t0
    ms_per_view = (dt / max(1, num_views)) * 1000.0
    _pretty(f"  ✅ chunk {chunk_idx + 1} done | {dt:.2f}s total | {ms_per_view:.1f} ms/view")
    
    # 1. Move EVERYTHING to CPU immediately to prevent GPU memory leaks
    preds_chunk_cpu_raw = _recursive_to_cpu(preds_chunk)
    
    # 2. Format for storage
    final_preds = []
    for pred in preds_chunk_cpu_raw:
        final_preds.append(_convert_pred_format(pred))
    
    # 3. Clean GPU memory inside the function
    del preds_chunk
    return final_preds


# ---------------- main loop ----------------
def run(args, max_frames_per_chunk=40):
    base_in = args.input_dir
    base_out = args.output_dir

    if not os.path.isdir(base_in):
        raise FileNotFoundError(base_in)
    os.makedirs(base_out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config & model
    cfg = _load_cfg(args.config)
    _pretty("🔧 loading model...")
    model = _build_model_from_cfg(cfg, ckpt_path=args.ckpt, device=device)
    _pretty("✅ model ready")

    # iterate scenes
    for scene in sorted(os.listdir(base_in)):
        in_dir = os.path.join(base_in, scene)
        if not os.path.isdir(in_dir):
            continue
        out_dir = os.path.join(base_out, scene)
        masks_dir = os.path.join(out_dir, "masks")
        images_dir = os.path.join(out_dir, "images")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        _pretty(f"\n📂 Scene: {scene}")
        _pretty("🖼️  loading images...")
        # Images loaded to CPU
        views, total_frames = _load_images_with_original_indices(in_dir)
        _pretty(f"🧮 {len(views)} views loaded (total frames: {total_frames})")

        all_preds = []
        all_views_cpu = []

        # Determine chunking
        if len(views) <= max_frames_per_chunk:
            # --- Single Batch Path ---
            _pretty("🚀 inference (single batch)...")
            
            # Must move to GPU for the model
            views_gpu = []
            for v in views:
                views_gpu.append({
                    "img": v["img"].to(device),
                    "time_step": v["time_step"]
                })
            
            preds = _process_chunk(model, views_gpu, 0, 1)

            # Annotate predictions with global frame and chunk metadata
            annotated_preds = []
            for local_idx, (pred, view) in enumerate(zip(preds, views)):
                pred["original_idx"] = view.get("original_idx", local_idx)
                pred["chunk_idx"] = 0
                pred["chunk_frame_idx"] = local_idx
                pred["chunk_start_idx"] = 0
                annotated_preds.append(pred)
            
            all_preds = annotated_preds
            all_views_cpu = views
            
            # Cleanup
            del views_gpu
            torch.cuda.empty_cache()

        else:
            # --- Chunked Path ---
            num_chunks = (len(views) + max_frames_per_chunk - 1) // max_frames_per_chunk
            _pretty(f"🚀 inference (chunked: {num_chunks} chunks, max {max_frames_per_chunk} frames/chunk)...")
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_frames_per_chunk
                end_idx = min(start_idx + max_frames_per_chunk, len(views))
                views_chunk_raw = views[start_idx:end_idx]
                
                # Prepare chunk on GPU
                views_chunk_gpu = []
                chunk_size = len(views_chunk_raw)
                for j, view in enumerate(views_chunk_raw):
                    chunk_time_step = j / (chunk_size - 1) if chunk_size > 1 else 0.0
                    views_chunk_gpu.append({
                        "img": view["img"].to(device), # Move to GPU
                        "time_step": chunk_time_step
                    })
                
                # Inference
                preds_chunk = _process_chunk(model, views_chunk_gpu, chunk_idx, num_chunks)

                # Annotate predictions in this chunk with global frame and chunk metadata
                for local_idx, (pred, view) in enumerate(zip(preds_chunk, views_chunk_raw)):
                    pred["original_idx"] = view.get("original_idx", start_idx + local_idx)
                    pred["chunk_idx"] = chunk_idx
                    pred["chunk_frame_idx"] = local_idx
                    pred["chunk_start_idx"] = start_idx
                
                # Store results (already CPU safe from _process_chunk)
                all_preds.extend(preds_chunk)
                all_views_cpu.extend(views_chunk_raw) # Keep original CPU references
                
                # --- AGGRESSIVE CLEANUP ---
                del views_chunk_gpu
                gc.collect() 
                torch.cuda.empty_cache()

            total_time = len(all_views_cpu)
            _pretty(f"✅ all chunks done | processed {total_time} frames total")

        # ---- compute + save FG masks and images ----
        _pretty("🧪 computing FG masks + saving frames...")
        for i, (pred, view) in enumerate(zip(all_preds, all_views_cpu)):
            # variance map over control points (K), mean over xyz -> [H,W]
            ctrl_pts3d = pred["ctrl_pts3d"] # pointclouds
            var_map = torch.var(ctrl_pts3d, dim=0, unbiased=False).mean(-1)
            thr = _smart_var_threshold(var_map)
            fg_mask = (~(var_map <= thr)).numpy().astype(bool)

            original_idx = view.get("original_idx", i)
            
            cv2.imwrite(os.path.join(masks_dir, f"{original_idx:03d}.png"), (fg_mask.astype(np.uint8) * 255))
            pred["fg_mask"] = torch.from_numpy(fg_mask)

            # Image is already on CPU
            img = view["img"].squeeze(0) # [3,H,W]
            img_np = (img.permute(1, 2, 0).numpy() + 1.0) * 127.5
            img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(images_dir, f"{original_idx:03d}.png"), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

            # trim off heavy intermediates
            # these are technically the points that we need, but we can't save them to GPU due to memory constraints
            # best to build these from sampling the B-spline.
            pred.pop("track_pts3d", None)
            pred.pop("track_conf", None)

        # ! no longer do this as it can destroy the correspondence to the local chunked run
        # ! this does however somewhat break visualization
        # Update time field in predictions to use full-sequence time_step from views
        # This ensures that chunked processing uses correct full-sequence times (0-1 across all frames)
        # instead of chunk-normalized times (0-1 within each chunk)
        # for pred, view in zip(all_preds, all_views_cpu):
        #     if "time" in pred:
        #         # Overwrite chunk-normalized time with full-sequence time_step
        #         pred["time"] = view["time_step"]

        # Create views list without extra metadata for saving
        views_for_save = [{"img": v["img"], "time_step": v["time_step"]} for v in all_views_cpu]
        
        save_path = os.path.join(out_dir, "output.pt")
        torch.save({"preds": all_preds, "views": views_for_save}, save_path)
        _pretty(f"💾 saved: {save_path}")
        _pretty(f"🖼️  masks → {masks_dir} | images → {images_dir}")

def parse_args():
    p = argparse.ArgumentParser("TraceAnything inference")
    p.add_argument("--config", type=str, default="TraceAnything/configs/eval.yaml",
                   help="Path to YAML config")
    p.add_argument("--ckpt", type=str, default="TraceAnything/checkpoints/trace_anything.pt",
                   help="Path to the checkpoint")
    p.add_argument("--input_dir", type=str, default="./TraceAnything/examples/input",
                   help="Directory containing scenes (each subfolder is a scene)")
    p.add_argument("--output_dir", type=str, default="./TraceAnything/examples/output",
                   help="Directory to write scene outputs")
    p.add_argument("--max_frames_per_chunk", type=int, default=41,
                   help="Maximum number of frames to process per chunk (default: 41)")
    return p.parse_args()

def main():
    args = parse_args()
    run(args, max_frames_per_chunk=args.max_frames_per_chunk)

if __name__ == "__main__":
    main()