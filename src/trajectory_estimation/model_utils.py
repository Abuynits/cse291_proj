import os
import torch
from omegaconf import OmegaConf
from TraceAnything.trace_anything.trace_anything import TraceAnything
import numpy as np


# ---------------- ckpt + model ----------------
def _get_state_dict(ckpt: dict) -> dict:
    """Accept either a pure state_dict or a Lightning .ckpt."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def _to_dict(x):
    # OmegaConf -> plain dict
    return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x


def build_model_from_cfg(cfg, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    # net config
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


def smart_var_threshold(var_map_t: torch.Tensor) -> float:
    """
    1) log-transform variance
    2) Otsu on histogram
    3) fallback to 65–80% mid-quantile midpoint
    Returns threshold in original variance domain.
    """
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