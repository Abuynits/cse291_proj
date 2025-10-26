from typing import List, Dict
from PIL import Image
import torchvision.transforms as tvf
import torch
import cv2

def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)


def process_video(frames: List, device: torch.device) -> List[Dict]:

    tfm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5,)*3, (0.5,)*3)])
    views, target = [], None
    n_frames= len(frames)
    for i, arr in enumerate(frames):
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

        W0, H0 = pil.size
        if H0 > W0:  # portrait -> landscape
            pil = pil.transpose(Image.Transpose.ROTATE_90)

        pil = _resize_long_side(pil, 512)
        if target is None:
            H, W = pil.size[1], pil.size[0]
            target = (H - H % 16, W - W % 16)
            print(f"target size: {target[0]}x{target[1]} (16-multiple)")
        Ht, Wt = target
        pil = pil.crop((0, 0, Wt, Ht))

        tensor = tfm(pil).unsqueeze(0).to(device)  # [1,3,H,W]
        t = i / (n_frames - 1) if n_frames > 1 else 0.0
        views.append({"img": tensor, "time_step": t})
    return views

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames