# core/postproc.py
from typing import List
import numpy as np
from PIL import Image
import cv2
import torch

def run_nms(boxes: List[List[float]], scores: List[float], iou_thr: float) -> List[int]:
    """Single Responsibility: Post-processing utilities (NMS only)."""
    try:
        from torchvision.ops import nms as torch_nms
        bt = torch.as_tensor(boxes, dtype=torch.float32)
        st = torch.as_tensor(scores, dtype=torch.float32)
        keep = torch_nms(bt, st, float(iou_thr)).cpu().tolist()
        return keep
    except Exception:
        return list(range(len(boxes)))

def draw_boxes(image: Image.Image, boxes, labels, scores, label_names):
    """Draw detection results on image."""
    img = np.array(image).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b, lid, sc in zip(boxes, labels, scores):
        x0, y0, x1, y1 = map(int, (b if hasattr(b, "__iter__") else b.tolist()))
        name = label_names[lid] if isinstance(lid, int) and 0 <= lid < len(label_names) else str(lid)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        txt = f"{name} {float(sc):.2f}"
        (tw, th), bl = cv2.getTextSize(txt, font, 0.6, 2)
        cv2.rectangle(img, (x0, max(0, y0 - th - bl - 6)), (x0 + tw + 8, y0), (0, 0, 0), -1)
        cv2.putText(img, txt, (x0 + 4, y0 - 6), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(img)

def draw_topk_legend(image: Image.Image, lines: list[str]):
    if not lines:
        return image
    img = np.array(image).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.7, 2
    pad = 8

    # วัดขนาดข้อความต่อบรรทัด
    sizes = [cv2.getTextSize(l, font, fs, th)[0] for l in lines]  # [(w,h), ...]
    max_w = max(w for (w, h) in sizes)
    total_h = sum(h for (w, h) in sizes) + (len(lines) - 1) * 6 + pad * 2

    x0, y0 = 10, 10
    cv2.rectangle(img, (x0, y0), (x0 + max_w + pad * 2, y0 + total_h), (0, 0, 0), -1)

    y = y0 + pad + sizes[0][1]
    for i, l in enumerate(lines):
        cv2.putText(img, l, (x0 + pad, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        if i < len(lines) - 1:
            y += sizes[i + 1][1] + 6
    return Image.fromarray(img)
