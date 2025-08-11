# core/adapters/owlvit.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from ..interfaces import VLMAdapter
from ..postproc import run_nms, draw_boxes

class OwlVitAdapter(VLMAdapter):
    name = "OWL-ViT"
    sizes = ["", "google/owlvit-base-patch32"]
    default_prompt = "ring, chain, pendant, bracelet, necklace, hand, tray, circle, earrings"
    supports_image_output = True

    def __init__(self):
        self.processor = None
        self.model = None
        self._repo = None

    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "google/owlvit-base-patch32"
        if self.model is not None and self._repo == repo:
            return
        self.processor = OwlViTProcessor.from_pretrained(repo)
        self.model = OwlViTForObjectDetection.from_pretrained(repo)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._repo = repo

    def infer(self, image: Image.Image, prompt: str, params: Optional[Dict[str, Any]] = None):
        self.load(params.get("size_repo") if params else None)
        det_thr = float((params or {}).get("score_thr", 0.05))
        iou_thr = float((params or {}).get("iou_thr", 0.5))

        phrases = [p.strip() for p in (prompt or "").split(",") if p.strip()]
        if not phrases:
            return None, image

        device = next(self.model.parameters()).device
        inputs = self.processor(text=phrases, images=image, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        H, W = image.size[1], image.size[0]
        try:
            # ✅ ลองแบบ grounded ก่อน (API ใหม่)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.get("input_ids"),
                box_threshold=det_thr,
                text_threshold=0.25,
                image_sizes=[(H, W)],
            )[0]
            boxes  = [b.tolist() for b in results.get("boxes", [])]
            scores = [float(s) for s in results.get("scores", [])]
            # map text_labels → label_names + index list
            text_labels = results.get("text_labels") or []
            label_names = [str(t) for t in text_labels]
            labels_idx = list(range(len(label_names)))
        except Exception:
            # 🔁 fallback เก่า
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=torch.tensor([[H, W]], device=device),
                threshold=det_thr
            )[0]
            boxes  = [b.tolist() for b in results.get("boxes", [])]
            scores = [float(s) for s in results.get("scores", [])]
            labels = [int(l) for l in results.get("labels", [])]
            # แปลงเป็นชื่อจาก prompt โดยตรง
            label_names = phrases
            labels_idx = labels

        keep = run_nms(boxes, scores, iou_thr) if boxes else []
        boxes  = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        labels_idx = [labels_idx[i] for i in keep]

        out_img = draw_boxes(image, boxes, labels_idx, scores, label_names)
        return None, out_img

