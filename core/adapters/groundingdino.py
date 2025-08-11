# core/adapters/groundingdino.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
from ..interfaces import VLMAdapter
from ..postproc import run_nms, draw_boxes

class GroundingDinoAdapter(VLMAdapter):
    name = "GroundingDINO"
    sizes = ["", "IDEA-Research/grounding-dino-base"]
    default_prompt = "ring, chain, pendant, bracelet, necklace, hand, tray, circle, earrings"
    supports_image_output = True

    def __init__(self):
        self.processor = None
        self.model = None
        self._repo = None

    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "IDEA-Research/grounding-dino-base"
        if self.model is not None and self._repo == repo:
            return
        self.processor = GroundingDinoProcessor.from_pretrained(repo)
        self.model = GroundingDinoForObjectDetection.from_pretrained(repo)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._repo = repo

    def infer(
        self, image: Image.Image, prompt: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[Image.Image]]:
        self.load(params.get("size_repo") if params else None)
        box_thr = float((params or {}).get("score_thr", 0.25))
        iou_thr = float((params or {}).get("iou_thr", 0.5))

        # 1) create name list from prompt (for name mapping)
        phrases = [p.strip() for p in (prompt or "").split(",") if p.strip()]
        if not phrases:
            return None, image

        # 2) inferencing
        device = next(self.model.parameters()).device
        H, W = image.size[1], image.size[0]
        text = ". ".join(phrases) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3) Post-process: supports new API (text_labels) with old fallback
        try:
            # HF >= 4.51 จะมี text_labels เป็นชื่อวัตถุตรงจากข้อความ
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                box_threshold=box_thr,
                text_threshold=0.25,
                image_sizes=[(H, W)],
            )[0]

            boxes  = [b.tolist() for b in results.get("boxes", [])]
            scores = [float(s) for s in results.get("scores", [])]

            text_labels = results.get("text_labels")
            if text_labels is not None and len(text_labels) == len(boxes):
                # using name from the model
                label_names = [str(t) for t in text_labels]
                labels_idx  = list(range(len(label_names)))
            else:
                # Fallback: map id -> using name from prompt (ถ้ามี)
                ids = results.get("labels", [])
                ids = [int(i) if not isinstance(i, int) else i for i in ids]
                label_names = phrases
                labels_idx  = [i if 0 <= i < len(phrases) else 0 for i in ids]

        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], box_thr, 0.25, [(H, W)]
            )[0]

            boxes  = [b.tolist() for b in results.get("boxes", [])]
            scores = [float(s) for s in results.get("scores", [])]
            ids = results.get("labels", [])
            idxs = []
            for l in ids:
                if isinstance(l, (int,)):
                    idxs.append(l)
                else:
                    try:
                        idxs.append(phrases.index(str(l)))
                    except ValueError:
                        idxs.append(0)
            label_names = phrases
            labels_idx  = [i if 0 <= i < len(phrases) else 0 for i in idxs]

        # 4) NMS + วาดผล โดยใช้ชื่อจาก prompt เสมอ
        keep = run_nms(boxes, scores, iou_thr) if boxes else []
        boxes      = [boxes[i] for i in keep]
        scores     = [scores[i] for i in keep]
        labels_idx = [labels_idx[i] for i in keep]

        out_img = draw_boxes(image, boxes, labels_idx, scores, label_names)
        return None, out_img
