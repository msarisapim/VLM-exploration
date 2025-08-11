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

        # 1) สร้างรายการคำจาก prompt (ไว้ใช้ map ชื่อ)
        phrases = [p.strip() for p in (prompt or "").split(",") if p.strip()]
        if not phrases:
            return None, image

        # 2) อินเฟอเรนซ์
        device = next(self.model.parameters()).device
        H, W = image.size[1], image.size[0]
        text = ". ".join(phrases) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3) Post-process: รองรับ API ใหม่ (text_labels) และ fallback เก่า
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
                # ใช้ชื่อจากโมเดลโดยตรง
                label_names = [str(t) for t in text_labels]
                labels_idx  = list(range(len(label_names)))
            else:
                # Fallback: map id -> ชื่อจาก prompt (ถ้ามี)
                ids = results.get("labels", [])
                ids = [int(i) if not isinstance(i, int) else i for i in ids]
                label_names = phrases
                labels_idx  = [i if 0 <= i < len(phrases) else 0 for i in ids]

        except TypeError:
            # เวอร์ชันเก่าที่ใช้ signature เดิม
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], box_thr, 0.25, [(H, W)]
            )[0]

            boxes  = [b.tolist() for b in results.get("boxes", [])]
            scores = [float(s) for s in results.get("scores", [])]
            # labels อาจเป็นสตริงหรือดัชนี → บังคับ map เป็นชื่อจาก prompt
            ids = results.get("labels", [])
            # แปลงเป็นดัชนีเท่าที่ทำได้
            idxs = []
            for l in ids:
                if isinstance(l, (int,)):
                    idxs.append(l)
                else:
                    # ถ้าเป็นสตริง ให้ลองค้นตำแหน่งใน phrases ไม่เจอให้ใช้ 0
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
