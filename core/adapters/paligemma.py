from typing import Optional, Tuple, Dict, Any, List
from PIL import Image
import os
import re
import cv2
import numpy as np
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
from transformers import AutoProcessor, AutoModelForVision2Seq

from ..interfaces import VLMAdapter

# ---------------------------- Regex / Utils ----------------------------
LOC_TAG = re.compile(r"<loc(\d{4})>", flags=re.IGNORECASE)
SEG_TAG = re.compile(r"<seg(\d{3})>", flags=re.IGNORECASE)

def _scale1023(val: int, maxv: int) -> int:
    """map loc (0..1023) -> pixel [0..maxv-1]"""
    v = max(0, min(1023, int(val)))
    return int(round(v / 1023.0 * max(1, (maxv - 1))))

def _labels_from_prompt(prompt: str, mode: str) -> List[str]:
    """
    ดึง label จาก prompt
    - mode == 'detect'  : รองรับ 'detect a; b; c' หรือคอมม่า
    - mode == 'segment' : รองรับ 'segment a; b; c' หรือคอมม่า
    """
    p = (prompt or "").strip()
    head = mode.lower().strip()
    if p.lower().startswith(head):
        p = p[len(head):].strip(": ").strip()
    parts = [s.strip() for s in re.split(r"[;,]", p) if s.strip()]
    return parts

def _build_prefixed_prompt(prompt: str, prefix: str, default_label: str = "object") -> Tuple[str, List[str]]:
    labels = _labels_from_prompt(prompt, prefix)
    if not labels:
        labels = [default_label]
    return f"{prefix} " + "; ".join(labels), labels

# ---------------------------- NMS helpers ----------------------------
def _iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _nms_numpy(boxes, scores, iou_thr=0.5):
    """NMS แบบ numpy (ใช้เมื่อไม่มี torchvision)"""
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        rest = []
        for j in idxs[1:]:
            if _iou_xyxy(boxes[i], boxes[j]) <= iou_thr:
                rest.append(j)
        idxs = np.array(rest, dtype=int)
    return keep

def _nms_per_class(boxes, label_idx, scores, iou_thr=0.5):
    """ทำ NMS แยกตามคลาส → รวมผลกลับ"""
    try:
        from torchvision.ops import nms as torch_nms
        use_torch = True
    except Exception:
        use_torch = False

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    label_idx = np.asarray(label_idx, dtype=np.int32)

    kept_boxes, kept_labels, kept_scores = [], [], []
    for cls in np.unique(label_idx):
        cls_mask = (label_idx == cls)
        b = boxes[cls_mask]
        s = scores[cls_mask]
        if len(b) == 0:
            continue

        if use_torch:
            import torch as _t
            keep = torch_nms(_t.tensor(b, dtype=_t.float32),
                             _t.tensor(s, dtype=_t.float32),
                             float(iou_thr)).cpu().numpy().tolist()
        else:
            keep = _nms_numpy(b, s, iou_thr)

        kept_boxes.extend(b[keep].astype(np.int32).tolist())
        kept_scores.extend(s[keep].tolist())
        kept_labels.extend([int(cls)] * len(keep))

    return kept_boxes, kept_labels, kept_scores

# --- วาดกรอบโดยไม่ทำ NMS เพิ่มเติม (เราทำ NMS ไปแล้วข้างบน หากเปิดใช้) ---
def _draw_boxes_keep_all(image: Image.Image,
                         boxes: List[List[int]],
                         label_idx: List[int],
                         scores: List[float],
                         labels: List[str]) -> Image.Image:
    img = np.array(image).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x0, y0, x1, y1), li, sc in zip(boxes, label_idx, scores):
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        name = labels[li] if 0 <= li < len(labels) else str(li)
        text = f"{name} ({sc:.2f})"
        (tw, th), bl = cv2.getTextSize(text, font, 0.6, 2)
        cv2.rectangle(img, (x0, max(0, y0 - th - bl - 6)), (x0 + tw + 8, y0), (0, 0, 0), -1)
        cv2.putText(img, text, (x0 + 4, y0 - 6), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(img)

# ---------------------------- VAE for segmentation ----------------------------
# ถอดรหัส <seg###> 16 ตัว (codebook indices) → mask ขนาด 64x64 ด้วย PyTorch (CPU)
def _load_vae_npz_local() -> Optional[np.ndarray]:
    try:
        local = os.path.join(os.path.dirname(__file__), "vae-oid.npz")
        if os.path.exists(local):
            return np.load(local)

        # ดึงจาก Space ทางการ (ต้องระบุ repo_type="space")
        path = hf_hub_download(
            repo_id="big-vision/paligemma-hf",
            filename="vae-oid.npz",
            repo_type="space",
            local_dir=os.path.dirname(__file__),
            local_dir_use_symlinks=False
        )
        return np.load(path)
    except Exception:
        # fallback: community mirror แบบโมเดลรีโป
        try:
            path = hf_hub_download(
                repo_id="ndkhanh95/Paligemma",
                filename="vae-oid.npz",
                repo_type="model",
                local_dir=os.path.dirname(__file__),
                local_dir_use_symlinks=False
            )
            return np.load(path)
        except Exception:
            return None

def _decode_masks_pt(code_idx_16: np.ndarray, vae_chk) -> Optional[np.ndarray]:
    """
    ถอดรหัสชุดโค้ด (B,16) → mask (B,64,64) โดยเดารูปแบบน้ำหนักจาก shape/bias อัตโนมัติ
    รองรับทั้ง Flax/PyTorch (และเคสสลับแกนอื่น ๆ)
    """
    try:
        import torch, torch.nn as nn
        from itertools import permutations
        device = torch.device("cpu")

        need = [
            "_vq_vae._embedding",
            "decoder.0.weight","decoder.0.bias",
            "decoder.2.net.0.weight","decoder.2.net.0.bias",
            "decoder.2.net.2.weight","decoder.2.net.2.bias",
            "decoder.2.net.4.weight","decoder.2.net.4.bias",
            "decoder.3.net.0.weight","decoder.3.net.0.bias",
            "decoder.3.net.2.weight","decoder.3.net.2.bias",
            "decoder.3.net.4.weight","decoder.3.net.4.bias",
            "decoder.4.weight","decoder.4.bias",
            "decoder.6.weight","decoder.6.bias",
            "decoder.8.weight","decoder.8.bias",
            "decoder.10.weight","decoder.10.bias",
            "decoder.12.weight","decoder.12.bias",
        ]
        for k in need:
            if k not in vae_chk:
                return None

        emb = torch.from_numpy(vae_chk["_vq_vae._embedding"]).to(device)

        def choose_weight_conv(w_np, bias_len):
            """หา w ที่เป็น (out,in,kH,kW) โดยให้ out == bias_len และ kH,kW เล็ก"""
            w0 = torch.from_numpy(w_np).to(device)
            best = None
            for p in permutations(range(4)):
                wp = w0.permute(*p).contiguous()
                o,i,kH,kW = wp.shape
                if kH <= 9 and kW <= 9 and o == bias_len:
                    score = (kH*kW,)
                    if best is None or score < best[0]:
                        best = (score, wp)
            if best is not None:
                return best[1]
            # fallback: เลือก permute ที่ทำให้สองแกนท้ายเล็กที่สุด
            best = None
            for p in permutations(range(4)):
                wp = w0.permute(*p).contiguous()
                o,i,kH,kW = wp.shape
                score = (kH*kW, abs(o - bias_len))
                if best is None or score < best[0]:
                    best = (score, wp)
            return best[1]

        def choose_weight_deconv(w_np, bias_len):
            """หา w ที่เป็น (in,out,kH,kW) โดยให้ out == bias_len และ kH,kW เล็ก"""
            w0 = torch.from_numpy(w_np).to(device)
            best = None
            for p in permutations(range(4)):
                wp = w0.permute(*p).contiguous()
                i,o,kH,kW = wp.shape
                if kH <= 9 and kW <= 9 and o == bias_len:
                    score = (kH*kW,)
                    if best is None or score < best[0]:
                        best = (score, wp)
            if best is not None:
                return best[1]
            # fallback: แบบเดียวกับ conv
            best = None
            for p in permutations(range(4)):
                wp = w0.permute(*p).contiguous()
                i,o,kH,kW = wp.shape
                score = (kH*kW, abs(o - bias_len))
                if best is None or score < best[0]:
                    best = (score, wp)
            return best[1]

        def conv_from_npz(w_key, b_key):
            b = torch.from_numpy(vae_chk[b_key]).to(device)
            w = choose_weight_conv(vae_chk[w_key], b.numel())
            o,i,kH,kW = w.shape
            layer = nn.Conv2d(i, o, (kH, kW), stride=1, padding=(kH//2, kW//2), bias=True).to(device)
            layer.weight.data.copy_(w); layer.bias.data.copy_(b)
            return layer

        def deconv_from_npz(w_key, b_key):
            b = torch.from_numpy(vae_chk[b_key]).to(device)
            w = choose_weight_deconv(vae_chk[w_key], b.numel())
            i,o,kH,kW = w.shape
            layer = nn.ConvTranspose2d(i, o, (kH, kW), stride=2, padding=1, bias=True).to(device)
            layer.weight.data.copy_(w); layer.bias.data.copy_(b)
            return layer

        # ประกอบดีโค้ดเดอร์
        conv0  = conv_from_npz("decoder.0.weight","decoder.0.bias")
        rb0_0  = conv_from_npz("decoder.2.net.0.weight","decoder.2.net.0.bias")
        rb0_1  = conv_from_npz("decoder.2.net.2.weight","decoder.2.net.2.bias")
        rb0_2  = conv_from_npz("decoder.2.net.4.weight","decoder.2.net.4.bias")
        rb1_0  = conv_from_npz("decoder.3.net.0.weight","decoder.3.net.0.bias")
        rb1_1  = conv_from_npz("decoder.3.net.2.weight","decoder.3.net.2.bias")
        rb1_2  = conv_from_npz("decoder.3.net.4.weight","decoder.3.net.4.bias")
        de4    = deconv_from_npz("decoder.4.weight","decoder.4.bias")
        de6    = deconv_from_npz("decoder.6.weight","decoder.6.bias")
        de8    = deconv_from_npz("decoder.8.weight","decoder.8.bias")
        de10   = deconv_from_npz("decoder.10.weight","decoder.10.bias")
        conv1  = conv_from_npz("decoder.12.weight","decoder.12.bias")

        # เตรียมโค้ดบุ๊ก → (B,D,4,4)
        idx   = torch.from_numpy(code_idx_16.astype(np.int64)).to(device)  # (B,16)
        quant = emb[idx]                                                   # (B,16,D)
        B, N, D = quant.shape
        x = quant.view(B, 4, 4, D).permute(0, 3, 1, 2).contiguous()        # (B,D,4,4)

        # forward
        x = torch.relu(conv0(x))
        y = torch.relu(rb0_0(x)); y = torch.relu(rb0_1(y)); y = rb0_2(y); x = x + y
        y = torch.relu(rb1_0(x)); y = torch.relu(rb1_1(y)); y = rb1_2(y); x = x + y
        x = torch.relu(de4(x)); x = torch.relu(de6(x)); x = torch.relu(de8(x)); x = torch.relu(de10(x))
        x = conv1(x)
        x = torch.sigmoid(x)
        return x.squeeze(1).detach().cpu().numpy()
    except Exception:
        return None

# ---------------------------- Main Adapter ----------------------------
class PaliGemmaAdapter(VLMAdapter):
    name = "PaliGemma"
    sizes = ["", "google/paligemma-3b-mix-224", "google/paligemma-3b-mix-448"]
    default_prompt = "Answer the question about this image."
    supports_image_output = True  # มีโหมด detect/segment ที่วาดภาพได้

    def __init__(self):
        self.processor = None
        self.model = None
        self._repo = None

    # ----------- load (รองรับ gated repo) -----------
    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "google/paligemma-3b-mix-224"
        if self.model is not None and self._repo == repo:
            return

        token = os.getenv("HF_TOKEN", None)
        try:
            try:
                self.processor = AutoProcessor.from_pretrained(repo, token=token)
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(repo, use_auth_token=token)

            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    repo,
                    device_map="auto",
                    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                    token=token,
                )
            except TypeError:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    repo,
                    device_map="auto",
                    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
                    use_auth_token=token,
                )
        except (GatedRepoError, OSError) as e:
            raise gr.Error(
                "PaliGemma เป็นโมเดลแบบ gated บน Hugging Face:\n"
                "1) ไปหน้าโมเดลแล้วกด Request/Agree access\n"
                "2) รัน `huggingface-cli login` หรือกำหนดตัวแปรแวดล้อม `HF_TOKEN`\n"
                "3) เปิดแอปใหม่แล้วลองอีกครั้ง\n\n"
                f"รายละเอียด: {e}"
            )
        self._repo = repo

    # ----------- helpers: parsing outputs -----------
    @staticmethod
    def _group_locs_to_boxes(locs: List[Tuple[int, int]], W: int, H: int) -> Tuple[List[List[int]], List[int]]:
        """
        รับรายการ locs = [(pos_in_text, value)] สำหรับทุก <loc####>
        จัดกลุ่มต่อเนื่องทีละ 4 เป็น 1 กล่อง (y0,x0,y1,x1)
        คืน: boxes (พิกเซล), last_pos_of_quad ต่อกล่อง (ใช้จับคู่ seg ถัดไป)
        """
        boxes, last_pos = [], []
        locs_sorted = sorted(locs, key=lambda x: x[0])
        for i in range(0, len(locs_sorted) - 3, 4):
            quad = locs_sorted[i:i+4]
            (p0, y0), (p1, x0), (p2, y1), (p3, x1) = quad
            x0p, x1p = _scale1023(x0, W), _scale1023(x1, W)
            y0p, y1p = _scale1023(y0, H), _scale1023(y1, H)
            x0p, x1p = min(x0p, x1p), max(x0p, x1p)
            y0p, y1p = min(y0p, y1p), max(y0p, y1p)
            boxes.append([x0p, y0p, x1p, y1p])
            last_pos.append(max(p0, p1, p2, p3))
        return boxes, last_pos

    @staticmethod
    def _assign_labels(text: str, labels: List[str], n_boxes: int) -> List[int]:
        """
        พยายามจับคู่ labels กับกล่องด้วยการหาตำแหน่งคำ label ในข้อความ
        แล้วไล่กำหนด index ให้กล่องตามลำดับ ถ้าหาไม่พอ → เติมแบบวนรอบ
        """
        hits = []
        for li, lab in enumerate(labels):
            for m in re.finditer(rf"\b{re.escape(lab)}\b", text, flags=re.IGNORECASE):
                hits.append((m.start(), li))
        hits.sort(key=lambda x: x[0])
        out = []
        for _, li in hits:
            if len(out) >= n_boxes: break
            out.append(li)
        while len(out) < n_boxes:
            out.append(len(out) % max(1, len(labels)))
        return out

    @staticmethod
    def _collect_seg_codes_after_positions(seg_positions: List[Tuple[int, int]], box_last_pos: List[int]) -> List[Optional[List[int]]]:
        """
        จับคู่ seg-16 ต่อกล่อง โดยใช้โทเค็น seg ที่ปรากฏหลังตำแหน่ง loc ตัวสุดท้ายของกล่องนั้น
        """
        seg_sorted = sorted(seg_positions, key=lambda x: x[0])
        out = []
        start = 0
        for bp in box_last_pos:
            batch = []
            i = start
            while i < len(seg_sorted) and len(batch) < 16:
                pos, idx = seg_sorted[i]
                if pos > bp:
                    batch.append(idx)
                i += 1
            out.append(batch if len(batch) == 16 else None)
            start = i
        return out

    @staticmethod
    def _collect_seg_codes_global(seg_positions: List[Tuple[int, int]], n_boxes: int) -> List[Optional[List[int]]]:
        """fallback: จัด seg เป็นก้อน ๆ 16 ตัวไปตามลำดับของข้อความ โดยไม่ผูกกับกล่อง"""
        seg_sorted = [idx for _, idx in sorted(seg_positions, key=lambda x: x[0])]
        out, i = [], 0
        for _ in range(n_boxes):
            out.append(seg_sorted[i:i+16] if i + 16 <= len(seg_sorted) else None)
            i += 16
        return out

    # ----------- inference -----------
    def infer(
        self, image: Image.Image, prompt: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[Image.Image]]:
        self.load(params.get("size_repo") if params else None)

        if image is None:
            raise gr.Error("Please upload the image ✨")
        if hasattr(image, "mode") and image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        text_prompt = (prompt or self.default_prompt).strip()

        # ปรับพารามิเตอร์ NMS จาก UI (ถ้ามี)
        apply_nms = (params.get("apply_nms") if params else None)
        if apply_nms is None:
            apply_nms = True
        nms_iou = (params.get("nms_iou") if params else None) or 0.5

        # ------------------ Detection mode ------------------
        if text_prompt.lower().startswith("detect"):
            det_prompt, labels = _build_prefixed_prompt(text_prompt, "detect", default_label="object")
            inputs = self.processor(text=det_prompt, images=[image], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=192, do_sample=False)
            txt = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            W, H = image.size
            locs = [(m.start(), int(m.group(1))) for m in LOC_TAG.finditer(txt)]
            boxes, _ = self._group_locs_to_boxes(locs, W, H)

            label_idx = self._assign_labels(txt, labels, len(boxes))
            scores = [0.99] * len(boxes)

            if apply_nms and boxes:
                boxes, label_idx, scores = _nms_per_class(boxes, label_idx, scores, nms_iou)

            if boxes:
                out_img = _draw_boxes_keep_all(image, boxes, label_idx, scores, labels)
                return None, out_img
            else:
                return f"[detect] no boxes parsed:\n{txt}", None

        # ------------------ Segmentation mode ------------------
        if text_prompt.lower().startswith("segment"):
            seg_prompt, labels = _build_prefixed_prompt(text_prompt, "segment", default_label="object")
            inputs = self.processor(text=seg_prompt, images=[image], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            txt = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            W, H = image.size
            # loc → boxes
            locs = [(m.start(), int(m.group(1))) for m in LOC_TAG.finditer(txt)]
            boxes, last_pos = self._group_locs_to_boxes(locs, W, H)

            # seg tokens
            segs = [(m.start(), int(m.group(1))) for m in SEG_TAG.finditer(txt)]
            no_seg_msg = None
            if not segs:
                no_seg_msg = "[segment] model returned no <seg###> tokens; falling back to boxes."

            # จับคู่ seg-16 ต่อกล่อง
            seg_codes_per_box = self._collect_seg_codes_after_positions(segs, last_pos) if segs else [None]*len(boxes)
            if all(c is None for c in seg_codes_per_box) and segs:
                seg_codes_per_box = self._collect_seg_codes_global(segs, len(boxes))

            # ถอดรหัส mask ทั้งก้อน (ถ้ามีครบ 16)
            masks: List[Optional[np.ndarray]] = [None] * len(boxes)
            need_decode = [codes for codes in seg_codes_per_box if codes is not None]
            idx_map = [i for i, codes in enumerate(seg_codes_per_box) if codes is not None]
            vae_problem_msg = None
            if need_decode:
                vae_chk = _load_vae_npz_local()
                if vae_chk is None:
                    vae_problem_msg = "[segment] VAE weights not available; showing bboxes only."
                else:
                    arr = np.stack([np.array(codes, dtype=np.int32) for codes in need_decode], axis=0)  # (B,16)
                    ms = _decode_masks_pt(arr, vae_chk)  # (B,64,64) หรือ None
                    if ms is None:
                        vae_problem_msg = "[segment] VAE decode failed; showing bboxes only."
                    else:
                        # วางแมสก์ตามขนาดกล่อง
                        for bx_i, m64 in zip(idx_map, ms):
                            x0, y0, x1, y1 = boxes[bx_i]
                            hh, ww = max(0, y1 - y0), max(0, x1 - x0)
                            if hh > 0 and ww > 0:
                                mm = cv2.resize((m64 * 255).astype(np.uint8), (ww, hh), interpolation=cv2.INTER_LINEAR)
                                full = np.zeros((H, W), dtype=np.uint8)
                                full[y0:y1, x0:x1] = mm
                                masks[bx_i] = full

            # ไฮไลต์ภาพด้วย mask (ถ้ามี)
            overlay = np.array(image).copy()
            used_mask = False
            for m in masks:
                if m is None:
                    continue
                used_mask = True
                overlay[m > 127, 1] = np.clip(overlay[m > 127, 1].astype(np.int32) + 60, 0, 255).astype(np.uint8)
            if used_mask:
                image = Image.fromarray(overlay)

            # วาดกรอบ + ป้าย (พร้อม NMS per-class ถ้าเปิดใช้)
            label_idx = self._assign_labels(txt, labels, len(boxes))
            scores = [0.99] * len(boxes)

            if apply_nms and boxes:
                boxes, label_idx, scores = _nms_per_class(boxes, label_idx, scores, nms_iou)

            if boxes:
                out_img = _draw_boxes_keep_all(image, boxes, label_idx, scores, labels)
                msg = None
                if no_seg_msg and vae_problem_msg:
                    msg = no_seg_msg + " " + vae_problem_msg
                elif no_seg_msg:
                    msg = no_seg_msg
                elif vae_problem_msg:
                    msg = vae_problem_msg
                return msg, out_img
            else:
                return f"[segment] no boxes/masks parsed:\n{txt}", None

        # ------------------ QA / Captioning (ปกติ) ------------------
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=192, do_sample=False)
        out_txt = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return out_txt, None
