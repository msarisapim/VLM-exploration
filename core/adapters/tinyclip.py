# core/adapters/tinyclip.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from ..interfaces import VLMAdapter
from ..postproc import draw_topk_legend

class TinyClipAdapter(VLMAdapter):
    name = "TinyCLIP"
    sizes = ["", "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"]
    default_prompt = "ring, chain, pendant, bracelet, necklace, hand, tray, circle, earrings"
    supports_image_output = True

    def __init__(self):
        self.processor = None
        self.model = None
        self._repo = None

    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
        if self.model is not None and self._repo == repo:
            return
        self.processor = CLIPProcessor.from_pretrained(repo)
        self.model = CLIPModel.from_pretrained(repo)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._repo = repo

    def infer(self, image: Image.Image, prompt: str, params: Optional[Dict[str, Any]] = None):
        self.load(params.get("size_repo") if params else None)
        label_str = prompt or ""
        labels = [s.strip() for s in label_str.split(",") if s.strip()]
        if not labels:
            return "Provide labels separated by commas, e.g., 'diamond, ruby, sapphire'", None

        device = next(self.model.parameters()).device
        inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
        k = min(3, len(labels))
        vals, idxs = torch.topk(probs, k=k)
        top_lines = [f"{labels[int(i)]} ({float(v):.2f})" for v, i in zip(vals, idxs)]
        out_img = draw_topk_legend(image, top_lines)
        return "\n".join(top_lines), out_img
