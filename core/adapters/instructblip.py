# core/adapters/instructblip.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from ..interfaces import VLMAdapter

class InstructBlipAdapter(VLMAdapter):
    name = "InstructBLIP"
    sizes = ["", "Salesforce/instructblip-flan-t5-xl"]
    default_prompt = "How many objects are on the tray?"
    supports_image_output = False

    def __init__(self):
        self.processor = None
        self.model = None
        self._repo = None

    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "Salesforce/instructblip-flan-t5-xl"
        if self.model is not None and self._repo == repo:
            return
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForVision2Seq.from_pretrained(
            repo, device_map="auto",
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        self._repo = repo

    def infer(self, image: Image.Image, prompt: str, params: Optional[Dict[str, Any]] = None):
        self.load(params.get("size_repo") if params else None)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=64)
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text, None
