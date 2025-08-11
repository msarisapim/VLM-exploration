# core/adapters/smolvlm.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from ..interfaces import VLMAdapter

class SmolVLMAdapter(VLMAdapter):
    name = "SmolVLM"
    sizes = ["", "HuggingFaceTB/SmolVLM-Instruct"]
    default_prompt = "How many objects are on the tray?"
    supports_image_output = False

    def __init__(self):
        self.processor = None
        self.model = None
        self._repo = None

    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "HuggingFaceTB/SmolVLM-Instruct"
        if self.model is not None and self._repo == repo:
            return
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForVision2Seq.from_pretrained(
            repo, torch_dtype=torch.float32
        ).to("cpu")
        self._repo = repo

    def infer(self, image: Image.Image, prompt: str, params: Optional[Dict[str, Any]] = None):
        self.load(params.get("size_repo") if params else None)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        template = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=template, images=[image], return_tensors="pt").to("cpu")
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=200)
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text, None
