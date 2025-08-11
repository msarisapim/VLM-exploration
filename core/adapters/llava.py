# core/adapters/llava.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from ..interfaces import VLMAdapter

class LlavaAdapter(VLMAdapter):
    name = "LLaVA"
    sizes = ["", "llava-hf/llava-1.5-7b-hf"]
    default_prompt = "How many objects are on the tray?"
    supports_image_output = False

    def __init__(self):
        self.processor = None
        self.model = None
        self._repo = None

    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "llava-hf/llava-1.5-7b-hf"
        if self.model is not None and self._repo == repo:
            return
        self.processor = LlavaProcessor.from_pretrained(repo)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            repo, device_map="auto",
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        self._repo = repo

    def infer(self, image: Image.Image, prompt: str, params: Optional[Dict[str, Any]] = None):
        self.load(params.get("size_repo") if params else None)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        template = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=template, images=[image], return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=128)
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text, None
