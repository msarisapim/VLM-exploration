# core/adapters/qwen2vl.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
)

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

from qwen_vl_utils import process_vision_info
from ..interfaces import VLMAdapter

class Qwen2VLAdapter(VLMAdapter):
    name = "Qwen2-VL"
    sizes = [
        "", "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct",
    ]
    default_prompt = "Describe this image in detail."
    supports_image_output = False

    def __init__(self):
        self.tokenizer = None
        self.processor = None
        self.model = None
        self._repo = None

    def load(self, size_repo: Optional[str] = None) -> None:
        repo = size_repo or "Qwen/Qwen2-VL-7B-Instruct"
        if self.model is not None and self._repo == repo:
            return

        cfg = AutoConfig.from_pretrained(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(repo)

        # select classes by structures
        if cfg.model_type == "qwen2_5_vl" and Qwen2_5_VLForConditionalGeneration:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repo, torch_dtype="auto", device_map="auto"
            )
        elif cfg.model_type == "qwen2_vl" and Qwen2VLForConditionalGeneration:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                repo, torch_dtype="auto", device_map="auto"
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                repo, torch_dtype="auto", device_map="auto"
            )
        self._repo = repo

    def infer(self, image, prompt, params=None):
        self.load(params.get("size_repo") if params else None)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},             
                {"type": "text", "text": prompt or "Describe this image."}
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # supports both tuple and dict from qwen_vl_utils
        res = process_vision_info(messages)
        if isinstance(res, tuple):
            vision_images, vision_videos = res
        else:
            vision_images = res.get("images")
            vision_videos = res.get("videos")

        inputs = self.processor(
            text=[text],
            images=vision_images,
            videos=vision_videos,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=192)  
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output_text, None