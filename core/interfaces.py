# core/interfaces.py
from typing import Optional, Tuple, Dict, Any
from PIL import Image

class VLMAdapter:
    """Base contract for all VLM adapters (Single Responsibility: define an interface)."""
    name: str = "BaseVLM"
    sizes: list[str] = []
    default_prompt: str = ""
    supports_image_output: bool = False

    def load(self, size_repo: Optional[str] = None) -> None:
        raise NotImplementedError

    def infer(
        self,
        image: Image.Image,
        prompt: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[Image.Image]]:
        raise NotImplementedError
