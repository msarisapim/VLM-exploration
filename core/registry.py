# core/registry.py
from typing import Dict, Type, List
from .interfaces import VLMAdapter
from .adapters.blip2 import Blip2Adapter
from .adapters.instructblip import InstructBlipAdapter
from .adapters.smolvlm import SmolVLMAdapter
from .adapters.llava import LlavaAdapter
from .adapters.owlvit import OwlVitAdapter
from .adapters.tinyclip import TinyClipAdapter
from .adapters.groundingdino import GroundingDinoAdapter
from .adapters.paligemma import PaliGemmaAdapter
from .adapters.qwen2vl import Qwen2VLAdapter

_REGISTRY: Dict[str, Type[VLMAdapter]] = {
    Blip2Adapter.name: Blip2Adapter,
    InstructBlipAdapter.name: InstructBlipAdapter,
    SmolVLMAdapter.name: SmolVLMAdapter,
    LlavaAdapter.name: LlavaAdapter,
    OwlVitAdapter.name: OwlVitAdapter,
    TinyClipAdapter.name: TinyClipAdapter,
    GroundingDinoAdapter.name: GroundingDinoAdapter,
    PaliGemmaAdapter.name: PaliGemmaAdapter,
    Qwen2VLAdapter.name: Qwen2VLAdapter,
}

def list_models() -> List[str]:
    return list(_REGISTRY.keys())

def get_adapter(name: str) -> VLMAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return _REGISTRY[name]()
