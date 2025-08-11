# core/config.py

DEFAULT_PROMPTS = {
    "BLIP-2": "How many objects are on the tray?",
    "InstructBLIP": "How many objects are on the tray?",
    "SmolVLM": "How many objects are on the tray?",
    "LLaVA": "How many objects are on the tray?",
    "OWL-ViT": "ring, chain, pendant, bracelet, necklace, hand, tray, circle, earrings",
    "TinyCLIP": "ring, chain, pendant, bracelet, necklace, hand, tray, circle, earrings",
    "GroundingDINO": "ring, chain, pendant, bracelet, necklace, hand, tray, circle, earrings",
    "PaliGemma": "detect ring; pendant; bracelet; necklace; earrings", # segment, caption
    "Qwen2-VL": "Describe this image in detail.",
}

DEFAULT_SLIDERS = {
    "score_thr": {"default": 0.05, "visible_for": {"OWL-ViT", "GroundingDINO"}},
    "iou_thr":   {"default": 0.50, "visible_for": {"OWL-ViT", "GroundingDINO"}},
}
