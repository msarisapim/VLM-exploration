# VLM Exploration 

Easy all-in-one Vision Language Model (VLM) inference tool via Gradio UI — supports QA/Captioning, Grounding/Detection, Segmentation, and Classification via a Gradio UI. Quickly check if your dataset style is compatible with various models, and easily add new models by placing them in the core/adapters/ folders.

<img width="1288" height="942" alt="image" src="https://github.com/user-attachments/assets/7288efa9-1362-4fca-8b09-9bc1987018b8" />


## What's inside?
```
vlm_gradio_app/
  app.py
  requirements.txt
  core/
    config.py                # defaults (prompts, slider visibility)
    interfaces.py            # VLMAdapter ABC
    postproc.py              # drawing, NMS helpers
    registry.py              # model registry
    adapters/
      blip2.py               # QA/Caption
      instructblip.py        # QA/Caption
      smolvlm.py             # QA/Caption (CPU-friendly)
      llava.py               # QA/Caption
      owlvit.py              # Grounding (detection)
      groundingdino.py       # Grounding (detection)
      tinyclip.py            # Classification (image–text matching)
      paligemma.py           # QA/Caption + Detect + Segment (+ per-class NMS, mask colors)
      qwen2vl.py             # QA/Caption (Qwen2-VL Instruct)

```

## Run
```
pip install -r requirements.txt
python app.py
```

## Requirements & Notes
- Python: 3.10+

- Suggested VRAM:

   - Small QA/Caption (BLIP-2 2.7B, LLaVA-7B, PaliGemma-3B): ≥ 8–12 GB

   - Qwen2-VL-7B / 2.5-7B: ≥ 16 GB (or allow CPU offload = slower)

   - SmolVLM and TinyCLIP can run on CPU

- Windows / HF cache: If you see symlink warnings from huggingface_hub, enable Windows Developer Mode or run the shell as Administrator (optional, otherwise it just uses more disk).

- Gated models (Hugging Face):

   - google/paligemma-3b-mix-224 / -448 require Request/Agree access on the model page.

   - Authenticate before running:
    ```
    huggingface-cli login    # or set HF_TOKEN=<your_token>
    ```
- Segmentation (PaliGemma):

   - Uses model tokens <loc####> + <seg###>. If <seg> are missing, it falls back to bboxes only.

   - vae-oid.npz will be auto-downloaded on first run (cached inside core/adapters/).

- NMS:

   - Includes per-class NMS and cross-class dedup (prefers per-label fallback boxes when overlaps are very high).

   - Adjust Score/IoU in the Advanced: NMS / Thresholds section of the UI.
 
## How to use (prompts)
- QA/Caption: free-form text, e.g. What is on the tray?

- Grounding/Detect: start with detect ...
e.g. detect ring; chain; pendant; bracelet

- Segment (PaliGemma): start with segment ...
e.g. segment cat; dog

- Classification (TinyCLIP): provide labels, e.g. cat, dog, rabbit (returns top-k scores)

## Add a new model
1. Create a new adapter in core/adapters/<name>.py inheriting VLMAdapter.

2. Register it in core/registry.py.

3. (Optional) Add its size/variant options to MODEL_SIZES in app.py.


## Supported devices

This app uses `device_map="auto"` and FP16 where possible.

**NVIDIA (CUDA)**
- ✅ 8 GB (e.g., RTX 3060 Ti): TinyCLIP, OWL-ViT, GroundingDINO, SmolVLM (CPU), BLIP-2 OPT-2.7B, PaliGemma-3B-224 are comfortable.
- ⚠️ Borderline on 8 GB: PaliGemma-3B-448, LLaVA-1.5-7B, Qwen2.5-VL-3B — use CPU offload / smaller variants / 224 inputs.
- ❌ Typically too large for 8 GB: Qwen2-/2.5-VL-7B (needs ≥14–16 GB or heavy offload).

**Apple Silicon (M-series via MPS)**
- Works for BLIP-2, PaliGemma, OWL-ViT, GroundingDINO, TinyCLIP.
- 7B chat VLMs run but are slower; prefer smaller variants and 224px inputs.

**AMD (ROCm on Linux)**
- Many models work on recent ROCm; Windows ROCm is not supported.

**CPU-only**
- TinyCLIP and SmolVLM are OK; others are slow.

### Low-VRAM tips
- Prefer 224px checkpoints (e.g. `google/paligemma-3b-mix-224`).
- Keep batch size = 1; lower `max_new_tokens` for text models.
- Install `accelerate` and enable CPU offload if needed.
- On Windows, try: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.

## License

**Code:** MIT (see [LICENSE](./LICENSE)).

**Models & weights:** Not included and governed by their own licenses/terms on Hugging Face or upstream repos.  
Before downloading or using any model, please review and accept its license. Some models are **gated** and require access approval and/or authentication (e.g., `google/paligemma-3b-mix-224` / `-448`). You must comply with all third-party model/data licenses.

**Notes**
- This app downloads weights at runtime; distribution of those weights is not part of this repository.
- By using the app, you agree to the licenses of any third-party models you select.


### Third-party models (examples)
- Salesforce/BLIP-2 variants (QA/Caption)
- llava-hf/LLaVA-1.5 (QA/Caption)
- HuggingFaceTB/SmolVLM-Instruct (QA/Caption; CPU-friendly)
- google/owlvit-base-patch32 (Grounding)
- IDEA-Research/grounding-dino-base (Grounding)
- wkcn/TinyCLIP-* (Classification)
- google/paligemma-3b-mix-224 / -448 (QA/Caption/Detect/Segment; gated)
- Qwen/Qwen2-VL / Qwen2.5-VL Instruct (QA/Caption)
