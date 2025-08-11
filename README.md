# VLM Exploration 

Quick all-in-one VLM tester (QA/Caption, Grounding/Detect, Segment, Classification) — Gradio UI

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
