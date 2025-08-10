# vlm_gradio_app.py
import gradio as gr
from PIL import Image
import numpy as np
import torch
import cv2

from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    OwlViTProcessor, OwlViTForObjectDetection,
    AutoProcessor, AutoModelForVision2Seq,
    CLIPProcessor, CLIPModel,
    LlavaProcessor, LlavaForConditionalGeneration
)

# Default prompts per model
DEFAULT_PROMPTS = {
    "BLIP-2": "Describe this image in one sentence.",
    "InstructBLIP": "Answer the question about this image.",
    "SmolVLM": "How many objects are on the tray?",
    "LLaVA": "How many objects are on the tray?",
    "OWL-ViT": "ring, jewelry, hand, chain, charm, pendant, tray",
    "TinyCLIP": "ring, jewelry, hand, chain, charm, pendant, tray",
}

# ============================== Global handles (cached by repo id) ==============================
blip_processor, blip_model = None, None
iblip_processor, iblip_model = None, None
smol_processor, smol_model = None, None
llava_processor, llava_model = None, None
owlvit_processor, owlvit_model = None, None
tclip_processor, tclip_model = None, None

# ============================== Loaders =========================================================
def load_blip(repo_id="Salesforce/blip2-opt-2.7b"):
    global blip_processor, blip_model
    if blip_processor is None or getattr(blip_processor, "_repo_id", None) != repo_id:
        blip_processor = Blip2Processor.from_pretrained(repo_id)
        blip_processor._repo_id = repo_id
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            repo_id, device_map="auto",
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
        )

def load_instructblip(repo_id="Salesforce/instructblip-flan-t5-xl"):
    global iblip_processor, iblip_model
    if iblip_processor is None or getattr(iblip_processor, "_repo_id", None) != repo_id:
        iblip_processor = AutoProcessor.from_pretrained(repo_id)
        iblip_processor._repo_id = repo_id
        iblip_model = AutoModelForVision2Seq.from_pretrained(
            repo_id, device_map="auto",
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
        )

def load_smolvlm(repo_id="HuggingFaceTB/SmolVLM-Instruct"):
    global smol_processor, smol_model
    if smol_processor is None or getattr(smol_processor, "_repo_id", None) != repo_id:
        smol_processor = AutoProcessor.from_pretrained(repo_id)
        smol_processor._repo_id = repo_id
        smol_model = AutoModelForVision2Seq.from_pretrained(
            repo_id, torch_dtype=torch.float32
        ).to("cpu")  # SmolVLM is tiny; keep on CPU for portability

def load_llava(repo_id="llava-hf/llava-1.5-7b-hf"):
    global llava_processor, llava_model
    if llava_processor is None or getattr(llava_processor, "_repo_id", None) != repo_id:
        llava_processor = LlavaProcessor.from_pretrained(repo_id)
        llava_processor._repo_id = repo_id
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            repo_id, device_map="auto",
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
        )

def load_owlvit(repo_id="google/owlvit-base-patch32"):
    global owlvit_processor, owlvit_model
    if owlvit_processor is None or getattr(owlvit_processor, "_repo_id", None) != repo_id:
        owlvit_processor = OwlViTProcessor.from_pretrained(repo_id)
        owlvit_processor._repo_id = repo_id
        owlvit_model = OwlViTForObjectDetection.from_pretrained(repo_id)
        owlvit_model = owlvit_model.to("cuda" if torch.cuda.is_available() else "cpu")

def load_tinyclip(repo_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"):
    global tclip_processor, tclip_model
    if tclip_processor is None or getattr(tclip_processor, "_repo_id", None) != repo_id:
        tclip_processor = CLIPProcessor.from_pretrained(repo_id)
        tclip_processor._repo_id = repo_id
        tclip_model = CLIPModel.from_pretrained(repo_id)
        tclip_model = tclip_model.to("cuda" if torch.cuda.is_available() else "cpu")

# ============================== Inference =======================================================
def blip_infer(image, question, size_repo):
    repo_id = size_repo or "Salesforce/blip2-opt-2.7b"
    load_blip(repo_id)
    inputs = blip_processor(images=image, text=question, return_tensors="pt")
    device = next(blip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = blip_model.generate(**inputs, max_new_tokens=64)
    return blip_processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)

def iblip_infer(image, question, size_repo):
    repo_id = size_repo or "Salesforce/instructblip-flan-t5-xl"
    load_instructblip(repo_id)
    inputs = iblip_processor(images=image, text=question, return_tensors="pt")
    device = next(iblip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = iblip_model.generate(**inputs, max_new_tokens=64)
    return iblip_processor.batch_decode(out_ids, skip_special_tokens=True)[0]

def smol_infer(image, question, size_repo):
    repo_id = size_repo or "HuggingFaceTB/SmolVLM-Instruct"
    load_smolvlm(repo_id)
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    prompt = smol_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = smol_processor(text=prompt, images=[image], return_tensors="pt").to("cpu")
    with torch.no_grad():
        out_ids = smol_model.generate(**inputs, max_new_tokens=200)
    return smol_processor.batch_decode(out_ids, skip_special_tokens=True)[0]

def llava_infer(image, question, size_repo):
    repo_id = size_repo or "llava-hf/llava-1.5-7b-hf"
    load_llava(repo_id)
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    prompt = llava_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = llava_processor(text=prompt, images=[image], return_tensors="pt")
    device = next(llava_model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = llava_model.generate(**inputs, max_new_tokens=128)
    return llava_processor.batch_decode(out_ids, skip_special_tokens=True)[0]

def owlvit_infer(image, prompt_str, size_repo):
    repo_id = size_repo or "google/owlvit-base-patch32"
    load_owlvit(repo_id)

    prompts = [p.strip() for p in prompt_str.split(",") if p.strip()]
    if not prompts:
        return image

    device = next(owlvit_model.parameters()).device

    inputs = owlvit_processor(text=prompts, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = owlvit_model(**inputs)

    H, W = image.size[1], image.size[0]
    target_sizes_tensor = torch.tensor([[H, W]], device=device)

    # ---- Robust post-process: prefer old stable API ----
    try:
        results = owlvit_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes_tensor, threshold=0.05
        )[0]
    except Exception:
        try:
            results = owlvit_processor.post_process_object_detection(
                outputs, target_sizes_tensor, 0.05
            )[0]
        except Exception:
            results = owlvit_processor.post_process_grounded_object_detection(
                outputs, [(H, W)], 0.05
            )[0]

    # Draw with readable label background
    img_np = np.array(image)
    font = cv2.FONT_HERSHEY_SIMPLEX

    boxes = results.get("boxes", [])
    labels = results.get("labels", [])
    scores = results.get("scores", [])

    for box, lid, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]

        # convert label id robustly to int
        try:
            li = int(lid) if isinstance(lid, (int, np.integer)) else int(lid.item() if hasattr(lid, "item") else int(lid))
        except Exception:
            li = 0
        cls = prompts[li] if 0 <= li < len(prompts) else str(li)

        # bbox
        cv2.rectangle(img_np, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # label text + score
        label_text = f"{cls} {float(score):.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, font, 0.6, 2)

        # background rectangle (clamped in image)
        bx0 = x0
        by0 = max(0, y0 - th - baseline - 6)
        bx1 = min(img_np.shape[1] - 1, x0 + tw + 8)
        by1 = y0

        cv2.rectangle(img_np, (bx0, by0), (bx1, by1), (0, 0, 0), thickness=-1)
        cv2.putText(img_np, label_text, (x0 + 4, y0 - 6),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return Image.fromarray(img_np)

def tinyclip_infer(image, label_str, size_repo, topk=3, draw=True):
    repo_id = size_repo or "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
    try:
        load_tinyclip(repo_id)
    except Exception as e:
        return f"Failed to load TinyCLIP: {e}", None

    labels = [s.strip() for s in label_str.split(",") if s.strip()]
    if not labels:
        return "Provide labels separated by commas, e.g., 'diamond, ruby, sapphire'", None

    device = next(tclip_model.parameters()).device
    inputs = tclip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = tclip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)  # [num_labels]
    k = min(topk, len(labels))
    vals, idxs = torch.topk(probs, k=k)
    top_lines = [f"{labels[int(i)]} ({float(v):.2f})" for v, i in zip(vals, idxs)]

    text_out = "\n".join(top_lines)

    if not draw:
        return text_out, None

    # Draw a small legend on the image
    img_np = np.array(image).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, thickness = 0.7, 2
    pad = 8
    # compute box size
    tw = th_total = 0
    for line in top_lines:
        (tw_i, th_i), base = cv2.getTextSize(line, font, fs, thickness)
        tw = max(tw, tw_i)
        th_total += th_i + base + 4
    x0, y0 = 10, 10 + th_total
    x1, y1 = x0 + tw + 2 * pad, y0 + pad
    # background
    cv2.rectangle(img_np, (x0 - pad, 10), (x1, y1), (0, 0, 0), -1)

    # put lines
    y = 10 + pad + 20
    for line in top_lines:
        cv2.putText(img_np, line, (x0, y), font, fs, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 24

    return text_out, Image.fromarray(img_np)

# ============================== Gradio wiring ===================================================
def run(model, size_repo, image, prompt):
    # Return (text, image)
    if model == "OWL-ViT":
        img = owlvit_infer(image, prompt, size_repo)
        return "", img
    if model == "TinyCLIP":
        txt, img = tinyclip_infer(image, prompt, size_repo, topk=3, draw=True)
        return txt, img
    if model == "BLIP-2":
        txt = blip_infer(image, prompt, size_repo)
        return txt, None
    if model == "InstructBLIP":
        txt = iblip_infer(image, prompt, size_repo)
        return txt, None
    if model == "SmolVLM":
        txt = smol_infer(image, prompt, size_repo)
        return txt, None
    if model == "LLaVA":
        txt = llava_infer(image, prompt, size_repo)
        return txt, None
    return "Unknown model", None

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 VLM Inference (Gradio)")
    edited_flag = gr.State(False)  # becomes True once user types in the prompt

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")
        model_choice = gr.Dropdown(
            ["BLIP-2", "InstructBLIP", "SmolVLM", "LLaVA", "OWL-ViT", "TinyCLIP"],
            label="Model Type", value="BLIP-2"
        )

    model_size = gr.Dropdown(
        label="Model Size / Variant (you can also type)",
        choices=[
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-flan-t5-xl",
            "Salesforce/instructblip-flan-t5-xl",
            "HuggingFaceTB/SmolVLM-Instruct",
            "llava-hf/llava-1.5-7b-hf",
            "google/owlvit-base-patch32",
            "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
            "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
        ],
        value="",
        interactive=True,
        allow_custom_value=True
    )

    prompt_input = gr.Textbox(label="Prompt or Question", value=DEFAULT_PROMPTS["BLIP-2"])
    reset_prompt_btn = gr.Button("Reset prompt to model default")

    with gr.Row():
        run_button = gr.Button("Run Inference")
        text_output = gr.Textbox(label="Text Output")
        image_output = gr.Image(label="Image Output")

    # --- helpers ---
    def update_on_model_change(model_type, edited):
        # size dropdown (same logic as before)
        options = {
            "BLIP-2": ["", "Salesforce/blip2-opt-2.7b", "Salesforce/blip2-flan-t5-xl"],
            "InstructBLIP": ["", "Salesforce/instructblip-flan-t5-xl"],
            "SmolVLM": ["", "HuggingFaceTB/SmolVLM-Instruct"],
            "LLaVA": ["", "llava-hf/llava-1.5-7b-hf"],
            "OWL-ViT": ["", "google/owlvit-base-patch32"],
            "TinyCLIP": ["", "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"],
        }
        size_update = gr.update(choices=options.get(model_type, []), value="", interactive=True)

        # only set the default prompt if user hasn't typed yet
        if not edited:
            prompt_update = gr.update(value=DEFAULT_PROMPTS.get(model_type, ""))
        else:
            prompt_update = gr.update()  # no change

        return size_update, prompt_update

    def mark_prompt_edited(_):
        return True  # once user types, stop auto-overwriting

    def reset_prompt(model_type):
        return gr.update(value=DEFAULT_PROMPTS.get(model_type, "")), False  # also clear edited flag

    # wire events
    model_choice.change(fn=update_on_model_change,
                        inputs=[model_choice, edited_flag],
                        outputs=[model_size, prompt_input])

    prompt_input.change(fn=mark_prompt_edited, inputs=prompt_input, outputs=edited_flag)

    reset_prompt_btn.click(fn=reset_prompt, inputs=model_choice, outputs=[prompt_input, edited_flag])

    run_button.click(fn=run, inputs=[model_choice, model_size, image_input, prompt_input],
                     outputs=[text_output, image_output])

demo.launch()
