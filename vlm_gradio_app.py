# Gradio web app that unifies BLIP, InstructBLIP, SmolVLM, LLaVA, OWL-ViT, and TinyCLIP
# Upload an image, pick a model, run inference

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

# Device
# Avoid accelerate/meta-device crash by not calling .to(device) on offloaded models
# Use model-specific device handling

# ============================================ Load Models ============================================================
blip_processor, blip_model = None, None
iblip_processor, iblip_model = None, None
smol_processor, smol_model = None, None
llava_processor, llava_model = None, None
owlvit_processor, owlvit_model = None, None
tclip_processor, tclip_model = None, None

def load_blip():
    global blip_processor, blip_model
    if blip_processor is None:
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16
        )

def load_owlvit():
    global owlvit_processor, owlvit_model
    if owlvit_processor is None:
        owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        owlvit_model = owlvit_model.to("cuda" if torch.cuda.is_available() else "cpu")

def load_instructblip():
    global iblip_processor, iblip_model
    if iblip_processor is None:
        iblip_processor = AutoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        iblip_model = AutoModelForVision2Seq.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            device_map="auto",
            torch_dtype=torch.float16
        )

def load_smolvlm():
    global smol_processor, smol_model
    if smol_processor is None:
        smol_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        smol_model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.float32
        ).to("cpu")

def load_llava():
    global llava_processor, llava_model
    if llava_processor is None:
        llava_processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            device_map="auto"
        )

def load_tinyclip():
    global tclip_processor, tclip_model
    if tclip_processor is None:
        tclip_processor = CLIPProcessor.from_pretrained("laion/TinyCLIP-ViT-Tiny-patch16-224")
        tclip_model = CLIPModel.from_pretrained("laion/TinyCLIP-ViT-Tiny-patch16-224")
        tclip_model = tclip_model.to("cuda" if torch.cuda.is_available() else "cpu")

# ======================================== Inference ==================================================================

def blip_infer(image, question):
    load_blip()
    inputs = blip_processor(images=image, text=question, return_tensors="pt")
    device = next(blip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def iblip_infer(image, question):
    load_instructblip()
    inputs = iblip_processor(images=image, text=question, return_tensors="pt")
    device = next(iblip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = iblip_model.generate(**inputs, max_new_tokens=50)
    return iblip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def smol_infer(image, question):
    load_smolvlm()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = smol_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = smol_processor(text=prompt, images=[image], return_tensors="pt").to("cpu")
    with torch.no_grad():
        generated_ids = smol_model.generate(**inputs, max_new_tokens=200)
    return smol_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def llava_infer(image, question):
    load_llava()
    inputs = llava_processor(images=image, text=question, return_tensors="pt")
    device = next(llava_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = llava_model.generate(**inputs, max_new_tokens=50)
    return llava_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def owlvit_infer(image, prompt_str):
    load_owlvit()
    prompts = [p.strip() for p in prompt_str.split(",") if p.strip()]
    device = next(owlvit_model.parameters()).device
    inputs = owlvit_processor(text=prompts, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owlvit_model(**inputs)
    size = torch.tensor([image.size[::-1]], device=device)
    results = owlvit_processor.post_process_object_detection(outputs, size, threshold=0.05)[0]
    # Draw
    img_np = np.array(image)
    for box, label_id, score in zip(results["boxes"], results["labels"], results["scores"]):
        box = box.int().tolist()
        label = prompts[label_id]
        cv2.rectangle(img_np, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
        cv2.putText(img_np, f"{label} {score:.2f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return Image.fromarray(img_np)

def tinyclip_infer(image, label_str):
    load_tinyclip()
    labels = label_str.split()
    device = next(tclip_model.parameters()).device
    inputs = tclip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = tclip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze()
    best = torch.argmax(probs).item()
    return f"{labels[best]} ({probs[best]:.2f})"

# Gradio App

def run(model, size, image, prompt):
    if model == "BLIP-2":
        return blip_infer(image, prompt)
    elif model == "InstructBLIP":
        return iblip_infer(image, prompt)
    elif model == "SmolVLM":
        return smol_infer(image, prompt)
    elif model == "LLaVA":
        return llava_infer(image, prompt)
    elif model == "OWL-ViT":
        return owlvit_infer(image, prompt)
    elif model == "TinyCLIP":
        return tinyclip_infer(image, prompt)

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 VLM Inference Web App (Gradio)")

    with gr.Row():
        image_input = gr.Image(type="pil")
        model_choice = gr.Dropdown(
            ["BLIP-2", "InstructBLIP", "SmolVLM", "LLaVA", "OWL-ViT", "TinyCLIP"],
            label="Model Type",
            value="BLIP-2"
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
                        "HuggingFaceM4/tiny-clip-vit-small-patch16"
                    ],
                    value="",
                    interactive=True,
                    allow_custom_value=True  # 🆕 let users type custom model
                )


    prompt_input = gr.Textbox(label="Prompt or Question", value="How many objects are on the tray?")
    run_button = gr.Button("Run Inference")
    output = gr.Textbox(label="Output (or image for OWL-ViT)")

    # Dynamically update size options based on selected model
    def update_model_size(model_type):
        options = {
            "BLIP-2": ["", "Salesforce/blip2-opt-2.7b", "Salesforce/blip2-flan-t5-xl"],
            "InstructBLIP": ["", "Salesforce/instructblip-flan-t5-xl"],
            "SmolVLM": ["", "HuggingFaceTB/SmolVLM-Instruct"],
            "LLaVA": ["", "llava-hf/llava-1.5-7b-hf"],
            "OWL-ViT": ["", "google/owlvit-base-patch32"],
            "TinyCLIP": ["", "HuggingFaceM4/tiny-clip-vit-small-patch16"],
        }
        if model_type in options:
            return gr.update(choices=options[model_type], value="", interactive=True)
        return gr.update(choices=[], value=None, interactive=False)

    model_choice.change(fn=update_model_size, inputs=model_choice, outputs=model_size)

    run_button.click(fn=run, inputs=[model_choice, model_size, image_input, prompt_input], outputs=output)

demo.launch()
