# app.py (เฉพาะส่วนที่ต่างจากเดิมของผม)

import gradio as gr
from core.registry import list_models, get_adapter
from core.config import DEFAULT_PROMPTS, DEFAULT_SLIDERS

# === จัดกลุ่มโมเดล ===
GROUNDING_MODELS = {"OWL-ViT", "GroundingDINO", "PaliGemma"}
QA_MODELS        = {"BLIP-2", "InstructBLIP", "SmolVLM", "LLaVA", "PaliGemma", "Qwen2-VL"}
CLASSIFY_MODELS  = {"TinyCLIP"}   # 🆕 กลุ่มใหม่: Classification (Image–Text Matching)

# mapping model -> size/variant choices
MODEL_SIZES = {
    "BLIP-2": ["", "Salesforce/blip2-opt-2.7b", "Salesforce/blip2-flan-t5-xl"],
    "InstructBLIP": ["", "Salesforce/instructblip-flan-t5-xl"],
    "SmolVLM": ["", "HuggingFaceTB/SmolVLM-Instruct"],
    "LLaVA": ["", "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"],
    "OWL-ViT": ["", "google/owlvit-base-patch32"],
    "GroundingDINO": ["", "IDEA-Research/grounding-dino-base"],
    "TinyCLIP": ["", "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"],
    "PaliGemma": ["", "google/paligemma-3b-mix-224", "google/paligemma-3b-mix-448"],
    "Qwen2-VL": ["", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"],
}                 

def models_for_type(group: str):
    if group == "Grounding":
        return sorted(GROUNDING_MODELS)
    if group == "QA (Text)":
        return sorted(QA_MODELS)
    # 🆕
    if group == "Classification":
        return sorted(CLASSIFY_MODELS)
    return []

def update_size_dropdown(model_name: str):
    return gr.update(choices=MODEL_SIZES.get(model_name, []), value="")

def toggle_sliders(model_name: str):
    # โชว์ slider เฉพาะ Grounding
    show_score = model_name in DEFAULT_SLIDERS["score_thr"]["visible_for"]
    show_iou   = model_name in DEFAULT_SLIDERS["iou_thr"]["visible_for"]
    s = gr.update(visible=show_score)
    i = gr.update(visible=show_iou)
    p = gr.update(value=DEFAULT_PROMPTS.get(model_name, ""))
    return s, i, p

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 VLM Inference")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Image")

        with gr.Column():
            # 🆕 มี 3 กลุ่ม
            model_type = gr.Radio(
                choices=["Grounding", "QA (Text)", "Classification"],
                value="QA (Text)",
                label="Model Type"
            )

            model_dd = gr.Dropdown(
                choices=models_for_type("QA (Text)"),
                value=models_for_type("QA (Text)")[0],
                label="Model"
            )

            # 🆕 dropdown ที่พิมพ์เองได้
            size_dd = gr.Dropdown(
                label="Model Size / Variant",
                choices=MODEL_SIZES.get(models_for_type("QA (Text)")[0], []),
                value="",
                allow_custom_value=True,
                interactive=True,
            )

            prompt_tb = gr.Textbox(
                value=DEFAULT_PROMPTS.get(models_for_type("QA (Text)")[0], ""),
                label="Prompt or Question"
            )

            with gr.Accordion("Advanced: NMS / Thresholds", open=False):
                score = gr.Slider(0.0, 1.0, value=DEFAULT_SLIDERS["score_thr"]["default"],
                                  step=0.01, label="Score threshold", visible=False)
                iou   = gr.Slider(0.0, 1.0, value=DEFAULT_SLIDERS["iou_thr"]["default"],
                                  step=0.01, label="IoU for NMS", visible=False)

            run_button = gr.Button("Run Inference")

    gr.Markdown("## Output")
    with gr.Row():
        text_output  = gr.Textbox(label="Text Output")
        image_output = gr.Image(label="Image Output")

    # เมื่อเปลี่ยนกลุ่ม → อัปเดต model list + sliders + prompt + size choices
    def on_change_group(group):
        models = models_for_type(group)
        first  = models[0] if models else None
        model_update = gr.update(choices=models, value=first)
        s, i, p = toggle_sliders(first)
        size_update = update_size_dropdown(first)
        return model_update, s, i, p, size_update

    model_type.change(
        on_change_group,
        inputs=[model_type],
        outputs=[model_dd, score, iou, prompt_tb, size_dd]
    )

    # เมื่อเปลี่ยนรุ่น → อัปเดต slider visibility / prompt / size choices
    def on_change_model(m):
        s, i, p = toggle_sliders(m)
        size_update = update_size_dropdown(m)
        return s, i, p, size_update

    model_dd.change(on_change_model, [model_dd], [score, iou, prompt_tb, size_dd])

    def run_infer(model_name, size_repo, image, prompt, score_thr, iou_thr):
        if image is None:
            import gradio as gr
            raise gr.Error("กรุณาอัปโหลดรูปก่อน ✨")
        adapter = get_adapter(model_name)
        params = {"size_repo": size_repo, "score_thr": score_thr, "iou_thr": iou_thr}
        text, img = adapter.infer(image, prompt, params)
        return text or "", img


    run_button.click(
        run_infer,
        [model_dd, size_dd, image_input, prompt_tb, score, iou],
        [text_output, image_output]
    )

if __name__ == "__main__":
    demo.launch()
