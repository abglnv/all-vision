import os
import numpy as np
import gradio as gr
from PIL import Image
from ai_edge_litert.interpreter import Interpreter 

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

INPUT_SIZE = (224, 224)
MODEL_PATH = os.getenv("TFLITE_PATH", "model.tflite")

class_names = ["Benign", "Early", "Pre", "Pro"]

_InterpreterClass = None
def _resolve_interpreter_class():
    global _InterpreterClass
    if _InterpreterClass is not None:
        return _InterpreterClass
    try:
        _InterpreterClass = Interpreter
        print("Using ai-edge-litert", flush=True)
    except Exception as e_litert:
        raise RuntimeError(
                "No TFLite interpreter available. Install one of:\n"
                "  - pip install ai-edge-litert  (recommended lightweight)\n"
                "  - pip install tensorflow      (fallback)\n"
                f"ai-edge-litert import error: {e_litert}\n"
            )
    return _InterpreterClass

_interpreter = None
def get_interpreter():
    global _interpreter
    if _interpreter is None:
        Interpreter = _resolve_interpreter_class()
        _interpreter = Interpreter(model_path=MODEL_PATH, num_threads=1)
        _interpreter.allocate_tensors()
    return _interpreter

def preprocess(pil_image: Image.Image, in_detail):
    img = pil_image.convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(img)

    arr = np.expand_dims(arr, axis=0)

    dtype = in_detail["dtype"]
    if dtype == np.uint8:
        scale, zero = in_detail.get("quantization", (0.0, 0))
        if scale == 0.0:
            qp = in_detail.get("quantization_parameters", {})
            scales = qp.get("scales", [])
            zeros = qp.get("zero_points", [])
            scale = scales[0] if len(scales) else 1.0
            zero = zeros[0] if len(zeros) else 0
        x = arr.astype(np.float32) / 255.0
        xq = x / scale + zero
        xq = np.clip(np.round(xq), 0, 255).astype(np.uint8)
        return xq
    else:
        x = arr.astype(np.float32) / 255.0
        return x

def postprocess(y, out_detail):
    dtype = out_detail["dtype"]
    preds = y
    if dtype == np.uint8:
        scale, zero = out_detail.get("quantization", (0.0, 0))
        if scale == 0.0:
            qp = out_detail.get("quantization_parameters", {})
            scales = qp.get("scales", [])
            zeros = qp.get("zero_points", [])
            scale = scales[0] if len(scales) else 1.0
            zero = zeros[0] if len(zeros) else 0
        preds = (y.astype(np.float32) - zero) * scale
    preds = np.squeeze(preds)

    if preds.ndim == 1:
        if class_names and len(class_names) == preds.shape[0]:
            scores = {name: float(p) for name, p in zip(class_names, preds)}
        else:
            scores = {f"class_{i}": float(p) for i, p in enumerate(preds)}
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return {k: v for k, v in top}
    return {"score": float(preds)}

def predict(image: Image.Image):
    interpreter = get_interpreter()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    x = preprocess(image, in_detail)
    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out_detail["index"])

    return postprocess(y, out_detail)

def build_ui():
    with gr.Blocks(title="LiteRT TFLite CNN (244x244)") as demo:
        gr.Markdown("LiteRT TFLite CNN Inference (244x244)")
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload image")
            out = gr.Label(num_top_classes=4, label="Prediction")

        inp.change(predict, inputs=inp, outputs=out)
        gr.Button("Run").click(predict, inputs=inp, outputs=out)
    return demo

demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))