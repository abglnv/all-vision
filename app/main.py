# pip install onnx onnxruntime gradio pillow numpy
import os
import numpy as np
import gradio as gr
from PIL import Image
import onnx
import onnxruntime as ort

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Config
MODEL_PATH = os.getenv("ONNX_PATH", "resnet50.onnx")
INPUT_SIZE = (224, 224)  # fallback (W, H) if model shape is dynamic
CLASS_NAMES = ["Benign", "Early", "Pre", "Pro"]  # adjust to your model
USE_CUDA = False  # set True if onnxruntime-gpu installed and CUDA is available

# ORT providers
PROVIDERS = ["CUDAExecutionProvider"] if USE_CUDA else ["CPUExecutionProvider"]

# Globals for lazy init
_sess = None
_input_info = None
_output_info = None


def _np_dtype_from_onnx_type(onnx_type: str):
    mapping = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(onnx_type, np.float32)


def _shape_dim(val, fallback):
    # val may be int, None, or a string like 'N'/'unk__822'
    return val if isinstance(val, int) else fallback


def get_session():
    global _sess, _input_info, _output_info
    if _sess is not None:
        return _sess

    # Validate model file
    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=PROVIDERS)

    inputs = _sess.get_inputs()
    outputs = _sess.get_outputs()
    assert len(inputs) == 1, "This app expects a single-input ONNX model."
    assert len(outputs) >= 1, "Model must have at least one output."

    _input_info = {
        "name": inputs[0].name,
        "shape": inputs[0].shape,  # may contain None or strings for dynamic dims
        "dtype": inputs[0].type,
    }
    _output_info = {
        "name": outputs[0].name,
        "shape": outputs[0].shape,
        "dtype": outputs[0].type,
    }

    print(f"Using ONNX Runtime providers: {_sess.get_providers()}", flush=True)
    print(f"Input:  {_input_info}", flush=True)
    print(f"Output: {_output_info}", flush=True)

    return _sess


def preprocess(pil_image: Image.Image):
    if pil_image is None:
        raise ValueError("No image provided.")

    sess = get_session()
    input_name = _input_info["name"]
    in_shape = _input_info["shape"]  # e.g., [None, 224, 224, 3] or [1, 3, 224, 224]
    in_dtype = _np_dtype_from_onnx_type(_input_info["dtype"])

    # Detect NHWC vs NCHW
    if len(in_shape) == 4 and in_shape[-1] == 3:
        # NHWC
        is_nhwc = True
        height = _shape_dim(in_shape[1], INPUT_SIZE[1])
        width = _shape_dim(in_shape[2], INPUT_SIZE[0])
    elif len(in_shape) == 4:
        # Assume NCHW
        is_nhwc = False
        height = _shape_dim(in_shape[2], INPUT_SIZE[1])
        width = _shape_dim(in_shape[3], INPUT_SIZE[0])
    else:
        # Fallback
        is_nhwc = True
        height, width = INPUT_SIZE[1], INPUT_SIZE[0]

    # EXACT same preprocessing as your working script:
    # 1) Resize
    img = pil_image.convert("RGB").resize((width, height), Image.BILINEAR)
    x = np.array(img).astype(np.float32)
    # 2) RGB -> BGR
    x = x[:, :, ::-1]
    # 3) Mean subtraction (ImageNet-style for ResNet50)
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    x = x - mean

    # Layout
    if is_nhwc:
        x = np.expand_dims(x, axis=0)  # [1, H, W, 3]
    else:
        x = np.transpose(x, (2, 0, 1))  # [3, H, W]
        x = np.expand_dims(x, axis=0)   # [1, 3, H, W]

    # Cast to expected dtype (usually float32)
    if in_dtype != x.dtype:
        x = x.astype(in_dtype)

    return input_name, x


def softmax(vec: np.ndarray) -> np.ndarray:
    # stable softmax for 1D
    v = vec - np.max(vec)
    e = np.exp(v)
    s = e.sum()
    return e / s if s != 0 else np.zeros_like(vec)


def postprocess(outputs):
    # Use probabilities (softmax) as requested
    y = outputs[0]
    y = np.squeeze(y)
    if y.ndim == 1:
        scores = y
    elif y.ndim == 2:
        scores = y[0]
    else:
        # Unexpected shape; return scalar if possible
        return {"score": float(np.ravel(y)[0])}

    probs = softmax(scores)

    if CLASS_NAMES and len(CLASS_NAMES) == probs.shape[0]:
        pairs = list(zip(CLASS_NAMES, probs.tolist()))
    else:
        pairs = [(f"class_{i}", float(v)) for i, v in enumerate(probs.tolist())]

    top = sorted(pairs, key=lambda kv: kv[1], reverse=True)[:3]
    return {k: float(v) for k, v in top}


def predict(image: Image.Image):
    if image is None:
        return {"error": "Upload an image first."}
    sess = get_session()
    input_name, x = preprocess(image)
    outputs = sess.run(None, {input_name: x})
    return postprocess(outputs)


def debug_pre(image: Image.Image):
    # Returns preprocessing diagnostics
    if image is None:
        return {"error": "Upload an image first."}

    _ = get_session()  # ensure metadata is initialized
    try:
        input_name, x = preprocess(image)
    except Exception as e:
        return {"error": f"Preprocess failed: {e}"}

    stats = {
        "input_meta": {
            "name": _input_info["name"],
            "shape": _input_info["shape"],
            "dtype": _input_info["dtype"],
        },
        "output_meta": {
            "name": _output_info["name"],
            "shape": _output_info["shape"],
            "dtype": _output_info["dtype"],
        },
        "tensor_shape": list(x.shape),
        "tensor_dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "providers": get_session().get_providers(),
    }
    return stats


def build_ui():
    with gr.Blocks(title="ONNX CNN (224x224) with Probabilities") as demo:
        gr.Markdown("ONNX Runtime CNN Inference (224x224). Returns probabilities.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload image")
        with gr.Row():
            out = gr.Label(num_top_classes=4, label="Prediction (Top-3)")
        with gr.Row():
            btn_run = gr.Button("Run")
            btn_debug = gr.Button("Debug Preprocessing")
        with gr.Row():
            dbg = gr.JSON(label="Debug Output")

        btn_run.click(predict, inputs=inp, outputs=out)
        btn_debug.click(debug_pre, inputs=inp, outputs=dbg)

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))