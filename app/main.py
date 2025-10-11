# pip install onnx onnxruntime gradio pillow numpy
import os
import numpy as np
import gradio as gr
from PIL import Image
import onnx
import onnxruntime as ort

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

INPUT_SIZE = (224, 224)
MODEL_PATH = os.getenv("ONNX_PATH", "resnet50.onnx")

# Adjust to your labels
class_names = ["Benign", "Early", "Pre", "Pro"]

# Choose providers: CPU or CUDA
PROVIDERS = ["CPUExecutionProvider"]
# For GPU, install onnxruntime-gpu and use:
# PROVIDERS = ["CUDAExecutionProvider"]

# Lazy-initialized session
_sess = None
_input_info = None
_output_info = None


def get_session():
    global _sess, _input_info, _output_info
    if _sess is not None:
        return _sess

    # Optional: validate model file
    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=PROVIDERS)

    # Fetch input/output metadata
    inputs = _sess.get_inputs()
    outputs = _sess.get_outputs()
    assert len(inputs) == 1, "This app expects a single input ONNX model."
    assert len(outputs) >= 1, "Model must have at least 1 output."

    _input_info = {
        "name": inputs[0].name,
        "shape": inputs[0].shape,  # may contain None or 'N'
        "dtype": inputs[0].type,   # e.g., 'tensor(float)'
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


def _np_dtype_from_onnx_type(onnx_type: str):
    # Map ONNX tensor types to numpy dtypes (common cases)
    # 'tensor(float)' -> np.float32, 'tensor(uint8)' -> np.uint8, etc.
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
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


def preprocess(pil_image: Image.Image):
    sess = get_session()
    # Use metadata to infer layout and dtype
    input_name = _input_info["name"]
    in_shape = _input_info["shape"]  # e.g., [1, 224, 224, 3] or [1, 3, 224, 224]
    in_dtype = _np_dtype_from_onnx_type(_input_info["dtype"])

    # Determine layout: NHWC if last dim is 3; NCHW if second dim is 3
    # Batch dim may be None or 'N'. We only care about H/W/C positions.
    # For dynamic shapes, in_shape could be [None, 224, 224, 3] or similar.
    is_nhwc = False
    if len(in_shape) == 4:
        # Try to detect: last dim equals 3 -> NHWC; second dim equals 3 -> NCHW
        if in_shape[-1] == 3:
            is_nhwc = True
            height = in_shape[1] if isinstance(in_shape[1], int) else INPUT_SIZE[1]
            width = in_shape[2] if isinstance(in_shape[2], int) else INPUT_SIZE[0]
        else:
            # Assume NCHW, with channels=3
            is_nhwc = False
            height = in_shape[2] if isinstance(in_shape[2], int) else INPUT_SIZE[1]
            width = in_shape[3] if isinstance(in_shape[3], int) else INPUT_SIZE[0]
    else:
        # Fallback
        is_nhwc = True
        height, width = INPUT_SIZE[1], INPUT_SIZE[0]

    # Resize
    img = pil_image.convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(img)

    # Normalize to [0,1] by default; adjust to your training normalization if needed
    x = arr.astype(np.float32) / 255.0

    # If your model expects ImageNet/ResNet preprocessing, replace with:
    # x = arr.astype(np.float32)
    # x = x[:, :, ::-1]  # RGB -> BGR
    # mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    # x = x - mean

    if is_nhwc:
        x = np.expand_dims(x, axis=0)  # [1, H, W, 3]
    else:
        x = np.transpose(x, (2, 0, 1))  # [3, H, W]
        x = np.expand_dims(x, axis=0)   # [1, 3, H, W]

    # Cast to expected dtype if needed
    if in_dtype != x.dtype:
        x = x.astype(in_dtype)

    return input_name, x


def postprocess(outputs):
    # Assume first output is logits or probabilities
    y = outputs[0]
    y = np.squeeze(y)

    if y.ndim == 1:
        scores = y
    elif y.ndim == 2:
        scores = y[0]
    else:
        # Unusual shape; return scalar if possible
        return {"score": float(np.ravel(y)[0])}

    # If outputs are logits, optionally apply softmax for probabilities:
    # probs = np.exp(scores - np.max(scores))
    # probs /= probs.sum()
    # For now, use raw scores; change to probs if you prefer normalized outputs
    probs = scores

    if class_names and len(class_names) == probs.shape[0]:
        pairs = list(zip(class_names, probs.tolist()))
    else:
        pairs = [(f"class_{i}", float(v)) for i, v in enumerate(probs.tolist())]

    top = sorted(pairs, key=lambda kv: kv[1], reverse=True)[:3]
    return {k: float(v) for k, v in top}


def predict(image: Image.Image):
    sess = get_session()
    input_name, x = preprocess(image)
    outputs = sess.run(None, {input_name: x})
    return postprocess(outputs)


def build_ui():
    with gr.Blocks(title="ONNX CNN (224x224)") as demo:
        gr.Markdown("ONNX Runtime CNN Inference (224x224)")
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload image")
            out = gr.Label(num_top_classes=4, label="Prediction")
        inp.change(predict, inputs=inp, outputs=out)
        gr.Button("Run").click(predict, inputs=inp, outputs=out)
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))