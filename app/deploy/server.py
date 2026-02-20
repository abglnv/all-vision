"""
RAM Agent – Recurrent Attention Model Deployment Server
=======================================================
FastAPI backend that runs the RAM ONNX model and returns rich
visualisation data (glimpse locations, preprocessing stages,
extracted patches, per-class probabilities).
"""

import io, os, base64, math, time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ───────────────────────────────────────────────────────────────────
_deploy_dir = Path(__file__).resolve().parent
_app_dir = _deploy_dir.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(_app_dir / "ram_agent.onnx"))
DYNAMIC_MODEL_PATH = str(_deploy_dir / "ram_agent_dynamic.onnx")
CLASS_NAMES = ["Benign", "Early Pre-B ALL", "Pre-B ALL", "Pro-B ALL"]
CONTENT_SIZE = 224          # resize the image content to this
PAD = 144                   # padding around the 224 → gives 512
FINAL_SIZE = CONTENT_SIZE + 2 * PAD   # = 512
PATCH_SIZE = 96             # the agent's glimpse patch size
NUM_GLIMPSES = 8

app = FastAPI(title="RAM Agent – Leukemia Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── ONNX session (lazy) ─────────────────────────────────────────────────────
_sess: Optional[ort.InferenceSession] = None

def _make_dynamic_model():
    """Patch the original ONNX file so spatial dims are dynamic."""
    import onnx
    model = onnx.load(MODEL_PATH)
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        if tt.HasField("shape"):
            for i, dim in enumerate(tt.shape.dim):
                if inp.name == "image" and i >= 2:
                    dim.ClearField("dim_value")
                    dim.dim_param = "height" if i == 2 else "width"
    # drop cached shapes so ORT doesn't complain
    while model.graph.value_info:
        model.graph.value_info.pop()
    onnx.save(model, DYNAMIC_MODEL_PATH)

def get_session() -> ort.InferenceSession:
    global _sess
    if _sess is not None:
        return _sess
    if not Path(DYNAMIC_MODEL_PATH).exists():
        _make_dynamic_model()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _sess = ort.InferenceSession(DYNAMIC_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    print(f"[RAM] Model loaded – inputs: {[(i.name, i.shape) for i in _sess.get_inputs()]}")
    return _sess


# ── Image pre-processing (numpy, no PyTorch) ────────────────────────────────

def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _np_gray_to_b64(arr: np.ndarray) -> str:
    """arr in [0,1] float → grayscale PNG base64."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    return _pil_to_b64(img)


def percentile_clip(arr: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    low_val = np.quantile(arr, lo)
    high_val = np.quantile(arr, hi)
    clipped = np.clip(arr, low_val, high_val)
    return (clipped - low_val) / (high_val - low_val + 1e-8)


def sobel_x(arr: np.ndarray) -> np.ndarray:
    """Horizontal Sobel filter (matches the training SobelEdgeDetection)."""
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    from scipy.signal import convolve2d
    return np.abs(convolve2d(arr, kernel, mode="same", boundary="symm"))


def preprocess_image(pil_img: Image.Image):
    """
    Replicate the training pipeline **exactly**:
        Resize(224) → Grayscale → ToTensor → PercentileClip → Sobel → Pad(144) → Normalize
    Returns the model-ready tensor *and* intermediate stage images (for the UI).
    """
    # 0. Original (thumbnail for display)
    orig_b64 = _pil_to_b64(pil_img.copy().convert("RGB"), "JPEG")

    # 1. Square-pad & resize to 224×224
    w, h = pil_img.size
    max_dim = max(w, h)
    pad_l = (max_dim - w) // 2
    pad_t = (max_dim - h) // 2
    square = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
    square.paste(pil_img, (pad_l, pad_t))
    resized = square.resize((CONTENT_SIZE, CONTENT_SIZE), Image.BILINEAR)
    resized_b64 = _pil_to_b64(resized, "JPEG")

    # 2. Grayscale [0,1]
    gray = np.array(resized.convert("L")).astype(np.float32) / 255.0
    gray_b64 = _np_gray_to_b64(gray)

    # 3. Percentile clip
    clipped = percentile_clip(gray)
    clip_b64 = _np_gray_to_b64(clipped)

    # 4. Sobel edge
    edges = sobel_x(clipped)
    edges_norm = edges / (edges.max() + 1e-8)
    edge_b64 = _np_gray_to_b64(edges_norm)

    # 5. Pad 144 each side → 512×512
    padded = np.full((FINAL_SIZE, FINAL_SIZE), 0.5, dtype=np.float32)
    padded[PAD:PAD + CONTENT_SIZE, PAD:PAD + CONTENT_SIZE] = edges_norm
    padded_b64 = _np_gray_to_b64(padded)

    # 6. Normalize (mean=0.5, std=0.5) → range ~ [-1, 1]
    tensor = (padded - 0.5) / 0.5
    tensor = tensor[np.newaxis, np.newaxis, :, :].astype(np.float32)   # [1,1,512,512]

    stages = {
        "original": orig_b64,
        "resized": resized_b64,
        "grayscale": gray_b64,
        "clipped": clip_b64,
        "sobel": edge_b64,
        "padded": padded_b64,
    }
    return tensor, stages, (w, h)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def extract_patch_np(padded_img: np.ndarray, loc: np.ndarray, patch_size: int = PATCH_SIZE):
    """
    Mimics the agent's grid_sample patch extraction.
    padded_img: [H, W] in [0,1]  loc: [2] in [-1,1]
    Returns [patch_size, patch_size] numpy array.
    """
    H, W = padded_img.shape
    # Convert [-1,1] to pixel center
    cx = (loc[0] + 1) / 2 * W
    cy = (loc[1] + 1) / 2 * H
    # The model's scale = patch_size / H
    scale = patch_size / H
    half_out = patch_size / 2

    patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    for py in range(patch_size):
        for px in range(patch_size):
            # output coord in [-1,1]
            ox = -1 + 2 * px / (patch_size - 1)
            oy = -1 + 2 * py / (patch_size - 1)
            # apply affine: sx = scale*ox + loc_x, sy = scale*oy + loc_y
            sx = scale * ox + loc[0]
            sy = scale * oy + loc[1]
            # to pixel
            src_x = (sx + 1) / 2 * (W - 1)
            src_y = (sy + 1) / 2 * (H - 1)
            # bilinear sample
            x0, y0 = int(math.floor(src_x)), int(math.floor(src_y))
            x1, y1 = x0 + 1, y0 + 1
            if 0 <= x0 < W and 0 <= x1 < W and 0 <= y0 < H and 0 <= y1 < H:
                fx, fy = src_x - x0, src_y - y0
                val = (padded_img[y0, x0] * (1-fx)*(1-fy)
                     + padded_img[y0, x1] * fx*(1-fy)
                     + padded_img[y1, x0] * (1-fx)*fy
                     + padded_img[y1, x1] * fx*fy)
                patch[py, px] = val
    return patch


def extract_patch_fast(padded_img: np.ndarray, loc: np.ndarray, patch_size: int = PATCH_SIZE):
    """
    Vectorised version of extract_patch_np (much faster).
    """
    H, W = padded_img.shape
    scale = patch_size / H

    # Build output grid in [-1, 1]
    t = np.linspace(-1, 1, patch_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(t, t)  # each (patch_size, patch_size)

    # Apply affine transform
    src_x_norm = scale * grid_x + loc[0]
    src_y_norm = scale * grid_y + loc[1]

    # To pixel coords
    src_x = (src_x_norm + 1) / 2 * (W - 1)
    src_y = (src_y_norm + 1) / 2 * (H - 1)

    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = src_x - x0
    fy = src_y - y0

    # Clamp
    def clamp_get(arr, yy, xx):
        yy = np.clip(yy, 0, H - 1)
        xx = np.clip(xx, 0, W - 1)
        return arr[yy, xx]

    # Check bounds → zero outside
    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    val = (clamp_get(padded_img, y0, x0) * (1 - fx) * (1 - fy)
         + clamp_get(padded_img, y0, x1) * fx * (1 - fy)
         + clamp_get(padded_img, y1, x0) * (1 - fx) * fy
         + clamp_get(padded_img, y1, x1) * fx * fy)
    patch = np.where(valid, val, 0.0).astype(np.float32)
    return patch


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    start_x: float = Query(0.0, ge=-1, le=1),
    start_y: float = Query(0.0, ge=-1, le=1),
):
    t0 = time.time()

    # 1. Load image
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

    # 2. Preprocess
    tensor, stages, orig_size = preprocess_image(pil_img)

    # 3. Run ONNX
    sess = get_session()
    start_loc = np.array([[start_x, start_y]], dtype=np.float32)
    logits, locations = sess.run(None, {"image": tensor, "start_loc": start_loc})
    # logits: [1,4]  locations: [1,8,2]

    probs = softmax(logits[0]).tolist()
    pred_idx = int(np.argmax(probs))
    locs = locations[0].tolist()   # [[x,y], ...] in [-1,1]

    # 4. Map locations to pixel coords on the PADDED 512×512
    pixel_locs = []
    for lx, ly in locs:
        px = (lx + 1) / 2 * FINAL_SIZE
        py = (ly + 1) / 2 * FINAL_SIZE
        pixel_locs.append({"x": round(px, 1), "y": round(py, 1)})

    # Also map to content (224×224) coords for overlay on the resized image
    content_locs = []
    for lx, ly in locs:
        cx = (lx + 1) / 2 * FINAL_SIZE - PAD
        cy = (ly + 1) / 2 * FINAL_SIZE - PAD
        content_locs.append({"x": round(cx, 1), "y": round(cy, 1)})

    # 5. Extract patches as base64
    padded_01 = tensor[0, 0] * 0.5 + 0.5  # undo normalize → [0,1]

    patch_images = []
    for lx, ly in locs:
        loc_arr = np.array([lx, ly], dtype=np.float32)
        patch = extract_patch_fast(padded_01, loc_arr, PATCH_SIZE)
        patch_images.append(_np_gray_to_b64(patch))

    # 6. Compute per-glimpse stats
    distances = [0.0]
    for i in range(1, len(locs)):
        d = math.sqrt((locs[i][0] - locs[i-1][0])**2 + (locs[i][1] - locs[i-1][1])**2)
        distances.append(round(d, 4))

    total_distance = round(sum(distances), 4)

    # Box size relative to content
    box_ratio = PATCH_SIZE / FINAL_SIZE
    box_content = round(box_ratio * CONTENT_SIZE, 1)

    elapsed = round(time.time() - t0, 3)

    return JSONResponse({
        "prediction": {
            "class": CLASS_NAMES[pred_idx],
            "index": pred_idx,
            "probabilities": {CLASS_NAMES[i]: round(p, 5) for i, p in enumerate(probs)},
            "confidence": round(max(probs), 5),
        },
        "glimpses": {
            "count": NUM_GLIMPSES,
            "patch_size": PATCH_SIZE,
            "box_content_px": box_content,
            "locations_norm": [{"x": round(lx, 5), "y": round(ly, 5)} for lx, ly in locs],
            "locations_pixel": pixel_locs,
            "locations_content": content_locs,
            "distances": distances,
            "total_distance": total_distance,
            "patches_b64": patch_images,
        },
        "stages": stages,
        "meta": {
            "original_size": list(orig_size),
            "model_input_size": [FINAL_SIZE, FINAL_SIZE],
            "content_size": CONTENT_SIZE,
            "padding": PAD,
            "inference_ms": round(elapsed * 1000, 1),
            "start_location": [start_x, start_y],
        },
    })


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
