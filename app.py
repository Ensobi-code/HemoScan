"""
app.py — Flask web server for blood cell classifier inference.

Usage
─────
pip install flask pillow torch torchvision

python app.py
# Then open http://localhost:5000

Changes from original
─────────────────────
  - Uses EfficientNet-B4 (matches updated train.py)
  - Exposes hard_class / hard_note fields from updated disease_mapping.py
  - /api/status now reports hard classes so the frontend can flag them
"""

import json
import random
from io import BytesIO
import base64
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import torchvision.transforms as T

from disease_mapping import (
    get_disease_info, get_hard_note, is_hard_class,
    SEVERITY_EMOJI, DISEASE_MAP, HARD_CLASSES,
)

app = Flask(__name__, static_folder="static")

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL       = None
CLASS_NAMES = []
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    global MODEL, CLASS_NAMES

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"[Warning] Checkpoint not found at {ckpt_path}. Running in DEMO mode.")
        return False

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    CLASS_NAMES = ckpt.get("class_names") or json.load(
        open(ckpt_path.parent / "classes.json")
    )

    num_classes = len(CLASS_NAMES)

    # EfficientNet-B4 — matches updated train.py
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    MODEL = model

    epoch   = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", None)
    val_str = f"{val_acc:.3f}" if val_acc is not None else "N/A"
    print(f"[App] Model loaded — epoch {epoch}, val_acc={val_str}")
    print(f"[App] Classes ({num_classes}): {CLASS_NAMES}")
    return True


# ── Inference helpers ─────────────────────────────────────────────────────────
def get_transform(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _build_prediction_entry(cls: str, confidence: float) -> dict:
    info = get_disease_info(cls)
    return {
        "class":          cls,
        "confidence":     round(confidence * 100, 2),
        "severity":       info["severity"],
        "severity_emoji": SEVERITY_EMOJI.get(info["severity"], "❓"),
        "diseases":       info["diseases"],
        "description":    info["description"],
        "action":         info["action"],
        "hard_class":     is_hard_class(cls),
        "hard_note":      get_hard_note(cls),
    }


@torch.no_grad()
def run_inference(pil_image: Image.Image, top_k: int = 5) -> list:
    transform = get_transform()
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = MODEL(tensor)
    probs  = F.softmax(logits, dim=1)[0]

    k = min(top_k, len(CLASS_NAMES))
    top_probs, top_idxs = probs.topk(k)

    return [
        _build_prediction_entry(CLASS_NAMES[idx.item()], prob.item())
        for idx, prob in zip(top_idxs, top_probs)
    ]


def demo_result() -> list:
    """Simulated result when no checkpoint is available."""
    classes  = list(DISEASE_MAP.keys())
    top_cls  = random.choice(classes)
    result   = [_build_prediction_entry(top_cls, random.uniform(0.70, 0.99))]
    alt_pool = [c for c in classes if c != top_cls]
    for cls in random.sample(alt_pool, min(4, len(alt_pool))):
        result.append(_build_prediction_entry(cls, random.uniform(0.01, 0.20)))
    return result


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_image = Image.open(BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    # Thumbnail for display
    thumb = pil_image.copy()
    thumb.thumbnail((400, 400))
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    if MODEL is None:
        predictions = demo_result()
        demo = True
    else:
        predictions = run_inference(pil_image)
        demo = False

    return jsonify({
        "demo":        demo,
        "predictions": predictions,
        "thumbnail":   f"data:image/jpeg;base64,{img_b64}",
    })


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": MODEL is not None,
        "device":       str(DEVICE),
        "classes":      CLASS_NAMES,
        "hard_classes": list(HARD_CLASSES),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
