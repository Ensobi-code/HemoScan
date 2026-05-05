"""
predict.py — run inference on a single image or folder of images.

Usage
─────
# Single image
python predict.py --image path/to/cell.jpg

# Folder of images
python predict.py --folder path/to/images/

# Use a specific checkpoint
python predict.py --image cell.tif --checkpoint checkpoints/best_model.pt

Output
──────
Prints a structured report per image:
  - Predicted class + confidence
  - Top-3 alternative predictions
  - Disease associations, severity, and recommended action
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from disease_mapping import get_disease_info, SEVERITY_EMOJI


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load model and class names from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    class_names = ckpt.get("class_names")
    if class_names is None:
        # Fall back to classes.json in same directory
        classes_json = Path(checkpoint_path).parent / "classes.json"
        if not classes_json.exists():
            sys.exit(
                f"[Error] No class names found in checkpoint and no classes.json at {classes_json}.\n"
                "Re-run train.py to regenerate the checkpoint with class names embedded."
            )
        with open(classes_json) as f:
            class_names = json.load(f)

    num_classes = len(class_names)
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", None)
    val_info = f"val_acc={val_acc:.3f}" if val_acc is not None else "val_acc=unknown"
    print(f"[Predict] Loaded checkpoint (epoch {epoch}, {val_info})")
    print(f"[Predict] Classes ({num_classes}): {class_names}\n")

    return model, class_names


# ── Image preprocessing ───────────────────────────────────────────────────────

def get_transform(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: str, transform) -> torch.Tensor:
    """Open an image, convert to RGB, apply transform, add batch dim."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)   # (1, C, H, W)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, class_names: list, device: torch.device, top_k: int = 3):
    """
    Returns a dict:
      {
        "class":       str,           # top predicted class
        "confidence":  float,         # 0-1
        "top_k":       [(class, prob), ...],
      }
    """
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)                # (1, num_classes)
    probs = F.softmax(logits, dim=1)[0]         # (num_classes,)

    top_k = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(top_k)

    top_list = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {
        "class":      top_list[0][0],
        "confidence": top_list[0][1],
        "top_k":      top_list,
    }


# ── Report formatting ─────────────────────────────────────────────────────────

def format_report(image_path: str, prediction: dict) -> str:
    cls        = prediction["class"]
    confidence = prediction["confidence"]
    info       = get_disease_info(cls)

    severity   = info["severity"]
    emoji      = SEVERITY_EMOJI.get(severity, "❓")
    diseases   = ", ".join(info["diseases"]) if info["diseases"] else "None"

    lines = [
        f"{'─' * 60}",
        f"  Image      : {image_path}",
        f"  Prediction : {cls}  ({confidence*100:.1f}% confidence)",
        f"  Severity   : {emoji}  {severity.upper()}",
        f"  Diseases   : {diseases}",
        f"  Info       : {info['description']}",
        f"  Action     : {info['action']}",
    ]

    if len(prediction["top_k"]) > 1:
        lines.append("  Alternatives:")
        for alt_cls, alt_prob in prediction["top_k"][1:]:
            alt_info = get_disease_info(alt_cls)
            alt_emoji = SEVERITY_EMOJI.get(alt_info["severity"], "❓")
            lines.append(f"    {alt_emoji}  {alt_cls:<22} {alt_prob*100:.1f}%")

    lines.append(f"{'─' * 60}")
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def collect_images(folder: str) -> list[Path]:
    folder = Path(folder)
    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        sys.exit(f"[Error] No supported images found in {folder}")
    return images


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Using device: {device}")

    model, class_names = load_model(args.checkpoint, device)
    transform = get_transform(args.image_size)

    # Collect images to process
    if args.image:
        image_paths = [Path(args.image)]
    elif args.folder:
        image_paths = collect_images(args.folder)
        print(f"[Predict] Found {len(image_paths)} image(s) in {args.folder}\n")
    else:
        sys.exit("[Error] Provide --image or --folder.")

    # Run inference
    results = []
    for path in image_paths:
        try:
            tensor = load_image(str(path), transform)
            pred   = predict(model, tensor, class_names, device, top_k=args.top_k)
            print(format_report(str(path), pred))
            results.append({"file": str(path), **pred})
        except Exception as e:
            print(f"[Warning] Could not process {path}: {e}")

    # Optionally save results to JSON
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Predict] Saved results to {out}")

    # Summary when processing a folder
    if len(results) > 1:
        from collections import Counter
        counts = Counter(r["class"] for r in results)
        print("\nSummary")
        print("─" * 40)
        for cls, count in counts.most_common():
            info  = get_disease_info(cls)
            emoji = SEVERITY_EMOJI.get(info["severity"], "❓")
            print(f"  {emoji}  {cls:<22} {count} image(s)")
        print("─" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood cell classifier — inference")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  help="Path to a single image")
    src.add_argument("--folder", help="Path to a folder of images")

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--top_k",       type=int, default=3, help="Show top-K predictions")
    parser.add_argument("--output_json", default=None, help="Optional path to save results as JSON")

    args = parser.parse_args()
    main(args)
