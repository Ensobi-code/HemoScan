"""
train.py — fine-tune EfficientNet-B4 on blood cell images.

Usage
─────
python train.py --data_dir data/blood_cells --epochs 30 --batch_size 32

Changes from original
─────────────────────
  1. Upgraded backbone: EfficientNet-B3 → B4 (better capacity for fine-grained WBC subtypes)
  2. Real class weights: computed from actual sample counts (fixes np.ones placeholder bug)
  3. Label smoothing (0.1): prevents overconfidence on ambiguous cells like lymphoblasts
  4. Two-stage training: freeze backbone for warmup epochs, then full fine-tune at lower LR
  5. Confusion matrix: saved to checkpoints/ after test evaluation so you can see exact errors
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from dataset import build_loaders


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = False):
    # Upgraded to B4 for better fine-grained feature extraction
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Always keep classifier head trainable
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Class weights from real counts ────────────────────────────────────────────
def compute_class_weights(train_ds, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency weights from actual training labels."""
    labels = [train_ds.subset.dataset.samples[i][1] for i in train_ds.subset.indices]
    counts = Counter(labels)
    class_counts = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes   # normalize so weights avg to 1
    print(f"[Train] Class counts : {dict(sorted(counts.items()))}")
    print(f"[Train] Class weights: { {i: round(w, 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Confusion matrix ──────────────────────────────────────────────────────────
def save_confusion_matrix(labels, preds, class_names: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    ConfusionMatrixDisplay.from_predictions(
        labels, preds,
        display_labels=class_names,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Train] Confusion matrix saved to {save_path}")


# ── Training stage helper ─────────────────────────────────────────────────────
def run_stage(
    stage_name: str,
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    epochs: tuple,          # (start_epoch, end_epoch) inclusive
    total_epochs: int,
    device,
    ckpt_path: Path,
    class_names: list,
    best_val_acc: float,
):
    print(f"\n{'═'*60}")
    print(f"  {stage_name}")
    print(f"{'═'*60}")

    for epoch in range(epochs[0], epochs[1] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    return best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Data ──
    train_loader, val_loader, test_loader, class_names = build_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)
    print(f"[Train] Classes ({num_classes}): {class_names}")

    # ── Checkpoint dir ──
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    with open(ckpt_dir / "classes.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Train] Saved class list → {ckpt_dir / 'classes.json'}")

    # ── Real class weights (fixes np.ones placeholder) ──
    train_ds = train_loader.dataset
    class_weights = compute_class_weights(train_ds, num_classes, device)

    # ── Loss with label smoothing (helps lymphoblast/lymphocyte confusion) ──
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ══════════════════════════════════════════════════════════════
    # TWO-STAGE TRAINING
    # Stage 1 — Warmup: freeze backbone, train head only (fast, stable)
    # Stage 2 — Fine-tune: unfreeze all, low LR (squeezes out accuracy)
    # ══════════════════════════════════════════════════════════════
    warmup_epochs = args.warmup_epochs
    finetune_epochs = args.epochs - warmup_epochs
    best_val_acc = 0.0

    # ── Stage 1: Warmup ──
    model = build_model(num_classes, freeze_backbone=True)
    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs)

    best_val_acc = run_stage(
        stage_name=f"Stage 1 / Warmup — head only  (epochs 1–{warmup_epochs})",
        model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        epochs=(1, warmup_epochs),
        total_epochs=args.epochs,
        device=device,
        ckpt_path=ckpt_path,
        class_names=class_names,
        best_val_acc=best_val_acc,
    )

    # ── Stage 2: Full fine-tune ──
    if finetune_epochs > 0:
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)

        best_val_acc = run_stage(
            stage_name=f"Stage 2 / Full fine-tune  (epochs {warmup_epochs+1}–{args.epochs})",
            model=model,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            epochs=(warmup_epochs + 1, args.epochs),
            total_epochs=args.epochs,
            device=device,
            ckpt_path=ckpt_path,
            class_names=class_names,
            best_val_acc=best_val_acc,
        )

    # ── Final test evaluation ──
    print("\n[Train] Evaluating best checkpoint on test set…")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)

    print(f"\nTest accuracy: {test_acc:.3f}")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    save_confusion_matrix(
        test_labels, test_preds, class_names,
        save_path=ckpt_dir / "confusion_matrix.png",
    )

    print(f"\n[Train] Done. Best val_acc={best_val_acc:.3f}  Test acc={test_acc:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 on blood cell images")
    parser.add_argument("--data_dir",        default="data/blood_cells", help="Root data directory")
    parser.add_argument("--checkpoint_dir",  default="checkpoints",      help="Where to save model")
    parser.add_argument("--epochs",          type=int,   default=30,     help="Total epochs (both stages)")
    parser.add_argument("--warmup_epochs",   type=int,   default=5,      help="Stage-1 head-only epochs")
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--image_size",      type=int,   default=224)
    parser.add_argument("--lr",              type=float, default=1e-4,   help="Stage-1 LR (Stage-2 uses lr/10)")
    parser.add_argument("--num_workers",     type=int,   default=4)
    args = parser.parse_args()
    main(args)
