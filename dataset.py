"""
dataset.py — loads blood cell images from disk.

Supported layouts
─────────────────
1. ImageFolder layout (one subfolder per class):
   data/
     myeloblast/   img1.tif  img2.tif ...
     normal/       ...

2. CSV layout:
   data/images/img001.jpg
   labels.csv:  filename, label

Changes from original
─────────────────────
  - Hard-class augmentation: lymphoblast, reactive_lymphocyte, lymphocyte get
    stronger augmentation (extra blur + elastic distortion) to force the model
    to learn morphology rather than staining artefacts.
  - Augmentation wrapper: uses a per-sample wrapper so train/val/test splits
    can each have their own transform without mutating the shared dataset object.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T


# ── Classes that need harder augmentation ─────────────────────────────────────
HARD_CLASSES = {"lymphoblast", "reactive_lymphocyte", "lymphocyte"}


# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(image_size: int = 224, augment: bool = True, hard: bool = False):
    """
    augment=False  →  plain resize + normalize  (val / test)
    augment=True   →  standard augmentation     (train, most classes)
    hard=True      →  stronger augmentation     (train, lymphocyte family)
    """
    norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])

    if not augment:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            norm,
        ])

    base_aug = [
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(15),
    ]

    if hard:
        # Extra transforms that stress morphological features
        base_aug += [
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.4),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]

    return T.Compose(base_aug + [T.ToTensor(), norm])


# ── Augmentation wrapper ───────────────────────────────────────────────────────
class AugmentedSubset(Dataset):
    """
    Wraps a Subset and applies a per-sample transform that can differ by class.
    Avoids mutating the shared parent dataset's transform.
    """
    def __init__(self, subset: Subset, default_transform, hard_transform=None, hard_class_indices: set = None):
        self.subset = subset
        self.default_transform = default_transform
        self.hard_transform = hard_transform
        self.hard_class_indices = hard_class_indices or set()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Load raw image (subset's dataset should have transform=None)
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")

        if self.hard_transform and label in self.hard_class_indices:
            img = self.hard_transform(img)
        else:
            img = self.default_transform(img)

        return img, label


# ── Dataset: ImageFolder layout ───────────────────────────────────────────────
class BloodCellDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = self._load_samples()
        print(f"[Dataset] {len(self.samples)} images | {len(self.classes)} classes: {self.classes}")

    def _load_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in self.EXTENSIONS:
                    samples.append((str(f), self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Dataset: CSV layout ───────────────────────────────────────────────────────
class CSVBloodDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        classes = set()
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                classes.add(row["label"])
                self.samples.append((row["filename"], row["label"]))
        self.classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label_str = self.samples[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label_str]


# ── Convenience factory ───────────────────────────────────────────────────────
def build_loaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 4,
    csv_path: str = None,
):
    """Returns (train_loader, val_loader, test_loader, class_names)."""

    # Load dataset with NO transform — transforms applied per-split below
    if csv_path:
        dataset = CSVBloodDataset(csv_path, data_dir, transform=None)
    else:
        dataset = BloodCellDataset(data_dir, transform=None)

    n = len(dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    train_subset, val_subset, test_subset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Hard-class indices (e.g. lymphoblast → idx 3)
    hard_indices = {
        dataset.class_to_idx[c]
        for c in HARD_CLASSES
        if c in dataset.class_to_idx
    }
    if hard_indices:
        hard_class_names = [dataset.classes[i] for i in hard_indices]
        print(f"[Dataset] Hard-augmentation classes: {hard_class_names}")

    # Wrap splits with appropriate transforms
    train_ds = AugmentedSubset(
        train_subset,
        default_transform=get_transforms(image_size, augment=True,  hard=False),
        hard_transform=   get_transforms(image_size, augment=True,  hard=True),
        hard_class_indices=hard_indices,
    )
    val_ds = AugmentedSubset(
        val_subset,
        default_transform=get_transforms(image_size, augment=False),
    )
    test_ds = AugmentedSubset(
        test_subset,
        default_transform=get_transforms(image_size, augment=False),
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, dataset.classes
