"""Train Stage 1 high-recall candidate heatmap model.

Dataset: data/dataset_stages/stage1/{train,test}/{images,grid}
Images are 960x540 RGB; target grids are 17x30 uint8 arrays.
Requires: torch, torchvision, numpy, Pillow, opencv-python-headless.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_H, IMAGE_W = 540, 960
GRID_H, GRID_W = 17, 30
CELL_SIZE = 32


def enhance_lab_dog(image_rgb: np.ndarray, clahe_clip: float = 2.0,
                    dog_gain: float = 0.35) -> np.ndarray:
    """Enhance luminance contrast in Lab and add a light DoG edge response.

    Input and output are uint8 RGB arrays. Keeping the result RGB means the
    detector still uses a three-channel stem, while L-channel contrast and
    edges are emphasized before training and inference.
    """
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("enhance_lab_dog expects a uint8 HxWx3 RGB image")
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    contrast = clahe.apply(lightness)
    blur_small = cv2.GaussianBlur(contrast, (0, 0), sigmaX=0.8)
    blur_large = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.6)
    dog_edges = cv2.absdiff(blur_small, blur_large)
    enhanced_l = np.clip(
        contrast.astype(np.float32) + dog_gain * dog_edges.astype(np.float32),
        0, 255,
    ).astype(np.uint8)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def prepare_stage1_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Apply the shared preprocessor and return [3,544,960] float tensor."""
    if image_rgb.shape[:2] != (IMAGE_H, IMAGE_W):
        raise ValueError(f"Stage 1 expects {IMAGE_W}x{IMAGE_H}, got {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    enhanced = enhance_lab_dog(image_rgb)
    canvas = np.full((GRID_H * CELL_SIZE, IMAGE_W, 3), 114, dtype=np.uint8)
    canvas[:IMAGE_H] = enhanced
    return TF.to_tensor(Image.fromarray(canvas))


class Stage1Dataset(Dataset):
    """Pair each image with its same-stem 17x30 grid target."""

    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.image_dir = self.root / "stage1" / split / "images"
        self.grid_dir = self.root / "stage1" / split / "grid"
        if not self.image_dir.is_dir() or not self.grid_dir.is_dir():
            raise FileNotFoundError(
                f"Missing Stage 1 folders: {self.image_dir} and/or {self.grid_dir}"
            )
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        missing = [p for p in self.images if not (self.grid_dir / f"{p.stem}.npy").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no matching grid label; example: {missing[0].name}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if image.shape[:2] != (IMAGE_H, IMAGE_W):
            raise ValueError(f"{image_path.name}: expected 960x540, got {image.shape[1]}x{image.shape[0]}")

        image_tensor = prepare_stage1_tensor(image)
        target = np.load(self.grid_dir / f"{image_path.stem}.npy")
        if target.shape != (GRID_H, GRID_W):
            raise ValueError(f"{image_path.stem}.npy: expected {(GRID_H, GRID_W)}, got {target.shape}")
        target = torch.from_numpy((target > 0).astype(np.float32)).unsqueeze(0)

        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            target = torch.flip(target, dims=[2])
        return image_tensor, target


class SqueezeExcitation(nn.Module):
    """Channel attention block applied after the final depthwise block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class DWBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CandidateNet(nn.Module):
    """Shallow high-resolution detector: 960x544 -> one 17x30 logit map."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            DWBlock(16, 24, 2),
            DWBlock(24, 32, 2),
            DWBlock(32, 48, 2),
            DWBlock(48, 64, 2),
            SqueezeExcitation(64, reduction=8),
        )
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        return self.head(self.features(x))


def focal_loss(logits, targets, alpha: float = 0.75, gamma: float = 2.0):
    probability = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    positive = targets.eq(1).float()
    negative = targets.eq(0).float()
    positive_loss = -alpha * (1 - probability).pow(gamma) * probability.log() * positive
    negative_loss = -(1 - alpha) * probability.pow(gamma) * (1 - probability).log() * negative
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    total_loss = total_positive = total_hit = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += focal_loss(logits, targets).item()
        positive = targets.bool()
        predicted = logits.sigmoid() >= threshold
        total_positive += positive.sum().item()
        total_hit += (predicted & positive).sum().item()
    return total_loss / max(1, len(loader)), total_hit / max(1.0, total_positive)


def main():
    parser = argparse.ArgumentParser(description="Train the Stage 1 candidate heatmap model")
    parser.add_argument("--data", type=Path, default=Path("data/dataset_stages"))
    parser.add_argument("--out", type=Path, default=Path("runs/stage1/best_enhanced.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = Stage1Dataset(args.data, "train", augment=True)
    indices = list(range(len(full_train)))
    random.Random(args.seed).shuffle(indices)
    val_count = max(1, round(len(indices) * args.val_ratio))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    train_ds = Subset(full_train, train_indices)
    val_ds = Subset(Stage1Dataset(args.data, "train", augment=False), val_indices)
    test_ds = Stage1Dataset(args.data, "test", augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)

    model = CandidateNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0

    print(f"device={device}; train={len(train_ds)}; val={len(val_ds)}; test={len(test_ds)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=pin):
                loss = focal_loss(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_sum += loss.item()
        scheduler.step()

        val_loss, val_recall = evaluate(model, val_loader, device, args.threshold)
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={train_sum/max(1,len(train_loader)):.4f} "
              f"val_loss={val_loss:.4f} val_cell_recall@{args.threshold:.2f}={val_recall:.4f}",
              flush=True)
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save({
                "model": model.state_dict(),
                "input_size": (IMAGE_W, GRID_H * CELL_SIZE),
                "grid_size": (GRID_H, GRID_W),
                "threshold": args.threshold,
                "val_cell_recall": val_recall,
                "epoch": epoch,
                "preprocessing": "Lab CLAHE + DoG v1",
                "se_attention": True,
            }, args.out)

    checkpoint = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_recall = evaluate(model, test_loader, device, args.threshold)
    print(f"best_epoch={checkpoint['epoch']} best_val_cell_recall={best_recall:.4f}")
    print(f"test_loss={test_loss:.4f} test_cell_recall@{args.threshold:.2f}={test_recall:.4f}")
    print(f"checkpoint={args.out}")


if __name__ == "__main__":
    main()
