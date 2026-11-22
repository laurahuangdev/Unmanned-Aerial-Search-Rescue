"""Train Stage 2 person/not-person classifier on the provided crop dataset.

Expected folders:
  data/dataset_stages/stage2/{train,test}/{person,not_person}/*
The crops are enhanced with the same Lab CLAHE + DoG preprocessing as Stage 1.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from stage1_train import enhance_lab_dog


class Stage2Dataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.folder = ImageFolder(str(self.root / "stage2" / split))
        if not self.folder.samples:
            raise FileNotFoundError(f"No Stage 2 crops found in {self.root / 'stage2' / split}")
        if "person" not in self.folder.class_to_idx or "not_person" not in self.folder.class_to_idx:
            raise ValueError("Stage 2 requires class folders named 'person' and 'not_person'")
        self.class_to_idx = self.folder.class_to_idx

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        path, label = self.folder.samples[index]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        enhanced = enhance_lab_dog(rgb)
        image = Image.fromarray(enhanced)
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
        return TF.to_tensor(image), int(label)


class Stage2Classifier(nn.Module):
    """ResNet-18 adapted for two classes of 64x64 candidate crops."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.network = resnet18(weights=None)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x):
        return self.network(x)


def stratified_split(labels, val_ratio: float, seed: int):
    by_class = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        val_count = max(1, round(len(class_indices) * val_ratio))
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


@torch.no_grad()
def evaluate(model, loader, device, person_idx):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = correct = tp = fp = fn = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += loss_fn(logits, labels).item()
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        tp += ((predicted == person_idx) & (labels == person_idx)).sum().item()
        fp += ((predicted == person_idx) & (labels != person_idx)).sum().item()
        fn += ((predicted != person_idx) & (labels == person_idx)).sum().item()
        total += labels.numel()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return loss_sum / max(1, len(loader)), correct / max(1, total), precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 person/not-person ResNet-18")
    parser.add_argument("--data", type=Path, default=Path("data/dataset_stages"))
    parser.add_argument("--out", type=Path, default=Path("runs/stage2/best_stage2.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
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

    train_full = Stage2Dataset(args.data, "train", augment=True)
    labels = [label for _, label in train_full.folder.samples]
    train_indices, val_indices = stratified_split(labels, args.val_ratio, args.seed)
    train_ds = Subset(train_full, train_indices)
    val_full = Stage2Dataset(args.data, "train", augment=False)
    val_ds = Subset(val_full, val_indices)
    test_ds = Stage2Dataset(args.data, "test", augment=False)
    class_to_idx = train_full.class_to_idx
    person_idx = class_to_idx["person"]

    counts = np.bincount([labels[i] for i in train_indices], minlength=2).astype(np.float32)
    class_weights = len(train_indices) / (2.0 * np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)

    model = Stage2Classifier(num_classes=len(class_to_idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    print(f"device={device}; train={len(train_ds)}; val={len(val_ds)}; test={len(test_ds)}; "
          f"classes={class_to_idx}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels_batch in train_loader:
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=pin):
                logits = model(images)
                loss = loss_fn(logits, labels_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        scheduler.step()

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, device, person_idx
        )
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={loss_sum/max(1,len(train_loader)):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"person_precision={val_precision:.4f} person_recall={val_recall:.4f} "
              f"person_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "person_idx": person_idx,
                "input_size": (64, 64),
                "preprocessing": "Lab CLAHE + DoG v1",
                "epoch": epoch,
                "val_f1": val_f1,
            }, args.out)

    checkpoint = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model, test_loader, device, person_idx
    )
    print(f"best_epoch={checkpoint['epoch']} best_val_f1={best_f1:.4f}")
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
          f"person_precision={test_precision:.4f} person_recall={test_recall:.4f} "
          f"person_f1={test_f1:.4f}")
    print(f"checkpoint={args.out}")


if __name__ == "__main__":
    main()
