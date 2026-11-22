"""Run Stage 1 heatmap proposals through the Stage 2 person classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from stage1_train import CandidateNet, IMAGE_H, IMAGE_W, enhance_lab_dog, prepare_stage1_tensor
from stage2_train import Stage2Classifier


GRID_H, GRID_W = 17, 30
CELL_SIZE = 32
STAGE2_CROP_SOURCE = 96
STAGE2_INPUT = 64


def crop_with_padding(image: np.ndarray, center_x: float, center_y: float,
                      width: int, height: int) -> np.ndarray:
    x0 = int(round(center_x - width / 2))
    y0 = int(round(center_y - height / 2))
    x1, y1 = x0 + width, y0 + height
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - image.shape[1]), max(0, y1 - image.shape[0])
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                   borderType=cv2.BORDER_REFLECT_101)
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    return image[y0:y1, x0:x1]


def merged_candidates(probability: np.ndarray, threshold: float,
                      min_distance: float = 1.5, max_candidates: int = 50):
    """Choose high-scoring grid cells as proposals instead of component centroids.

    Large connected heatmap regions can have centroids far from the cells that
    produced the high scores. Ranking the active cells and applying spatial
    suppression keeps proposal crops near those score peaks.
    """
    rows, cols = np.where(probability >= threshold)
    if len(rows) == 0:
        return []

    scores = probability[rows, cols]
    order = np.argsort(scores)[::-1]
    min_distance_sq = float(min_distance) ** 2
    selected_points = []
    candidates = []

    for idx in order:
        row, col = int(rows[idx]), int(cols[idx])
        if any((col - prev_col) ** 2 + (row - prev_row) ** 2 < min_distance_sq
               for prev_col, prev_row in selected_points):
            continue

        candidates.append({
            "center_grid": (float(col), float(row)),
            "score": float(scores[idx]),
            "cells": 1,
        })
        selected_points.append((col, row))
        if len(candidates) >= max_candidates:
            break

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Stage 1 -> Stage 2 inference")
    parser.add_argument("--image", type=Path, required=True,
                        help="960x540 Stage 1 image or its 1920x1080 source image")
    parser.add_argument("--stage1", type=Path, default=Path("runs/stage1/best_enhanced.pt"))
    parser.add_argument("--stage2", type=Path, default=Path("runs/stage2/best_stage2.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/cascade/result.jpg"))
    parser.add_argument("--stage1-threshold", type=float, default=0.15)
    parser.add_argument("--person-threshold", type=float, default=0.5)
    parser.add_argument("--proposal-min-distance", type=float, default=1.5,
                        help="Minimum spacing between proposal centers, in grid cells")
    parser.add_argument("--max-proposals", type=int, default=50,
                        help="Maximum number of Stage 1 proposals sent to Stage 2")
    parser.add_argument("--save-crops", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PIL handles Windows paths containing Chinese characters; cv2.imread
    # may fail on such paths even when the file exists.
    try:
        frame = np.asarray(Image.open(args.image).convert("RGB"))
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"Could not read image: {args.image}") from exc
    original_h, original_w = frame.shape[:2]
    stage1_frame = cv2.resize(frame, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)

    stage1_checkpoint = torch.load(args.stage1, map_location=device, weights_only=False)
    stage1_model = CandidateNet().to(device)
    stage1_model.load_state_dict(stage1_checkpoint["model"])
    stage1_model.eval()

    stage2_checkpoint = torch.load(args.stage2, map_location=device, weights_only=False)
    stage2_model = Stage2Classifier(num_classes=len(stage2_checkpoint["class_to_idx"])).to(device)
    stage2_model.load_state_dict(stage2_checkpoint["model"])
    stage2_model.eval()
    person_idx = int(stage2_checkpoint["person_idx"])

    stage1_tensor = prepare_stage1_tensor(stage1_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = stage1_model(stage1_tensor).sigmoid()[0, 0].cpu().numpy()
    proposals = merged_candidates(
        probability,
        args.stage1_threshold,
        min_distance=args.proposal_min_distance,
        max_candidates=args.max_proposals,
    )

    # 96x96 in the original 1920x1080 source corresponds to 48x48 at 960x540.
    crop_w = max(1, int(round(STAGE2_CROP_SOURCE * original_w / 1920)))
    crop_h = max(1, int(round(STAGE2_CROP_SOURCE * original_h / 1080)))
    output = frame.copy()
    crop_dir = args.output.parent / f"{args.output.stem}_crops"
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    person_count = 0
    for index, proposal in enumerate(proposals):
        grid_x, grid_y = proposal["center_grid"]
        center_x_stage1 = (grid_x + 0.5) * CELL_SIZE
        center_y_stage1 = (grid_y + 0.5) * CELL_SIZE
        center_x = center_x_stage1 * original_w / IMAGE_W
        center_y = center_y_stage1 * original_h / IMAGE_H

        # Draw every Stage 1 proposal, including ones later rejected by Stage 2.
        # Coordinates are clipped to the original frame; yellow marks candidates.
        x0 = max(0, int(round(center_x - crop_w / 2)))
        y0 = max(0, int(round(center_y - crop_h / 2)))
        x1 = min(original_w - 1, int(round(center_x + crop_w / 2)))
        y1 = min(original_h - 1, int(round(center_y + crop_h / 2)))
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 0), 2)

        crop = crop_with_padding(frame, center_x, center_y, crop_w, crop_h)
        crop = cv2.resize(crop, (STAGE2_INPUT, STAGE2_INPUT), interpolation=cv2.INTER_LINEAR)
        crop = enhance_lab_dog(crop)
        crop_tensor = TF.to_tensor(Image.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = stage2_model(crop_tensor).softmax(dim=1)[0]
        person_score = float(scores[person_idx].item())

        # Keep Stage 1 and Stage 2 scores visible for debugging each proposal.
        label = f"C{index:02d} S1:{proposal['score']:.2f} S2:{person_score:.2f}"
        cv2.putText(output, label, (x0, max(18, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        if args.save_crops:
            Image.fromarray(crop).save(crop_dir / f"candidate_{index:04d}_{person_score:.3f}.jpg")
        if person_score < args.person_threshold:
            continue

        person_count += 1
        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(output, f"person {person_score:.2f}", (x0, max(20, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(output, "Yellow: Stage 1 candidate | Green: Stage 2 person",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output)
    print(f"device={device}; proposals={len(proposals)}; persons={person_count}")
    print(f"annotated_image={args.output}")
    if args.save_crops:
        print(f"candidate_crops={crop_dir}")


if __name__ == "__main__":
    main()
