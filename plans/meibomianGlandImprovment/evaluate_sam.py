"""
Evaluate Segment Anything (SAM) on MGD-1k with oracle box prompts.

This script is a benchmark, not a deployable automatic pipeline: it builds a
box prompt from the ground-truth mask and measures whether SAM can recover the
target structure when prompted with a good anatomical box.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import DEVICE, RESULTS_ROOT
from src.dataset import MGD1kDataModule
from src.preprocessing import PreprocessingPipeline


def import_sam():
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment-anything is not installed. Install it with:\n"
            "pip install -r plans/meibomianGlandImprovment/requirements-sam.txt"
        ) from exc
    return SamPredictor, sam_model_registry


def load_rgb_image(path: Path, use_clahe: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def load_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to load mask: {path}")
    return (mask > 127).astype(np.uint8)


def mask_to_box(mask: np.ndarray, margin_ratio: float) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0, 0, mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    margin = int(round(max(x2 - x1 + 1, y2 - y1 + 1) * margin_ratio))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(mask.shape[1] - 1, x2 + margin)
    y2 = min(mask.shape[0] - 1, y2 + margin)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def calculate_binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = np.logical_or(pred, target).sum()
    smooth = 1e-5

    return {
        "dice": float((2 * intersection + smooth) / (pred_sum + target_sum + smooth)),
        "iou": float((intersection + smooth) / (union + smooth)),
        "pixel_acc": float((pred == target).mean()),
        "pred_foreground": float(pred.mean()),
        "target_foreground": float(target.mean()),
    }


def save_overlay(image_rgb: np.ndarray, target: np.ndarray, pred: np.ndarray, box: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    canvas[target > 0] = (0, 255, 0)
    pred_pixels = pred > 0
    canvas[pred_pixels] = np.maximum(canvas[pred_pixels], np.array([0, 0, 255]))

    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(str(path), canvas)


def save_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_predictor(model_type: str, checkpoint: Path, device: torch.device):
    SamPredictor, sam_model_registry = import_sam()
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model type {model_type!r}. Available: {sorted(sam_model_registry)}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM on MGD-1k with oracle box prompts")
    parser.add_argument("--mgd1k-root", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to SAM .pth checkpoint")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--mask-type", default="gland", choices=["gland", "eyelid"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--box-margin", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate the full split")
    parser.add_argument("--prediction-samples", type=int, default=24)
    parser.add_argument("--use-clahe", action="store_true", help="Apply CLAHE before sending image to SAM")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor = load_predictor(args.model_type, Path(args.checkpoint), device)

    data_module = MGD1kDataModule(
        args.mgd1k_root,
        mask_type=args.mask_type,
        batch_size=1,
        num_workers=0,
        crop_to_eyelid_roi=False,
        augment=False,
    )
    dataset = {"val": data_module.val_dataset, "test": data_module.test_dataset}[args.split]
    n_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))

    rows = []
    figure_dir = RESULTS_ROOT / "figures" / f"sam_{args.mask_type}_{run_id}" / args.split

    for idx in range(n_samples):
        image_path = Path(dataset.image_paths[idx])
        mask_path = Path(dataset.mask_paths[idx])

        image_rgb = load_rgb_image(image_path, use_clahe=args.use_clahe)
        target = load_binary_mask(mask_path)
        box = mask_to_box(target, margin_ratio=args.box_margin)

        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        pred = masks[best_idx].astype(np.uint8)

        row = {
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "sam_score": float(scores[best_idx]),
            "box_x1": float(box[0]),
            "box_y1": float(box[1]),
            "box_x2": float(box[2]),
            "box_y2": float(box[3]),
            **calculate_binary_metrics(pred, target),
        }
        rows.append(row)

        if idx < args.prediction_samples:
            save_overlay(
                image_rgb,
                target,
                pred,
                box,
                figure_dir / f"{idx:03d}_{image_path.stem}.png",
            )

    summary = {
        "run_id": run_id,
        "model_type": args.model_type,
        "mask_type": args.mask_type,
        "split": args.split,
        "n_samples": len(rows),
        "oracle_prompt": "ground_truth_mask_box",
        "mean_sam_score": float(np.mean([r["sam_score"] for r in rows])),
        "mean_dice": float(np.mean([r["dice"] for r in rows])),
        "mean_iou": float(np.mean([r["iou"] for r in rows])),
        "mean_pixel_acc": float(np.mean([r["pixel_acc"] for r in rows])),
        "mean_pred_foreground": float(np.mean([r["pred_foreground"] for r in rows])),
        "mean_target_foreground": float(np.mean([r["target_foreground"] for r in rows])),
        "args": vars(args),
    }

    metrics_dir = RESULTS_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, metrics_dir / f"sam_summary_{args.mask_type}_{run_id}.json")

    csv_path = metrics_dir / f"sam_per_image_{args.mask_type}_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics to {csv_path}")
    print(f"Saved overlays to {figure_dir}")


if __name__ == "__main__":
    main()
