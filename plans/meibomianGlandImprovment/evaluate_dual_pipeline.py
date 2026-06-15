"""
Evaluate a two-stage MGD-1k pipeline:
1. Predict eyelid ROI from the original image.
2. Crop the original image with the predicted eyelid ROI.
3. Predict meibomian glands inside that ROI.
4. Compute gland segmentation metrics and MG ratio.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import BACKBONE, DEVICE, IMAGE_SIZE, OUTPUT_STRIDE, RESULTS_ROOT
from src.dataset import MGD1kDataModule
from src.metrics import SegmentationMetrics
from src.model import create_model
from src.preprocessing import PreprocessingPipeline


def load_state_dict(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def logits_to_mask(logits: torch.Tensor) -> np.ndarray:
    pred = torch.argmax(logits, dim=1)[0]
    return pred.detach().cpu().numpy().astype(np.uint8)


def preprocess_tensor(preprocessing: PreprocessingPipeline, image: np.ndarray, device: torch.device):
    image = preprocessing.preprocess(image)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def square_prediction_to_original(pred_square: np.ndarray, original_shape) -> np.ndarray:
    h, w = original_shape[:2]
    square_size = max(h, w)
    pred_square = cv2.resize(
        pred_square,
        (square_size, square_size),
        interpolation=cv2.INTER_NEAREST,
    )

    if h > w:
        pad = h - w
        left = pad // 2
        return pred_square[:, left:left + w]

    if w > h:
        pad = w - h
        top = pad // 2
        return pred_square[top:top + h, :]

    return pred_square


def bbox_from_mask(mask: np.ndarray, margin_ratio: float):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    margin = int(round(max(x2 - x1, y2 - y1) * margin_ratio))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(mask.shape[1], x2 + margin)
    y2 = min(mask.shape[0], y2 + margin)
    return x1, y1, x2, y2


def paste_crop_prediction(pred_crop_320: np.ndarray, full_shape, bbox):
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1
    pred_crop = cv2.resize(
        pred_crop_320,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST,
    )
    full = np.zeros(full_shape[:2], dtype=np.uint8)
    full[y1:y2, x1:x2] = pred_crop
    return full


def find_matching_mask(mask_dir: Path, image_path: Path) -> Path:
    candidates = sorted(mask_dir.glob(f"{image_path.stem}*"))
    if not candidates:
        raise FileNotFoundError(f"No matching mask found for image: {image_path}")
    return candidates[0]


def save_overlay(image: np.ndarray, gland_gt: np.ndarray, gland_pred: np.ndarray, eyelid_pred: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image_norm = image.astype(np.float32)
    image_norm = image_norm - image_norm.min()
    image_norm = image_norm / (image_norm.max() + 1e-8)
    canvas = cv2.cvtColor((image_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Eyelid boundary in blue.
    contours, _ = cv2.findContours(eyelid_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 0, 0), 2)

    # GT gland green, predicted gland red, overlap yellow-ish.
    canvas[gland_gt > 0] = (0, 255, 0)
    pred_pixels = gland_pred > 0
    canvas[pred_pixels] = np.maximum(canvas[pred_pixels], np.array([0, 0, 255]))
    cv2.imwrite(str(path), canvas)


def evaluate_sample(eyelid_model, gland_model, preprocessing, image_path, gland_path, eyelid_path, device, roi_margin):
    image = preprocessing.load_image(image_path)
    gland_gt = (preprocessing.load_mask(gland_path) > 127).astype(np.uint8)
    eyelid_gt = (preprocessing.load_mask(eyelid_path) > 127).astype(np.uint8)

    with torch.no_grad():
        eyelid_logits = eyelid_model(preprocess_tensor(preprocessing, image, device))["out"]
        eyelid_pred_square = logits_to_mask(eyelid_logits)

    eyelid_pred = square_prediction_to_original(eyelid_pred_square, image.shape)
    bbox = bbox_from_mask(eyelid_pred, margin_ratio=roi_margin)
    x1, y1, x2, y2 = bbox

    image_crop = image[y1:y2, x1:x2]
    with torch.no_grad():
        gland_logits = gland_model(preprocess_tensor(preprocessing, image_crop, device))["out"]
        gland_pred_320 = logits_to_mask(gland_logits)

    gland_pred = paste_crop_prediction(gland_pred_320, image.shape, bbox)

    gt_t = torch.from_numpy(gland_gt).unsqueeze(0)
    pred_logits = torch.stack(
        [
            torch.from_numpy(1 - gland_pred),
            torch.from_numpy(gland_pred),
        ],
        dim=0,
    ).float().unsqueeze(0)
    gland_metrics = SegmentationMetrics.calculate_metrics(pred_logits, gt_t)

    eyelid_gt_t = torch.from_numpy(eyelid_gt).unsqueeze(0)
    eyelid_logits_full = torch.stack(
        [
            torch.from_numpy(1 - eyelid_pred),
            torch.from_numpy(eyelid_pred),
        ],
        dim=0,
    ).float().unsqueeze(0)
    eyelid_metrics = SegmentationMetrics.calculate_metrics(eyelid_logits_full, eyelid_gt_t)

    pred_eyelid_area = int(eyelid_pred.sum())
    pred_gland_area = int(gland_pred.sum())
    gt_eyelid_area = int(eyelid_gt.sum())
    gt_gland_area = int(gland_gt.sum())

    return {
        "image": image,
        "gland_gt": gland_gt,
        "gland_pred": gland_pred,
        "eyelid_pred": eyelid_pred,
        "metrics": {
            "image_path": str(image_path),
            "gland_dice": gland_metrics["dice"],
            "gland_iou": gland_metrics["iou"],
            "eyelid_dice": eyelid_metrics["dice"],
            "eyelid_iou": eyelid_metrics["iou"],
            "pred_gland_area": pred_gland_area,
            "pred_eyelid_area": pred_eyelid_area,
            "pred_mg_ratio": pred_gland_area / max(pred_eyelid_area, 1),
            "gt_gland_area": gt_gland_area,
            "gt_eyelid_area": gt_eyelid_area,
            "gt_mg_ratio": gt_gland_area / max(gt_eyelid_area, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-stage eyelid + gland segmentation pipeline")
    parser.add_argument("--mgd1k-root", required=True, type=str)
    parser.add_argument("--eyelid-checkpoint", required=True, type=str)
    parser.add_argument("--gland-checkpoint", required=True, type=str)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--output-stride", type=int, default=OUTPUT_STRIDE, choices=[8, 16])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--roi-margin", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate the whole split")
    parser.add_argument("--prediction-samples", type=int, default=24)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    preprocessing = PreprocessingPipeline(image_size=IMAGE_SIZE)

    eyelid_model = create_model(
        num_classes=2,
        backbone=args.backbone,
        pretrained=False,
        output_stride=args.output_stride,
        device=str(device),
    )
    gland_model = create_model(
        num_classes=2,
        backbone=args.backbone,
        pretrained=False,
        output_stride=args.output_stride,
        device=str(device),
    )
    eyelid_model.load_state_dict(load_state_dict(Path(args.eyelid_checkpoint), device))
    gland_model.load_state_dict(load_state_dict(Path(args.gland_checkpoint), device))
    eyelid_model.eval()
    gland_model.eval()

    gland_dm = MGD1kDataModule(
        args.mgd1k_root,
        mask_type="gland",
        batch_size=1,
        num_workers=0,
        crop_to_eyelid_roi=False,
        augment=False,
    )
    split_dataset = {
        "val": gland_dm.val_dataset,
        "test": gland_dm.test_dataset,
    }[args.split]

    eyelid_dataset = gland_dm.full_dataset.__class__(
        args.mgd1k_root,
        mask_type="eyelid",
        preprocessing=preprocessing,
        crop_to_eyelid_roi=False,
        augment=False,
    )

    rows = []
    figure_dir = RESULTS_ROOT / "figures" / f"dual_{run_id}" / args.split
    n_samples = len(split_dataset) if args.max_samples <= 0 else min(args.max_samples, len(split_dataset))

    for idx in range(n_samples):
        image_path = Path(split_dataset.image_paths[idx])
        gland_path = Path(split_dataset.mask_paths[idx])
        eyelid_path = find_matching_mask(eyelid_dataset.masks_dir, image_path)
        result = evaluate_sample(
            eyelid_model,
            gland_model,
            preprocessing,
            image_path,
            gland_path,
            eyelid_path,
            device,
            args.roi_margin,
        )
        row = result["metrics"]
        rows.append(row)

        if idx < args.prediction_samples:
            save_overlay(
                result["image"],
                result["gland_gt"],
                result["gland_pred"],
                result["eyelid_pred"],
                figure_dir / f"{idx:03d}_{image_path.stem}.png",
            )

    summary = {
        "run_id": run_id,
        "split": args.split,
        "n_samples": len(rows),
        "mean_gland_dice": float(np.mean([r["gland_dice"] for r in rows])),
        "mean_gland_iou": float(np.mean([r["gland_iou"] for r in rows])),
        "mean_eyelid_dice": float(np.mean([r["eyelid_dice"] for r in rows])),
        "mean_eyelid_iou": float(np.mean([r["eyelid_iou"] for r in rows])),
        "mean_pred_mg_ratio": float(np.mean([r["pred_mg_ratio"] for r in rows])),
        "mean_gt_mg_ratio": float(np.mean([r["gt_mg_ratio"] for r in rows])),
        "mean_abs_mg_ratio_error": float(np.mean([abs(r["pred_mg_ratio"] - r["gt_mg_ratio"]) for r in rows])),
        "args": vars(args),
    }

    metrics_dir = RESULTS_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"dual_summary_{run_id}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = metrics_dir / f"dual_per_image_{run_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics to {csv_path}")
    print(f"Saved overlays to {figure_dir}")


if __name__ == "__main__":
    main()
