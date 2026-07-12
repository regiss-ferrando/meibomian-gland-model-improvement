"""Compare segmentation checkpoints with per-image topology-aware figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import MGD1K_OFFICIAL
from src.dataset import MGD1kDataModule
from src.model import create_model


MODEL_ORDER = ("baseline480", "betti001", "betti0005")
MODEL_TITLES = {
    "baseline480": "Baseline 480",
    "betti001": "Betti H0 (0.01)",
    "betti0005": "Betti H0 (0.005)",
}


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int(np.count_nonzero(prediction & target))
    pred_area = int(np.count_nonzero(prediction))
    target_area = int(np.count_nonzero(target))
    union = pred_area + target_area - intersection
    dice = (2.0 * intersection + 1e-5) / (pred_area + target_area + 1e-5)
    iou = (intersection + 1e-5) / (union + 1e-5)
    beta0 = cv2.connectedComponents(prediction.astype(np.uint8), connectivity=8)[0] - 1
    target_beta0 = cv2.connectedComponents(target.astype(np.uint8), connectivity=8)[0] - 1
    return {
        "dice": float(dice),
        "iou": float(iou),
        "beta0": int(beta0),
        "beta0_error": int(abs(beta0 - target_beta0)),
        "foreground_fraction": pred_area / prediction.size,
    }


def parse_model(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Models must use NAME=/path/to/checkpoint.pt")
    name, path = value.split("=", 1)
    if name not in MODEL_ORDER:
        raise argparse.ArgumentTypeError(f"Model name must be one of {MODEL_ORDER}, got {name!r}")
    return name, Path(path).expanduser()


def load_predictions(
    checkpoint: Path,
    data_loader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = create_model(
        num_classes=2,
        backbone="resnet50",
        pretrained=False,
        output_stride=16,
        device=str(device),
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    predictions: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch["image"].to(device))["out"]
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            for path, prediction in zip(batch["image_path"], batch_predictions):
                predictions[str(path)] = prediction
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def collect_reference_data(data_loader) -> Dict[str, Dict[str, np.ndarray]]:
    references = {}
    for batch in data_loader:
        images = batch["image"][:, 0].numpy()
        targets = batch["mask"].numpy().astype(np.uint8)
        for path, image, target in zip(batch["image_path"], images, targets):
            references[str(path)] = {"image": image, "target": target}
    return references


def overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray | None = None):
    normalized = image - image.min()
    normalized /= normalized.max() + 1e-8
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    if prediction is None:
        rgb[target == 1] = np.array([0.1, 1.0, 0.1])
        return rgb
    target_only = (target == 1) & (prediction == 0)
    prediction_only = (target == 0) & (prediction == 1)
    overlap_pixels = (target == 1) & (prediction == 1)
    rgb[target_only] = np.array([0.1, 1.0, 0.1])
    rgb[prediction_only] = np.array([1.0, 0.1, 0.1])
    rgb[overlap_pixels] = np.array([1.0, 0.9, 0.0])
    return rgb


def difference_panel(
    image: np.ndarray,
    baseline: np.ndarray,
    comparison: np.ndarray,
) -> np.ndarray:
    normalized = image - image.min()
    normalized /= normalized.max() + 1e-8
    rgb = np.repeat(normalized[..., None], 3, axis=2) * 0.55
    added = (baseline == 0) & (comparison == 1)
    removed = (baseline == 1) & (comparison == 0)
    unchanged_foreground = (baseline == 1) & (comparison == 1)
    rgb[unchanged_foreground] = np.array([0.75, 0.75, 0.75])
    rgb[added] = np.array([0.0, 1.0, 1.0])
    rgb[removed] = np.array([1.0, 0.0, 1.0])
    return rgb


def save_figure(row: dict, output_path: Path) -> None:
    image = row["image_array"]
    target = row["target_array"]
    predictions = row["prediction_arrays"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title(Path(row["image"]).name)
    axes[0, 1].imshow(overlay(image, target))
    axes[0, 1].set_title(f"Target | beta0={row['target_beta0']}")
    axes[0, 2].imshow(
        difference_panel(image, predictions["baseline480"], predictions["betti0005"])
    )
    axes[0, 2].set_title(
        f"Baseline -> Betti 0.005 | disagreement={row['baseline_vs_betti0005_disagreement']:.2%}\n"
        "cyan=added, magenta=removed, gray=shared"
    )

    for axis, name in zip(axes[1], MODEL_ORDER):
        axis.imshow(overlay(image, target, predictions[name]))
        axis.set_title(
            f"{MODEL_TITLES[name]}\n"
            f"Dice={row[name + '_dice']:.3f} | IoU={row[name + '_iou']:.3f} | "
            f"beta0={row[name + '_beta0']} | |Delta beta0|={row[name + '_beta0_error']}"
        )

    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(
        "Overlay legend: yellow=correct foreground, green=missed target, red=false positive",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def serializable_row(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"image_array", "target_array", "prediction_arrays"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgd1k-root", default=str(MGD1K_OFFICIAL))
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    model_paths = dict(args.model)
    missing_names = set(MODEL_ORDER) - set(model_paths)
    if missing_names:
        parser.error(f"Missing models: {sorted(missing_names)}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_module = MGD1kDataModule(
        mgd1k_root=args.mgd1k_root,
        mask_type="gland",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        crop_to_eyelid_roi=True,
        roi_margin=0.05,
        augment=False,
    )
    test_loader = data_module.get_test_loader()
    references = collect_reference_data(test_loader)
    all_predictions = {
        name: load_predictions(model_paths[name], test_loader, device) for name in MODEL_ORDER
    }

    rows: List[dict] = []
    for image_path, reference in references.items():
        target = reference["target"]
        target_beta0 = cv2.connectedComponents(target, connectivity=8)[0] - 1
        prediction_arrays = {name: all_predictions[name][image_path] for name in MODEL_ORDER}
        row = {
            "image": image_path,
            "target_beta0": int(target_beta0),
            "image_array": reference["image"],
            "target_array": target,
            "prediction_arrays": prediction_arrays,
        }
        for name in MODEL_ORDER:
            for metric, value in binary_metrics(prediction_arrays[name], target).items():
                row[f"{name}_{metric}"] = value
        row["baseline_vs_betti001_disagreement"] = float(
            np.mean(prediction_arrays["baseline480"] != prediction_arrays["betti001"])
        )
        row["baseline_vs_betti0005_disagreement"] = float(
            np.mean(prediction_arrays["baseline480"] != prediction_arrays["betti0005"])
        )
        row["betti001_vs_betti0005_disagreement"] = float(
            np.mean(prediction_arrays["betti001"] != prediction_arrays["betti0005"])
        )
        row["betti0005_topology_improvement"] = (
            row["baseline480_beta0_error"] - row["betti0005_beta0_error"]
        )
        rows.append(row)

    csv_rows = [serializable_row(row) for row in rows]
    with (output_dir / "per_image_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "n_images": len(rows),
        "models": {name: str(model_paths[name]) for name in MODEL_ORDER},
        "means": {
            key: float(np.mean([row[key] for row in rows]))
            for key in csv_rows[0]
            if key != "image"
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rankings = {
        "topology_improved": sorted(
            rows, key=lambda row: row["betti0005_topology_improvement"], reverse=True
        ),
        "topology_regressed": sorted(
            rows, key=lambda row: row["betti0005_topology_improvement"]
        ),
        "most_changed": sorted(
            rows,
            key=lambda row: row["baseline_vs_betti0005_disagreement"],
            reverse=True,
        ),
    }
    for category, ranked_rows in rankings.items():
        for index, row in enumerate(ranked_rows[: args.top_k]):
            stem = Path(row["image"]).stem
            save_figure(row, output_dir / category / f"{index:02d}_{stem}.png")

    print(f"Compared {len(rows)} test images")
    print(f"CSV: {output_dir / 'per_image_comparison.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Figures: {output_dir}")


if __name__ == "__main__":
    main()
