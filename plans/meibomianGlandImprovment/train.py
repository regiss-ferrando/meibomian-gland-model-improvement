"""
Training script for DeepLabv3+ on MGD-1k dataset
"""

import argparse
import cv2
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.config import *
from src.dataset import MGD1kDataModule
from src.losses import CombinedLoss
from src.model import create_model
from src.metrics import SegmentationMetrics


def setup_logging(log_dir: Path, run_id: str) -> logging.Logger:
    """Setup logging configuration"""
    log_file = log_dir / f"training_{run_id}.log"
    
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs['out']
    return outputs


def train_epoch(model: nn.Module,
                train_loader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: str,
                logger: logging.Logger) -> dict:
    """Train for one epoch"""
    model.train()
    metrics = {
        'loss': 0.0,
        'dice': 0.0,
        'iou': 0.0,
        'pixel_acc': 0.0,
        'pred_foreground': 0.0,
        'target_foreground': 0.0,
    }
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        logits = extract_logits(outputs)
        
        # Calculate loss
        loss = criterion(logits, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_metrics = SegmentationMetrics.calculate_metrics(logits, masks)
        
        # Accumulate metrics
        metrics['loss'] += loss.item()
        metrics['dice'] += batch_metrics['dice']
        metrics['iou'] += batch_metrics['iou']
        metrics['pixel_acc'] += batch_metrics['pixel_accuracy']
        metrics['pred_foreground'] += batch_metrics['pred_foreground']
        metrics['target_foreground'] += batch_metrics['target_foreground']
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Dice: {batch_metrics['dice']:.4f} "
                f"IoU: {batch_metrics['iou']:.4f} "
                f"PredFG: {batch_metrics['pred_foreground']:.4f} "
                f"TargetFG: {batch_metrics['target_foreground']:.4f}"
            )
    
    # Average metrics
    n_batches = len(train_loader)
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def evaluate(model: nn.Module,
             data_loader,
             criterion: nn.Module,
             device: str) -> dict:
    """Evaluate the model on a dataloader."""
    model.eval()
    metrics = {
        'loss': 0.0,
        'dice': 0.0,
        'iou': 0.0,
        'pixel_acc': 0.0,
        'mean_iou': 0.0,
        'pred_foreground': 0.0,
        'target_foreground': 0.0,
    }
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            logits = extract_logits(outputs)
            
            # Calculate loss
            loss = criterion(logits, masks)
            
            # Calculate metrics
            batch_metrics = SegmentationMetrics.calculate_metrics(logits, masks)
            
            # Accumulate metrics
            metrics['loss'] += loss.item()
            metrics['dice'] += batch_metrics['dice']
            metrics['iou'] += batch_metrics['iou']
            metrics['pixel_acc'] += batch_metrics['pixel_accuracy']
            metrics['mean_iou'] += batch_metrics['mean_iou']
            metrics['pred_foreground'] += batch_metrics['pred_foreground']
            metrics['target_foreground'] += batch_metrics['target_foreground']
    
    # Average metrics
    n_batches = len(data_loader)
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   scheduler,
                   epoch: int,
                   metrics: dict,
                   checkpoint_dir: Path,
                   args: argparse.Namespace):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'args': vars(args),
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def save_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_prediction_samples(
    model: nn.Module,
    data_loader,
    device: str,
    output_dir: Path,
    max_samples: int = 12,
):
    """Save image/ground-truth/prediction overlays for quick qualitative review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    saved = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            logits = extract_logits(model(images))
            preds = torch.argmax(logits, dim=1)

            for i in range(images.shape[0]):
                image = images[i, 0].detach().cpu().numpy()
                target = masks[i].detach().cpu().numpy().astype(np.uint8)
                pred = preds[i].detach().cpu().numpy().astype(np.uint8)

                image = image - image.min()
                image = image / (image.max() + 1e-8)
                gray = (image * 255).astype(np.uint8)
                canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                # Ground truth in green, prediction in red, overlap becomes yellow.
                canvas[target == 1] = (0, 255, 0)
                pred_pixels = pred == 1
                canvas[pred_pixels] = np.maximum(canvas[pred_pixels], np.array([0, 0, 255]))

                image_name = Path(batch['image_path'][i]).stem
                output_path = output_dir / f"{saved:03d}_{image_name}.png"
                cv2.imwrite(str(output_path), canvas)
                saved += 1

                if saved >= max_samples:
                    return


def main():
    parser = argparse.ArgumentParser(description="Train DeepLabv3+ on MGD-1k dataset")
    parser.add_argument('--mgd1k-root', type=str, 
                       default=str(MGD1K_OFFICIAL),
                       help='Path to MGD-1k dataset root')
    parser.add_argument('--mask-type', type=str, default='gland',
                       help='Type of mask: gland or eyelid')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--backbone', type=str, default=BACKBONE,
                       help='Backbone architecture')
    parser.add_argument('--output-stride', type=int, default=OUTPUT_STRIDE,
                       choices=[8, 16],
                       help='DeepLabv3+ encoder output stride')
    parser.add_argument('--device', type=str, default=DEVICE,
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                       help='Number of workers for dataloader')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Disable ImageNet-pretrained backbone weights')
    parser.add_argument('--ce-weight', type=float, default=CE_LOSS_WEIGHT,
                       help='Cross-entropy weight in the combined loss')
    parser.add_argument('--dice-weight', type=float, default=DICE_LOSS_WEIGHT,
                       help='Dice weight in the combined loss')
    parser.add_argument('--foreground-weight', type=float, default=FOREGROUND_LOSS_WEIGHT,
                       help='Foreground class weight for cross-entropy')
    parser.add_argument('--no-eyelid-roi', action='store_true',
                       help='Disable eyelid ROI cropping for gland segmentation')
    parser.add_argument('--roi-margin', type=float, default=ROI_MARGIN,
                       help='Fractional margin around the eyelid ROI crop')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable paired training augmentations')
    parser.add_argument('--early-stopping-patience', type=int, default=EARLY_STOPPING_PATIENCE,
                       help='Stop if validation IoU does not improve for this many epochs')
    parser.add_argument('--lr-scheduler-patience', type=int, default=LR_SCHEDULER_PATIENCE,
                       help='Reduce LR if validation IoU plateaus for this many epochs')
    parser.add_argument('--lr-scheduler-factor', type=float, default=LR_SCHEDULER_FACTOR,
                       help='Multiplicative LR reduction factor')
    parser.add_argument('--prediction-samples', type=int, default=PREDICTION_SAMPLES,
                       help='Number of validation/test prediction overlays to save')
    
    args = parser.parse_args()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging
    logger = setup_logging(LOG_DIR, run_id)
    logger.info(f"Starting training with args: {args}")
    save_json(vars(args), CONFIG_DIR / f"config_{run_id}.json")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading MGD-1k dataset...")
    data_module = MGD1kDataModule(
        mgd1k_root=args.mgd1k_root,
        mask_type=args.mask_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_to_eyelid_roi=not args.no_eyelid_roi,
        roi_margin=args.roi_margin,
        augment=not args.no_augmentation,
    )
    
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    test_loader = data_module.get_test_loader()
    logger.info(
        f"Dataset loaded: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches, {len(test_loader)} test batches"
    )
    
    # Create model
    logger.info(f"Creating DeepLabv3+ model with {args.backbone} backbone...")
    model = create_model(
        num_classes=2,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        output_stride=args.output_stride,
        device=str(device)
    )
    logger.info(f"Model created and moved to {device}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=MIN_LR,
    )
    criterion = CombinedLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        foreground_weight=args.foreground_weight,
    ).to(device)
    
    # Setup tensorboard
    writer = SummaryWriter(TENSORBOARD_DIR / run_id)
    
    # Training loop
    best_val_iou = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, str(device), logger)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, "
                   f"IoU: {train_metrics['iou']:.4f}, PredFG: {train_metrics['pred_foreground']:.4f}, "
                   f"TargetFG: {train_metrics['target_foreground']:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, str(device))
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
                   f"IoU: {val_metrics['iou']:.4f}, Mean IoU: {val_metrics['mean_iou']:.4f}, "
                   f"PredFG: {val_metrics['pred_foreground']:.4f}, "
                   f"TargetFG: {val_metrics['target_foreground']:.4f}")
        scheduler.step(val_metrics['iou'])
        
        # Log to tensorboard
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        history.append({
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train': train_metrics,
            'val': val_metrics,
        })
        save_json({'run_id': run_id, 'history': history}, RESULTS_ROOT / "metrics" / f"history_{run_id}.json")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_metrics,
                CHECKPOINT_DIR,
                args,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"Best model saved with IoU: {best_val_iou:.4f}")
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            logger.info(
                f"Early stopping triggered after {epochs_without_improvement} epochs "
                f"without validation IoU improvement. Best epoch: {best_epoch}"
            )
            break
    
    best_path = CHECKPOINT_DIR / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger.info(f"Loaded best model from epoch {best_epoch} for final evaluation")

    val_metrics = evaluate(model, val_loader, criterion, str(device))
    test_metrics = evaluate(model, test_loader, criterion, str(device))
    final_payload = {
        'run_id': run_id,
        'best_epoch': best_epoch,
        'best_val_iou': best_val_iou,
        'final_val': val_metrics,
        'test': test_metrics,
        'args': vars(args),
    }
    save_json(final_payload, RESULTS_ROOT / "metrics" / f"final_{run_id}.json")
    logger.info(f"Final Val - {val_metrics}")
    logger.info(f"Final Test - {test_metrics}")

    save_prediction_samples(
        model,
        val_loader,
        str(device),
        RESULTS_ROOT / "figures" / run_id / "val",
        max_samples=args.prediction_samples,
    )
    save_prediction_samples(
        model,
        test_loader,
        str(device),
        RESULTS_ROOT / "figures" / run_id / "test",
        max_samples=args.prediction_samples,
    )

    logger.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()
