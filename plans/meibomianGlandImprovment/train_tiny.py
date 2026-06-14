"""
Tiny training script for CPU validation
Trains on a small subset (50 images) to verify training loop works
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import *
from src.dataset import MGD1kDataModule
from src.losses import CombinedLoss
from src.model import create_model
from src.metrics import SegmentationMetrics


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_file = log_dir / f"training_tiny_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def train_epoch_tiny(model: nn.Module,
                     train_loader,
                     optimizer: optim.Optimizer,
                     criterion: nn.Module,
                     device: str,
                     logger: logging.Logger) -> dict:
    """Train for one epoch"""
    model.train()
    metrics = {'loss': 0.0, 'dice': 0.0, 'iou': 0.0, 'pixel_acc': 0.0}
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle DeepLabv3+ output format
        if isinstance(outputs, dict):
            logits = outputs['out']
        else:
            logits = outputs
        
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
        
        logger.info(
            f"Batch [{batch_idx+1}/{len(train_loader)}] "
            f"Loss: {loss.item():.4f} "
            f"Dice: {batch_metrics['dice']:.4f} "
            f"IoU: {batch_metrics['iou']:.4f}"
        )
    
    # Average metrics
    n_batches = len(train_loader)
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def validate_tiny(model: nn.Module,
                  val_loader,
                  criterion: nn.Module,
                  device: str) -> dict:
    """Validate the model"""
    model.eval()
    metrics = {'loss': 0.0, 'dice': 0.0, 'iou': 0.0, 'pixel_acc': 0.0, 'mean_iou': 0.0}
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            if isinstance(outputs, dict):
                logits = outputs['out']
            else:
                logits = outputs
            
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
    
    # Average metrics
    n_batches = len(val_loader)
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Tiny training on CPU for validation")
    parser.add_argument('--mgd1k-root', type=str, 
                       default=str(MGD1K_OFFICIAL),
                       help='Path to MGD-1k dataset root')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples to use')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(LOG_DIR)
    logger.info("=" * 80)
    logger.info("TINY TRAINING SESSION (CPU VALIDATION)")
    logger.info("=" * 80)
    logger.info(f"Args: {args}\n")
    
    # Setup device - force CPU
    device = torch.device("cpu")
    logger.info(f"Using device: {device}\n")
    
    # Load dataset
    logger.info("Loading MGD-1k dataset...")
    data_module = MGD1kDataModule(
        mgd1k_root=args.mgd1k_root,
        mask_type="gland",
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Create small subsets for testing
    n_samples = min(args.n_samples, len(data_module.full_dataset))
    train_size = int(n_samples * 0.7)
    val_size = n_samples - train_size
    
    # Use only first n_samples
    from torch.utils.data import Subset
    indices = list(range(n_samples))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = Subset(data_module.full_dataset, train_indices)
    val_subset = Subset(data_module.full_dataset, val_indices)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Dataset loaded: {len(train_subset)} train, {len(val_subset)} val")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # Create model (without pretrained to speed up)
    logger.info("Creating DeepLabv3+ model...")
    model = create_model(
        num_classes=2,
        backbone="resnet50",
        pretrained=False,  # No pretrained for speed
        device="cpu"
    )
    logger.info(f"Model created and moved to {device}\n")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    criterion = CombinedLoss(ce_weight=CE_LOSS_WEIGHT, dice_weight=DICE_LOSS_WEIGHT)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...\n")
    best_val_iou = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch_tiny(model, train_loader, optimizer, criterion, "cpu", logger)
        logger.info(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                   f"Dice: {train_metrics['dice']:.4f}, "
                   f"IoU: {train_metrics['iou']:.4f}")
        
        # Validate
        val_metrics = validate_tiny(model, val_loader, criterion, "cpu")
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                   f"Dice: {val_metrics['dice']:.4f}, "
                   f"IoU: {val_metrics['iou']:.4f}, "
                   f"Mean IoU: {val_metrics['mean_iou']:.4f}")
        
        # Track best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            logger.info(f"New best model (IoU: {best_val_iou:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info("TINY TRAINING SESSION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Final best validation IoU: {best_val_iou:.4f}")
    logger.info("\nNext steps:")
    logger.info("  1. Run sanity_check.py for detailed validation")
    logger.info("  2. Once GPU available, run: python train.py")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
