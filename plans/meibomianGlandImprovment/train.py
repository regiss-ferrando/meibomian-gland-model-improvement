"""
Training script for DeepLabv3+ on MGD-1k dataset
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.config import *
from src.dataset import MGD1kDataModule
from src.losses import CombinedLoss
from src.model import create_model
from src.metrics import SegmentationMetrics


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def train_epoch(model: nn.Module,
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
        
        if (batch_idx + 1) % 10 == 0:
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


def validate(model: nn.Module,
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


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   metrics: dict,
                   checkpoint_dir: Path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


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
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(LOG_DIR)
    logger.info(f"Starting training with args: {args}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading MGD-1k dataset...")
    data_module = MGD1kDataModule(
        mgd1k_root=args.mgd1k_root,
        mask_type=args.mask_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    logger.info(f"Dataset loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    logger.info(f"Creating DeepLabv3+ model with {args.backbone} backbone...")
    model = create_model(
        num_classes=2,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        device=str(device)
    )
    logger.info(f"Model created and moved to {device}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    criterion = CombinedLoss(ce_weight=args.ce_weight, dice_weight=args.dice_weight)
    
    # Setup tensorboard
    writer = SummaryWriter(TENSORBOARD_DIR / datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # Training loop
    best_val_iou = 0.0
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, str(device), logger)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, "
                   f"IoU: {train_metrics['iou']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, str(device))
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
                   f"IoU: {val_metrics['iou']:.4f}, Mean IoU: {val_metrics['mean_iou']:.4f}")
        
        # Log to tensorboard
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        
        # Save checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = save_checkpoint(model, optimizer, epoch+1, val_metrics, CHECKPOINT_DIR)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"Best model saved with IoU: {best_val_iou:.4f}")
    
    logger.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()
