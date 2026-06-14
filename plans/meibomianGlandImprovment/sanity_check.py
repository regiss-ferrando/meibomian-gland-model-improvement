"""
Sanity check script for MGD-1k training pipeline
Validates data loading, preprocessing, model creation, and training loop
"""

import torch
import logging

from src.config import *
from src.dataset import MGD1kDataModule
from src.losses import CombinedLoss
from src.model import create_model
from src.metrics import SegmentationMetrics


def setup_logging():
    """Setup logging"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)


def check_dataset(logger):
    """Check dataset loading"""
    logger.info("=" * 60)
    logger.info("CHECK 1: Dataset Loading")
    logger.info("=" * 60)
    
    try:
        mgd1k_root = MGD1K_OFFICIAL
        if not mgd1k_root.exists():
            logger.error(f"MGD-1k root not found: {mgd1k_root}")
            return False
        
        data_module = MGD1kDataModule(
            mgd1k_root=str(mgd1k_root),
            mask_type="gland",
            batch_size=4,
            num_workers=0,  # CPU only
            crop_to_eyelid_roi=USE_EYELID_ROI,
            roi_margin=ROI_MARGIN,
        )
        
        logger.info("Dataset loaded successfully")
        logger.info(f"  Total samples: {len(data_module.full_dataset)}")
        logger.info(f"  Train: {len(data_module.train_dataset)}")
        logger.info(f"  Val: {len(data_module.val_dataset)}")
        logger.info(f"  Test: {len(data_module.test_dataset)}")
        
        return True, data_module
    
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return False, None


def check_dataloader(logger, data_module):
    """Check dataloader functionality"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECK 2: DataLoader Functionality")
    logger.info("=" * 60)
    
    try:
        train_loader = data_module.get_train_loader()
        batch = next(iter(train_loader))
        
        logger.info("DataLoader working correctly")
        logger.info(f"  Image shape: {batch['image'].shape}")
        logger.info(f"  Mask shape: {batch['mask'].shape}")
        logger.info(f"  Image dtype: {batch['image'].dtype}")
        logger.info(f"  Mask dtype: {batch['mask'].dtype}")
        logger.info(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        logger.info(f"  Mask unique values: {torch.unique(batch['mask']).numpy()}")
        
        # Check for NaN values
        if torch.isnan(batch['image']).any():
            logger.error("NaN values found in images!")
            return False
        if torch.isnan(batch['mask'].float()).any():
            logger.error("NaN values found in masks!")
            return False
        
        logger.info("No NaN values detected")
        return True
    
    except Exception as e:
        logger.error(f"DataLoader check failed: {e}")
        return False


def check_model(logger):
    """Check model creation"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECK 3: Model Creation")
    logger.info("=" * 60)
    
    try:
        device = "cpu"  # Force CPU for this check
        model = create_model(
            num_classes=2,
            backbone="resnet50",
            pretrained=False,  # No pretrained to speed up
            device=device
        )
        
        logger.info(f"Model created successfully on {device}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"  Total parameters: {n_params:,}")
        logger.info(f"  Trainable parameters: {n_trainable:,}")
        logger.info(f"  Model memory (MB): {n_params * 4 / 1024 / 1024:.1f}")
        
        return True, model
    
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return False, None


def check_forward_pass(logger, model, batch):
    """Check forward pass"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECK 4: Forward Pass")
    logger.info("=" * 60)
    
    try:
        device = next(model.parameters()).device
        images = batch['image'].to(device)
        
        outputs = model(images)
        
        if isinstance(outputs, dict):
            logits = outputs['out']
        else:
            logits = outputs
        
        logger.info("Forward pass successful")
        logger.info(f"  Input shape: {images.shape}")
        logger.info(f"  Output shape: {logits.shape}")
        logger.info(f"  Output dtype: {logits.dtype}")
        logger.info(f"  Output value range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        if torch.isnan(logits).any():
            logger.error("NaN values in model output!")
            return False
        
        logger.info("No NaN values in output")
        return True, logits
    
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return False, None


def check_loss(logger, logits, batch):
    """Check loss calculation"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECK 5: Loss Calculation")
    logger.info("=" * 60)
    
    try:
        device = logits.device
        masks = batch['mask'].to(device)
        
        criterion = CombinedLoss(
            ce_weight=CE_LOSS_WEIGHT,
            dice_weight=DICE_LOSS_WEIGHT,
            foreground_weight=FOREGROUND_LOSS_WEIGHT,
        )
        loss = criterion(logits, masks)
        
        logger.info("Loss calculation successful")
        logger.info(f"  Loss value: {loss.item():.4f}")
        logger.info(f"  Loss dtype: {loss.dtype}")
        
        if torch.isnan(loss).item():
            logger.error("NaN loss value!")
            return False
        
        if loss.item() > 1e6:
            logger.warning(f"Loss is very high: {loss.item():.4f}")
        
        return True, loss
    
    except Exception as e:
        logger.error(f"Loss calculation failed: {e}")
        return False, None


def check_metrics(logger, logits, batch):
    """Check metrics calculation"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECK 6: Metrics Calculation")
    logger.info("=" * 60)
    
    try:
        masks = batch['mask']
        
        metrics = SegmentationMetrics.calculate_metrics(logits, masks, num_classes=2)
        
        logger.info("Metrics calculation successful")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return False


def check_backward_pass(logger, model, logits, batch):
    """Check backward pass and gradient flow"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECK 7: Backward Pass & Gradients")
    logger.info("=" * 60)
    
    try:
        device = logits.device
        masks = batch['mask'].to(device)
        
        criterion = CombinedLoss(
            ce_weight=CE_LOSS_WEIGHT,
            dice_weight=DICE_LOSS_WEIGHT,
            foreground_weight=FOREGROUND_LOSS_WEIGHT,
        )
        loss = criterion(logits, masks)
        
        # Backward
        loss.backward()
        
        logger.info("Backward pass successful")
        
        # Check gradients
        n_no_grad = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                n_no_grad += 1
        
        logger.info(f"  Parameters with gradients: {sum(1 for p in model.parameters() if p.grad is not None)}")
        logger.info(f"  Parameters without gradients: {n_no_grad}")
        
        # Check for exploding gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if grad_norm > 100:
                    logger.warning(f"Large gradient in {name}: {grad_norm:.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Backward pass failed: {e}")
        return False


def main():
    logger = setup_logging()
    
    logger.info("\n" + "=" * 60)
    logger.info("MGD-1k PIPELINE SANITY CHECK")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # Check 1: Dataset
    result, data_module = check_dataset(logger)
    results['dataset'] = result
    if not result:
        logger.error("\nSanity check failed at dataset loading!")
        return
    
    # Check 2: DataLoader
    result = check_dataloader(logger, data_module)
    results['dataloader'] = result
    if not result:
        logger.error("\nSanity check failed at dataloader!")
        return
    
    # Get a batch for further checks
    train_loader = data_module.get_train_loader()
    batch = next(iter(train_loader))
    
    # Check 3: Model
    result, model = check_model(logger)
    results['model'] = result
    if not result:
        logger.error("\nSanity check failed at model creation!")
        return
    
    # Check 4: Forward pass
    result, logits = check_forward_pass(logger, model, batch)
    results['forward_pass'] = result
    if not result:
        logger.error("\nSanity check failed at forward pass!")
        return
    
    # Check 5: Loss
    result, loss = check_loss(logger, logits, batch)
    results['loss'] = result
    if not result:
        logger.error("\nSanity check failed at loss calculation!")
        return
    
    # Check 6: Metrics
    result = check_metrics(logger, logits, batch)
    results['metrics'] = result
    if not result:
        logger.error("\nSanity check failed at metrics!")
        return
    
    # Check 7: Backward pass
    result = check_backward_pass(logger, model, logits, batch)
    results['backward_pass'] = result
    if not result:
        logger.error("\nSanity check failed at backward pass!")
        return
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("=" * 60)
    
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {check}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("ALL CHECKS PASSED - PIPELINE IS READY!")
        logger.info("You can now run: python train_tiny.py")
    else:
        logger.info("SOME CHECKS FAILED - SEE ABOVE FOR DETAILS")


if __name__ == "__main__":
    main()
