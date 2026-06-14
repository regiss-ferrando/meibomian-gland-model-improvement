"""
MGD-1k Dataset loader
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import PreprocessingPipeline


class MGD1kDataset(Dataset):
    """MGD-1k dataset for meibomian gland segmentation"""
    
    def __init__(self,
                 mgd1k_root: Union[str, Path],
                 mask_type: str = "gland",  # "gland" or "eyelid"
                 preprocessing: Optional[PreprocessingPipeline] = None,
                 image_paths: Optional[List[Path]] = None,
                 mask_paths: Optional[List[Path]] = None,
                 crop_to_eyelid_roi: bool = True,
                 roi_margin: float = 0.05):
        """
        Initialize MGD-1k dataset
        
        Args:
            mgd1k_root: Root path to MGD-1k "Expore MGD1k Dataset" folder
            mask_type: Type of mask - "gland" or "eyelid"
            preprocessing: PreprocessingPipeline instance
            image_paths: Specific image paths to use (if None, loads all)
            mask_paths: Specific mask paths to use
            crop_to_eyelid_roi: Crop gland samples to the eyelid ROI before resizing
            roi_margin: Fractional margin added around the eyelid ROI crop
        """
        self.mgd1k_root = Path(mgd1k_root)
        self.mask_type = mask_type
        self.preprocessing = preprocessing or PreprocessingPipeline()
        self.crop_to_eyelid_roi = crop_to_eyelid_roi
        self.roi_margin = roi_margin

        if self.mask_type not in {"gland", "eyelid"}:
            raise ValueError(f"mask_type must be 'gland' or 'eyelid', got {mask_type!r}")
        
        # Define paths
        self.images_dir = self.mgd1k_root / "Original Images"
        eyelid_options = [
            self.mgd1k_root / "Eyelid Lebels" / "Eyelid Lebels",
            self.mgd1k_root / "Eyelid Lebels",
        ]
        self.eyelid_dir = next((p for p in eyelid_options if p.exists()), eyelid_options[0])
        
        # Try to find masks directory (handle both nested and flat structures)
        if mask_type == "gland":
            gland_options = [
                self.mgd1k_root / "Meibomian Gland Labels" / "Meibomian Gland Labels",
                self.mgd1k_root / "Meibomian Gland Labels",
            ]
            self.masks_dir = next((p for p in gland_options if p.exists()), gland_options[0])
        else:  # eyelid
            self.masks_dir = next((p for p in eyelid_options if p.exists()), eyelid_options[0])
        
        # Load image and mask pairs
        if image_paths is not None:
            if mask_paths is None:
                raise ValueError("mask_paths must be provided when image_paths is provided")
            if len(image_paths) != len(mask_paths):
                raise ValueError(
                    f"image_paths and mask_paths must have the same length, got "
                    f"{len(image_paths)} and {len(mask_paths)}"
                )
            self.image_paths = image_paths
            self.mask_paths = mask_paths
        else:
            self.image_paths, self.mask_paths = self._load_file_pairs()
    
    def _load_file_pairs(self) -> Tuple[List[Path], List[Path]]:
        """Load all image-mask pairs from MGD-1k structure"""
        image_paths = []
        mask_paths = []

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # Get all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in sorted(self.images_dir.glob('*')):
            if image_path.suffix.lower() in image_extensions:
                # Try to find corresponding mask
                mask_candidates = list(self.masks_dir.glob(f"{image_path.stem}*"))
                if mask_candidates:
                    mask_path = mask_candidates[0]
                    image_paths.append(image_path)
                    mask_paths.append(mask_path)
        
        print(f"Found {len(image_paths)} image-mask pairs in {self.mask_type} masks")
        if not image_paths:
            raise ValueError(
                f"No image-mask pairs found. Checked images in {self.images_dir} "
                f"and masks in {self.masks_dir}"
            )
        return image_paths, mask_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """Get image and mask pair"""
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load images
        image = self.preprocessing.load_image(image_path)
        mask = self.preprocessing.load_mask(mask_path)

        if self.crop_to_eyelid_roi and self.mask_type == "gland":
            eyelid_mask = self.preprocessing.load_mask(self._find_eyelid_mask(image_path))
            image, mask = self.preprocessing.crop_to_mask_bbox(
                image,
                mask,
                eyelid_mask,
                margin_ratio=self.roi_margin,
            )
        
        # Preprocess
        image = self.preprocessing.preprocess(image)
        mask = self.preprocessing.preprocess_mask(mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
            'mask_path': str(mask_path)
        }

    def _find_eyelid_mask(self, image_path: Path) -> Path:
        candidates = sorted(self.eyelid_dir.glob(f"{image_path.stem}*"))
        if not candidates:
            raise FileNotFoundError(f"Eyelid ROI mask not found for image: {image_path}")
        return candidates[0]


class MGD1kDataModule:
    """Data module for managing train/val/test splits"""
    
    def __init__(self,
                 mgd1k_root: Union[str, Path],
                 mask_type: str = "gland",
                 batch_size: int = 8,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 num_workers: int = 4,
                 seed: int = 42,
                 crop_to_eyelid_roi: bool = True,
                 roi_margin: float = 0.05):
        """
        Initialize data module
        
        Args:
            mgd1k_root: Root path to MGD-1k data
            mask_type: Type of mask
            batch_size: Batch size for dataloaders
            train_split: Fraction for training
            val_split: Fraction for validation
            num_workers: Number of workers for dataloader
            seed: Random seed for reproducibility
            crop_to_eyelid_roi: Crop gland samples to eyelid ROI before resizing
            roi_margin: Fractional margin added around the eyelid ROI crop
        """
        self.mgd1k_root = mgd1k_root
        self.mask_type = mask_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = torch.cuda.is_available()
        self.crop_to_eyelid_roi = crop_to_eyelid_roi
        self.roi_margin = roi_margin

        if not 0.0 < train_split < 1.0:
            raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
        if not 0.0 <= val_split < 1.0:
            raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
        if train_split + val_split >= 1.0:
            raise ValueError(
                f"train_split + val_split must be less than 1.0, got {train_split + val_split}"
            )
        
        # Calculate splits
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split
        
        # Create preprocessing pipeline
        self.preprocessing = PreprocessingPipeline()
        
        # Load full dataset
        self.full_dataset = MGD1kDataset(
            mgd1k_root=mgd1k_root,
            mask_type=mask_type,
            preprocessing=self.preprocessing,
            crop_to_eyelid_roi=crop_to_eyelid_roi,
            roi_margin=roi_margin
        )
        
        # Create splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._split_dataset()
    
    def _split_dataset(self):
        """Split dataset into train/val/test"""
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.full_dataset))
        if len(indices) < 3:
            raise ValueError(f"Need at least 3 samples for train/val/test split, got {len(indices)}")
        
        n_train = int(len(indices) * self.train_split)
        n_val = int(len(indices) * self.val_split)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Create subset datasets
        train_image_paths = [self.full_dataset.image_paths[i] for i in train_indices]
        train_mask_paths = [self.full_dataset.mask_paths[i] for i in train_indices]
        
        val_image_paths = [self.full_dataset.image_paths[i] for i in val_indices]
        val_mask_paths = [self.full_dataset.mask_paths[i] for i in val_indices]
        
        test_image_paths = [self.full_dataset.image_paths[i] for i in test_indices]
        test_mask_paths = [self.full_dataset.mask_paths[i] for i in test_indices]
        
        self.train_dataset = MGD1kDataset(
            mgd1k_root=self.mgd1k_root,
            mask_type=self.mask_type,
            preprocessing=self.preprocessing,
            image_paths=train_image_paths,
            mask_paths=train_mask_paths,
            crop_to_eyelid_roi=self.crop_to_eyelid_roi,
            roi_margin=self.roi_margin
        )
        
        self.val_dataset = MGD1kDataset(
            mgd1k_root=self.mgd1k_root,
            mask_type=self.mask_type,
            preprocessing=self.preprocessing,
            image_paths=val_image_paths,
            mask_paths=val_mask_paths,
            crop_to_eyelid_roi=self.crop_to_eyelid_roi,
            roi_margin=self.roi_margin
        )
        
        self.test_dataset = MGD1kDataset(
            mgd1k_root=self.mgd1k_root,
            mask_type=self.mask_type,
            preprocessing=self.preprocessing,
            image_paths=test_image_paths,
            mask_paths=test_mask_paths,
            crop_to_eyelid_roi=self.crop_to_eyelid_roi,
            roi_margin=self.roi_margin
        )
        
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def get_train_loader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
