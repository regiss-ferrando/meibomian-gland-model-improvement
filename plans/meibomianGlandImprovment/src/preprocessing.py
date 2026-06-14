"""
Preprocessing pipeline for MGD-1k infrared meibography images.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class PreprocessingPipeline:
    """
    Preprocessing pipeline for meibography segmentation.

    Steps:
    1. Load grayscale image or mask.
    2. Pad to square.
    3. Resize to target size.
    4. Apply CLAHE to images only.
    5. Normalize images and binarize masks.
    """

    def __init__(
        self,
        image_size: int = 320,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        self.image_size = image_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size),
        )
        self.mean = mean
        self.std = std

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load a grayscale meibography image."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")
        return image

    def load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """Load a grayscale segmentation mask."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to load mask: {mask_path}")
        return mask

    def pad_to_square(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Pad an image to a square shape."""
        h, w = image.shape[:2]
        if h == w:
            return image

        border_type = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101

        if h > w:
            pad = h - w
            left = pad // 2
            right = pad - left
            return cv2.copyMakeBorder(
                image,
                0,
                0,
                left,
                right,
                border_type,
                value=0,
            )

        pad = w - h
        top = pad // 2
        bottom = pad - top
        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            0,
            0,
            border_type,
            value=0,
        )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement."""
        return self.clahe.apply(image)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using dataset stats if available, otherwise scale to [0, 1]."""
        image = image.astype(np.float32)

        if self.mean is not None and self.std is not None:
            return (image - self.mean) / (self.std + 1e-8)

        return image / 255.0

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Run full image preprocessing."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = self.pad_to_square(image, is_mask=False)
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        image = self.apply_clahe(image)
        image = self.normalize(image)

        return image.astype(np.float32)

    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Run mask preprocessing and return class indices in {0, 1}."""
        mask = self.pad_to_square(mask, is_mask=True)
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return (mask > 127).astype(np.uint8)

    def compute_dataset_stats(self, image_paths) -> Tuple[float, float]:
        """Compute dataset mean and std after resizing and CLAHE."""
        pixels = []

        for image_path in image_paths:
            image = self.load_image(image_path)
            image = self.pad_to_square(image, is_mask=False)
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            image = self.apply_clahe(image)
            pixels.append(image.astype(np.float32).ravel())

        if not pixels:
            raise ValueError("Cannot compute dataset statistics from an empty image list")

        all_pixels = np.concatenate(pixels)
        self.mean = float(np.mean(all_pixels))
        self.std = float(np.std(all_pixels))

        return self.mean, self.std
