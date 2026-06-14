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

    def crop_to_mask_bbox(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        roi_mask: np.ndarray,
        margin_ratio: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop image and target mask to the bounding box of a binary ROI mask."""
        if image.shape[:2] != mask.shape[:2] or image.shape[:2] != roi_mask.shape[:2]:
            raise ValueError(
                "Image, mask, and ROI mask must have matching spatial dimensions: "
                f"image={image.shape}, mask={mask.shape}, roi_mask={roi_mask.shape}"
            )

        ys, xs = np.where(roi_mask > 127)
        if len(xs) == 0 or len(ys) == 0:
            return image, mask

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        margin = int(round(max(x2 - x1, y2 - y1) * margin_ratio))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)

        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def augment_pair(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply light paired augmentations for meibography segmentation."""
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        if np.random.rand() < 0.2:
            image = np.ascontiguousarray(np.flipud(image))
            mask = np.ascontiguousarray(np.flipud(mask))

        if np.random.rand() < 0.7:
            angle = float(np.random.uniform(-10.0, 10.0))
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        if np.random.rand() < 0.5:
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = float(np.random.uniform(-10.0, 10.0))
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return image, mask

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
