"""Tests for configurable preprocessing resolution."""

import unittest

import numpy as np

from src.preprocessing import PreprocessingPipeline


class ConfigurableImageSizeTests(unittest.TestCase):
    def test_default_resolution_preserves_baseline_shape(self):
        preprocessing = PreprocessingPipeline()
        image = np.zeros((40, 80), dtype=np.uint8)
        mask = np.zeros((40, 80), dtype=np.uint8)

        self.assertEqual(preprocessing.preprocess(image).shape, (320, 320))
        self.assertEqual(preprocessing.preprocess_mask(mask).shape, (320, 320))

    def test_480_resolution_applies_to_image_and_mask(self):
        preprocessing = PreprocessingPipeline(image_size=480)
        image = np.zeros((40, 80), dtype=np.uint8)
        mask = np.zeros((40, 80), dtype=np.uint8)
        mask[10:30, 20:60] = 255

        processed_image = preprocessing.preprocess(image)
        processed_mask = preprocessing.preprocess_mask(mask)

        self.assertEqual(processed_image.shape, (480, 480))
        self.assertEqual(processed_mask.shape, (480, 480))
        self.assertEqual(set(np.unique(processed_mask)), {0, 1})

    def test_non_positive_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "image_size must be >= 1"):
            PreprocessingPipeline(image_size=0)


if __name__ == "__main__":
    unittest.main()
