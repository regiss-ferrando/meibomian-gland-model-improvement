import random
import unittest

import numpy as np
import torch

from train import set_training_seed


class TrainingSeedTests(unittest.TestCase):
    def test_same_seed_repeats_random_sequences(self):
        set_training_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        set_training_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_none_preserves_existing_rng_state(self):
        random.seed(456)
        expected = random.random()
        random.seed(456)
        set_training_seed(None)
        self.assertEqual(random.random(), expected)


if __name__ == "__main__":
    unittest.main()
