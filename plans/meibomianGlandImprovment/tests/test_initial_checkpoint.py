import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from train import load_initial_model_weights


class InitialCheckpointTests(unittest.TestCase):
    def test_loads_plain_model_state_dict(self):
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.pt"
            torch.save(source.state_dict(), path)
            resolved = load_initial_model_weights(target, path, "cpu")
        self.assertEqual(resolved, path.resolve())
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_loads_periodic_checkpoint_payload(self):
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.pt"
            torch.save({"model_state_dict": source.state_dict(), "epoch": 4}, path)
            load_initial_model_weights(target, path, "cpu")
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_missing_checkpoint_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_initial_model_weights(nn.Linear(1, 1), "missing-checkpoint.pt", "cpu")


if __name__ == "__main__":
    unittest.main()
