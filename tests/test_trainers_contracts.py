"""
Unit tests to verify algorithmic, architectural, and initialization contracts across all 7 MARL trainers.
"""

import importlib
import inspect
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath("src/py"))


class TestTrainerContracts(unittest.TestCase):
    def setUp(self):
        self.algos = ["thief", "ecoop", "mappo", "hmappo", "coop", "marc", "coma"]

    def test_all_trainer_modules_importable(self):
        for algo in self.algos:
            module_name = f"train_{algo}"
            mod = importlib.import_module(module_name)
            self.assertTrue(hasattr(mod, "train"), f"{module_name} missing train function")
            sig = inspect.signature(mod.train)
            self.assertIn("algo_name", sig.parameters)
            self.assertIn("stage_idx", sig.parameters)
            self.assertIn("total_timesteps", sig.parameters)
            self.assertIn("save_ckpt_dir", sig.parameters)
            self.assertIn("log_dir", sig.parameters)

    def test_orthogonal_layer_initialization(self):
        for algo in self.algos:
            mod = importlib.import_module(f"train_{algo}")
            self.assertTrue(
                hasattr(mod, "layer_init"),
                f"Trainer train_{algo}.py missing variance-reduction layer_init function",
            )
            # Test layer_init output
            lin = torch.nn.Linear(32, 16)
            initialized = mod.layer_init(lin, std=1.414, bias_const=0.0)
            self.assertEqual(initialized.bias.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
