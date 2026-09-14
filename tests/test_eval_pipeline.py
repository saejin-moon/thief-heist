"""
Unit tests for the parallel evaluation suite, checkpoint loaders, and evaluation parquet persistence.
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath("src/py"))

from eval import (
    MODEL_LOADERS,
    parse_eval_seeds,
    parse_eval_stages,
)
from train_coma import ComaNetwork
from train_coop import CoopNetwork
from train_ecoop import EcoopNetwork
from train_hmappo import HierarchicalNetwork
from train_mappo import MappoNetwork
from train_marc import MarcNetwork
from train_thief import ThiefNetwork


class TestEvalPipeline(unittest.TestCase):
    def test_model_loaders_registered(self):
        expected_algos = ["thief", "ecoop", "hmappo", "mappo", "coop", "marc", "coma"]
        for algo in expected_algos:
            self.assertIn(algo, MODEL_LOADERS, f"Loader missing for algo: {algo}")

    def test_parse_eval_seeds(self):
        self.assertEqual(parse_eval_seeds("0-4"), [0, 1, 2, 3, 4])
        self.assertEqual(parse_eval_seeds("0,2"), [0, 2])
        self.assertEqual(parse_eval_seeds("5"), [5])

    def test_parse_eval_stages(self):
        self.assertEqual(parse_eval_stages("all"), [0, 1, 2, 3, 4])
        self.assertEqual(parse_eval_stages("2,4"), [2, 4])
        self.assertEqual(parse_eval_stages("0-2"), [0, 1, 2])

    def test_model_loaders_execution(self):
        state_dim = 128
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test THIEF Loader
            thief_model = ThiefNetwork(state_dim, num_initial_experts=2)
            thief_path = os.path.join(tmpdir, "thief.pt")
            torch.save(
                {
                    "model_state": thief_model.state_dict(),
                    "active_experts": 2,
                    "dormant_experts": [],
                },
                thief_path,
            )
            policy_fn = MODEL_LOADERS["thief"](thief_path, state_dim, device)
            self.assertTrue(callable(policy_fn))

            # 2. Test MAPPO Loader
            mappo_model = MappoNetwork(state_dim)
            mappo_path = os.path.join(tmpdir, "mappo.pt")
            torch.save({"model_state": mappo_model.state_dict()}, mappo_path)
            policy_fn = MODEL_LOADERS["mappo"](mappo_path, state_dim, device)
            self.assertTrue(callable(policy_fn))


if __name__ == "__main__":
    unittest.main()
