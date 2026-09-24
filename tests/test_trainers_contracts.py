"""
Unit tests to verify algorithmic, architectural, and initialization contracts across all 9 MARL trainers.
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
        self.algos = [
            "thief",
            "ecoop",
            "mappo",
            "hmappo",
            "coop",
            "marc",
            "coma",
            "roma",
            "rode",
        ]

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

    def test_all_trainers_smoke_execution(self):
        """Smoke test executing actual training updates for all 7 algorithms in Stage 0."""
        import tempfile
        from constants import CURRICULUM_STAGES

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CURRICULUM_STAGES[0].copy()
            cfg.pop("timesteps", None)

            for algo in self.algos:
                mod = importlib.import_module(f"train_{algo}")
                stage_dir = os.path.join(tmpdir, algo, "stage_0")
                os.makedirs(stage_dir, exist_ok=True)

                # Run a smoke pass (500 timesteps)
                mod.train(
                    algo_name=algo,
                    stage_idx=0,
                    env_config=cfg,
                    total_timesteps=500,
                    seed=0,
                    use_rust=True,
                    save_ckpt_dir=stage_dir,
                    log_dir=stage_dir,
                )
                ckpt_path = os.path.join(stage_dir, "model.pt")
                self.assertTrue(
                    os.path.exists(ckpt_path),
                    f"{algo} failed to produce checkpoint {ckpt_path}",
                )

    def test_all_trainers_checkpoint_transfer(self):
        """Smoke test verifying checkpoint loading and transfer from Stage 0 to Stage 1 across all 7 algorithms."""
        import tempfile
        from constants import CURRICULUM_STAGES

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg0 = CURRICULUM_STAGES[0].copy()
            cfg0.pop("timesteps", None)
            cfg1 = CURRICULUM_STAGES[1].copy()
            cfg1.pop("timesteps", None)

            for algo in self.algos:
                mod = importlib.import_module(f"train_{algo}")
                stage0_dir = os.path.join(tmpdir, algo, "stage_0")
                stage1_dir = os.path.join(tmpdir, algo, "stage_1")
                os.makedirs(stage0_dir, exist_ok=True)
                os.makedirs(stage1_dir, exist_ok=True)

                # Train 500 timesteps Stage 0
                mod.train(
                    algo_name=algo,
                    stage_idx=0,
                    env_config=cfg0,
                    total_timesteps=500,
                    seed=0,
                    use_rust=True,
                    save_ckpt_dir=stage0_dir,
                    log_dir=stage0_dir,
                )
                ckpt0_path = os.path.join(stage0_dir, "model.pt")
                self.assertTrue(os.path.exists(ckpt0_path))

                # Train 500 timesteps Stage 1 loading Stage 0 checkpoint
                mod.train(
                    algo_name=algo,
                    stage_idx=1,
                    env_config=cfg1,
                    total_timesteps=500,
                    seed=0,
                    load_ckpt_path=ckpt0_path,
                    use_rust=True,
                    save_ckpt_dir=stage1_dir,
                    log_dir=stage1_dir,
                )
                ckpt1_path = os.path.join(stage1_dir, "model.pt")
                self.assertTrue(
                    os.path.exists(ckpt1_path),
                    f"{algo} failed to transfer checkpoint to Stage 1 at {ckpt1_path}",
                )

    def test_all_model_loaders_contract(self):
        """Verify that all 9 algorithms are registered in MODEL_LOADERS and policy_fn executes cleanly."""
        import tempfile
        from constants import ACTION_SPACE_SIZE, CURRICULUM_STAGES, GOAL_VECTOR_DIM, N_AGENTS, OBSERVATION_SIZE
        from eval import MODEL_LOADERS

        for algo in self.algos:
            self.assertIn(algo, MODEL_LOADERS, f"{algo} not found in eval.MODEL_LOADERS")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CURRICULUM_STAGES[0].copy()
            cfg.pop("timesteps", None)

            for algo in ["roma", "rode"]:
                mod = importlib.import_module(f"train_{algo}")
                stage_dir = os.path.join(tmpdir, algo)
                os.makedirs(stage_dir, exist_ok=True)
                mod.train(
                    algo_name=algo,
                    stage_idx=0,
                    env_config=cfg,
                    total_timesteps=500,
                    seed=0,
                    use_rust=True,
                    save_ckpt_dir=stage_dir,
                    log_dir=stage_dir,
                )
                ckpt_path = os.path.join(stage_dir, "model.pt")
                device = torch.device("cpu")
                policy_fn = MODEL_LOADERS[algo](ckpt_path, state_dim=64, device=device)

                num_envs = 2
                obs = torch.zeros((N_AGENTS, num_envs, *OBSERVATION_SIZE), dtype=torch.float32)
                role = torch.zeros((N_AGENTS, num_envs, N_AGENTS), dtype=torch.float32)
                mask = torch.ones((N_AGENTS, num_envs, ACTION_SPACE_SIZE), dtype=torch.float32)
                goal = torch.zeros((N_AGENTS, num_envs, GOAL_VECTOR_DIM), dtype=torch.float32)
                state_rep = torch.zeros((N_AGENTS * num_envs, 64), dtype=torch.float32)

                actions, chosen = policy_fn(obs, role, mask, goal, state_rep, prev_experts=None)
                self.assertEqual(actions.shape, (N_AGENTS * num_envs,))


if __name__ == "__main__":
    unittest.main()
