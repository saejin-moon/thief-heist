import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src/py"))

from ascii import RustEnvWrapper
from constants import (
    ACTION_SPACE_SIZE,
    AGENTS,
    GOAL_VECTOR_DIM,
    MAP_SIZE,
    N_AGENTS,
    OBSERVATION_SIZE,
)
from train_coma import ComaNetwork
from train_coop import CoopNetwork
from train_ecoop import EcoopNetwork
from train_hmappo import HierarchicalNetwork
from train_mappo import MappoNetwork
from train_marc import MarcNetwork
from train_thief import ThiefNetwork
from vec_env import make_vec_env


class TestNativeHeistEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "map_size": (17, 17),
            "guard_count": 2,
            "camera_count": 1,
            "door_count": 1,
            "max_steps": 50,
            "alarm_max": 150.0,
            "spawn_mode": "role",
        }
        self.num_envs = 4
        self.vec_env = make_vec_env(
            num_envs=self.num_envs, config=self.config, base_seed=42
        )
        self.single_env = RustEnvWrapper(self.config, base_seed=42)

    def test_vector_env_contract(self):
        obs, state = self.vec_env.reset(seed=42)
        self.assertIn("_stacked", obs)
        self.assertEqual(state.shape, (self.num_envs, self.vec_env.state_dim))

        for key in ["observation", "action_mask", "role_id", "goal_vector"]:
            self.assertIn(key, obs["_stacked"])

        self.assertEqual(
            obs["_stacked"]["observation"].shape,
            (N_AGENTS, self.num_envs, OBSERVATION_SIZE[0], OBSERVATION_SIZE[1]),
        )
        self.assertEqual(
            obs["_stacked"]["action_mask"].shape,
            (N_AGENTS, self.num_envs, ACTION_SPACE_SIZE),
        )
        self.assertEqual(
            obs["_stacked"]["role_id"].shape,
            (N_AGENTS, self.num_envs, N_AGENTS),
        )
        self.assertEqual(
            obs["_stacked"]["goal_vector"].shape,
            (N_AGENTS, self.num_envs, GOAL_VECTOR_DIM),
        )

        for a in AGENTS:
            self.assertIn(a, obs)
            self.assertEqual(obs[a]["observation"].shape, (self.num_envs, 7, 7))
            self.assertEqual(obs[a]["action_mask"].shape, (self.num_envs, 6))

        # Test stepping
        actions = {a: np.zeros(self.num_envs, dtype=np.int32) for a in AGENTS}
        next_obs, rews, terms, truncs, infos = self.vec_env.step(actions)

        self.assertEqual(len(infos), self.num_envs)
        for a in AGENTS:
            self.assertEqual(rews[a].shape, (self.num_envs,))
            self.assertEqual(terms[a].shape, (self.num_envs,))
            self.assertEqual(truncs[a].shape, (self.num_envs,))
            self.assertEqual(next_obs[a]["observation"].shape, (self.num_envs, 7, 7))

    def test_single_env_render_wrapper(self):
        obs, _ = self.single_env.reset(seed=42)
        for a in AGENTS:
            self.assertIn(a, obs)
            self.assertEqual(obs[a]["observation"].shape, (7, 7))
            self.assertEqual(obs[a]["action_mask"].shape, (6,))

        self.assertEqual(self.single_env.map_h, 17)
        self.assertEqual(self.single_env.map_w, 17)
        self.assertEqual(self.single_env.grid.shape, (17, 17))
        self.assertEqual(self.single_env.explored_map.shape, (17, 17))
        self.assertEqual(len(self.single_env.agent_positions), N_AGENTS)

        actions = {a: 4 for a in AGENTS}
        next_obs, _rews, _terms, _truncs, info = self.single_env.step(actions)
        self.assertIn("scout", next_obs)
        self.assertIn("win", info["scout"])
        self.assertIn("alarm", info["scout"])

    def test_neural_network_forward_passes(self):
        state_dim = 6 + (MAP_SIZE[0] * MAP_SIZE[1]) + (N_AGENTS * 2) + 24 + 12
        device = "cuda" if torch.cuda.is_available() else "cpu"

        obs_t = torch.zeros(N_AGENTS, 7, 7, device=device)
        role_t = torch.eye(N_AGENTS, device=device)
        mask_t = torch.ones(N_AGENTS, 6, device=device)
        goal_t = torch.zeros(N_AGENTS, 2, device=device)
        state_t = torch.zeros(N_AGENTS, state_dim, device=device)

        models = {
            "mappo": MappoNetwork(state_dim).to(device),
            "coop": CoopNetwork(state_dim, num_experts=2).to(device),
            "ecoop": EcoopNetwork(state_dim, num_initial_experts=2).to(device),
            "hmappo": HierarchicalNetwork(state_dim).to(device),
            "marc": MarcNetwork(state_dim).to(device),
            "coma": ComaNetwork(state_dim).to(device),
            "thief": ThiefNetwork(state_dim, num_initial_experts=2).to(device),
        }

        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                if name in ("mappo", "marc"):
                    act, _, _, _ = model.get_action_and_value(
                        obs_t, role_t, mask_t, goal_t, state_t
                    )
                elif name == "coop":
                    act, _, _, _, _ = model.get_action_and_value(
                        obs_t, role_t, mask_t, goal_t, state_t
                    )
                elif name == "ecoop":
                    act, _, _, _, _ = model.get_action_and_value(
                        obs_t,
                        role_t,
                        mask_t,
                        goal_t,
                        state_t,
                        active_experts=len(model.experts),
                    )
                elif name == "thief":
                    act, _, _, _, _ = model.get_action_and_value(
                        obs_t,
                        role_t,
                        mask_t,
                        goal_t,
                        state_t,
                        active_experts=len(model.experts),
                        num_envs=1,
                    )
                elif name == "coma":
                    act, _, _, _ = model.get_action(obs_t, role_t, mask_t, goal_t)
                elif name == "hmappo":
                    m_act, _, _, _ = model.get_manager_action_and_value(state_t, role_t)
                    act, _, _, _ = model.get_worker_action_and_value(
                        obs_t, role_t, mask_t, m_act, state_t
                    )

                self.assertEqual(act.shape, (N_AGENTS,))


if __name__ == "__main__":
    unittest.main()
