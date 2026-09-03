import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("rust/python"))

from vec_env_rust import RustVectorEnv

from constants import AGENTS
from vec_env import VectorEnv as PyVectorEnv


class TestRustVectorEnvParity(unittest.TestCase):
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

    def test_tensor_contract_parity(self):
        py_env = PyVectorEnv(num_envs=self.num_envs, config=self.config, base_seed=42)
        rs_env = RustVectorEnv(num_envs=self.num_envs, config=self.config, base_seed=42)

        py_obs, py_state = py_env.reset(seed=42)
        rs_obs, rs_state = rs_env.reset(seed=42)

        # 1. State dimension & shape
        self.assertEqual(py_state.shape, rs_state.shape)
        self.assertEqual(py_env.state_dim, rs_env.state_dim)

        # 2. Observation structure and shapes
        self.assertIn("_stacked", py_obs)
        self.assertIn("_stacked", rs_obs)

        for key in ["observation", "action_mask", "role_id", "goal_vector"]:
            self.assertIn(key, py_obs["_stacked"])
            self.assertIn(key, rs_obs["_stacked"])
            self.assertEqual(
                py_obs["_stacked"][key].shape,
                rs_obs["_stacked"][key].shape,
                f"Shape mismatch for key {key}",
            )
            self.assertEqual(
                py_obs["_stacked"][key].dtype,
                rs_obs["_stacked"][key].dtype,
                f"Dtype mismatch for key {key}",
            )

        for a in AGENTS:
            self.assertIn(a, py_obs)
            self.assertIn(a, rs_obs)
            for key in ["observation", "action_mask", "role_id", "goal_vector"]:
                self.assertEqual(
                    py_obs[a][key].shape,
                    rs_obs[a][key].shape,
                    f"Shape mismatch for agent {a}, key {key}",
                )

        # 3. Stepping output parity
        actions = {a: np.zeros(self.num_envs, dtype=np.int32) for a in AGENTS}
        _py_next_obs, py_r, py_terms, py_truncs, py_infos = py_env.step(actions)
        _rs_next_obs, rs_r, rs_terms, rs_truncs, rs_infos = rs_env.step(actions)

        for a in AGENTS:
            self.assertEqual(py_r[a].shape, rs_r[a].shape)
            self.assertEqual(py_terms[a].shape, rs_terms[a].shape)
            self.assertEqual(py_truncs[a].shape, rs_truncs[a].shape)
            self.assertEqual(len(py_infos), len(rs_infos))

        for e in range(self.num_envs):
            for a in AGENTS:
                self.assertIn("win", rs_infos[e][a])
                self.assertIn("alarm", rs_infos[e][a])
                self.assertIn("steps", rs_infos[e][a])
                self.assertIn("pos", rs_infos[e][a])
                self.assertIn("scout_pois_tagged", rs_infos[e][a])
                self.assertIn("hacker_hack_success", rs_infos[e][a])
                self.assertIn("muscle_neutralize_success", rs_infos[e][a])
                self.assertIn("extractor_loot_success", rs_infos[e][a])

        py_env.close()
        rs_env.close()


if __name__ == "__main__":
    unittest.main()
