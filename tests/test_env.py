import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))

from constants import (
    ACTION_SPACE_SIZE,
    AGENTS,
    DOOR,
    GOAL_VECTOR_DIM,
    INTERACT,
    N_AGENTS,
    OBSERVATION_SIZE,
    WALL,
)
from env import HeistEnv, manhattan
from train_coma import ComaNetwork
from train_coop import CoopNetwork
from train_ecoop import EcoopNetwork
from train_hmappo import HierarchicalNetwork
from train_mappo import MappoNetwork
from train_marc import MarcNetwork
from train_thief import ThiefNetwork
from vec_env import VectorEnv
from vision import bfs_next_step


class TestHeistEnv(unittest.TestCase):
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
        self.env = HeistEnv(self.config)

    def test_reset(self):
        obs, _infos = self.env.reset()
        self.assertIn("scout", obs)
        self.assertEqual(len(self.env.agents), len(AGENTS))
        self.assertEqual(self.env.action_spaces["scout"].n, ACTION_SPACE_SIZE)
        self.assertEqual(ACTION_SPACE_SIZE, 6)
        # Check observation space has all keys including goal_vector
        for a in AGENTS:
            self.assertIn("observation", self.env.observation_spaces[a].spaces)
            self.assertIn("action_mask", self.env.observation_spaces[a].spaces)
            self.assertIn("role_id", self.env.observation_spaces[a].spaces)
            self.assertIn("goal_vector", self.env.observation_spaces[a].spaces)
            self.assertEqual(obs[a]["goal_vector"].shape, (GOAL_VECTOR_DIM,))

    def test_agent_and_guard_spawning(self):
        self.env.reset(seed=42)
        # Agents should be clustered in a spawn room
        agent_positions = list(self.env.agent_positions.values())
        for a_pos in agent_positions:
            # Distance among agents in spawn room should be small
            min_dist_to_teammate = min(
                manhattan(a_pos, other) for other in agent_positions if other != a_pos
            )
            self.assertLessEqual(min_dist_to_teammate, 8)

        # Guards should spawn at a safe distance from agents
        for g_pos in self.env.guard_positions:
            for a_pos in agent_positions:
                dist = manhattan(g_pos, a_pos)
                self.assertGreaterEqual(dist, 2)

    def test_step(self):
        self.env.reset()
        actions = {a: 0 for a in AGENTS}
        _, rewards, terms, _, _ = self.env.step(actions)
        self.assertIn("scout", rewards)
        self.assertFalse(terms["scout"])

    def test_single_step_loot_and_extraction(self):
        self.env.reset()
        self.env.terminal_disabled = True
        self.env.agent_positions["extractor"] = self.env.loot_pos

        actions = {a: 4 for a in AGENTS}
        actions["extractor"] = INTERACT
        self.env.step(actions)

        self.assertTrue(self.env.loot_acquired)
        self.assertTrue(self.env.extraction_triggered)

    def test_permanent_guard_neutralization(self):
        self.env.reset()
        if self.env.guard_positions:
            gpos = self.env.guard_positions[0]
            self.env.agent_positions["muscle"] = gpos

            actions = {a: 4 for a in AGENTS}
            actions["muscle"] = INTERACT
            self.env.step(actions)

            self.assertEqual(self.env.neutralized[0], 1)

            # Step 10 times and verify guard stays neutralized (permanent)
            for _ in range(10):
                self.env.step({a: 4 for a in AGENTS})
                self.assertEqual(self.env.neutralized[0], 1)

    def test_alarm_capping_and_scaling(self):
        self.env.reset()
        self.assertEqual(self.env.alarm_max, 150.0)
        rewards = {a: 0.0 for a in AGENTS}
        for _ in range(30):
            self.env._add_alarm(10.0, rewards)
        self.assertLessEqual(self.env.alarm, 150.0)
        self.assertGreater(self.env.alarm, 100.0)

    def test_guard_bfs_pathfinding(self):
        self.env.reset(seed=42)
        empty_cells = [
            tuple(int(x) for x in c) for c in np.argwhere(self.env.grid == 0)
        ]
        self.assertGreater(len(empty_cells), 2)
        start = empty_cells[0]
        target = empty_cells[-1]

        step_r, _step_c = bfs_next_step(
            self.env.grid,
            start[0],
            start[1],
            target[0],
            target[1],
            WALL,
            DOOR,
            self.env._bfs_queue,
            self.env._bfs_previous,
            self.env._bfs_reset,
        )
        self.assertTrue(step_r >= 0 or (start == target))

    def test_vector_env(self):
        vec = VectorEnv(num_envs=2, config=self.config, base_seed=0)
        obs, state = vec.reset()
        self.assertIn("_stacked", obs)
        self.assertEqual(state.shape[0], 2)
        actions = {a: np.zeros(2, dtype=np.int32) for a in AGENTS}
        _next_obs, rewards, _terms, _truncs, _infos = vec.step(actions)
        self.assertEqual(rewards["scout"].shape[0], 2)
        vec.close()

    def test_network_forward_passes(self):
        state_dim = self.env.state().shape[0]
        obs = torch.zeros(1, *OBSERVATION_SIZE)
        role = torch.zeros(1, N_AGENTS)
        mask = torch.ones(1, ACTION_SPACE_SIZE)
        goal = torch.zeros(1, GOAL_VECTOR_DIM)
        state = torch.zeros(1, state_dim)

        # 1. MAPPO
        mappo = MappoNetwork(state_dim)
        act, _logp, _ent, _val = mappo.get_action_and_value(
            obs, role, mask, goal, state
        )
        self.assertEqual(act.shape, (1,))

        # 2. CO-OP
        coop = CoopNetwork(state_dim, num_experts=2)
        act, _logp, _ent, _val, _ = coop.get_action_and_value(
            obs, role, mask, goal, state
        )
        self.assertEqual(act.shape, (1,))

        # 3. E-COOP
        ecoop = EcoopNetwork(state_dim, num_initial_experts=2)
        act, _logp, _ent, _val, _ = ecoop.get_action_and_value(
            obs, role, mask, goal, state, active_experts=2
        )
        self.assertEqual(act.shape, (1,))

        # 4. H-MAPPO
        hmappo = HierarchicalNetwork(state_dim)
        m_act, _m_logp, _, _m_val = hmappo.get_manager_action_and_value(
            state, role
        )
        w_act, _w_logp, _, _w_val = hmappo.get_worker_action_and_value(
            obs, role, mask, m_act, state
        )
        self.assertEqual(m_act.shape, (1, 2))
        self.assertEqual(w_act.shape, (1,))

        # 5. MARC
        marc = MarcNetwork(state_dim)
        act, _logp, _ent, _val = marc.get_action_and_value(
            obs, role, mask, goal, state
        )
        self.assertEqual(act.shape, (1,))

        # 6. COMA
        coma = ComaNetwork(state_dim)
        act, _logp, _ent, _probs = coma.get_action(obs, role, mask, goal)
        other_acts = torch.zeros(1, (N_AGENTS - 1) * ACTION_SPACE_SIZE)
        q_vals = coma.get_q_values(state, other_acts, role, goal)
        self.assertEqual(act.shape, (1,))
        self.assertEqual(q_vals.shape, (1, ACTION_SPACE_SIZE))

        # 7. THIEF
        thief = ThiefNetwork(state_dim, num_initial_experts=2)
        act, _logp, _ent, _val, _ = thief.get_action_and_value(
            obs, role, mask, goal, state, active_experts=2, num_envs=1
        )
        self.assertEqual(act.shape, (1,))


if __name__ == "__main__":
    unittest.main()
