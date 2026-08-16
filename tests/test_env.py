import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('src'))

from constants import ACTION_SPACE_SIZE, AGENTS, INTERACT
from env import HeistEnv, manhattan


class TestHeistEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "map_size": (17, 17),
            "guard_count": 2,
            "camera_count": 1,
            "door_count": 1,
            "max_steps": 50,
            "spawn_mode": "role"
        }
        self.env = HeistEnv(self.config)

    def test_reset(self):
        obs, _infos = self.env.reset()
        self.assertIn("scout", obs)
        self.assertEqual(len(self.env.agents), len(AGENTS))
        self.assertEqual(self.env.action_spaces["scout"].n, ACTION_SPACE_SIZE)
        self.assertEqual(ACTION_SPACE_SIZE, 6)

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


if __name__ == "__main__":
    unittest.main()
