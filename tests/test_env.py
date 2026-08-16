import sys
import os
import unittest

sys.path.insert(0, os.path.abspath('src'))

from env import HeistEnv
from constants import AGENTS

class TestHeistEnv(unittest.TestCase):
    def setUp(self):
        self.config = {
            "map_size": (11, 11),
            "guard_count": 1,
            "camera_count": 1,
            "door_count": 1,
            "max_steps": 50,
            "spawn_mode": "role"
        }
        self.env = HeistEnv(self.config)

    def test_reset(self):
        obs, infos = self.env.reset()
        self.assertIn("scout", obs)
        self.assertEqual(len(self.env.agents), len(AGENTS))

    def test_step(self):
        self.env.reset()
        actions = {a: 0 for a in AGENTS}
        _, rewards, terms, _, _ = self.env.step(actions)
        self.assertIn("scout", rewards)
        self.assertFalse(terms["scout"])

if __name__ == "__main__":
    unittest.main()
