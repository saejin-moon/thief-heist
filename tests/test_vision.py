import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath('src'))

from constants import WALL
from vision import line_is_clear


class TestVision(unittest.TestCase):
    def test_line_clear(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[5, 5] = WALL
        self.assertTrue(line_is_clear(grid, 0, 0, 4, 4, WALL, -1))
        self.assertFalse(line_is_clear(grid, 0, 0, 9, 9, WALL, -1))

if __name__ == "__main__":
    unittest.main()
