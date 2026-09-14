"""
Unit tests for curriculum execution, seed parsing, priority ordering, and CLI argument contracts.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src/py"))

from curriculum import (
    PRIORITY_ALGO_ORDER,
    parse_algos,
    parse_seeds,
    parse_stages,
)


class TestCurriculumContracts(unittest.TestCase):
    def test_priority_algo_order(self):
        expected_order = ["thief", "ecoop", "mappo", "hmappo", "coop", "marc", "coma"]
        self.assertEqual(PRIORITY_ALGO_ORDER, expected_order)

    def test_parse_algos(self):
        # 'all' returns the full priority order
        self.assertEqual(parse_algos("all"), PRIORITY_ALGO_ORDER)
        self.assertEqual(parse_algos(None), PRIORITY_ALGO_ORDER)

        # Subsets are sorted according to priority order
        self.assertEqual(parse_algos("coma,thief"), ["thief", "coma"])
        self.assertEqual(parse_algos("marc,ecoop,mappo"), ["ecoop", "mappo", "marc"])
        self.assertEqual(parse_algos("thief"), ["thief"])

    def test_parse_seeds_inclusive(self):
        # Range '0-10' must be inclusive (11 seeds: 0 through 10)
        seeds_10 = parse_seeds("0-10")
        self.assertEqual(seeds_10, list(range(11)))
        self.assertEqual(len(seeds_10), 11)

        # Range '0-3'
        self.assertEqual(parse_seeds("0-3"), [0, 1, 2, 3])

        # Comma separated
        self.assertEqual(parse_seeds("0,2,5"), [0, 2, 5])

        # Single seed integer and string
        self.assertEqual(parse_seeds("7"), [7])
        self.assertEqual(parse_seeds(7), [7])
        self.assertEqual(parse_seeds(None), [0])

    def test_parse_stages(self):
        self.assertEqual(parse_stages("all"), [0, 1, 2, 3, 4])
        self.assertEqual(parse_stages(None), [0, 1, 2, 3, 4])
        self.assertEqual(parse_stages("0-2"), [0, 1, 2])
        self.assertEqual(parse_stages("2,4"), [2, 4])
        self.assertEqual(parse_stages("3"), [3])


if __name__ == "__main__":
    unittest.main()
