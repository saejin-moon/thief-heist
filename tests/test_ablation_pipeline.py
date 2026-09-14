"""
Unit tests for the THIEF component micro-ablation suite and LaTeX table generator.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src/py"))

from ablation import (
    ABLATION_VARIANTS,
    generate_latex_ablation_table,
    parse_ablation_seeds,
    parse_ablation_stages,
    parse_ablation_variants,
)


class TestAblationPipeline(unittest.TestCase):
    def test_ablation_variants_contract(self):
        expected_variants = [
            "none",
            "no_incubation",
            "uniform_recomb",
            "clone_best",
            "no_balance",
            "no_hysteresis",
            "fixed_schedule",
        ]
        self.assertEqual(ABLATION_VARIANTS, expected_variants)
        self.assertEqual(len(ABLATION_VARIANTS), 7)

    def test_parse_ablation_variants(self):
        self.assertEqual(parse_ablation_variants("all"), ABLATION_VARIANTS)
        self.assertEqual(parse_ablation_variants(None), ABLATION_VARIANTS)
        self.assertEqual(
            parse_ablation_variants("none,clone_best"), ["none", "clone_best"]
        )

    def test_parse_ablation_seeds(self):
        self.assertEqual(parse_ablation_seeds("0-2"), [0, 1, 2])
        self.assertEqual(parse_ablation_seeds("0,1"), [0, 1])
        self.assertEqual(parse_ablation_seeds(5), [5])
        self.assertEqual(parse_ablation_seeds(None), [0, 1, 2])

    def test_parse_ablation_stages(self):
        self.assertEqual(parse_ablation_stages("2,4"), [2, 4])
        self.assertEqual(parse_ablation_stages("all"), [0, 1, 2, 3, 4])
        self.assertEqual(parse_ablation_stages("2"), [2])

    def test_latex_table_compilation(self):
        # Create mock evaluation results
        os.makedirs("results/eval", exist_ok=True)
        test_episodes = 9999
        mock_eval = [
            {"stage": 2, "win_rate": 0.85, "train_seed": 0},
            {"stage": 4, "win_rate": 0.72, "train_seed": 0},
        ]
        mock_file = f"results/eval/ablation_none_eval_n{test_episodes}.json"
        with open(mock_file, "w") as f:
            json.dump(mock_eval, f)

        out_tex = "paper/tables/test_ablation.tex"
        res = generate_latex_ablation_table(episodes=test_episodes, table_path=out_tex)
        self.assertTrue(os.path.exists(out_tex))

        with open(out_tex) as f:
            content = f.read()
        self.assertIn(r"\begin{table*}", content)
        self.assertIn("Full THIEF", content)
        self.assertIn(r"\end{table*}", content)

        # Cleanup
        if os.path.exists(mock_file):
            os.remove(mock_file)
        if os.path.exists(out_tex):
            os.remove(out_tex)


if __name__ == "__main__":
    unittest.main()
