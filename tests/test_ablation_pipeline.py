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
            "local_router",
        ]
        self.assertEqual(ABLATION_VARIANTS, expected_variants)
        self.assertEqual(len(ABLATION_VARIANTS), 8)

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

    def test_local_gating_network_contract(self):
        import torch
        from train_thief import LocalGatingNetwork, ThiefNetwork

        gate = LocalGatingNetwork(num_experts=1)
        self.assertEqual(gate.num_experts, 1)

        # 55-dim local input: 49 (7x7 obs) + 4 (role) + 2 (compass)
        x_local = torch.randn(8, 55)
        out = gate(x_local)
        self.assertEqual(out.shape, (8, 1))

        # Dynamic expert addition
        gate.add_expert()
        self.assertEqual(gate.num_experts, 2)
        out2 = gate(x_local)
        self.assertEqual(out2.shape, (8, 2))

        # ThiefNetwork with decentralized_eval
        state_dim = 2550
        net = ThiefNetwork(state_dim=state_dim, num_initial_experts=2)
        batch_size = 4
        obs = torch.randn(batch_size, 7, 7)
        role = torch.eye(4)[:batch_size]
        mask = torch.ones(batch_size, 6)
        goal = torch.randn(batch_size, 2)

        # Under decentralized_eval=True, state is literally None
        actions, logprobs, entropies, values, chosen_expert = (
            net.get_action_and_value(
                obs=obs,
                role=role,
                mask=mask,
                goal=goal,
                state=None,
                active_experts=2,
                decentralized_eval=True,
            )
        )
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(chosen_expert.shape, (batch_size,))
        self.assertTrue(torch.all((chosen_expert >= 0) & (chosen_expert < 2)))


if __name__ == "__main__":
    unittest.main()
