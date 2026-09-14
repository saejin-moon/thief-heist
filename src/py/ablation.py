"""
THIEF Component Micro-Ablation Suite.

Trains each THIEF ablation variant STANDALONE (fresh initialization, no curriculum
checkpoint transfer) on diagnostic stages (default: Stage 2 and Stage 4), evaluates
the resulting checkpoints with the parallel evaluation suite, and compiles the LaTeX
ablation table into paper/tables/ablation.tex.

Variants:
  - none:           Full THIEF (control)
  - no_incubation:  w/o warmup (tau_iso = 0, child joins pool immediately)
  - uniform_recomb: w/o weighted recombination (uniform parameter blend)
  - clone_best:     clone-best-parent (child = best parent + noise)
  - no_balance:     w/o load balancing (alpha_bal = 0)
  - no_hysteresis:  w/o hysteresis (epsilon = 0)
  - fixed_schedule: fixed-schedule spawning (plateau trigger bypassed)
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from constants import CURRICULUM_STAGES
from dataset import get_effective_run_id

ABLATION_VARIANTS = [
    "none",
    "no_incubation",
    "uniform_recomb",
    "clone_best",
    "no_balance",
    "no_hysteresis",
    "fixed_schedule",
]

VARIANT_META = {
    "none":           ("Full THIEF", "nothing ablated"),
    "no_incubation":  ("w/o warmup", r"$\tau_{\mathrm{iso}}=0$ (child joins pool immediately)"),
    "uniform_recomb": ("w/o weighted recombination", "uniform recombination"),
    "clone_best":     ("clone-best-parent", r"child $=$ best parent $+$ noise"),
    "no_balance":     (r"w/o load balancing", r"$\alpha_{\mathrm{bal}}=0$"),
    "no_hysteresis":  ("w/o hysteresis", r"$\epsilon=0$"),
    "fixed_schedule": ("fixed-schedule spawning", "plateau trigger bypassed"),
}

STAGE_LABELS = {
    0: r"Stage 0 ($10^2$)",
    1: r"Stage 1 ($15^2$)",
    2: r"Stage 2 ($25^2$)",
    3: r"Stage 3 ($35^2$)",
    4: r"Stage 4 ($50^2$)",
}

STAGE_BUDGETS = {
    0: 200_000,
    1: 200_000,
    2: 500_000,
    3: 900_000,
    4: 1_600_000,
}


def parse_ablation_seeds(seeds_arg):
    """Parse seed argument (e.g. '0,1,2', '0-2', '5') into a list of seed integers."""
    if seeds_arg is None:
        return [0, 1, 2]
    if isinstance(seeds_arg, int):
        return [seeds_arg]
    s = str(seeds_arg).strip()
    if not s:
        return [0, 1, 2]
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if "-" in s:
        parts = s.split("-")
        return list(range(int(parts[0].strip()), int(parts[1].strip()) + 1))
    if s.isdigit():
        return [int(s)]
    return [0, 1, 2]


def parse_ablation_stages(stages_arg):
    """Parse stages argument (e.g. '2,4', '2-4', 'all') into a list of stage integers."""
    if stages_arg is None:
        return [2, 4]
    s = str(stages_arg).strip()
    if s.lower() == "all":
        return list(range(len(CURRICULUM_STAGES)))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if "-" in s:
        parts = s.split("-")
        return list(range(int(parts[0].strip()), int(parts[1].strip()) + 1))
    if s.isdigit():
        return [int(s)]
    return [2, 4]


def parse_ablation_variants(variants_arg):
    """Parse variants argument into list of valid variant strings."""
    if variants_arg is None or variants_arg.lower() == "all":
        return list(ABLATION_VARIANTS)
    items = [x.strip() for x in str(variants_arg).split(",") if x.strip()]
    valid = [v for v in items if v in ABLATION_VARIANTS]
    if not valid:
        print(f"Warning: No valid variants found in '{variants_arg}', defaulting to all.")
        return list(ABLATION_VARIANTS)
    return valid


def train_ablation_variant(
    variant: str,
    stage_idx: int,
    seed: int,
    total_timesteps: int,
    use_rust: bool = False,
    force: bool = False,
    num_envs: int = 16,
):
    """Trains a single ablation variant standalone on a specific stage and seed."""
    from train_thief import train

    save_dir = f"results/ablation/{variant}/thief/seed_{seed}/stage_{stage_idx}"
    ckpt_path = os.path.join(save_dir, "model.pt")

    if os.path.exists(ckpt_path) and not force:
        print(f"[SKIP] {variant} | stage {stage_idx} | seed {seed} (checkpoint exists: {ckpt_path})")
        return ckpt_path

    print(f"\n[TRAIN] Variant: {variant} | Stage: {stage_idx} | Seed: {seed} | Budget: {total_timesteps:,} steps")
    stage_config = CURRICULUM_STAGES[stage_idx].copy()
    env_config = stage_config.copy()
    env_config.pop("timesteps", None)

    os.makedirs(save_dir, exist_ok=True)

    train(
        algo_name="thief",
        stage_idx=stage_idx,
        env_config=env_config,
        total_timesteps=total_timesteps,
        load_ckpt_path=None,  # Standalone training: fresh initialization, no curriculum transfer
        save_ckpt_dir=save_dir,
        log_dir=save_dir,
        seed=seed,
        use_rust=use_rust,
        ablation=variant,
        run_id=f"ablation/{variant}",
    )
    return ckpt_path


def evaluate_ablation_variant(
    variant: str,
    stages: list[int],
    seeds: list[int],
    episodes: int = 1000,
    num_envs: int = 16,
    use_rust: bool = False,
):
    """Evaluates checkpoints for a given ablation variant across seeds and stages."""
    from eval import evaluate_checkpoint

    print(f"\n{'=' * 75}")
    print(f"EVALUATING ABLATION: {variant} (Seeds: {seeds}, Stages: {stages}, Episodes: {episodes})")
    print(f"{'=' * 75}")

    all_eval_results = []
    for seed in seeds:
        for stage_idx in stages:
            ckpt_path = f"results/ablation/{variant}/thief/seed_{seed}/stage_{stage_idx}/model.pt"
            if not os.path.exists(ckpt_path):
                print(f"[WARN] Checkpoint not found: {ckpt_path}, skipping evaluation.")
                continue

            print(f"Evaluating {variant} | seed {seed} | stage {stage_idx} ({episodes} episodes)...", end="\r", flush=True)
            try:
                res = evaluate_checkpoint(
                    algo_name="thief",
                    stage_idx=stage_idx,
                    ckpt_path=ckpt_path,
                    target_episodes=episodes,
                    num_envs=num_envs,
                    base_seed=10000 + seed * 1000 + stage_idx * 100,
                    use_rust=use_rust,
                    train_seed=seed,
                    run_id=f"ablation/{variant}",
                    save_parquet=True,
                )
                res["variant"] = variant
                res["train_seed"] = seed
                all_eval_results.append(res)
                print(
                    f"Eval {variant} | seed {seed} | stage {stage_idx} -> "
                    f"Win: {res['win_rate']*100:.1f}% | Ret: {res['mean_return']:.2f} | Time: {res['evaluation_time_sec']}s"
                )
            except Exception as e:
                print(f"\n[ERROR] Evaluating {variant} seed {seed} stage {stage_idx}: {e}")

    # Save variant evaluation JSON
    eval_dir = "results/eval"
    os.makedirs(eval_dir, exist_ok=True)
    out_json = f"{eval_dir}/ablation_{variant}_eval_n{episodes}.json"
    with open(out_json, "w") as f:
        json.dump(all_eval_results, f, indent=4)
    print(f"Saved evaluation metrics to {out_json}")
    return all_eval_results


def load_mechanism_metrics(variant: str):
    """Aggregate mechanism telemetry from per-run results.json files."""
    rows = sorted(glob.glob(f"results/ablation/{variant}/thief/seed_*/stage_*/results.json"))
    spawns, switches, entropies, actives = [], [], [], []
    for path in rows:
        try:
            with open(path) as f:
                r = json.load(f)
            spawns.append(r.get("total_spawns", 0))
            actives.append(r.get("active_experts", 1))
            ent = r.get("routing_entropy_nats", r.get("routing_entropy"))
            if ent is not None and not np.isnan(ent):
                entropies.append(ent)
            sw = r.get("expert_switch_rate")
            if sw is not None and not np.isnan(sw):
                switches.append(sw)
        except Exception:
            continue

    return {
        "spawns": float(np.mean(spawns)) if spawns else None,
        "switch_rate": float(np.mean(switches)) if switches else None,
        "entropy": float(np.mean(entropies)) if entropies else None,
        "active": float(np.mean(actives)) if actives else None,
    }


def generate_latex_ablation_table(
    episodes: int = 1000,
    table_path: str = "paper/tables/ablation.tex",
):
    """Compiles LaTeX micro-ablation table from evaluation and mechanism JSON files."""
    raw = {}
    for variant in ABLATION_VARIANTS:
        eval_path = f"results/eval/ablation_{variant}_eval_n{episodes}.json"
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    raw[variant] = json.load(f)
            except Exception:
                pass

    if not raw:
        print("[WARN] No ablation evaluation data found for LaTeX table generation.")
        return None

    stages = sorted({row["stage"] for data in raw.values() for row in data})
    agg = defaultdict(dict)
    for variant, data in raw.items():
        by_stage = defaultdict(list)
        for row in data:
            by_stage[row["stage"]].append(row["win_rate"] * 100.0)
        for stage, vals in by_stage.items():
            agg[variant][stage] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            )

    best = {}
    for stage in stages:
        candidates = [agg[v][stage][0] for v in raw if stage in agg[v]]
        best[stage] = max(candidates) if candidates else None

    def wr_cell(mean, std, best_mean):
        if mean is None:
            return "--"
        text = f"{mean:.1f}\\tiny$\\pm${std:.1f}\\%"
        if best_mean is not None and abs(mean - best_mean) < 0.05:
            text = f"\\textbf{{{text}}}"
        return text

    lines = [
        "% Auto-generated by src/py/ablation.py. Do not edit manually.",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Component micro-ablations of THIEF, trained standalone (fresh initialization, no curriculum transfer) on diagnostic stages with identical per-stage budgets, evaluated over $N={episodes:,}$ held-out episodes per seed. Mechanism telemetry is averaged across seeds and stages. Bold denotes best held-out win rate per stage.}}",
        "\\label{tab:ablations}",
        "\\vspace{0.05in}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l l " + "c" * len(stages) + " ccc}",
        "\\toprule",
        f" & & \\multicolumn{{{len(stages)}}}{{c}}{{\\textbf{{Held-out win rate (\\%)}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Mechanism telemetry}}}} \\\\",
        f"\\cmidrule(lr){{3-{2 + len(stages)}}} \\cmidrule(lr){{{3 + len(stages)}-{5 + len(stages)}}}",
        "\\textbf{Variant} & \\textbf{Ablated component} & "
        + " & ".join(STAGE_LABELS.get(s, f"Stage {s}") for s in stages)
        + " & Spawns & Switch (\\%) & $H(f)$ (nats) \\\\",
        "\\midrule",
    ]

    for variant in ABLATION_VARIANTS:
        if variant not in raw:
            continue
        name, desc = VARIANT_META.get(variant, (variant, ""))
        cells = [wr_cell(*agg[variant].get(s, (None, None)), best.get(s)) for s in stages]

        mech = load_mechanism_metrics(variant)
        sp = f"{mech['spawns']:.1f}" if mech["spawns"] is not None else "--"
        sw = f"{mech['switch_rate']:.1f}" if mech["switch_rate"] is not None else "--"
        en = f"{mech['entropy']:.2f}" if mech["entropy"] is not None else "--"

        lines.append(f"{name} & {desc} & " + " & ".join(cells) + f" & {sp} & {sw} & {en} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])

    os.makedirs(os.path.dirname(os.path.abspath(table_path)), exist_ok=True)
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote LaTeX ablation table to {table_path}")
    return table_path


def run_ablation_suite(
    variants: list[str] | None = None,
    stages: list[int] | None = None,
    seeds: list[int] | None = None,
    episodes: int = 1000,
    timesteps_override: int | None = None,
    use_rust: bool = False,
    force: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    num_envs: int = 16,
):
    """Master runner for the ablation suite."""
    if variants is None:
        variants = list(ABLATION_VARIANTS)
    if stages is None:
        stages = [2, 4]
    if seeds is None:
        seeds = [0, 1, 2]

    print(f"\n{'=' * 80}")
    print(f"THIEF MICRO-ABLATION SUITE")
    print(f"Variants: {variants}")
    print(f"Stages:   {stages}")
    print(f"Seeds:    {seeds}")
    print(f"Episodes: {episodes}")
    print(f"{'=' * 80}\n")

    # 1. Training Phase
    if not skip_train:
        for variant in variants:
            for stage_idx in stages:
                budget = timesteps_override if timesteps_override is not None else STAGE_BUDGETS.get(stage_idx, 500_000)
                for seed in seeds:
                    train_ablation_variant(
                        variant=variant,
                        stage_idx=stage_idx,
                        seed=seed,
                        total_timesteps=budget,
                        use_rust=use_rust,
                        force=force,
                        num_envs=num_envs,
                    )

    # 2. Evaluation Phase
    if not skip_eval:
        for variant in variants:
            evaluate_ablation_variant(
                variant=variant,
                stages=stages,
                seeds=seeds,
                episodes=episodes,
                num_envs=num_envs,
                use_rust=use_rust,
            )

    # 3. Generate LaTeX Table
    generate_latex_ablation_table(episodes=episodes)
    print(f"\n[DONE] THIEF Micro-ablation suite complete.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="THIEF Component Micro-Ablation Suite")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help=f"Comma-separated list of variants to run or 'all' (options: {','.join(ABLATION_VARIANTS)})",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="2,4",
        help="Diagnostic stages to run (e.g. '2,4', '2', or 'all'). Default: '2,4'",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Convenience alias to run a single stage (e.g. --stage 2)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Seeds or seed range to evaluate (e.g. '0,1,2' or '0-2'). Default: '0,1,2'",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Evaluation episodes per seed and stage (default: 1000)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps per stage (useful for fast smoke tests)",
    )
    parser.add_argument(
        "--rust",
        "--use-rust",
        dest="rust",
        action="store_true",
        default=False,
        help="Use high-throughput native Rust environment",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Retrain checkpoints even if they already exist",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        help="Skip training and only execute evaluation + table generation",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip evaluation and table generation",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_stages = [args.stage] if args.stage is not None else parse_ablation_stages(args.stages)
    selected_seeds = parse_ablation_seeds(args.seeds)
    selected_variants = parse_ablation_variants(args.variants)

    run_ablation_suite(
        variants=selected_variants,
        stages=selected_stages,
        seeds=selected_seeds,
        episodes=args.episodes,
        timesteps_override=args.timesteps,
        use_rust=args.rust,
        force=args.force,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        num_envs=args.num_envs,
    )
