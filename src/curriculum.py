import argparse
import importlib
import inspect
import json
import os
import random

import numpy as np
import torch

from constants import CURRICULUM_STAGES


def parse_stages(stages_arg):
    """Parses stage string like '0', '0-2', '0,1,3', or 'all' into a list of stage integers."""
    if stages_arg is None or stages_arg.lower() == "all":
        return list(range(len(CURRICULUM_STAGES)))

    if "-" in stages_arg:
        parts = stages_arg.split("-")
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))

    if "," in stages_arg:
        return [int(x.strip()) for x in stages_arg.split(",") if x.strip()]

    return [int(stages_arg)]


def parse_seeds(seeds_arg):
    """Parses seed argument like '0', '0,1,2', or '3' (count) into a list of seed integers."""
    if seeds_arg is None:
        return [0]

    if isinstance(seeds_arg, int):
        return list(range(seeds_arg))

    seeds_str = str(seeds_arg).strip()
    if "," in seeds_str:
        return [int(x.strip()) for x in seeds_str.split(",") if x.strip()]

    if seeds_str.isdigit():
        val = int(seeds_str)
        # If user passed a single digit count >= 2, treat as count [0..val-1], else single seed [val]
        if val > 1:
            return list(range(val))
        return [val]

    return [0]


def aggregate_multiseed_results(algo_name, stages_to_run, seeds):
    """Aggregates per-seed results.json files into a unified multi-seed statistical report."""
    summary = {}
    print(f"\n{'=' * 75}")
    print(f"MULTI-SEED STATISTICAL SUMMARY: {algo_name.upper()} (Seeds: {seeds})")
    print(f"{'=' * 75}")
    print(
        f"{'Stage':<8} | {'Win Rate (Mean±Std)':<22} | {'Return (Mean±Std)':<20} | {'Steps':<12} | {'Alarm':<10}"
    )
    print("-" * 75)

    for stage_idx in stages_to_run:
        seed_win_rates = []
        seed_returns = []
        seed_steps = []
        seed_alarms = []
        seed_hacks = []
        seed_neuts = []
        seed_loots = []
        seed_extracts = []

        for s in seeds:
            # Check seed-specific path first, then legacy flat path
            seed_path = f"results/{algo_name}/seed_{s}/stage_{stage_idx}/results.json"
            flat_path = f"results/{algo_name}/stage_{stage_idx}/results.json"
            target_path = seed_path if os.path.exists(seed_path) else flat_path

            if os.path.exists(target_path):
                with open(target_path) as f:
                    d = json.load(f)
                seed_win_rates.append(d.get("win_rate", 0.0) * 100.0)
                seed_returns.append(d.get("mean_reward", 0.0))
                seed_steps.append(d.get("avg_episode_steps", 0.0))
                seed_alarms.append(d.get("avg_alarm", 0.0))
                seed_hacks.append(d.get("hacker_hack_rate", 0.0) * 100.0)
                seed_neuts.append(d.get("muscle_neutralize_rate", 0.0) * 100.0)
                seed_loots.append(d.get("extractor_loot_rate", 0.0) * 100.0)
                seed_extracts.append(d.get("avg_agents_at_extract", 0.0))

        if not seed_win_rates:
            continue

        mean_wr, std_wr = float(np.mean(seed_win_rates)), float(np.std(seed_win_rates))
        mean_ret, std_ret = float(np.mean(seed_returns)), float(np.std(seed_returns))
        mean_steps = float(np.mean(seed_steps))
        mean_alarm = float(np.mean(seed_alarms))

        wr_str = f"{mean_wr:.1f}% ± {std_wr:.1f}%"
        ret_str = f"{mean_ret:.2f} ± {std_ret:.2f}"
        steps_str = f"{mean_steps:.1f}"
        alarm_str = f"{mean_alarm:.1f}"

        print(
            f"Stage {stage_idx:<2} | {wr_str:<22} | {ret_str:<20} | {steps_str:<12} | {alarm_str:<10}"
        )

        summary[f"stage_{stage_idx}"] = {
            "win_rate_mean": mean_wr,
            "win_rate_std": std_wr,
            "mean_reward_mean": mean_ret,
            "mean_reward_std": std_ret,
            "avg_steps_mean": mean_steps,
            "avg_alarm_mean": mean_alarm,
            "hacker_hack_mean": float(np.mean(seed_hacks)) if seed_hacks else 0.0,
            "muscle_neut_mean": float(np.mean(seed_neuts)) if seed_neuts else 0.0,
            "extractor_loot_mean": float(np.mean(seed_loots)) if seed_loots else 0.0,
            "agents_extract_mean": float(np.mean(seed_extracts))
            if seed_extracts
            else 0.0,
            "seeds_evaluated": seeds,
        }

    summary_file = f"results/{algo_name}/multiseed_summary.json"
    os.makedirs(f"results/{algo_name}", exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved aggregated multi-seed summary to {summary_file}\n")


def run_curriculum(
    algo_name, stages_to_run=None, seeds=None, total_timesteps_override=None
):
    if stages_to_run is None:
        stages_to_run = list(range(len(CURRICULUM_STAGES)))
    if seeds is None:
        seeds = [0]

    print(
        f"Starting Curriculum for {algo_name} | Stages: {stages_to_run} | Seeds: {seeds}"
    )

    # Dynamically import the requested trainer
    module_name = f"train_{algo_name}"
    try:
        trainer = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Could not import src/{module_name}.py")
        return

    is_multi_seed = len(seeds) > 1

    for seed in seeds:
        # Seed global RNGs
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        base_dir = (
            f"results/{algo_name}/seed_{seed}"
            if is_multi_seed
            else f"results/{algo_name}"
        )
        os.makedirs(base_dir, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(
            f"### RUNNING ALGO: {algo_name.upper()} | SEED: {seed} | STAGES: {stages_to_run}"
        )
        print(f"{'#' * 60}")

        # Automatically check if a checkpoint from the preceding stage exists
        first_stage = stages_to_run[0]
        load_ckpt_path = None
        if first_stage > 0:
            prev_ckpt = os.path.join(base_dir, f"stage_{first_stage - 1}", "model.pt")
            if os.path.exists(prev_ckpt):
                load_ckpt_path = prev_ckpt
                print(
                    f"Found preceding checkpoint for Stage {first_stage} at {load_ckpt_path}"
                )

        for stage_idx in stages_to_run:
            if stage_idx < 0 or stage_idx >= len(CURRICULUM_STAGES):
                print(
                    f"Warning: Stage {stage_idx} out of bounds (0-{len(CURRICULUM_STAGES) - 1}). Skipping."
                )
                continue

            stage_config = CURRICULUM_STAGES[stage_idx]
            print(f"\n{'=' * 40}")
            print(
                f"Executing Stage {stage_idx} (Seed {seed}) | Area: {stage_config['map_size'][0]}x{stage_config['map_size'][1]}"
            )
            print(f"{'=' * 40}")

            env_config = stage_config.copy()
            total_timesteps = (
                total_timesteps_override
                if total_timesteps_override is not None
                else env_config.pop("timesteps")
            )

            stage_dir = os.path.join(base_dir, f"stage_{stage_idx}")
            os.makedirs(stage_dir, exist_ok=True)
            save_ckpt_dir = stage_dir
            log_dir = stage_dir

            # Build kwargs dynamically based on trainer signature
            sig = inspect.signature(trainer.train)
            train_kwargs = {
                "algo_name": algo_name,
                "stage_idx": stage_idx,
                "env_config": env_config,
                "total_timesteps": total_timesteps,
                "load_ckpt_path": load_ckpt_path,
                "save_ckpt_dir": save_ckpt_dir,
                "log_dir": log_dir,
            }
            if "seed" in sig.parameters:
                train_kwargs["seed"] = seed

            # Call the train function
            trainer.train(**train_kwargs)

            # Update checkpoint for next stage in sequence
            load_ckpt_path = os.path.join(save_ckpt_dir, "model.pt")

    # If multi-seed run, generate statistical aggregate report
    if is_multi_seed:
        aggregate_multiseed_results(algo_name, stages_to_run, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent RL Curriculum Runner")
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["mappo", "coop", "marc", "hmappo", "ecoop", "coma", "thief", "all"],
        help="Algorithm to train across curriculum, or 'all' to run all benchmarks",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Stages to train, e.g. '0', '0-2', '0,1,3', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Convenience alias to run a single stage (e.g. --stage 0)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1",
        help="Number of random seeds (e.g. '3') or list of seeds (e.g. '0,1,2'). Default: '1' (seed 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Convenience alias to run a single specific seed (e.g. --seed 42)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps per stage (useful for fast testing/smoke tests)",
    )
    args = parser.parse_args()

    selected_stages = (
        [args.stage] if args.stage is not None else parse_stages(args.stages)
    )
    selected_seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)

    if args.algo == "all":
        for algo in ["thief", "ecoop", "mappo", "hmappo", "coop", "marc", "coma"]:
            run_curriculum(
                algo,
                stages_to_run=selected_stages,
                seeds=selected_seeds,
                total_timesteps_override=args.timesteps,
            )
    else:
        run_curriculum(
            args.algo,
            stages_to_run=selected_stages,
            seeds=selected_seeds,
            total_timesteps_override=args.timesteps,
        )
