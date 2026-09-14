"""
Multi-Agent RL Curriculum Runner with Dynamic Concurrency and Unified Parquet Analytics.

Features:
- Priority-ordered algorithm training: thief -> ecoop -> mappo -> hmappo -> coop -> marc -> coma.
- Dynamic concurrency (--concurrent-algos K): pops the next algorithm from the queue as soon
  as any active algorithm finishes.
- Evaluation suite trigger (--eval) with optimal sample-efficient evaluation (--eval-episodes 1000).
- Pure Python THIEF micro-ablation suite trigger (--ablations).
- Automatic LLM-ready master benchmark report generation (Markdown & JSON).
"""

import argparse
import importlib
import inspect
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from constants import CURRICULUM_STAGES
from dataset import (
    generate_master_summary_report,
    get_effective_run_id,
    load_stage_results,
)

PRIORITY_ALGO_ORDER = [
    "thief",
    "ecoop",
    "mappo",
    "hmappo",
    "coop",
    "marc",
    "coma",
]


def parse_stages(stages_arg):
    """Parses stage string like '0', '0-2', '0,1,3', or 'all' into a list of stage integers."""
    if stages_arg is None or str(stages_arg).lower() == "all":
        return list(range(len(CURRICULUM_STAGES)))

    stages_str = str(stages_arg).strip()
    if "-" in stages_str:
        parts = stages_str.split("-")
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))

    if "," in stages_str:
        return [int(x.strip()) for x in stages_str.split(",") if x.strip()]

    return [int(stages_str)]


def parse_seeds(seeds_arg):
    """Parses seed argument like '5', '0,1,2', or '0-10' (inclusive) into a list of seed integers."""
    if seeds_arg is None:
        return [0]

    if isinstance(seeds_arg, int):
        return [seeds_arg]

    seeds_str = str(seeds_arg).strip()
    if not seeds_str:
        return [0]

    if "," in seeds_str:
        return [int(x.strip()) for x in seeds_str.split(",") if x.strip()]

    if "-" in seeds_str:
        parts = seeds_str.split("-")
        return list(range(int(parts[0].strip()), int(parts[1].strip()) + 1))

    if seeds_str.isdigit():
        return [int(seeds_str)]

    return [0]


def parse_algos(algo_arg):
    """Parses algorithm argument into a priority-ordered list of algorithm strings."""
    if algo_arg is None or str(algo_arg).lower() == "all":
        return list(PRIORITY_ALGO_ORDER)

    algo_str = str(algo_arg).strip()
    if "," in algo_str:
        requested = [x.strip().lower() for x in algo_str.split(",") if x.strip()]
        # Sort according to canonical PRIORITY_ALGO_ORDER
        return sorted(
            requested,
            key=lambda a: PRIORITY_ALGO_ORDER.index(a) if a in PRIORITY_ALGO_ORDER else 99,
        )

    return [algo_str.lower()]


def aggregate_multiseed_results(algo_name, stages_to_run, seeds, run_id=None):
    """Aggregates per-seed stage results using Polars into a unified multi-seed statistical report."""
    import polars as pl

    summary = {}
    title_suffix = f" (Run: {run_id})" if run_id else ""
    print(f"\n{'=' * 85}")
    print(
        f"MULTI-SEED STATISTICAL SUMMARY: {algo_name.upper()} (Seeds: {seeds}){title_suffix}"
    )
    print(f"{'=' * 85}")
    print(
        f"{'Stage':<8} | {'Win Rate (Mean±Std)':<22} | {'Return (Mean±Std)':<20} | {'Stealth':<10} | {'Steps':<8} | {'Alarm':<8}"
    )
    print("-" * 85)

    base_prefix = f"results/{run_id}" if run_id else "results"

    # Try loading from global/local parquet first
    df_stages = load_stage_results(results_root="results", run_id=run_id, algo=algo_name)
    if df_stages.height == 0 and os.path.exists(f"{base_prefix}/stage_results.parquet"):
        try:
            df_stages = pl.read_parquet(f"{base_prefix}/stage_results.parquet")
            if "algo" in df_stages.columns:
                df_stages = df_stages.filter(pl.col("algo") == algo_name)
        except Exception:
            pass

    for stage_idx in stages_to_run:
        seed_win_rates = []
        seed_returns = []
        seed_steps = []
        seed_alarms = []
        seed_stealths = []
        seed_hacks = []
        seed_neuts = []
        seed_loots = []
        seed_extracts = []

        # If we have polars data for this stage and seeds
        if df_stages.height > 0 and "stage" in df_stages.columns:
            stage_rows = df_stages.filter(
                (pl.col("stage") == stage_idx) & (pl.col("seed").is_in(seeds))
            )
            if stage_rows.height > 0:
                seed_win_rates = (stage_rows["win_rate"] * 100.0).to_list()
                seed_returns = stage_rows["mean_return"].to_list()
                seed_steps = stage_rows["avg_steps"].to_list()
                seed_alarms = stage_rows["avg_alarm"].to_list()
                seed_stealths = (
                    (stage_rows["stealth_index"] * 100.0).to_list()
                    if "stealth_index" in stage_rows.columns
                    else []
                )
                if "hacker_hack_rate" in stage_rows.columns:
                    seed_hacks = (stage_rows["hacker_hack_rate"] * 100.0).to_list()
                if "muscle_neut_rate" in stage_rows.columns:
                    seed_neuts = (stage_rows["muscle_neut_rate"] * 100.0).to_list()
                if "extractor_loot_rate" in stage_rows.columns:
                    seed_loots = (stage_rows["extractor_loot_rate"] * 100.0).to_list()
                if "avg_agents_extract" in stage_rows.columns:
                    seed_extracts = stage_rows["avg_agents_extract"].to_list()

        # Fallback to json if polars didn't have this stage yet
        if not seed_win_rates:
            for s in seeds:
                seed_path = f"{base_prefix}/{algo_name}/seed_{s}/stage_{stage_idx}/results.json"
                flat_path = f"{base_prefix}/{algo_name}/stage_{stage_idx}/results.json"
                target_path = seed_path if os.path.exists(seed_path) else flat_path
                if os.path.exists(target_path):
                    with open(target_path) as f:
                        d = json.load(f)
                    seed_win_rates.append(d.get("win_rate", 0.0) * 100.0)
                    seed_returns.append(d.get("mean_reward", d.get("mean_return", 0.0)))
                    seed_steps.append(d.get("avg_episode_steps", d.get("avg_steps", 0.0)))
                    seed_alarms.append(d.get("avg_alarm", 0.0))
                    alarm_max = d.get("stage_alarm_max", 100.0)
                    stealth = d.get(
                        "stealth_index",
                        max(0.0, 1.0 - (d.get("avg_alarm", 0.0) / max(1.0, alarm_max))),
                    )
                    seed_stealths.append(stealth * 100.0)
                    seed_hacks.append(d.get("hacker_hack_rate", 0.0) * 100.0)
                    seed_neuts.append(
                        d.get("muscle_neutralize_rate", d.get("muscle_neut_rate", 0.0))
                        * 100.0
                    )
                    seed_loots.append(d.get("extractor_loot_rate", 0.0) * 100.0)
                    seed_extracts.append(
                        d.get("avg_agents_at_extract", d.get("avg_agents_extract", 0.0))
                    )

        if not seed_win_rates:
            continue

        mean_wr, std_wr = float(np.mean(seed_win_rates)), float(np.std(seed_win_rates))
        mean_ret, std_ret = float(np.mean(seed_returns)), float(np.std(seed_returns))
        mean_steps = float(np.mean(seed_steps))
        mean_alarm = float(np.mean(seed_alarms))
        mean_stealth = float(np.mean(seed_stealths)) if seed_stealths else 0.0

        wr_str = f"{mean_wr:.1f}% ± {std_wr:.1f}%"
        ret_str = f"{mean_ret:.2f} ± {std_ret:.2f}"
        stealth_str = f"{mean_stealth:.1f}%"
        steps_str = f"{mean_steps:.1f}"
        alarm_str = f"{mean_alarm:.1f}"

        print(
            f"Stage {stage_idx:<2} | {wr_str:<22} | {ret_str:<20} | {stealth_str:<10} | {steps_str:<8} | {alarm_str:<8}"
        )

        summary[f"stage_{stage_idx}"] = {
            "run_id": run_id,
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate_mean": mean_wr,
            "win_rate_std": std_wr,
            "mean_reward_mean": mean_ret,
            "mean_reward_std": std_ret,
            "avg_steps_mean": mean_steps,
            "avg_alarm_mean": mean_alarm,
            "stealth_index_mean": mean_stealth / 100.0,
            "hacker_hack_mean": float(np.mean(seed_hacks)) if seed_hacks else 0.0,
            "muscle_neut_mean": float(np.mean(seed_neuts)) if seed_neuts else 0.0,
            "extractor_loot_mean": float(np.mean(seed_loots)) if seed_loots else 0.0,
            "agents_extract_mean": float(np.mean(seed_extracts)) if seed_extracts else 0.0,
            "seeds_evaluated": seeds,
        }

    summary_file = f"{base_prefix}/{algo_name}/multiseed_summary.json"
    os.makedirs(f"{base_prefix}/{algo_name}", exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved aggregated multi-seed summary to {summary_file}\n")
    return summary


def run_single_seed(
    algo_name,
    seed,
    stages_to_run,
    total_timesteps_override=None,
    is_multi_seed=True,
    run_id=None,
    prev_run_id=None,
    use_rust=False,
    no_load_ckpt=False,
):
    """Executes the curriculum across stages for a single seed."""
    # Seed global RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dynamically import the requested trainer
    module_name = f"train_{algo_name}"
    try:
        trainer = importlib.import_module(module_name)
    except ImportError:
        print(f"[Seed {seed}] Error: Could not import src/{module_name}.py")
        return

    base_prefix = f"results/{run_id}" if run_id else "results"
    base_dir = f"{base_prefix}/{algo_name}/seed_{seed}"
    os.makedirs(base_dir, exist_ok=True)

    engine_tag = " [NATIVE RUST]" if use_rust else ""
    print(f"\n{'#' * 60}")
    print(
        f"### RUNNING ALGO: {algo_name.upper()}{engine_tag} | SEED: {seed} | STAGES: {stages_to_run}"
    )
    print(f"{'#' * 60}")

    # Check if a checkpoint from the preceding stage exists
    first_stage = stages_to_run[0]
    load_ckpt_path = None
    if first_stage > 0 and not no_load_ckpt:
        candidate_paths = []
        if prev_run_id:
            candidate_paths.extend(
                [
                    f"results/{prev_run_id}/{algo_name}/seed_{seed}/stage_{first_stage - 1}/model.pt",
                    f"results/{prev_run_id}/{algo_name}/stage_{first_stage - 1}/model.pt",
                ]
            )
        candidate_paths.extend(
            [
                os.path.join(base_dir, f"stage_{first_stage - 1}", "model.pt"),
                f"results/{algo_name}/seed_{seed}/stage_{first_stage - 1}/model.pt",
                f"results/{algo_name}/stage_{first_stage - 1}/model.pt",
            ]
        )
        for p in candidate_paths:
            if os.path.exists(p):
                load_ckpt_path = p
                print(
                    f"[Seed {seed}] Found preceding checkpoint for Stage {first_stage} at {load_ckpt_path}"
                )
                break

    for stage_idx in stages_to_run:
        if stage_idx < 0 or stage_idx >= len(CURRICULUM_STAGES):
            print(
                f"[Seed {seed}] Warning: Stage {stage_idx} out of bounds (0-{len(CURRICULUM_STAGES) - 1}). Skipping."
            )
            continue

        stage_config = CURRICULUM_STAGES[stage_idx]
        print(f"\n{'=' * 40}")
        print(
            f"[Seed {seed}] Executing Stage {stage_idx} | Area: {stage_config['map_size'][0]}x{stage_config['map_size'][1]}"
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
        if "use_rust" in sig.parameters:
            train_kwargs["use_rust"] = use_rust
        if "run_id" in sig.parameters:
            train_kwargs["run_id"] = run_id

        trainer.train(**train_kwargs)
        load_ckpt_path = os.path.join(save_ckpt_dir, "model.pt")


def run_curriculum(
    algo_name,
    stages_to_run=None,
    seeds=None,
    total_timesteps_override=None,
    parallel=False,
    run_id=None,
    prev_run_id=None,
    use_rust=False,
    no_load_ckpt=False,
):
    """Runs curriculum training across stages and seeds for a single algorithm."""
    effective_run_id = get_effective_run_id(run_id)

    if stages_to_run is None:
        stages_to_run = list(range(len(CURRICULUM_STAGES)))
    if seeds is None:
        seeds = [0]

    is_multi_seed = len(seeds) > 1
    run_in_parallel = parallel and is_multi_seed

    engine_tag = " [NATIVE RUST]" if use_rust else ""
    print(
        f"\nStarting Curriculum for {algo_name.upper()}{engine_tag} | Run ID: {effective_run_id} | Stages: {stages_to_run} | Seeds: {seeds} | Parallel: {run_in_parallel}"
    )

    if run_in_parallel:
        ctx = mp.get_context("spawn")
        processes = []
        for seed in seeds:
            p = ctx.Process(
                target=run_single_seed,
                args=(
                    algo_name,
                    seed,
                    stages_to_run,
                    total_timesteps_override,
                    is_multi_seed,
                    effective_run_id,
                    prev_run_id,
                    use_rust,
                    no_load_ckpt,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(
                    f"Warning: Seed process exited with non-zero exitcode {p.exitcode}"
                )
    else:
        for seed in seeds:
            run_single_seed(
                algo_name,
                seed,
                stages_to_run,
                total_timesteps_override,
                is_multi_seed,
                effective_run_id,
                prev_run_id,
                use_rust,
                no_load_ckpt,
            )

    if is_multi_seed:
        aggregate_multiseed_results(algo_name, stages_to_run, seeds, run_id=effective_run_id)


def run_concurrent_curriculum(
    algos_to_run: list[str],
    stages_to_run: list[int],
    seeds: list[int],
    total_timesteps_override: int | None = None,
    parallel_seeds: bool = False,
    run_id: str | None = None,
    prev_run_id: str | None = None,
    use_rust: bool = False,
    no_load_ckpt: bool = False,
    concurrent_algos: int = 1,
):
    """
    Executes training across a priority queue of algorithms with dynamic concurrency.
    When concurrent_algos > 1, maintains a pool of up to concurrent_algos processes.
    As soon as ANY active algorithm finishes, the next queued algorithm is popped and trained immediately!
    """
    if concurrent_algos <= 1 or len(algos_to_run) <= 1:
        for algo in algos_to_run:
            run_curriculum(
                algo_name=algo,
                stages_to_run=stages_to_run,
                seeds=seeds,
                total_timesteps_override=total_timesteps_override,
                parallel=parallel_seeds,
                run_id=run_id,
                prev_run_id=prev_run_id,
                use_rust=use_rust,
                no_load_ckpt=no_load_ckpt,
            )
        return

    ctx = mp.get_context("spawn")
    queue = list(algos_to_run)
    active_workers = {}  # {algo_name: Process}
    failed_algos = []

    print(f"\n{'=' * 85}")
    print(f"DYNAMIC CONCURRENT ALGORITHM RUNNER")
    print(f"Max Concurrent Algorithms: {concurrent_algos}")
    print(f"Priority Queue:            {queue}")
    print(f"{'=' * 85}\n")

    def launch_next_algo(algo_name):
        kwargs = {
            "algo_name": algo_name,
            "stages_to_run": stages_to_run,
            "seeds": seeds,
            "total_timesteps_override": total_timesteps_override,
            "parallel": parallel_seeds,
            "run_id": run_id,
            "prev_run_id": prev_run_id,
            "use_rust": use_rust,
            "no_load_ckpt": no_load_ckpt,
        }
        p = ctx.Process(target=run_curriculum, kwargs=kwargs)
        p.start()
        active_workers[algo_name] = p
        print(
            f"\n[DISPATCH] Started algorithm '{algo_name.upper()}' (PID: {p.pid}) | "
            f"Active algorithms: {list(active_workers.keys())} | "
            f"Remaining in queue: {len(queue)}"
        )

    # Fill active slots initially
    while queue and len(active_workers) < concurrent_algos:
        next_algo = queue.pop(0)
        launch_next_algo(next_algo)

    # Monitor and pop immediately upon any completion
    while active_workers:
        time.sleep(1.0)
        finished_algos = []
        for algo_name, p in list(active_workers.items()):
            if not p.is_alive():
                p.join()
                finished_algos.append(algo_name)
                if p.exitcode != 0:
                    print(
                        f"\n[ERROR] Algorithm '{algo_name.upper()}' exited with error code {p.exitcode}!"
                    )
                    failed_algos.append(algo_name)
                else:
                    print(
                        f"\n[SUCCESS] Algorithm '{algo_name.upper()}' finished training successfully!"
                    )

        for algo_name in finished_algos:
            del active_workers[algo_name]
            # Pop and launch next algorithm immediately
            if queue:
                next_algo = queue.pop(0)
                launch_next_algo(next_algo)

    if failed_algos:
        print(f"\n[ALERT] The following algorithms exited with errors: {failed_algos}")


def execute_evaluation(
    algos: list[str],
    stages: list[int],
    seeds: list[int],
    episodes: int = 1000,
    run_id: str | None = None,
    use_rust: bool = False,
):
    """Invokes the parallel evaluation suite across all trained algorithms."""
    print(f"\n{'=' * 85}")
    print(f"RUNNING PARALLEL EVALUATION SUITE")
    print(f"Algorithms: {algos}")
    print(f"Stages:     {stages}")
    print(f"Seeds:      {seeds}")
    print(f"Episodes:   {episodes}")
    print(f"{'=' * 85}\n")

    cmd = [
        sys.executable,
        "src/py/eval.py",
        "--algos",
        ",".join(algos),
        "--stages",
        ",".join(map(str, stages)),
        "--train-seeds",
        ",".join(map(str, seeds)),
        "--episodes",
        str(episodes),
    ]
    if run_id:
        cmd.extend(["--run-id", str(run_id)])
    if use_rust:
        cmd.append("--rust")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src/py"
    subprocess.run(cmd, check=True, env=env)


def execute_ablations(
    seeds: list[int],
    episodes: int = 1000,
    timesteps_override: int | None = None,
    use_rust: bool = False,
):
    """Invokes pure Python THIEF micro-ablation suite."""
    print(f"\n{'=' * 85}")
    print(f"RUNNING THIEF MICRO-ABLATION SUITE")
    print(f"Seeds:    {seeds}")
    print(f"Episodes: {episodes}")
    print(f"{'=' * 85}\n")

    cmd = [
        sys.executable,
        "src/py/ablation.py",
        "--seeds",
        ",".join(map(str, seeds)),
        "--episodes",
        str(episodes),
    ]
    if timesteps_override is not None:
        cmd.extend(["--timesteps", str(timesteps_override)])
    if use_rust:
        cmd.append("--rust")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src/py"
    subprocess.run(cmd, check=True, env=env)


def emit_llm_master_report(run_id: str | None = None):
    """Generates the comprehensive Markdown and JSON master benchmark reports for LLM digestion."""
    print(f"\n{'=' * 85}")
    print(f"GENERATING LLM-OPTIMIZED BENCHMARK MASTER REPORT")
    print(f"{'=' * 85}\n")

    effective_run_id = get_effective_run_id(run_id)
    summary_dict = generate_master_summary_report(
        results_root="results",
        run_id=effective_run_id,
        out_md="results/benchmark_master_summary.md",
        out_json="results/benchmark_master_summary.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent RL Curriculum Runner with Dynamic Concurrency"
    )
    parser.add_argument(
        "--algo",
        "--algos",
        dest="algo",
        type=str,
        default="all",
        help="Algorithm(s) to train across curriculum: 'all', single algo, or comma-separated list. "
        f"Priority order: {','.join(PRIORITY_ALGO_ORDER)}",
    )
    parser.add_argument(
        "--concurrent-algos",
        "--concurrent",
        "--concurrency",
        "--parallel-algos",
        dest="concurrent_algos",
        type=int,
        default=1,
        help="Number of algorithms to train simultaneously (default: 1). Pops next algorithm from priority queue as soon as any finishes.",
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
        default="0",
        help="Seed or list/range of seeds (e.g. '5', '0,1,2', or '0-10' inclusive). Default: '0'",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps per stage (useful for fast testing/smoke tests)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional experiment run subdirectory name under results/ (defaults to 4-char base62 timestamp hash)",
    )
    parser.add_argument(
        "--prev-run-id",
        type=str,
        default=None,
        help="Optional preceding run ID under results/ from which to load checkpoints for the initial stage",
    )
    parser.add_argument(
        "--no-load-ckpt",
        "--scratch",
        "--fresh",
        dest="no_load_ckpt",
        action="store_true",
        default=False,
        help="Do not load preceding stage checkpoints; train stage(s) completely from scratch",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run multiple seeds of the SAME algorithm simultaneously in parallel worker processes (default: False)",
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
        "--eval",
        action="store_true",
        default=False,
        help="Execute parallel evaluation suite across trained algorithms and seeds upon training completion",
    )
    parser.add_argument(
        "--eval-episodes",
        "--episodes",
        dest="eval_episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes per stage and seed (default: 1000, mathematically optimal for paper rigor)",
    )
    parser.add_argument(
        "--ablations",
        "--ablation",
        dest="ablations",
        action="store_true",
        default=False,
        help="Execute THIEF micro-ablation suite upon completion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_stages = (
        [args.stage] if args.stage is not None else parse_stages(args.stages)
    )
    selected_seeds = parse_seeds(args.seeds)
    selected_algos = parse_algos(args.algo)
    effective_run_id = get_effective_run_id(args.run_id)

    print(f"\n{'=' * 85}")
    print(f"MARL HEIST CURRICULUM PIPELINE")
    print(f"Run ID:                  {effective_run_id}")
    print(f"Algorithms:              {selected_algos}")
    print(f"Stages:                  {selected_stages}")
    print(f"Seeds:                   {selected_seeds}")
    print(f"Algorithm Concurrency:   {args.concurrent_algos}")
    print(f"Per-Seed Parallelism:    {args.parallel}")
    print(f"Post-Train Eval:         {args.eval} (Episodes: {args.eval_episodes})")
    print(f"Micro-Ablations:         {args.ablations}")
    print(f"Native Rust:             {args.rust}")
    print(f"{'=' * 85}\n")

    # 1. Training Phase (Dynamic Concurrency)
    run_concurrent_curriculum(
        algos_to_run=selected_algos,
        stages_to_run=selected_stages,
        seeds=selected_seeds,
        total_timesteps_override=args.timesteps,
        parallel_seeds=args.parallel,
        run_id=effective_run_id,
        prev_run_id=args.prev_run_id,
        use_rust=args.rust,
        no_load_ckpt=args.no_load_ckpt,
        concurrent_algos=args.concurrent_algos,
    )

    # 2. Evaluation Phase
    if args.eval:
        execute_evaluation(
            algos=selected_algos,
            stages=selected_stages,
            seeds=selected_seeds,
            episodes=args.eval_episodes,
            run_id=effective_run_id,
            use_rust=args.rust,
        )

    # 3. Ablation Phase
    if args.ablations:
        execute_ablations(
            seeds=selected_seeds,
            episodes=args.eval_episodes,
            timesteps_override=args.timesteps,
            use_rust=args.rust,
        )

    # 4. Generate Master Benchmark Report (Markdown & JSON)
    emit_llm_master_report(run_id=effective_run_id)


if __name__ == "__main__":
    main()
