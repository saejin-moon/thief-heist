"""Resume high-throughput evaluation runner for FbRL checkpoints."""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import polars as pl
import json
import numpy as np

# Ensure src/py is on path
sys.path.insert(0, "src/py")

from eval import evaluate_checkpoint
from constants import CURRICULUM_STAGES
from dataset import get_effective_run_id

def worker_task(worker_id: int, tasks: list[tuple[str, int, int]]):
    """
    Evaluates a designated list of (algo, stage, seed) checkpoints.
    Each checkpoint is evaluated over 1000 episodes with 64 parallel environments in Rust.
    """
    print(f"[Worker {worker_id}] Starting {len(tasks)} checkpoints...", flush=True)
    results = []
    t_start = time.time()
    for idx, (algo, stage, seed) in enumerate(tasks):
        ckpt_path = f"results/FbRL/{algo}/seed_{seed}/stage_{stage}/model.pt"
        if not os.path.exists(ckpt_path):
            print(f"[Worker {worker_id}] Missing checkpoint {ckpt_path}, skipping!", flush=True)
            continue
        
        t0 = time.time()
        print(f"[Worker {worker_id}][{idx+1}/{len(tasks)}] Evaluating {algo} s{seed} stage_{stage}...", flush=True)
        res = evaluate_checkpoint(
            algo_name=algo,
            stage_idx=stage,
            ckpt_path=ckpt_path,
            target_episodes=1000,
            num_envs=64,
            base_seed=10000 + seed * 1000 + stage * 100,
            use_rust=True,
            train_seed=seed,
            run_id="FbRL",
            save_parquet=True,
        )
        elapsed = time.time() - t0
        print(f"[Worker {worker_id}][{idx+1}/{len(tasks)}] {algo} s{seed} stage_{stage} -> WR: {res['win_rate']*100:.1f}% | Ret: {res['mean_return']:.2f} | Time: {elapsed:.1f}s", flush=True)
        results.append(res)

    total_time = time.time() - t_start
    print(f"[Worker {worker_id}] FINISHED all {len(tasks)} checkpoints in {total_time/60:.1f} min!", flush=True)
    return worker_id, results


def get_missing_tasks():
    # Find existing checkpoints in eval_episodes.parquet with >= 1000 episodes
    df = pl.read_parquet("results/eval_episodes.parquet").filter(pl.col("run_id") == "FbRL")
    existing = set()
    for row in df.group_by(["algo", "stage", "train_seed"]).len().filter(pl.col("len") >= 1000).iter_rows():
        existing.add((row[0], row[1], row[2]))

    algos = ["ecoop", "mappo", "hmappo", "coop"]
    tasks_by_seed = {s: [] for s in range(10)}
    for s in range(10):
        for stage in range(5):
            for algo in algos:
                if (algo, stage, s) not in existing:
                    tasks_by_seed[s].append((algo, stage, s))

    # Balanced 5-worker partition:
    # Worker 0: Seed 1 (20) + Seed 0 (1) = 21
    # Worker 1: Seed 3 (20) + Seed 2 (1) = 21
    # Worker 2: Seed 5 (20) + Seed 4 (1) = 21
    # Worker 3: Seed 7 (20) + Seed 6 (1) = 21
    # Worker 4: Seed 9 (20) + Seed 8 (1) = 21
    worker_queues = [
        tasks_by_seed[1] + tasks_by_seed[0],
        tasks_by_seed[3] + tasks_by_seed[2],
        tasks_by_seed[5] + tasks_by_seed[4],
        tasks_by_seed[7] + tasks_by_seed[6],
        tasks_by_seed[9] + tasks_by_seed[8],
    ]
    return worker_queues


def aggregate_all_benchmark_results():
    """
    Builds the authoritative eval_summary.json by aggregating all 350,000 episodes
    from eval_episodes.parquet (150,000 from G20x + 200,000 from FbRL).
    """
    print("\n" + "=" * 80)
    print("AGGREGATING COMPLETE 350-CHECKPOINT BENCHMARK EVALUATION METRICS")
    print("=" * 80)
    df = pl.read_parquet("results/eval_episodes.parquet")
    
    # Target configurations:
    # G20x: thief, marc, coma (seeds 0-9, stages 0-4)
    # FbRL: ecoop, mappo, hmappo, coop (seeds 0-9, stages 0-4)
    target_specs = [
        ("G20x", "thief"), ("G20x", "marc"), ("G20x", "coma"),
        ("FbRL", "ecoop"), ("FbRL", "mappo"), ("FbRL", "hmappo"), ("FbRL", "coop"),
    ]

    all_summary = []
    for run_id, algo in target_specs:
        for seed in range(10):
            for stage in range(5):
                sub = df.filter(
                    (pl.col("run_id") == run_id) &
                    (pl.col("algo") == algo) &
                    (pl.col("train_seed") == seed) &
                    (pl.col("stage") == stage)
                )
                if len(sub) == 0:
                    print(f"[WARN] Missing evaluation episodes for {run_id} {algo} seed {seed} stage {stage}")
                    continue
                
                # Take exactly the first 1000 episodes (or all if <= 1000)
                sub = sub.sort("episode_idx").head(1000)
                stage_cfg = CURRICULUM_STAGES[stage]
                rec = {
                    "run_id": run_id,
                    "algo": algo,
                    "ablation": "none",
                    "stage": stage,
                    "train_seed": seed,
                    "episodes_evaluated": len(sub),
                    "evaluation_time_sec": 0.0,
                    "episodes_per_sec": 0.0,
                    "win_rate": float(sub["win"].mean()),
                    "win_rate_std": float(sub["win"].std()),
                    "mean_return": float(sub["return"].mean()),
                    "mean_return_std": float(sub["return"].std()),
                    "avg_steps": float(sub["steps"].mean()),
                    "max_stage_steps": stage_cfg.get("max_steps", 300),
                    "avg_alarm": float(sub["final_alarm"].mean()),
                    "stage_alarm_max": stage_cfg.get("alarm_max", 100.0),
                    "stealth_index": float(sub["stealth_index"].mean()),
                    "ghost_run_rate": float(sub["ghost_run"].mean()),
                    "full_squad_extract_rate": float(sub["full_squad_extracted"].mean()),
                    "scout_tag_rate": float(sub["scout_interact"].mean()),
                    "scout_avg_pois": float(sub["scout_pois_tagged"].mean()),
                    "hacker_hack_rate": float(sub["hacker_hack"].mean()),
                    "muscle_neutralize_rate": float(sub["muscle_neutralize"].mean()),
                    "muscle_avg_guards": float(sub["muscle_guards_neutralized"].mean()),
                    "total_stage_guards": stage_cfg.get("guard_count", 0),
                    "extractor_loot_rate": float(sub["extractor_loot"].mean()),
                    "avg_agents_extract": float(sub["agents_at_extract"].mean()),
                }
                all_summary.append(rec)

    print(f"Total aggregated benchmark checkpoints: {len(all_summary)} / 350")
    summary_path = "results/eval/eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=4)
    print(f"Successfully wrote unified benchmark summary to {summary_path}!")


def main():
    worker_queues = get_missing_tasks()
    total_tasks = sum(len(q) for q in worker_queues)
    print(f"Resuming FbRL parallel evaluation: {total_tasks} checkpoints across 5 workers")

    if total_tasks > 0:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(worker_task, i, q)
                for i, q in enumerate(worker_queues)
                if len(q) > 0
            ]
            for f in as_completed(futures):
                w_id, res = f.result()
                print(f"[MAIN] Worker {w_id} successfully returned {len(res)} evaluated checkpoints.")
        print(f"\nAll remaining evaluations completed in {(time.time() - t0)/60:.1f} minutes!")

    # Aggregate full benchmark records across G20x and FbRL
    aggregate_all_benchmark_results()


if __name__ == "__main__":
    main()
