#!/usr/bin/env python3
"""
Rebuilds results/eval/eval_summary.json from the raw per-episode parquet
(results/eval_episodes.parquet), replicating the exact aggregation logic of
src/py/eval.py:evaluate_checkpoint().

Use case: eval_summary.json is overwritten on each eval invocation; this script
recovers the full multi-run canonical summary from the durable episode-level data.

Canonical run mapping (per algorithm) is configurable below; only episodes
belonging to the canonical runs are included.
"""

import json
import os

import polars as pl

from constants import CURRICULUM_STAGES

# Canonical evaluation runs per algorithm (leave-one-out mapping of the full
# 10-seed x 5-stage x 1000-episode benchmark sweep).
CANONICAL_RUNS = {
    "thief": "G20x",
    "marc": "G20x",
    "coma": "G20x",
    "mappo": "FbRL",
    "hmappo": "FbRL",
    "coop": "FbRL",
    "ecoop": "FbRL",
    "roma": "9xlf",
    "rode": "9xlf",
}


def main():
    parquet_path = "results/eval_episodes.parquet"
    if not os.path.exists(parquet_path):
        raise SystemExit(f"Missing {parquet_path}")

    df = pl.read_parquet(parquet_path)

    mask = pl.lit(False)
    ors = None
    for algo, run in CANONICAL_RUNS.items():
        cond = (pl.col("algo") == algo) & (pl.col("run_id") == run)
        ors = cond if ors is None else (ors | cond)
    mask = ors
    df = df.filter(mask)
    print(f"Filtered to canonical runs: {df.height} episodes")

    stage_ids = sorted(df.select("stage").unique().to_series().to_list())
    out = []
    for (algo, run), g in df.partition_by(["algo", "run_id"], as_dict=True).items():
        pass

    grouped = df.group_by(["run_id", "algo", "stage", "train_seed"]).agg(
        pl.len().alias("episodes_evaluated"),
        pl.col("win").mean().alias("win_rate"),
        pl.col("win").std().alias("win_rate_std"),
        pl.col("return").mean().alias("mean_return"),
        pl.col("return").std().alias("mean_return_std"),
        pl.col("steps").mean().alias("avg_steps"),
        pl.col("final_alarm").mean().alias("avg_alarm"),
        pl.col("stealth_index").mean().alias("stealth_index"),
        pl.col("ghost_run").mean().alias("ghost_run_rate"),
        pl.col("full_squad_extracted").mean().alias("full_squad_extract_rate"),
        pl.col("scout_interact").mean().alias("scout_tag_rate"),
        pl.col("scout_pois_tagged").mean().alias("scout_avg_pois"),
        pl.col("hacker_hack").mean().alias("hacker_hack_rate"),
        pl.col("muscle_neutralize").mean().alias("muscle_neutralize_rate"),
        pl.col("muscle_guards_neutralized").mean().alias("muscle_avg_guards"),
        pl.col("extractor_loot").mean().alias("extractor_loot_rate"),
        pl.col("agents_at_extract").mean().alias("avg_agents_extract"),
        pl.col("ablation").first().alias("ablation"),
    )

    rows = grouped.sort(["algo", "stage", "train_seed"]).to_dicts()
    for r in rows:
        stage = int(r["stage"])
        stage_config = CURRICULUM_STAGES[stage]
        # Guard against degenerate single-episode std (polars returns None)
        for k in ("win_rate_std", "mean_return_std"):
            if r[k] is None:
                r[k] = 0.0
        out.append(
            {
                "run_id": r["run_id"],
                "algo": r["algo"],
                "ablation": r["ablation"],
                "stage": stage,
                "train_seed": int(r["train_seed"]),
                "episodes_evaluated": int(r["episodes_evaluated"]),
                "evaluation_time_sec": 0.0,
                "episodes_per_sec": 0.0,
                "win_rate": float(r["win_rate"]),
                "win_rate_std": float(r["win_rate_std"]),
                "mean_return": float(r["mean_return"]),
                "mean_return_std": float(r["mean_return_std"]),
                "avg_steps": float(r["avg_steps"]),
                "max_stage_steps": stage_config.get("max_steps", 300),
                "avg_alarm": float(r["avg_alarm"]),
                "stage_alarm_max": stage_config.get("alarm_max", 100.0),
                "stealth_index": float(r["stealth_index"]),
                "ghost_run_rate": float(r["ghost_run_rate"]),
                "full_squad_extract_rate": float(r["full_squad_extract_rate"]),
                "scout_tag_rate": float(r["scout_tag_rate"]),
                "scout_avg_pois": float(r["scout_avg_pois"]),
                "hacker_hack_rate": float(r["hacker_hack_rate"]),
                "muscle_neutralize_rate": float(r["muscle_neutralize_rate"]),
                "muscle_avg_guards": float(r["muscle_avg_guards"]),
                "total_stage_guards": stage_config.get("guard_count", 0),
                "extractor_loot_rate": float(r["extractor_loot_rate"]),
                "avg_agents_extract": float(r["avg_agents_extract"]),
            }
        )

    os.makedirs("results/eval", exist_ok=True)
    with open("results/eval/eval_summary.json", "w") as f:
        json.dump(out, f, indent=4)
    print(f"Rebuilt results/eval/eval_summary.json with {len(out)} aggregate rows")
    algos = sorted({r["algo"] for r in out})
    print(f"Algorithms: {algos}")


if __name__ == "__main__":
    main()
