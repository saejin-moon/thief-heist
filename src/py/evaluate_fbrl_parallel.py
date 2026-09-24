"""
Parallel evaluation runner for FbRL checkpoints.
Spawns 5 concurrent workers across seed pairs [0,1], [2,3], [4,5], [6,7], [8,9]
evaluating 4 algorithms (ecoop, mappo, hmappo, coop) across all 5 stages for 1000 episodes each.
Merges results with existing G20x evaluation records to produce the complete 350-checkpoint benchmark evaluation.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SEED_BATCHES = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

ALGOS = ["ecoop", "mappo", "hmappo", "coop"]


def run_batch(batch_idx: int, seeds: list[int]):
    seeds_str = ",".join(map(str, seeds))
    output_file = f"results/eval/eval_summary_batch_{batch_idx}.json"
    cmd = [
        sys.executable,
        "src/py/eval.py",
        "--run-id", "FbRL",
        "--algos", ",".join(ALGOS),
        "--stages", "all",
        "--train-seeds", seeds_str,
        "--episodes", "1000",
        "--rust",
        "--num-envs", "64",
        "--output", output_file,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src/py"
    print(f"[WORKER {batch_idx}] Starting evaluation for seeds {seeds_str} -> {output_file}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[WORKER {batch_idx}] ERROR: exited with code {res.returncode} after {elapsed:.1f}s", flush=True)
        return batch_idx, False, output_file
    print(f"[WORKER {batch_idx}] SUCCESS in {elapsed/60:.1f} minutes", flush=True)
    return batch_idx, True, output_file


def main():
    os.makedirs("results/eval", exist_ok=True)
    print("=" * 80)
    print(f"PARALLEL EVALUATION FOR FbRL (5 Concurrent Batches | 10 Seeds | 4 Algorithms | 5 Stages)")
    print("=" * 80)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(run_batch, i, batch)
            for i, batch in enumerate(SEED_BATCHES)
        ]
        for f in as_completed(futures):
            idx, success, out = f.result()
            if not success:
                print(f"[FATAL] Batch {idx} failed!")

    print(f"\nAll worker batches finished in {(time.time() - t0)/60:.1f} minutes!")

    # Merge all batches with existing G20x results
    existing_file = "results/eval/eval_summary.json"
    g20x_records = []
    if os.path.exists(existing_file):
        with open(existing_file, "r") as f:
            try:
                data = json.load(f)
                g20x_records = [r for r in data if r.get("run_id") == "G20x"]
            except Exception as e:
                print(f"Warning loading existing eval_summary: {e}")

    fbrl_records = []
    for i in range(len(SEED_BATCHES)):
        batch_file = f"results/eval/eval_summary_batch_{i}.json"
        if os.path.exists(batch_file):
            with open(batch_file, "r") as f:
                try:
                    fbrl_records.extend(json.load(f))
                except Exception as e:
                    print(f"Warning reading {batch_file}: {e}")
            try:
                os.remove(batch_file)
            except OSError:
                pass

    total_records = g20x_records + fbrl_records
    print(f"\nMerged records summary:")
    print(f"  G20x (thief, marc, coma): {len(g20x_records)} records")
    print(f"  FbRL (ecoop, mappo, hmappo, coop): {len(fbrl_records)} records")
    print(f"  Total: {len(total_records)} records (Expected: 350)")

    with open(existing_file, "w") as f:
        json.dump(total_records, f, indent=4)
    print(f"Saved merged benchmark evaluation metrics to {existing_file}")


if __name__ == "__main__":
    main()
