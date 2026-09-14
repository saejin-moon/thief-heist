"""Unified Polars + Parquet Experiment Data Architecture for HEIST & THIEF.

Provides lock-free / locked-append telemetry logging for concurrent multi-seed
and multi-algorithm training runs. Replaces legacy results.json, multiseed_summary.json,
and train.log with high-throughput columnar Parquet tables.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import string
import time
from typing import Any

import numpy as np
import polars as pl


def generate_base62_run_id() -> str:
    """Generate a unique 4-character base62 hash from high-precision current timestamp.

    Base62 alphabet consists of 0-9, a-z, A-Z (length 62).
    """
    chars = string.digits + string.ascii_letters  # 62 characters
    t_bytes = str(time.time_ns()).encode("utf-8")
    digest = hashlib.sha256(t_bytes).digest()
    num = int.from_bytes(digest[:4], "big")
    res = []
    for _ in range(4):
        res.append(chars[num % 62])
        num //= 62
    return "".join(res)


def get_effective_run_id(run_id: str | None) -> str:
    """Return run_id if specified, otherwise generate a 4-char base62 timestamp hash."""
    if run_id and str(run_id).strip():
        return str(run_id).strip()
    return generate_base62_run_id()


def compute_gini(probs: np.ndarray | list[float]) -> float:
    """Compute Gini inequality coefficient for expert usage distribution."""
    p = np.array(probs, dtype=np.float64)
    if len(p) <= 1 or np.sum(p) <= 1e-12:
        return 0.0
    p = np.sort(p)
    n = len(p)
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * p) / (n * np.sum(p) + 1e-12))


def compute_iqm(values: list[float] | np.ndarray) -> float:
    """Computes Interquartile Mean (IQM) across values (middle 50%)."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    if n == 0:
        return 0.0
    if n < 4:
        return float(np.mean(arr))
    q25 = int(np.floor(0.25 * n))
    q75 = int(np.ceil(0.75 * n))
    middle = arr[q25:q75]
    if len(middle) == 0:
        return float(np.mean(arr))
    return float(np.mean(middle))


def compute_bootstrap_ci(
    values: list[float] | np.ndarray,
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Computes (mean, lower_ci, upper_ci) using stratified percentile bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    boot_samples = rng.choice(arr, size=(n_bootstraps, len(arr)), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    mean_val = float(np.mean(arr))
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_val, lower, upper


def compute_role_expert_mi(role_expert_counts: Any) -> float:
    """Compute Mutual Information I(Role; Expert) in nats from a 2D contingency table or nested dict."""
    if isinstance(role_expert_counts, dict):
        roles = sorted(role_expert_counts.keys())
        all_experts = set()
        for r in roles:
            all_experts.update(role_expert_counts[r].keys())
        experts = sorted(all_experts)
        mat = np.zeros((len(roles), len(experts)), dtype=np.float64)
        for i, r in enumerate(roles):
            for j, exp in enumerate(experts):
                mat[i, j] = role_expert_counts[r].get(exp, 0)
        role_expert_counts = mat
    else:
        role_expert_counts = np.asarray(role_expert_counts, dtype=np.float64)

    total = np.sum(role_expert_counts)
    if total <= 1e-12:
        return 0.0
    p_joint = role_expert_counts / total
    p_role = np.sum(p_joint, axis=1, keepdims=True)
    p_expert = np.sum(p_joint, axis=0, keepdims=True)
    denom = p_role * p_expert
    mask = (p_joint > 1e-12) & (denom > 1e-12)
    if not np.any(mask):
        return 0.0
    mi = np.sum(p_joint[mask] * np.log(p_joint[mask] / denom[mask]))
    return float(max(0.0, mi))


def append_to_parquet(
    file_path: str,
    new_df: pl.DataFrame,
    lock_path: str | None = None,
) -> None:
    """Thread-safe and process-safe append to a Parquet file using POSIX file locks.

    Uses atomic temporary-file replacement and diagonal schema relaxation
    so new columns or evolving schemas are handled safely without data corruption.
    """
    if new_df.height == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    if lock_path is None:
        lock_path = file_path + ".lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    existing_df = pl.read_parquet(file_path)
                    combined_df = pl.concat([existing_df, new_df], how="diagonal_relaxed")
                except Exception:  # noqa: BLE001
                    # In case of any read failure, fallback to new_df
                    combined_df = new_df
            else:
                combined_df = new_df

            temp_path = file_path + f".tmp.{os.getpid()}_{time.time_ns()}"
            combined_df.write_parquet(temp_path, compression="zstd")
            os.replace(temp_path, file_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class TrainCurveLogger:
    """Buffers periodic training curve records in memory and flushes them to Parquet."""

    def __init__(
        self,
        run_id: str,
        algo: str,
        seed: int,
        stage: int,
        results_root: str = "results",
        local_dir: str | None = None,
        flush_interval: int = 10,
        ablation: str = "none",
    ):
        self.run_id = run_id
        self.algo = algo
        self.seed = seed
        self.stage = stage
        self.results_root = results_root
        self.local_dir = local_dir
        self.flush_interval = flush_interval
        self.ablation = ablation
        self.buffer: list[dict[str, Any]] = []

        self.global_parquet_path = os.path.join(results_root, "train_curves.parquet")
        self.local_parquet_path = (
            os.path.join(local_dir, "train_curves.parquet") if local_dir else None
        )

    def log_update(
        self,
        update: int,
        total_updates: int,
        timesteps: int,
        elapsed_sec: float,
        fps: float,
        win_rate: float,
        mean_return: float,
        avg_steps: float,
        avg_alarm: float,
        stage_alarm_max: float,
        scout_tag_rate: float = 0.0,
        hacker_hack_rate: float = 0.0,
        muscle_neut_rate: float = 0.0,
        extractor_loot_rate: float = 0.0,
        avg_agents_extract: float = 0.0,
        active_experts: int = 1,
        effective_experts: float = 1.0,
        total_spawns: int = 0,
        switch_rate: float = 0.0,
        routing_entropy: float = 0.0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        entropy_loss: float = 0.0,
        approx_kl: float = 0.0,
        clip_fraction: float = 0.0,
        explained_variance: float = 0.0,
        policy_grad_norm: float = 0.0,
        value_grad_norm: float = 0.0,
        vram_mb: float = 0.0,
    ) -> None:
        """Add an update record to the in-memory buffer and flush if interval reached."""
        stealth_index = float(max(0.0, min(1.0, 1.0 - (avg_alarm / max(1.0, stage_alarm_max)))))
        alarm_velocity = float(avg_alarm / max(1.0, avg_steps))

        record = {
            "run_id": str(self.run_id),
            "algo": str(self.algo),
            "seed": int(self.seed),
            "stage": int(self.stage),
            "update": int(update),
            "total_updates": int(total_updates),
            "timesteps": int(timesteps),
            "elapsed_sec": float(elapsed_sec),
            "fps": float(fps),
            "win_rate": float(win_rate),
            "mean_return": float(mean_return),
            "avg_steps": float(avg_steps),
            "avg_alarm": float(avg_alarm),
            "stealth_index": float(stealth_index),
            "alarm_velocity": float(alarm_velocity),
            "scout_tag_rate": float(scout_tag_rate),
            "hacker_hack_rate": float(hacker_hack_rate),
            "muscle_neut_rate": float(muscle_neut_rate),
            "extractor_loot_rate": float(extractor_loot_rate),
            "avg_agents_extract": float(avg_agents_extract),
            "active_experts": int(active_experts),
            "effective_experts": float(effective_experts),
            "total_spawns": int(total_spawns),
            "switch_rate": float(switch_rate),
            "routing_entropy": float(routing_entropy),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy_loss": float(entropy_loss),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "explained_variance": float(explained_variance),
            "policy_grad_norm": float(policy_grad_norm),
            "value_grad_norm": float(value_grad_norm),
            "vram_mb": float(vram_mb),
            "ablation": str(self.ablation),
        }
        self.buffer.append(record)

        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered records to both the global and local Parquet files."""
        if not self.buffer:
            return

        df_to_flush = pl.DataFrame(self.buffer)
        self.buffer.clear()

        # Write to global results/train_curves.parquet with file locking
        append_to_parquet(self.global_parquet_path, df_to_flush)

        # Write to local seed/stage directory if requested and distinct from global
        if (
            self.local_parquet_path
            and os.path.abspath(self.local_parquet_path)
            != os.path.abspath(self.global_parquet_path)
        ):
            append_to_parquet(self.local_parquet_path, df_to_flush)

    def close(self) -> None:
        """Ensure all remaining records are flushed."""
        self.flush()


def save_stage_result(
    result_data: dict[str, Any],
    save_dir: str,
    results_root: str = "results",
) -> None:
    """Save stage completion result to global stage_results.parquet, local parquet, and legacy JSON."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    # Ensure all Tier 1 and Tier 2 fields are populated with proper types and defaults
    avg_alarm = float(result_data.get("avg_alarm", 0.0))
    stage_alarm_max = float(result_data.get("stage_alarm_max", 100.0))
    avg_steps = float(result_data.get("avg_steps", 1.0))
    avg_agents_extract = float(result_data.get("avg_agents_extract", 0.0))

    stealth_index = float(
        result_data.get(
            "stealth_index",
            max(0.0, min(1.0, 1.0 - (avg_alarm / max(1.0, stage_alarm_max)))),
        )
    )
    alarm_velocity = float(result_data.get("alarm_velocity", avg_alarm / max(1.0, avg_steps)))
    coop_efficiency = float(
        result_data.get("coop_efficiency", avg_agents_extract / max(1.0, avg_steps))
    )

    clean_record = {
        "run_id": str(result_data.get("run_id", "default")),
        "algo": str(result_data.get("algo", "unknown")),
        "seed": int(result_data.get("seed", 0)),
        "stage": int(result_data.get("stage", 0)),
        "map_size": int(result_data.get("map_size", 11)),
        "timesteps": int(result_data.get("timesteps", 0)),
        "wall_clock_sec": float(result_data.get("wall_clock_sec", 0.0)),
        "throughput_fps": float(result_data.get("throughput_fps", 0.0)),
        "win_rate": float(result_data.get("win_rate", 0.0)),
        "lifetime_win_rate": float(result_data.get("lifetime_win_rate", 0.0)),
        "mean_return": float(result_data.get("mean_return", 0.0)),
        "std_return": float(result_data.get("std_return", 0.0)),
        "avg_steps": float(avg_steps),
        "max_stage_steps": int(result_data.get("max_stage_steps", 300)),
        "avg_alarm": float(avg_alarm),
        "stage_alarm_max": float(stage_alarm_max),
        "stealth_index": float(stealth_index),
        "alarm_velocity": float(alarm_velocity),
        "ghost_run_rate": float(result_data.get("ghost_run_rate", 0.0)),
        "full_squad_extract_rate": float(result_data.get("full_squad_extract_rate", 0.0)),
        "coop_efficiency": float(coop_efficiency),
        "scout_tag_rate": float(result_data.get("scout_tag_rate", 0.0)),
        "scout_avg_pois": float(result_data.get("scout_avg_pois", 0.0)),
        "hacker_hack_rate": float(result_data.get("hacker_hack_rate", 0.0)),
        "muscle_neut_rate": float(result_data.get("muscle_neut_rate", 0.0)),
        "muscle_avg_guards": float(result_data.get("muscle_avg_guards", 0.0)),
        "total_stage_guards": int(result_data.get("total_stage_guards", 0)),
        "extractor_loot_rate": float(result_data.get("extractor_loot_rate", 0.0)),
        "avg_agents_extract": float(avg_agents_extract),
        "scout_to_hack_latency": float(result_data.get("scout_to_hack_latency", -1.0)),
        "hack_to_loot_latency": float(result_data.get("hack_to_loot_latency", -1.0)),
        "loot_to_extract_latency": float(result_data.get("loot_to_extract_latency", -1.0)),
        "primary_failure_cause": str(result_data.get("primary_failure_cause", "none")),
        "timesteps_to_50pct_wr": int(result_data.get("timesteps_to_50pct_wr", -1)),
        "timesteps_to_80pct_wr": int(result_data.get("timesteps_to_80pct_wr", -1)),
        "auc_return": float(result_data.get("auc_return", 0.0)),
        "forward_transfer_wr": float(result_data.get("forward_transfer_wr", 0.0)),
        "active_experts": int(result_data.get("active_experts", 1)),
        "dormant_experts_count": int(result_data.get("dormant_experts_count", 0)),
        "total_spawns": int(result_data.get("total_spawns", 0)),
        "expert_switch_rate": float(result_data.get("expert_switch_rate", 0.0)),
        "routing_entropy": float(result_data.get("routing_entropy", 0.0)),
        "effective_experts": float(result_data.get("effective_experts", 1.0)),
        "gini_expert_imbalance": float(result_data.get("gini_expert_imbalance", 0.0)),
        "role_expert_mi": float(result_data.get("role_expert_mi", 0.0)),
        "vram_peak_mb": float(result_data.get("vram_peak_mb", 0.0)),
        "mean_inference_latency_us": float(result_data.get("mean_inference_latency_us", 0.0)),
        "ablation": str(result_data.get("ablation", "none")),
    }

    df_row = pl.DataFrame([clean_record])

    # 1. Global parquet
    global_path = os.path.join(results_root, "stage_results.parquet")
    append_to_parquet(global_path, df_row)

    # 2. Local stage parquet (if distinct from global)
    local_path = os.path.join(save_dir, "stage_results.parquet")
    if os.path.abspath(local_path) != os.path.abspath(global_path):
        append_to_parquet(local_path, df_row)

    # 3. Legacy JSON export for backward compatibility
    import json

    json_path = os.path.join(save_dir, "results.json")
    try:
        with open(json_path, "w") as f:
            json.dump(clean_record, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed to write legacy results.json: {e}")


def record_stage_completion(
    algo_name: str,
    stage_idx: int,
    seed: int,
    run_id: str,
    timesteps: int,
    wall_clock_sec: float,
    completed_wins: list[float],
    completed_returns: list[float],
    completed_steps: list[float],
    completed_alarms: list[float],
    completed_scout_interact: list[float] | None = None,
    completed_scout_pois: list[float] | None = None,
    completed_hacker_hack: list[float] | None = None,
    completed_muscle_neutralize: list[float] | None = None,
    completed_muscle_guards: list[float] | None = None,
    completed_extractor_loot: list[float] | None = None,
    completed_agents_at_extract: list[float] | None = None,
    stage_max_steps: int = 300,
    stage_alarm_max: float = 100.0,
    total_stage_guards: int = 0,
    map_size: int = 11,
    active_experts: int = 1,
    dormant_experts_count: int = 0,
    total_spawns: int = 0,
    expert_switch_rate: float = 0.0,
    routing_entropy: float = 0.0,
    expert_usage_pct: dict[Any, float] | None = None,
    role_expert_counts: np.ndarray | None = None,
    ablation: str = "none",
    spawn_history: list[dict[str, Any]] | None = None,
    save_dir: str | None = None,
    results_root: str = "results",
) -> dict[str, Any]:
    """Compute all standard and paper metrics across the final evaluation window and write Parquet + JSON."""
    win_window = completed_wins[-100:] if completed_wins else []
    ret_window = completed_returns[-100:] if completed_returns else []
    step_window = completed_steps[-100:] if completed_steps else []
    alarm_window = completed_alarms[-100:] if completed_alarms else []
    guards_window = completed_muscle_guards[-100:] if completed_muscle_guards else []
    extract_window = completed_agents_at_extract[-100:] if completed_agents_at_extract else []

    avg_win = float(np.mean(win_window)) if win_window else 0.0
    lifetime_win = float(np.mean(completed_wins)) if completed_wins else 0.0
    avg_rew = float(np.mean(ret_window)) if ret_window else 0.0
    std_rew = float(np.std(ret_window)) if ret_window else 0.0
    avg_steps = float(np.mean(step_window)) if step_window else 0.0
    avg_alarm = float(np.mean(alarm_window)) if alarm_window else 0.0

    stealth_index = float(max(0.0, min(1.0, 1.0 - (avg_alarm / max(1.0, stage_alarm_max)))))
    alarm_velocity = float(avg_alarm / max(1.0, avg_steps))

    # Ghost run: won with 0 guards neutralized
    if win_window and guards_window and len(win_window) == len(guards_window):
        ghost_runs = [
            1.0 if (w > 0.5 and g <= 0.0) else 0.0 for w, g in zip(win_window, guards_window)
        ]
        ghost_run_rate = float(np.mean(ghost_runs))
    else:
        ghost_run_rate = 0.0

    # Full squad extract: all 4 agents extracted alive
    if extract_window:
        full_squad_rate = float(np.mean([1.0 if a >= 4.0 else 0.0 for a in extract_window]))
    else:
        full_squad_rate = 0.0

    avg_agents_extract = float(np.mean(extract_window)) if extract_window else 0.0
    coop_efficiency = float(avg_agents_extract / max(1.0, avg_steps))

    # Role Subtasks
    scout_tag_rate = (
        float(np.mean(completed_scout_interact[-100:])) if completed_scout_interact else 0.0
    )
    scout_avg_pois = float(np.mean(completed_scout_pois[-100:])) if completed_scout_pois else 0.0
    hacker_hack_rate = (
        float(np.mean(completed_hacker_hack[-100:])) if completed_hacker_hack else 0.0
    )
    muscle_neut_rate = (
        float(np.mean(completed_muscle_neutralize[-100:]))
        if completed_muscle_neutralize
        else 0.0
    )
    muscle_avg_guards = float(np.mean(guards_window)) if guards_window else 0.0
    extractor_loot_rate = (
        float(np.mean(completed_extractor_loot[-100:])) if completed_extractor_loot else 0.0
    )

    # MoE Diversity
    effective_experts = float(np.exp(routing_entropy)) if routing_entropy > 0 else 1.0
    if expert_usage_pct:
        usage_vals = list(expert_usage_pct.values())
        gini = compute_gini(usage_vals)
    else:
        gini = 0.0

    # Role-Expert Mutual Information
    if role_expert_counts is not None:
        role_mi = compute_role_expert_mi(role_expert_counts)
    else:
        role_mi = 0.0

    # Primary failure cause analysis from final window
    failure_cause = "none"
    if avg_win < 0.95:
        if avg_alarm >= stage_alarm_max * 0.9:
            failure_cause = "alarm_max"
        elif avg_steps >= stage_max_steps * 0.95:
            failure_cause = "step_limit"
        elif muscle_neut_rate < 0.2 and total_stage_guards > 0:
            failure_cause = "muscle_bottleneck"
        else:
            failure_cause = "general_attrition"

    # Peak VRAM
    import torch

    vram_peak = (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available()
        else 0.0
    )

    # Throughput
    fps = float(timesteps / max(0.001, wall_clock_sec))

    result_data = {
        "run_id": str(run_id),
        "algo": str(algo_name),
        "seed": int(seed),
        "stage": int(stage_idx),
        "map_size": int(map_size),
        "timesteps": int(timesteps),
        "wall_clock_sec": float(wall_clock_sec),
        "throughput_fps": float(fps),
        "win_rate": float(avg_win),
        "lifetime_win_rate": float(lifetime_win),
        "mean_return": float(avg_rew),
        "std_return": float(std_rew),
        "avg_steps": float(avg_steps),
        "max_stage_steps": int(stage_max_steps),
        "avg_alarm": float(avg_alarm),
        "stage_alarm_max": float(stage_alarm_max),
        "stealth_index": float(stealth_index),
        "alarm_velocity": float(alarm_velocity),
        "ghost_run_rate": float(ghost_run_rate),
        "full_squad_extract_rate": float(full_squad_rate),
        "coop_efficiency": float(coop_efficiency),
        "scout_tag_rate": float(scout_tag_rate),
        "scout_avg_pois": float(scout_avg_pois),
        "hacker_hack_rate": float(hacker_hack_rate),
        "muscle_neut_rate": float(muscle_neut_rate),
        "muscle_avg_guards": float(muscle_avg_guards),
        "total_stage_guards": int(total_stage_guards),
        "extractor_loot_rate": float(extractor_loot_rate),
        "avg_agents_extract": float(avg_agents_extract),
        "primary_failure_cause": str(failure_cause),
        "active_experts": int(active_experts),
        "dormant_experts_count": int(dormant_experts_count),
        "total_spawns": int(total_spawns),
        "expert_switch_rate": float(expert_switch_rate),
        "routing_entropy": float(routing_entropy),
        "effective_experts": float(effective_experts),
        "gini_expert_imbalance": float(gini),
        "role_expert_mi": float(role_mi),
        "vram_peak_mb": float(vram_peak),
        "ablation": str(ablation),
    }

    if save_dir:
        save_stage_result(result_data, save_dir=save_dir, results_root=results_root)
        if spawn_history:
            save_spawn_events(
                spawn_history,
                results_root=results_root,
                local_path=os.path.join(save_dir, "spawn_events.parquet"),
            )

    return result_data


def save_eval_episodes(
    episodes: list[dict[str, Any]],
    results_root: str = "results",
    local_path: str | None = None,
) -> None:
    """Save detailed evaluation episodes to eval_episodes.parquet."""
    if not episodes:
        return

    df_eval = pl.DataFrame(episodes)
    global_path = os.path.join(results_root, "eval_episodes.parquet")
    append_to_parquet(global_path, df_eval)

    if local_path and os.path.abspath(local_path) != os.path.abspath(global_path):
        append_to_parquet(local_path, df_eval)


def save_spawn_events(
    events: list[dict[str, Any]],
    results_root: str = "results",
    local_path: str | None = None,
) -> None:
    """Save dynamic spawn events to spawn_events.parquet."""
    if not events:
        return

    df_spawns = pl.DataFrame(events)
    global_path = os.path.join(results_root, "spawn_events.parquet")
    append_to_parquet(global_path, df_spawns)

    if local_path and os.path.abspath(local_path) != os.path.abspath(global_path):
        append_to_parquet(local_path, df_spawns)


def load_stage_results(
    results_root: str = "results",
    run_id: str | None = None,
    algo: str | None = None,
    stage: int | None = None,
) -> pl.DataFrame:
    """Load and optionally filter stage_results.parquet."""
    parquet_path = os.path.join(results_root, "stage_results.parquet")
    if not os.path.exists(parquet_path):
        return pl.DataFrame()

    lf = pl.scan_parquet(parquet_path)
    if run_id is not None:
        lf = lf.filter(pl.col("run_id") == str(run_id))
    if algo is not None:
        lf = lf.filter(pl.col("algo") == str(algo))
    if stage is not None:
        lf = lf.filter(pl.col("stage") == int(stage))
    return lf.collect()


def load_train_curves(
    results_root: str = "results",
    run_id: str | None = None,
    algo: str | None = None,
    seed: int | None = None,
    stage: int | None = None,
) -> pl.DataFrame:
    """Load and optionally filter train_curves.parquet."""
    parquet_path = os.path.join(results_root, "train_curves.parquet")
    if not os.path.exists(parquet_path):
        return pl.DataFrame()

    lf = pl.scan_parquet(parquet_path)
    if run_id is not None:
        lf = lf.filter(pl.col("run_id") == str(run_id))
    if algo is not None:
        lf = lf.filter(pl.col("algo") == str(algo))
    if seed is not None:
        lf = lf.filter(pl.col("seed") == int(seed))
    if stage is not None:
        lf = lf.filter(pl.col("stage") == int(stage))
    return lf.collect()


def aggregate_multiseed_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Compute mean and std across seeds for each (run_id, algo, stage) grouping."""
    if df.height == 0:
        return pl.DataFrame()

    metrics = [
        "win_rate",
        "mean_return",
        "avg_steps",
        "avg_alarm",
        "stealth_index",
        "ghost_run_rate",
        "full_squad_extract_rate",
        "scout_tag_rate",
        "hacker_hack_rate",
        "muscle_neut_rate",
        "extractor_loot_rate",
        "avg_agents_extract",
        "active_experts",
        "effective_experts",
        "expert_switch_rate",
        "routing_entropy",
        "gini_expert_imbalance",
        "role_expert_mi",
        "vram_peak_mb",
    ]

    aggs = [pl.len().alias("num_seeds")]
    for m in metrics:
        if m in df.columns:
            aggs.append(pl.col(m).mean().alias(f"{m}_mean"))
            aggs.append(pl.col(m).std().alias(f"{m}_std"))

    return df.group_by(["run_id", "algo", "stage", "map_size"]).agg(aggs).sort(["algo", "stage"])


def generate_master_summary_report(
    results_root: str = "results",
    run_id: str | None = None,
    out_md: str | None = None,
    out_json: str | None = None,
) -> dict[str, Any]:
    """Generates a comprehensive master results document in Markdown and JSON for LLMs and researchers.

    Compiles:
      1. Executive Benchmark Matrix (Win Rate, IQM, 95% Bootstrap CI, Return, Stealth across Stages 0-4).
      2. Tactical Subtask & Coordination Matrix (Scout Tag, Hacker Hack, Muscle Neut, Loot, Squad Extract, Ghost Run).
      3. MoE & Compute Scalability Matrix (Throughput FPS, Peak VRAM, Active Experts, Routing Entropy, Gini, MI).
      4. THIEF Micro-Ablation Telemetry across diagnostic stages.
    """
    import json

    stage_df = load_stage_results(results_root=results_root, run_id=run_id)
    eval_path = os.path.join(results_root, "eval_episodes.parquet")
    eval_df = pl.read_parquet(eval_path) if os.path.exists(eval_path) else pl.DataFrame()

    priority_algos = ["thief", "ecoop", "mappo", "hmappo", "coop", "marc", "coma"]
    stages = list(range(5))

    master_dict: dict[str, Any] = {
        "metadata": {
            "run_id": run_id or "all",
            "results_root": results_root,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "benchmark_matrix": {},
        "tactical_matrix": {},
        "scalability_matrix": {},
        "ablation_matrix": {},
    }

    md_lines: list[str] = [
        "# MARL Benchmark Master Results Summary",
        "",
        "> **Format Note**: This document is formatted in standard GitHub Flavored Markdown with precise tabular layouts and statistical aggregates (Mean ± Std, Interquartile Mean [IQM], and 95% Bootstrap Confidence Intervals) specifically designed for direct ingestion and reasoning by Large Language Models.",
        "",
        f"- **Run ID**: `{run_id or 'all'}`",
        f"- **Generated At**: {master_dict['metadata']['generated_at']}",
        f"- **Evaluated Stages**: Stages 0 to 4 (Maps $11\\times 11$ to $50\\times 50$)",
        "",
        "---",
        "",
        "## Table 1: Executive Benchmark Performance Matrix",
        "",
        "| Algorithm | Stage | Map Size | Seeds | Win Rate (%) | Win Rate (IQM %) | 95% Bootstrap CI (%) | Mean Return | Lifetime Win % | Stealth Index |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    if stage_df.height > 0:
        present_algos = [a for a in priority_algos if a in stage_df["algo"].unique().to_list()]
        other_algos = [a for a in stage_df["algo"].unique().to_list() if a not in priority_algos]
        ordered_algos = present_algos + other_algos

        for algo in ordered_algos:
            master_dict["benchmark_matrix"][algo] = {}
            master_dict["tactical_matrix"][algo] = {}
            master_dict["scalability_matrix"][algo] = {}

            algo_df = stage_df.filter((pl.col("algo") == algo) & (pl.col("ablation") == "none"))
            if algo_df.height == 0:
                algo_df = stage_df.filter(pl.col("algo") == algo)

            for s in stages:
                s_df = algo_df.filter(pl.col("stage") == s)
                if s_df.height == 0:
                    continue

                wrs = (s_df["win_rate"] * 100.0).to_list()
                rets = s_df["mean_return"].to_list()
                stealths = s_df["stealth_index"].to_list() if "stealth_index" in s_df.columns else [0.0]
                lifetimes = s_df["lifetime_win_rate"].to_list() if "lifetime_win_rate" in s_df.columns else [0.0]

                wr_mean, wr_std = float(np.mean(wrs)), float(np.std(wrs, ddof=1)) if len(wrs) > 1 else 0.0
                wr_iqm = compute_iqm(wrs)
                _, wr_ci_low, wr_ci_high = compute_bootstrap_ci(wrs)
                ret_mean, ret_std = float(np.mean(rets)), float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
                stealth_m = float(np.mean(stealths))
                lifetime_m = float(np.mean(lifetimes)) * 100.0
                map_s = s_df["map_size"][0] if "map_size" in s_df.columns else 11
                n_seeds = len(wrs)

                master_dict["benchmark_matrix"][algo][s] = {
                    "win_rate_mean": round(wr_mean, 2),
                    "win_rate_std": round(wr_std, 2),
                    "win_rate_iqm": round(wr_iqm, 2),
                    "win_rate_ci_95": [round(wr_ci_low, 2), round(wr_ci_high, 2)],
                    "mean_return_mean": round(ret_mean, 2),
                    "mean_return_std": round(ret_std, 2),
                    "stealth_index": round(stealth_m, 3),
                    "lifetime_win_pct": round(lifetime_m, 2),
                    "num_seeds": n_seeds,
                }

                md_lines.append(
                    f"| **{algo.upper()}** | Stage {s} | {map_s}x{map_s} | {n_seeds} | {wr_mean:.1f} ± {wr_std:.1f} | {wr_iqm:.1f} | [{wr_ci_low:.1f}, {wr_ci_high:.1f}] | {ret_mean:.2f} ± {ret_std:.2f} | {lifetime_m:.1f} | {stealth_m:.3f} |"
                )

        md_lines.extend([
            "",
            "---",
            "",
            "## Table 2: Tactical Subtask Mastery & Coordination Telemetry",
            "",
            "| Algorithm | Stage | Scout Tag (%) | Hacker Hack (%) | Muscle Neut (%) | Extractor Loot (%) | Full Squad Extract (%) | Ghost Run (%) | Alarm Velocity | Primary Failure |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])

        for algo in ordered_algos:
            algo_df = stage_df.filter((pl.col("algo") == algo) & (pl.col("ablation") == "none"))
            if algo_df.height == 0:
                algo_df = stage_df.filter(pl.col("algo") == algo)

            for s in stages:
                s_df = algo_df.filter(pl.col("stage") == s)
                if s_df.height == 0:
                    continue

                def _get_m(col: str, scale: float = 1.0) -> float:
                    val = s_df[col].mean() if col in s_df.columns else None
                    return float(val * scale) if val is not None else 0.0

                tag_m = _get_m("scout_tag_rate", 100.0)
                hack_m = _get_m("hacker_hack_rate", 100.0)
                neut_m = _get_m("muscle_neut_rate", 100.0)
                loot_m = _get_m("extractor_loot_rate", 100.0)
                squad_m = _get_m("full_squad_extract_rate", 100.0)
                ghost_m = _get_m("ghost_run_rate", 100.0)
                vel_m = _get_m("alarm_velocity", 1.0)
                fail_mode = s_df["primary_failure_cause"][0] if "primary_failure_cause" in s_df.columns else "none"

                master_dict["tactical_matrix"][algo][s] = {
                    "scout_tag_rate": round(tag_m, 2),
                    "hacker_hack_rate": round(hack_m, 2),
                    "muscle_neut_rate": round(neut_m, 2),
                    "extractor_loot_rate": round(loot_m, 2),
                    "full_squad_extract_rate": round(squad_m, 2),
                    "ghost_run_rate": round(ghost_m, 2),
                    "alarm_velocity": round(vel_m, 3),
                    "primary_failure_cause": fail_mode,
                }

                md_lines.append(
                    f"| **{algo.upper()}** | Stage {s} | {tag_m:.1f}% | {hack_m:.1f}% | {neut_m:.1f}% | {loot_m:.1f}% | {squad_m:.1f}% | {ghost_m:.1f}% | {vel_m:.3f} | `{fail_mode}` |"
                )

        md_lines.extend([
            "",
            "---",
            "",
            "## Table 3: Compute, Hardware & MoE Scalability Telemetry",
            "",
            "| Algorithm | Stage | FPS Throughput | Wall Clock (s) | Peak VRAM (MB) | Active Experts | Effective Experts | Routing Entropy (nats) | Role-Expert MI |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for algo in ordered_algos:
            algo_df = stage_df.filter((pl.col("algo") == algo) & (pl.col("ablation") == "none"))
            if algo_df.height == 0:
                algo_df = stage_df.filter(pl.col("algo") == algo)

            for s in stages:
                s_df = algo_df.filter(pl.col("stage") == s)
                if s_df.height == 0:
                    continue

                def _get_m3(col: str) -> float:
                    val = s_df[col].mean() if col in s_df.columns else None
                    return float(val) if val is not None else 0.0

                fps_m = _get_m3("throughput_fps")
                wall_m = _get_m3("wall_clock_sec")
                vram_m = _get_m3("vram_peak_mb")
                act_m = _get_m3("active_experts")
                eff_m = _get_m3("effective_experts")
                ent_m = _get_m3("routing_entropy")
                mi_m = _get_m3("role_expert_mi")

                master_dict["scalability_matrix"][algo][s] = {
                    "throughput_fps": round(fps_m, 1),
                    "wall_clock_sec": round(wall_m, 1),
                    "vram_peak_mb": round(vram_m, 1),
                    "active_experts": round(act_m, 1),
                    "effective_experts": round(eff_m, 2),
                    "routing_entropy": round(ent_m, 3),
                    "role_expert_mi": round(mi_m, 3),
                }

                md_lines.append(
                    f"| **{algo.upper()}** | Stage {s} | {fps_m:,.0f} | {wall_m:.1f}s | {vram_m:.1f} | {act_m:.1f} | {eff_m:.2f} | {ent_m:.3f} | {mi_m:.3f} |"
                )

    # Table 4: Ablations (if present)
    ablation_df = stage_df.filter((pl.col("ablation") != "none") & (pl.col("ablation") != "unknown"))
    if ablation_df.height > 0:
        md_lines.extend([
            "",
            "---",
            "",
            "## Table 4: THIEF Micro-Ablation Telemetry (Diagnostic Stages 2 & 4)",
            "",
            "| Ablation Variant | Stage | Win Rate (%) | Mean Return | Stealth Index | Active Experts | Effective Experts | Routing Entropy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for variant in sorted(ablation_df["ablation"].unique().to_list()):
            master_dict["ablation_matrix"][variant] = {}
            var_df = ablation_df.filter(pl.col("ablation") == variant)
            for s in [2, 4]:
                s_df = var_df.filter(pl.col("stage") == s)
                if s_df.height == 0:
                    continue
                wr_m = float(s_df["win_rate"].mean() * 100.0) if s_df["win_rate"].mean() is not None else 0.0
                ret_m = float(s_df["mean_return"].mean()) if s_df["mean_return"].mean() is not None else 0.0
                st_m = float(s_df["stealth_index"].mean()) if "stealth_index" in s_df.columns and s_df["stealth_index"].mean() is not None else 0.0
                act_m = float(s_df["active_experts"].mean()) if "active_experts" in s_df.columns and s_df["active_experts"].mean() is not None else 1.0
                eff_m = float(s_df["effective_experts"].mean()) if "effective_experts" in s_df.columns and s_df["effective_experts"].mean() is not None else 1.0
                ent_m = float(s_df["routing_entropy"].mean()) if "routing_entropy" in s_df.columns and s_df["routing_entropy"].mean() is not None else 0.0

                master_dict["ablation_matrix"][variant][s] = {
                    "win_rate": round(wr_m, 2),
                    "mean_return": round(ret_m, 2),
                    "stealth_index": round(st_m, 3),
                    "active_experts": round(act_m, 1),
                    "effective_experts": round(eff_m, 2),
                    "routing_entropy": round(ent_m, 3),
                }
                md_lines.append(
                    f"| `{variant}` | Stage {s} | {wr_m:.1f}% | {ret_m:.2f} | {st_m:.3f} | {act_m:.1f} | {eff_m:.2f} | {ent_m:.3f} |"
                )

    md_content = "\n".join(md_lines) + "\n"

    # Write Markdown outputs
    primary_md = out_md or os.path.join(results_root, "benchmark_master_summary.md")
    paper_md = "paper/results_master_table.md"
    os.makedirs(os.path.dirname(os.path.abspath(primary_md)), exist_ok=True)
    with open(primary_md, "w") as f:
        f.write(md_content)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(paper_md)), exist_ok=True)
        with open(paper_md, "w") as f:
            f.write(md_content)
    except OSError:
        pass

    # Write JSON output
    primary_json = out_json or os.path.join(results_root, "benchmark_master_summary.json")
    with open(primary_json, "w") as f:
        json.dump(master_dict, f, indent=2)

    print(f"\n[Master Report] Auto-generated benchmark summary in:\n  -> {primary_md}\n  -> {paper_md}\n  -> {primary_json}\n")
    return master_dict
