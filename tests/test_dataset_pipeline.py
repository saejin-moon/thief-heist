import os
import tempfile

import polars as pl
from dataset import (
    TrainCurveLogger,
    append_to_parquet,
    compute_gini,
    compute_role_expert_mi,
    generate_base62_run_id,
    get_effective_run_id,
    load_stage_results,
    load_train_curves,
    record_stage_completion,
    save_eval_episodes,
)


def test_base62_run_id():
    rid1 = generate_base62_run_id()
    rid2 = generate_base62_run_id()
    assert len(rid1) == 4
    assert len(rid2) == 4
    assert rid1.isalnum()
    assert get_effective_run_id(None) is not None
    assert len(get_effective_run_id(None)) == 4
    assert get_effective_run_id("CUSTOM") == "CUSTOM"


def test_gini_and_mi():
    # Pure uniform
    probs_uniform = [0.25, 0.25, 0.25, 0.25]
    assert compute_gini(probs_uniform) == 0.0

    # Completely concentrated
    probs_peak = [1.0, 0.0, 0.0, 0.0]
    assert compute_gini(probs_peak) > 0.0

    # Mutual information
    counts = {
        0: {0: 10, 1: 0},
        1: {0: 0, 1: 10},
    }
    mi = compute_role_expert_mi(counts)
    assert mi > 0.0


def test_parquet_append_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = os.path.join(tmpdir, "test.parquet")
        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        append_to_parquet(pq_path, df1)

        df2 = pl.DataFrame({"a": [3], "b": ["z"]})
        append_to_parquet(pq_path, df2)

        read_df = pl.read_parquet(pq_path)
        assert len(read_df) == 3
        assert read_df["a"].to_list() == [1, 2, 3]


def test_train_curve_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainCurveLogger(
            run_id="TEST",
            algo="thief",
            seed=0,
            stage=0,
            results_root=tmpdir,
            local_dir=tmpdir,
            flush_interval=2,
        )
        for u in range(1, 4):
            logger.log_update(
                update=u,
                total_updates=10,
                timesteps=u * 100,
                elapsed_sec=float(u),
                fps=100.0,
                win_rate=0.5,
                mean_return=1.2,
                avg_steps=50.0,
                avg_alarm=10.0,
                stage_alarm_max=100.0,
                scout_tag_rate=1.0,
                hacker_hack_rate=1.0,
                muscle_neut_rate=1.0,
                extractor_loot_rate=1.0,
                avg_agents_extract=4.0,
                active_experts=2,
                effective_experts=1.8,
                total_spawns=1,
                switch_rate=0.1,
                routing_entropy=0.6,
                policy_loss=0.05,
                value_loss=0.10,
                entropy_loss=0.01,
                approx_kl=0.002,
                clip_fraction=0.05,
                explained_variance=0.85,
                policy_grad_norm=0.3,
                vram_mb=256.0,
            )
        logger.close()

        loaded = load_train_curves(results_root=tmpdir, run_id="TEST", algo="thief")
        assert len(loaded) == 3
        assert "policy_loss" in loaded.columns
        assert "vram_mb" in loaded.columns
        assert loaded["run_id"][0] == "TEST"


def test_stage_completion_and_eval_episodes():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = record_stage_completion(
            algo_name="thief",
            stage_idx=0,
            seed=0,
            run_id="AB12",
            timesteps=1000,
            wall_clock_sec=5.0,
            completed_wins=[1.0, 1.0, 0.0, 1.0],
            completed_returns=[10.0, 12.0, -5.0, 8.0],
            completed_steps=[30, 40, 150, 45],
            completed_alarms=[0.0, 10.0, 100.0, 5.0],
            completed_scout_interact=[1.0, 1.0, 0.0, 1.0],
            completed_scout_pois=[2, 2, 0, 1],
            completed_hacker_hack=[1.0, 1.0, 0.0, 1.0],
            completed_muscle_neutralize=[1.0, 1.0, 0.0, 1.0],
            completed_muscle_guards=[1, 1, 0, 1],
            completed_extractor_loot=[1.0, 1.0, 0.0, 1.0],
            completed_agents_at_extract=[4, 4, 1, 4],
            stage_max_steps=150,
            stage_alarm_max=100.0,
            total_stage_guards=1,
            map_size=11,
            active_experts=2,
            dormant_experts_count=0,
            total_spawns=1,
            expert_switch_rate=0.05,
            routing_entropy=0.5,
            save_dir=tmpdir,
            results_root=tmpdir,
        )
        assert res["run_id"] == "AB12"
        assert res["win_rate"] == 0.75
        assert "stealth_index" in res
        assert "alarm_velocity" in res
        assert "ghost_run_rate" in res
        assert "full_squad_extract_rate" in res

        # Check stage_results.parquet loaded
        sr = load_stage_results(results_root=tmpdir, run_id="AB12")
        assert len(sr) == 1

        # Check eval episodes
        eval_eps = [
            {
                "run_id": "AB12",
                "algo": "thief",
                "stage": 0,
                "eval_seed": 1000,
                "train_seed": 0,
                "episode_idx": 0,
                "win": True,
                "return": 10.0,
                "steps": 35,
                "max_steps": 150,
                "final_alarm": 0.0,
                "max_alarm": 100.0,
                "stealth_index": 1.0,
                "ghost_run": True,
                "full_squad_extracted": True,
                "failure_cause": "none",
                "scout_interact": True,
                "scout_pois_tagged": 2,
                "hacker_hack": True,
                "muscle_neutralize": True,
                "muscle_guards_neutralized": 1,
                "extractor_loot": True,
                "agents_at_extract": 4,
            }
        ]
        save_eval_episodes(eval_eps, results_root=tmpdir)
        pq_eval = pl.read_parquet(os.path.join(tmpdir, "eval_episodes.parquet"))
        assert len(pq_eval) == 1
        assert pq_eval["ghost_run"][0] is True
