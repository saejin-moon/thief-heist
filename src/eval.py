"""
High-Throughput Parallel Evaluation Suite for Multi-Agent RL Heist Algorithms.
Evaluates checkpoints across all curriculum stages in parallel using VectorEnv.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from constants import (
    AGENTS,
    CURRICULUM_STAGES,
    N_AGENTS,
)
from vec_env import VectorEnv


# ----------------------------------------------------------------------
# Universal Model Loaders
# ----------------------------------------------------------------------
def load_thief_model(ckpt_path, state_dim, device):
    from train_thief import ThiefNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )

    expert_indices = {
        int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
    }
    needed = max(expert_indices) + 1 if expert_indices else 2

    model = ThiefNetwork(state_dim, num_initial_experts=needed).to(device)
    while len(model.experts) < needed:
        model.add_expert()
    if len(model.experts) > needed:
        model.prune_experts(list(range(needed)))

    model.load_state_dict(state_dict)
    model.eval()

    active_experts = (
        int(ckpt.get("active_experts", len(model.experts)))
        if isinstance(ckpt, dict)
        else len(model.experts)
    )

    def policy_fn(
        obs_all, role_all, mask_all, state_rep, prev_experts, deterministic=True
    ):
        with torch.no_grad():
            actions, _, _, _, chosen_expert = model.get_action_and_value(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                state_rep,
                active_experts=active_experts,
                previous_expert=prev_experts,
                deterministic=deterministic,
            )
            return actions, chosen_expert

    return policy_fn


def load_ecoop_model(ckpt_path, state_dim, device):
    from train_ecoop import EcoopNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )

    expert_indices = {
        int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
    }
    needed = max(expert_indices) + 1 if expert_indices else 1

    model = EcoopNetwork(state_dim, num_initial_experts=needed).to(device)
    while len(model.experts) < needed:
        model.add_expert()
    if len(model.experts) > needed:
        model.prune_experts(list(range(needed)))

    model.load_state_dict(state_dict)
    model.eval()

    active_experts = (
        int(ckpt.get("active_experts", len(model.experts)))
        if isinstance(ckpt, dict)
        else len(model.experts)
    )

    def policy_fn(
        obs_all, role_all, mask_all, state_rep, prev_experts, deterministic=True
    ):
        with torch.no_grad():
            actions, _, _, _, chosen_expert = model.get_action_and_value(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                state_rep,
                active_experts=active_experts,
                previous_expert=prev_experts,
            )
            return actions, chosen_expert

    return policy_fn


def load_hmappo_model(ckpt_path, state_dim, device):
    from constants import MACRO_STEP
    from train_hmappo import HierarchicalNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )

    agent = HierarchicalNetwork(state_dim).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    current_goals = None
    step_count = [0]

    def policy_fn(
        obs_all, role_all, mask_all, state_rep, prev_experts, deterministic=True
    ):
        nonlocal current_goals
        num_envs = obs_all.shape[1]
        with torch.no_grad():
            if step_count[0] % MACRO_STEP == 0 or current_goals is None:
                # state_rep is (N_AGENTS * num_envs, state_dim) -> per-env state is state_rep[:num_envs]
                env_state_rep = state_rep[:num_envs].repeat(N_AGENTS, 1)
                m_acts, _, _, _ = agent.get_manager_action_and_value(
                    env_state_rep, role_all.flatten(0, 1)
                )
                current_goals = m_acts.view(N_AGENTS, num_envs, 2)

            step_count[0] += 1

            actions_list = []
            for i, a in enumerate(AGENTS):
                w_act, _, _, _ = agent.get_worker_action_and_value(
                    obs_all[i],
                    role_all[i],
                    mask_all[i],
                    current_goals[i],
                )
                actions_list.append(w_act)

            actions = torch.cat(actions_list, dim=0)
            return actions, None

    return policy_fn


def load_mappo_model(ckpt_path, state_dim, device):
    from train_mappo import MappoNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )

    model = MappoNetwork(state_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def policy_fn(
        obs_all, role_all, mask_all, state_rep, prev_experts, deterministic=True
    ):
        with torch.no_grad():
            actions, _, _, _ = model.get_action_and_value(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                state_rep,
            )
            return actions, None

    return policy_fn


def load_coop_model(ckpt_path, state_dim, device):
    from train_coop import CoopNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )

    model = CoopNetwork(state_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def policy_fn(
        obs_all, role_all, mask_all, state_rep, prev_experts, deterministic=True
    ):
        with torch.no_grad():
            actions, _, _, _, _ = model.get_action_and_value(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                state_rep,
            )
            return actions, None

    return policy_fn


def load_marc_model(ckpt_path, state_dim, device):
    from train_marc import MarcNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )

    model = MarcNetwork(state_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def policy_fn(
        obs_all, role_all, mask_all, state_rep, prev_experts, deterministic=True
    ):
        with torch.no_grad():
            actions, _, _, _ = model.get_action_and_value(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                state_rep,
            )
            return actions, None

    return policy_fn


def load_coma_model(ckpt_path, state_dim, device):
    from train_coma import ComaNetwork

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )

    model = ComaNetwork(state_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def policy_fn(
        obs_all, role_all, mask_all, state_rep, prev_experts, deterministic=True
    ):
        with torch.no_grad():
            actions, _, _, _ = model.get_action(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
            )
            return actions, None

    return policy_fn


MODEL_LOADERS = {
    "thief": load_thief_model,
    "ecoop": load_ecoop_model,
    "hmappo": load_hmappo_model,
    "mappo": load_mappo_model,
    "coop": load_coop_model,
    "marc": load_marc_model,
    "coma": load_coma_model,
}


# ----------------------------------------------------------------------
# Parallel Vectorized Evaluation Engine
# ----------------------------------------------------------------------
def evaluate_checkpoint(
    algo_name,
    stage_idx,
    ckpt_path,
    target_episodes=100,
    num_envs=16,
    device=None,
    deterministic=True,
    base_seed=10000,
):
    """
    Evaluates a model checkpoint in parallel using VectorEnv.
    Returns rich dictionary of aggregated milestone statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_config = CURRICULUM_STAGES[stage_idx].copy()
    env_config = stage_config.copy()
    env_config.pop("timesteps", None)

    vec_env = VectorEnv(num_envs, config=env_config, base_seed=base_seed)
    state_dim = vec_env.state_dim

    # Load policy function
    loader = MODEL_LOADERS.get(algo_name.lower())
    if loader is None:
        raise ValueError(f"Unknown algorithm loader: {algo_name}")

    policy_fn = loader(ckpt_path, state_dim, device)

    # Evaluation accumulators
    completed_wins = []
    completed_returns = []
    completed_steps = []
    completed_alarms = []
    completed_scout_interact = []
    completed_scout_pois = []
    completed_hacker_hack = []
    completed_muscle_neutralize = []
    completed_muscle_guards = []
    completed_extractor_loot = []
    completed_agents_extract = []

    current_returns = np.zeros(num_envs)
    env_prev_experts = None

    next_obs, next_state = vec_env.reset(seed=base_seed)
    start_time = time.time()

    while len(completed_wins) < target_episodes:
        with torch.no_grad():
            stacked = next_obs["_stacked"]
            obs_all = torch.tensor(
                stacked["observation"], dtype=torch.float32, device=device
            )
            role_all = torch.tensor(
                stacked["role_id"], dtype=torch.float32, device=device
            )
            mask_all = torch.tensor(
                stacked["action_mask"], dtype=torch.float32, device=device
            )
            state_rep = (
                torch.tensor(next_state, dtype=torch.float32)
                .to(device)
                .repeat(N_AGENTS, 1)
            )

            actions, chosen_expert = policy_fn(
                obs_all,
                role_all,
                mask_all,
                state_rep,
                env_prev_experts,
                deterministic=deterministic,
            )

            if chosen_expert is not None:
                env_prev_experts = chosen_expert

            actions_unflattened = actions.view(N_AGENTS, num_envs)
            actions_dict = {
                a: actions_unflattened[i].cpu().numpy() for i, a in enumerate(AGENTS)
            }

        next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)
        next_state = vec_env.state

        # Accumulate per-agent mean step reward
        step_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
        current_returns += step_reward

        for e in range(num_envs):
            is_done = terms["scout"][e] or truncs["scout"][e]
            if is_done and len(completed_wins) < target_episodes:
                scout_info = infos[e]["scout"]
                is_win = bool(scout_info.get("win", False))

                completed_wins.append(float(is_win))
                completed_returns.append(float(current_returns[e]))
                current_returns[e] = 0.0

                completed_steps.append(float(scout_info.get("steps", 0)))
                completed_alarms.append(float(scout_info.get("alarm", 0.0)))
                completed_scout_interact.append(
                    float(scout_info.get("scout_interact_success", False))
                )
                completed_scout_pois.append(
                    float(scout_info.get("scout_pois_tagged", 0))
                )
                completed_hacker_hack.append(
                    float(scout_info.get("hacker_hack_success", False))
                )
                completed_muscle_neutralize.append(
                    float(scout_info.get("muscle_neutralize_success", False))
                )
                completed_muscle_guards.append(
                    float(scout_info.get("muscle_guards_neutralized", 0))
                )
                completed_extractor_loot.append(
                    float(scout_info.get("extractor_loot_success", False))
                )
                completed_agents_extract.append(
                    float(scout_info.get("agents_at_extract", 0))
                )

                if env_prev_experts is not None:
                    for ag_i in range(N_AGENTS):
                        env_prev_experts[ag_i * num_envs + e] = -1

    vec_env.close()
    elapsed_time = time.time() - start_time

    results = {
        "algo": algo_name,
        "stage": stage_idx,
        "episodes_evaluated": len(completed_wins),
        "evaluation_time_sec": round(elapsed_time, 2),
        "episodes_per_sec": round(len(completed_wins) / elapsed_time, 1),
        "win_rate": float(np.mean(completed_wins)),
        "win_rate_std": float(np.std(completed_wins)),
        "mean_return": float(np.mean(completed_returns)),
        "mean_return_std": float(np.std(completed_returns)),
        "avg_steps": float(np.mean(completed_steps)),
        "max_stage_steps": stage_config.get("max_steps", 300),
        "avg_alarm": float(np.mean(completed_alarms)),
        "stage_alarm_max": stage_config.get("alarm_max", 100.0),
        "scout_tag_rate": float(np.mean(completed_scout_interact)),
        "scout_avg_pois": float(np.mean(completed_scout_pois)),
        "hacker_hack_rate": float(np.mean(completed_hacker_hack)),
        "muscle_neutralize_rate": float(np.mean(completed_muscle_neutralize)),
        "muscle_avg_guards": float(np.mean(completed_muscle_guards)),
        "total_stage_guards": stage_config.get("guard_count", 0),
        "extractor_loot_rate": float(np.mean(completed_extractor_loot)),
        "avg_agents_extract": float(np.mean(completed_agents_extract)),
    }
    return results


# ----------------------------------------------------------------------
# CLI Runner & Multi-Model Multi-Stage Batch Evaluation
# ----------------------------------------------------------------------
def parse_eval_args():
    parser = argparse.ArgumentParser(description="High-Throughput Parallel Evaluator")
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        help="Algorithm to evaluate ('thief', 'hmappo', etc., comma-separated list, or 'all')",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Stages to evaluate ('0', '0-4', 'all')",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Convenience alias to evaluate single stage",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per stage (default: 100)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel worker environments (default: 16)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Direct path to checkpoint model.pt to evaluate",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic argmax action selection instead of default policy sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
        help="Base evaluation seed",
    )
    return parser.parse_args()


import warnings

warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    args = parse_eval_args()
    deterministic = args.greedy

    stages = (
        [args.stage]
        if args.stage is not None
        else (
            list(range(len(CURRICULUM_STAGES)))
            if args.stages == "all"
            else [int(x) for x in args.stages.split(",") if x]
        )
    )

    if args.algo == "all":
        algos = ["thief", "hmappo", "mappo", "ecoop", "coop", "marc", "coma"]
    elif "," in args.algo:
        algos = [x.strip().lower() for x in args.algo.split(",") if x.strip()]
    else:
        algos = [args.algo.strip().lower()]

    all_eval_results = []

    print(
        f"\n{'=' * 85}\n"
        f"PARALLEL EVALUATION SUITE (Envs: {args.num_envs} | Episodes: {args.episodes} | Greedy: {deterministic})\n"
        f"{'=' * 85}\n",
        flush=True,
    )

    for stage_idx in stages:
        print(
            f"\n--- STAGE {stage_idx} (Max Steps: {CURRICULUM_STAGES[stage_idx]['max_steps']}) ---",
            flush=True,
        )
        print(
            f"{'Algorithm':<10} | {'Win Rate':<10} | {'Mean Return':<12} | {'Steps':<14} | {'Alarm':<12} | {'Extract':<10} | {'Eval Time':<10}",
            flush=True,
        )
        print("-" * 85, flush=True)

        for algo in algos:
            # Checkpoint resolution
            if args.ckpt and os.path.exists(args.ckpt):
                ckpt_path = args.ckpt
            else:
                ckpt_candidates = [
                    f"results/{algo}/stage_{stage_idx}/model.pt",
                    f"results/{algo}/seed_0/stage_{stage_idx}/model.pt",
                ]
                ckpt_path = next(
                    (c for c in ckpt_candidates if os.path.exists(c)), None
                )

            if ckpt_path is None:
                print(f"{algo:<10} | {'[No Checkpoint Found]':<65}", flush=True)
                continue

            print(
                f"{algo:<10} | Evaluating {args.episodes} episodes...",
                end="\r",
                flush=True,
            )

            try:
                res = evaluate_checkpoint(
                    algo_name=algo,
                    stage_idx=stage_idx,
                    ckpt_path=ckpt_path,
                    target_episodes=args.episodes,
                    num_envs=args.num_envs,
                    deterministic=deterministic,
                    base_seed=args.seed,
                )
                all_eval_results.append(res)

                wr = f"{res['win_rate'] * 100:.1f}%"
                ret = f"{res['mean_return']:.2f}"
                steps = f"{res['avg_steps']:.1f}/{res['max_stage_steps']}"
                alarm = f"{res['avg_alarm']:.1f}/{res['stage_alarm_max']:.0f}"
                extract = f"{res['avg_agents_extract']:.2f}/4"
                time_str = (
                    f"{res['evaluation_time_sec']}s ({res['episodes_per_sec']} ep/s)"
                )

                print(
                    f"{algo:<10} | {wr:<10} | {ret:<12} | {steps:<14} | {alarm:<12} | {extract:<10} | {time_str:<10}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{algo:<10} | Error evaluating: {e}", flush=True)

    # Print Full Multi-Stage Summary if multiple stages evaluated
    if len(stages) > 1 and len(all_eval_results) > 0:
        print(f"\n\n{'=' * 85}")
        print("EVALUATION BENCHMARK SUMMARY (WIN RATES ACROSS STAGES)")
        print(f"{'=' * 85}")
        header_stages = " | ".join(f"Stage {s}" for s in stages)
        print(f"{'Algorithm':<10} | {'Avg Win%':<10} | {header_stages}")
        print("-" * 85)

        for algo in algos:
            algo_res = [r for r in all_eval_results if r["algo"] == algo]
            if not algo_res:
                continue
            wrs = [
                next(
                    (
                        f"{r['win_rate'] * 100:.1f}%"
                        for r in algo_res
                        if r["stage"] == s
                    ),
                    "N/A",
                )
                for s in stages
            ]
            num_wrs = [r["win_rate"] * 100 for r in algo_res]
            avg_wr = np.mean(num_wrs) if num_wrs else 0.0
            row_str = " | ".join(f"{w:>7}" for w in wrs)
            print(f"{algo:<10} | {avg_wr:>7.1f}%  | {row_str}")

    # Save output JSON report
    os.makedirs("results/eval", exist_ok=True)
    report_file = "results/eval/eval_summary.json"
    with open(report_file, "w") as f:
        json.dump(all_eval_results, f, indent=4)
    print(f"\nSaved complete evaluation metrics to {report_file}\n", flush=True)


if __name__ == "__main__":
    main()
