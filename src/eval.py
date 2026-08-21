"""
Unified Multi-Model Multi-Seed Evaluation Engine for Heist MARL Benchmark.
Evaluates trained checkpoints across standardized random seeds for any combination
of curriculum stages and algorithm paradigms under both Deterministic (argmax)
and Stochastic (sampling) action selection modes.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from constants import (
    AGENTS,
    CURRICULUM_STAGES,
    MACRO_STEP,
    N_AGENTS,
)
from env import HeistEnv


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
    """Parses seeds argument like '42,43,44' into a list of ints."""
    if isinstance(seeds_arg, list):
        return [int(x) for x in seeds_arg]
    if isinstance(seeds_arg, int):
        return [seeds_arg]
    return [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]


def parse_models(models_arg):
    """Parses models argument into a list of model identifiers or checkpoint paths."""
    all_known_algos = ["thief", "hmappo", "mappo", "ecoop", "coma", "marc", "coop"]
    if models_arg is None or models_arg.lower() == "all":
        return all_known_algos

    if "," in models_arg:
        return [x.strip() for x in models_arg.split(",") if x.strip()]

    return [models_arg.strip()]


def load_agent(model_key, stage_idx, state_dim, ckpt_path=None, device="cpu"):
    """
    Instantiates and loads the correct network architecture for each MARL paradigm.
    Returns: (agent, meta_dict)
    """
    is_direct_path = model_key.endswith(".pt") or os.path.isfile(model_key)
    if is_direct_path:
        path = model_key
        # Infer algo from path
        algo_name = "mappo"
        for candidate in ["thief", "hmappo", "ecoop", "coma", "marc", "coop", "mappo"]:
            if candidate in path.lower():
                algo_name = candidate
                break
    else:
        algo_name = model_key.lower()
        path = ckpt_path or f"results/{algo_name}/stage_{stage_idx}/model.pt"

    if not os.path.exists(path):
        return None, f"Checkpoint not found at {path}"

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:  # noqa: BLE001
        return None, f"Failed to load checkpoint ({e})"

    state_dict = (
        ckpt["model_state"]
        if (isinstance(ckpt, dict) and "model_state" in ckpt)
        else ckpt
    )
    meta = {}

    if algo_name == "thief":
        from train_thief import ThiefNetwork

        expert_indices = {
            int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
        }
        needed = max(expert_indices) + 1 if expert_indices else 2
        agent = ThiefNetwork(state_dim, num_initial_experts=needed).to(device)
        while len(agent.experts) < needed:
            agent.add_expert()
        agent.load_state_dict(state_dict)
        agent.eval()
        meta["active_experts"] = (
            ckpt.get("active_experts", len(agent.experts))
            if isinstance(ckpt, dict)
            else len(agent.experts)
        )
        return agent, meta

    elif algo_name == "ecoop":
        from train_ecoop import EcoopNetwork

        expert_indices = {
            int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
        }
        needed = max(expert_indices) + 1 if expert_indices else 1
        agent = EcoopNetwork(state_dim, num_initial_experts=needed).to(device)
        while len(agent.experts) < needed:
            agent.add_expert()
        agent.load_state_dict(state_dict)
        agent.eval()
        meta["active_experts"] = (
            ckpt.get("active_experts", len(agent.experts))
            if isinstance(ckpt, dict)
            else len(agent.experts)
        )
        return agent, meta

    elif algo_name == "hmappo":
        from train_hmappo import HierarchicalNetwork

        agent = HierarchicalNetwork(state_dim).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, meta

    elif algo_name == "mappo":
        from train_mappo import MappoNetwork

        agent = MappoNetwork(state_dim).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, meta

    elif algo_name == "coma":
        from train_coma import ComaNetwork

        agent = ComaNetwork(state_dim).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, meta

    elif algo_name == "marc":
        from train_marc import MarcNetwork

        agent = MarcNetwork(state_dim).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, meta

    elif algo_name == "coop":
        from train_coop import CoopNetwork

        agent = CoopNetwork(state_dim).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, meta

    else:
        # Fallback to standard MAPPO
        from train_mappo import MappoNetwork

        agent = MappoNetwork(state_dim).to(device)
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, meta


def get_actions(
    agent,
    algo_name,
    obs_dict,
    state_arr,
    step,
    hmappo_goals,
    deterministic=True,
    device="cpu",
    meta=None,
):
    """
    Executes policy action inference across all agents.
    Returns: actions_dict mapping agent_name -> action_int
    """
    meta = meta or {}
    actions = {}

    obs_t = {
        a: torch.tensor(
            obs_dict[a]["observation"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        for a in AGENTS
    }
    role_t = {
        a: torch.tensor(
            obs_dict[a]["role_id"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        for a in AGENTS
    }
    mask_t = {
        a: torch.tensor(
            obs_dict[a]["action_mask"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        for a in AGENTS
    }
    state_t = torch.tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        if algo_name == "thief":
            active_experts = meta.get("active_experts", len(agent.experts))
            prev_expert = meta.get("previous_expert")

            obs_stack = torch.cat([obs_t[a] for a in AGENTS], dim=0)
            role_stack = torch.cat([role_t[a] for a in AGENTS], dim=0)
            mask_stack = torch.cat([mask_t[a] for a in AGENTS], dim=0)
            state_stack = state_t.repeat(N_AGENTS, 1)

            acts, _, _, _, chosen_expert = agent.get_action_and_value(
                obs_stack,
                role_stack,
                mask_stack,
                state_stack,
                active_experts=active_experts,
                previous_expert=prev_expert,
                deterministic=deterministic,
            )
            meta["previous_expert"] = chosen_expert
            for i, a in enumerate(AGENTS):
                actions[a] = int(acts[i].item())

        elif algo_name == "ecoop":
            active_experts = meta.get("active_experts", len(agent.experts))
            prev_expert = meta.get("previous_expert")

            obs_stack = torch.cat([obs_t[a] for a in AGENTS], dim=0)
            role_stack = torch.cat([role_t[a] for a in AGENTS], dim=0)
            mask_stack = torch.cat([mask_t[a] for a in AGENTS], dim=0)
            state_stack = state_t.repeat(N_AGENTS, 1)

            acts, _, _, _, chosen_expert = agent.get_action_and_value(
                obs_stack,
                role_stack,
                mask_stack,
                state_stack,
                active_experts=active_experts,
                previous_expert=prev_expert,
            )
            meta["previous_expert"] = chosen_expert
            for i, a in enumerate(AGENTS):
                actions[a] = int(acts[i].item())

        elif algo_name == "hmappo":
            role_stack = torch.cat([role_t[a] for a in AGENTS], dim=0)
            state_stack = state_t.repeat(N_AGENTS, 1)

            # High-level Manager updates macro sub-goal every MACRO_STEP steps
            if step % MACRO_STEP == 0 or hmappo_goals is None:
                m_acts, _, _, _ = agent.get_manager_action_and_value(
                    state_stack, role_stack
                )
                hmappo_goals = m_acts

            for i, a in enumerate(AGENTS):
                goal_i = hmappo_goals[i].unsqueeze(0)
                x = torch.cat([obs_t[a].flatten(start_dim=1), role_t[a], goal_i], dim=1)
                logits = agent.worker_actor(x)
                masked_logits = logits + ((1.0 - mask_t[a]) * -1e9)
                if deterministic:
                    act = masked_logits.argmax(dim=-1).item()
                else:
                    dist = Categorical(logits=masked_logits)
                    act = dist.sample().item()
                actions[a] = int(act)

        elif algo_name in ["coma", "mappo", "marc", "coop"]:
            for a in AGENTS:
                x_actor = torch.cat([obs_t[a].flatten(start_dim=1), role_t[a]], dim=1)
                logits = agent.actor(x_actor)
                masked_logits = logits + ((1.0 - mask_t[a]) * -1e9)
                if deterministic:
                    act = masked_logits.argmax(dim=-1).item()
                else:
                    dist = Categorical(logits=masked_logits)
                    act = dist.sample().item()
                actions[a] = int(act)

    return actions, hmappo_goals


def evaluate_seed(
    agent,
    algo_name,
    stage_idx,
    env_config,
    seed,
    num_episodes=100,
    deterministic=True,
    device="cpu",
    meta_init=None,
):
    """
    Evaluates a model over `num_episodes` in a fixed environment instance seeded with `seed`.
    """
    env = HeistEnv(env_config)
    meta = dict(meta_init or {})

    wins = []
    returns = []
    episode_steps = []
    alarms = []
    scout_pois = []
    hacker_hacks = []
    muscle_guards = []
    extractor_loots = []
    agents_extracted = []

    for ep in range(num_episodes):
        ep_seed = seed * 10000 + ep
        obs, _info = env.reset(seed=ep_seed)
        state = env.state()
        done = False
        step = 0
        ep_rew = 0.0
        hmappo_goals = None
        meta["previous_expert"] = None

        while not done:
            acts, hmappo_goals = get_actions(
                agent,
                algo_name,
                obs,
                state,
                step,
                hmappo_goals,
                deterministic=deterministic,
                device=device,
                meta=meta,
            )

            obs, rewards, terms, truncs, infos = env.step(acts)
            state = env.state()
            step += 1
            step_reward = sum(rewards[a] for a in AGENTS) / float(N_AGENTS)
            ep_rew += step_reward

            done = terms["scout"] or truncs["scout"]

        scout_info = infos["scout"]
        is_win = bool(scout_info.get("win", False))
        wins.append(float(is_win))
        returns.append(float(ep_rew))
        episode_steps.append(float(scout_info.get("steps", step)))
        alarms.append(float(scout_info.get("alarm", 0.0)))
        scout_pois.append(float(scout_info.get("scout_pois_tagged", 0)))
        hacker_hacks.append(float(scout_info.get("hacker_hack_success", False)))
        muscle_guards.append(float(scout_info.get("muscle_guards_neutralized", 0)))
        extractor_loots.append(float(scout_info.get("extractor_loot_success", False)))
        agents_extracted.append(float(scout_info.get("agents_at_extract", 0)))

    total_stage_guards = env_config.get("guard_count", 0)
    neut_rate = (
        (np.mean(muscle_guards) / total_stage_guards) if total_stage_guards > 0 else 0.0
    )

    return {
        "win_rate": float(np.mean(wins)),
        "mean_reward": float(np.mean(returns)),
        "avg_episode_steps": float(np.mean(episode_steps)),
        "avg_alarm": float(np.mean(alarms)),
        "scout_avg_pois_tagged": float(np.mean(scout_pois)),
        "hacker_hack_rate": float(np.mean(hacker_hacks)),
        "muscle_avg_guards_neutralized": float(np.mean(muscle_guards)),
        "muscle_neutralize_rate": float(neut_rate),
        "extractor_loot_rate": float(np.mean(extractor_loots)),
        "avg_agents_at_extract": float(np.mean(agents_extracted)),
        "episodes": num_episodes,
    }


def aggregate_seeds(per_seed_results):
    """
    Computes mean and standard error across all evaluated seeds.
    """
    keys = [
        "win_rate",
        "mean_reward",
        "avg_episode_steps",
        "avg_alarm",
        "scout_avg_pois_tagged",
        "hacker_hack_rate",
        "muscle_avg_guards_neutralized",
        "muscle_neutralize_rate",
        "extractor_loot_rate",
        "avg_agents_at_extract",
    ]
    agg = {}
    n_seeds = len(per_seed_results)

    for k in keys:
        vals = [res[k] for res in per_seed_results.values()]
        mean_val = float(np.mean(vals))
        std_err = (
            float(np.std(vals, ddof=1) / math.sqrt(n_seeds)) if n_seeds > 1 else 0.0
        )
        agg[f"{k}_mean"] = round(mean_val, 4)
        agg[f"{k}_stderr"] = round(std_err, 4)

    return agg


def format_table(
    stage_idx, stage_config, results_dict, mode_label, n_seeds, episodes_per_seed
):
    """
    Formats a clean ASCII table of comparative metrics for a stage.
    """
    total_eps = n_seeds * episodes_per_seed
    stage_name = stage_config.get("name", f"Stage {stage_idx}")
    max_steps = stage_config.get("max_steps", 300)
    guards = stage_config.get("guard_count", 0)

    header = (
        f"\n{'=' * 115}\n"
        f"[{mode_label.upper()} EVALUATION] Stage {stage_idx}: {stage_name} "
        f"({max_steps} max steps, {guards} guards) | {n_seeds} seeds x {episodes_per_seed} eps ({total_eps} eps total)\n"
        f"{'=' * 115}\n"
        f"{'Algorithm':<12} | {'Win Rate (%)':<18} | {'Mean Return':<16} | {'Avg Steps':<18} | {'Avg Alarm':<16} | {'Hack (%)':<10} | {'Loot (%)':<10}\n"
        f"{'-' * 115}"
    )
    lines = [header]

    for algo, stage_data in results_dict.items():
        if f"stage_{stage_idx}" not in stage_data:
            continue
        st_data = stage_data[f"stage_{stage_idx}"]
        win_m = st_data["win_rate_mean"] * 100
        win_se = st_data["win_rate_stderr"] * 100
        ret_m = st_data["mean_reward_mean"]
        ret_se = st_data["mean_reward_stderr"]
        steps_m = st_data["avg_episode_steps_mean"]
        steps_se = st_data["avg_episode_steps_stderr"]
        alarm_m = st_data["avg_alarm_mean"]
        alarm_se = st_data["avg_alarm_stderr"]
        hack_m = st_data["hacker_hack_rate_mean"] * 100
        loot_m = st_data["extractor_loot_rate_mean"] * 100

        win_str = f"{win_m:5.1f} +/- {win_se:4.2f}"
        ret_str = f"{ret_m:5.2f} +/- {ret_se:4.2f}"
        steps_str = f"{steps_m:6.1f} +/- {steps_se:4.1f}"
        alarm_str = f"{alarm_m:4.1f} +/- {alarm_se:3.1f}"
        hack_str = f"{hack_m:4.1f}%"
        loot_str = f"{loot_m:4.1f}%"

        line = f"{algo.upper():<12} | {win_str:<18} | {ret_str:<16} | {steps_str:<18} | {alarm_str:<16} | {hack_str:<10} | {loot_str:<10}"
        lines.append(line)

    lines.append("=" * 115)
    return "\n".join(lines)


def run_eval():
    parser = argparse.ArgumentParser(
        description="Unified Multi-Model Multi-Seed Evaluation Engine for Heist"
    )
    parser.add_argument(
        "--models",
        "--algo",
        "--algos",
        type=str,
        default="all",
        help="Models to evaluate: comma-separated names ('thief,hmappo,mappo,ecoop,coma'), 'all', or path to model.pt",
    )
    parser.add_argument(
        "--stages",
        "--stage",
        type=str,
        default="all",
        help="Curriculum stages to evaluate: '0', '0-4', '0,1,3', or 'all'",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds (e.g. '42,43,44')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per seed per stage",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "determ", "stocha"],
        default="both",
        help="Evaluation action mode: 'both' (default, runs determ + stocha), 'determ', or 'stocha'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/eval",
        help="Directory to save evaluation results and summaries",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch compute device ('cuda' or 'cpu')",
    )

    args = parser.parse_args()

    models_to_eval = parse_models(args.models)
    stages_to_eval = parse_stages(args.stages)
    seeds = parse_seeds(args.seeds)
    episodes_per_seed = args.episodes
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    modes = []
    if args.mode in ["both", "determ"]:
        modes.append(("determ", True, "Deterministic (Argmax)"))
    if args.mode in ["both", "stocha"]:
        modes.append(("stocha", False, "Stochastic (Sampling)"))

    print(f"\n{'=' * 80}")
    print("HEIST MULTI-MODEL MULTI-SEED EVALUATION ENGINE")
    print(f"Models: {models_to_eval}")
    print(f"Stages: {stages_to_eval}")
    print(f"Seeds: {seeds} ({len(seeds)} seeds)")
    print(
        f"Episodes per seed: {episodes_per_seed} ({len(seeds) * episodes_per_seed} per stage)"
    )
    print(f"Modes: {[m[0] for m in modes]}")
    print(f"Device: {device}")
    print(f"{'=' * 80}\n")

    eval_json_data = {
        "determ": defaultdict(dict),
        "stocha": defaultdict(dict),
    }

    dummy_env = HeistEnv(CURRICULUM_STAGES[0])
    dummy_env.reset()
    state_dim = dummy_env.state().shape[0]

    for mode_key, is_determ, mode_desc in modes:
        print(f"\n>>> Running {mode_desc} Evaluation...")

        for stage_idx in stages_to_eval:
            stage_config = CURRICULUM_STAGES[stage_idx]

            for model_id in models_to_eval:
                agent, meta_or_err = load_agent(
                    model_id, stage_idx, state_dim, device=device
                )
                if agent is None:
                    print(f"  [Skipping] {model_id} Stage {stage_idx}: {meta_or_err}")
                    continue

                algo_name = model_id.split("/")[-1].split(".")[0].lower()
                for known in [
                    "thief",
                    "hmappo",
                    "ecoop",
                    "coma",
                    "marc",
                    "coop",
                    "mappo",
                ]:
                    if known in model_id.lower():
                        algo_name = known
                        break

                print(
                    f"  Evaluating {algo_name.upper()} on Stage {stage_idx} "
                    f"({mode_key}) across seeds {seeds}..."
                )

                per_seed_results = {}
                for seed in seeds:
                    seed_res = evaluate_seed(
                        agent,
                        algo_name,
                        stage_idx,
                        stage_config,
                        seed,
                        num_episodes=episodes_per_seed,
                        deterministic=is_determ,
                        device=device,
                        meta_init=meta_or_err,
                    )
                    per_seed_results[str(seed)] = seed_res

                agg_stats = aggregate_seeds(per_seed_results)
                agg_stats["seeds"] = seeds
                agg_stats["episodes_per_seed"] = episodes_per_seed
                agg_stats["total_episodes"] = len(seeds) * episodes_per_seed
                agg_stats["per_seed_results"] = per_seed_results

                eval_json_data[mode_key][algo_name][f"stage_{stage_idx}"] = agg_stats

    # Clean dict for JSON serialization
    output_payload = {
        "determ": {
            algo: dict(stages) for algo, stages in eval_json_data["determ"].items()
        },
        "stocha": {
            algo: dict(stages) for algo, stages in eval_json_data["stocha"].items()
        },
    }

    # Save JSON results
    json_path = os.path.join(args.output_dir, "eval.json")
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=4)
    print(f"\n[Saved JSON Summary]: {json_path}")

    # Generate and Print Formatted Tables
    markdown_lines = ["# Heist Multi-Model Benchmark Evaluation Summary\n"]

    for mode_key, _, mode_desc in modes:
        mode_data = output_payload[mode_key]
        if not mode_data:
            continue

        for stage_idx in stages_to_eval:
            stage_config = CURRICULUM_STAGES[stage_idx]
            table_str = format_table(
                stage_idx,
                stage_config,
                mode_data,
                mode_desc,
                len(seeds),
                episodes_per_seed,
            )
            print(table_str)
            markdown_lines.append(f"```text\n{table_str}\n```\n")

    md_path = os.path.join(args.output_dir, "eval_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(markdown_lines))
    print(f"[Saved Markdown Summary]: {md_path}\n")


if __name__ == "__main__":
    run_eval()
