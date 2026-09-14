import json
import logging
import os

import numpy as np
import torch
from constants import (
    ACTION_SPACE_SIZE,
    AFFORDANCE_COEF,
    AGENTS,
    ALPHA_ALARM,
    CLIP_COEF,
    CURRICULUM_STAGES,
    ENT_COEF,
    GAE_LAMBDA,
    GAMMA,
    GAMMA_CAUSAL,
    GOAL_VECTOR_DIM,
    LR,
    N_AGENTS,
    NUM_ENVS,
    NUM_STEPS,
    OBSERVATION_SIZE,
    UPDATE_EPOCHS,
    VF_COEF,
)
from torch.nn import functional as F
from torch import nn
from torch.distributions.categorical import Categorical
from vec_env import make_vec_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# CleanRL philosophy: Everything in one file.
class MarcNetwork(nn.Module):
    """
    Standard Actor-Critic network. In MARC, the architecture is flat like MAPPO,
    but the credit assignment (GAE calculation) is profoundly different.
    """

    def __init__(self, state_dim):
        super().__init__()
        actor_in_dim = (
            (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS + GOAL_VECTOR_DIM
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(actor_in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, ACTION_SPACE_SIZE), std=0.01),
        )
        critic_in_dim = state_dim + N_AGENTS + GOAL_VECTOR_DIM
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_action_and_value(self, obs, role, mask, goal, state=None, action=None):
        x_actor = torch.cat([obs.flatten(start_dim=1), role, goal], dim=1)
        logits = self.actor(x_actor)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        value = None
        if state is not None:
            x_critic = torch.cat([state, role, goal], dim=1)
            value = self.critic(x_critic).squeeze(-1)
        return action, probs.log_prob(action), probs.entropy(), value


def train(
    algo_name="marc",
    stage_idx=0,
    env_config=None,
    total_timesteps=1000,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    log_dir=None,
    seed=None,
    use_rust=False,
    run_id=None,
):
    import random
    import time
    from dataset import get_effective_run_id, TrainCurveLogger, record_stage_completion

    start_time = time.time()
    effective_run_id = get_effective_run_id(run_id)
    effective_seed = seed if seed is not None else 0
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if save_ckpt_dir is None:
        save_ckpt_dir = f"results/{algo_name}/seed_{effective_seed}/stage_{stage_idx}"
    if log_dir is None:
        log_dir = save_ckpt_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    curve_logger = TrainCurveLogger(
        run_id=effective_run_id,
        algo=algo_name,
        seed=effective_seed,
        stage=stage_idx,
        results_root="results",
        local_dir=log_dir,
        flush_interval=10,
    )

    console_logger = logging.getLogger(f"console_{algo_name}_{stage_idx}_{effective_seed}")
    console_logger.setLevel(logging.INFO)
    console_logger.handlers = [logging.StreamHandler()]
    console_logger.propagate = False

    file_logger = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_logger = logging.getLogger(f"file_{algo_name}_{stage_idx}_{effective_seed}")
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        file_logger.handlers = [file_handler]
        file_logger.propagate = False

    engine_tag = "Native Rust Rayon" if use_rust else "Python Multiprocessing"
    msg = f"Training {algo_name} Stage {stage_idx} on {device} ({engine_tag})..."
    console_logger.info(msg)
    if file_logger:
        file_logger.info(msg)

    if env_config is None:
        env_config = dict(CURRICULUM_STAGES[stage_idx])
    vec_env = make_vec_env(
        NUM_ENVS,
        config=env_config,
        base_seed=seed * 1000 if seed is not None else 0,
        use_rust=use_rust,
    )
    state_dim = vec_env.state_dim

    agent = MarcNetwork(state_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        agent.load_state_dict(torch.load(load_ckpt_path, map_location=device))
        logging.info(  # noqa: LOG015
            f"Loaded checkpoint from {load_ckpt_path}"
        )

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)

    num_steps = 250 if stage_idx >= 3 else NUM_STEPS
    num_updates = total_timesteps // (NUM_ENVS * num_steps)
    global_episodes = 0
    global_wins = 0
    current_env_returns = np.zeros(NUM_ENVS)
    completed_episode_returns = []
    completed_wins = []
    completed_episode_steps = []
    completed_episode_alarms = []
    completed_scout_interact = []
    completed_scout_pois = []
    completed_hacker_hack = []
    completed_muscle_neutralize = []
    completed_muscle_guards = []
    completed_extractor_loot = []
    completed_agents_at_extract = []

    interval_wins = []
    interval_episode_returns = []
    interval_episode_steps = []
    interval_episode_alarms = []
    interval_scout_interact = []
    interval_scout_pois = []
    interval_hacker_hack = []
    interval_muscle_neutralize = []
    interval_muscle_guards = []
    interval_extractor_loot = []
    interval_agents_at_extract = []

    for update in range(1, num_updates + 1):
        check_thermal_guard()
        b_obs = {
            a: torch.zeros((num_steps, NUM_ENVS, *OBSERVATION_SIZE)).to(device)
            for a in AGENTS
        }
        b_role = {
            a: torch.zeros((num_steps, NUM_ENVS, N_AGENTS)).to(device) for a in AGENTS
        }
        b_mask = {
            a: torch.zeros((num_steps, NUM_ENVS, ACTION_SPACE_SIZE)).to(device)
            for a in AGENTS
        }
        b_goal = {
            a: torch.zeros((num_steps, NUM_ENVS, GOAL_VECTOR_DIM)).to(device)
            for a in AGENTS
        }
        b_actions = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_logprobs = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_rewards = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_values = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_dones = torch.zeros((num_steps, NUM_ENVS)).to(device)
        b_states = torch.zeros((num_steps, NUM_ENVS, state_dim)).to(device)

        # MARC specialized buffers
        b_affordances = {
            a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS
        }
        b_alarms = torch.zeros((num_steps, NUM_ENVS)).to(device)
        b_wins = np.zeros((num_steps, NUM_ENVS), dtype=bool)

        # --- ROLLOUT PHASE ---
        for step in range(num_steps):
            b_states[step] = torch.tensor(next_state, dtype=torch.float32).to(device)
            b_dones[step] = next_done

            actions_dict = {}
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
                goals_all = torch.tensor(
                    stacked["goal_vector"], dtype=torch.float32, device=device
                )

                state_rep = b_states[step].repeat(N_AGENTS, 1)
                actions, logprobs, _, values = agent.get_action_and_value(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    goals_all.flatten(0, 1),
                    state_rep,
                )

                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                values = values.view(N_AGENTS, NUM_ENVS)

                for i, a in enumerate(AGENTS):
                    b_obs[a][step] = obs_all[i]
                    b_role[a][step] = role_all[i]
                    b_mask[a][step] = mask_all[i]
                    b_goal[a][step] = goals_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            # Step the environment!
            next_obs_new, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            # Accumulate per-agent average return per env
            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            for e in range(NUM_ENVS):
                if infos[e]["scout"].get("win", False):
                    b_wins[step, e] = True
                # Check for termination to update tracking metrics
                is_done = terms["scout"][e] or truncs["scout"][e]

                if is_done:
                    global_episodes += 1
                    scout_info = infos[e]["scout"]
                    is_win = bool(scout_info.get("win", False))
                    if is_win:
                        global_wins += 1
                    completed_wins.append(float(is_win))
                    completed_episode_returns.append(float(current_env_returns[e]))
                    interval_wins.append(float(is_win))
                    interval_episode_returns.append(float(current_env_returns[e]))
                    current_env_returns[e] = 0.0

                    interact_succ = float(
                        scout_info.get("scout_interact_success", False)
                    )
                    pois_tagged = float(scout_info.get("scout_pois_tagged", 0))
                    hack_succ = float(scout_info.get("hacker_hack_success", False))
                    neutralize_succ = float(
                        scout_info.get("muscle_neutralize_success", False)
                    )
                    guards_neutralized = float(
                        scout_info.get("muscle_guards_neutralized", 0)
                    )
                    loot_succ = float(scout_info.get("extractor_loot_success", False))
                    agents_extract = float(scout_info.get("agents_at_extract", 0))
                    ep_steps = float(scout_info.get("steps", 0))
                    ep_alarm = float(scout_info.get("alarm", 0.0))

                    completed_scout_interact.append(interact_succ)
                    completed_scout_pois.append(pois_tagged)
                    completed_hacker_hack.append(hack_succ)
                    completed_muscle_neutralize.append(neutralize_succ)
                    completed_muscle_guards.append(guards_neutralized)
                    completed_extractor_loot.append(loot_succ)
                    completed_agents_at_extract.append(agents_extract)
                    completed_episode_steps.append(ep_steps)
                    completed_episode_alarms.append(ep_alarm)

                    interval_scout_interact.append(interact_succ)
                    interval_scout_pois.append(pois_tagged)
                    interval_hacker_hack.append(hack_succ)
                    interval_muscle_neutralize.append(neutralize_succ)
                    interval_muscle_guards.append(guards_neutralized)
                    interval_extractor_loot.append(loot_succ)
                    interval_agents_at_extract.append(agents_extract)
                    interval_episode_steps.append(ep_steps)
                    interval_episode_alarms.append(ep_alarm)

            # The team unlocked an affordance if an agent interacted and the mask volume increased.
            mask_new = next_obs_new["_stacked"]["action_mask"]
            mask_vol_new = mask_new.sum(axis=(0, 2))  # sum over agents and action dims
            mask_vol_old = stacked["action_mask"].sum(axis=(0, 2))
            unlocked = mask_vol_new > mask_vol_old

            for i, a in enumerate(AGENTS):
                # 5 is INTERACT
                interacted = actions_dict[a] == 5
                # Assign affordance delta (1.0) if they interacted and unlocked something
                b_affordances[a][step] = torch.tensor(
                    interacted & unlocked, dtype=torch.float32
                ).to(device)

            b_alarms[step] = torch.tensor(
                [infos[e]["scout"]["alarm"] for e in range(NUM_ENVS)]
            ).to(device)

            next_state = vec_env.state
            next_done = torch.tensor(
                terms["scout"] | truncs["scout"], dtype=torch.float32
            ).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(
                    device
                )

            next_obs = next_obs_new

        # --- MARC CAUSAL REWARD & ADVANTAGE RESTRUCTURING ---
        alarm_weights = torch.exp(-ALPHA_ALARM * (b_alarms / 100.0))
        win_tensor = torch.tensor(b_wins, dtype=torch.float32, device=device)
        macro_weights = (1.0 + 2.0 * win_tensor) * alarm_weights

        b_shaped_rewards = {}
        for a in AGENTS:
            r_imm = b_rewards[a] + AFFORDANCE_COEF * b_affordances[a]
            b_shaped_rewards[a] = macro_weights * r_imm

        with torch.no_grad():
            state_next_t = torch.tensor(next_state, dtype=torch.float32, device=device)
            stacked_next = next_obs["_stacked"]
            next_obs_all = torch.tensor(
                stacked_next["observation"], dtype=torch.float32, device=device
            )
            next_role_all = torch.tensor(
                stacked_next["role_id"], dtype=torch.float32, device=device
            )
            next_mask_all = torch.tensor(
                stacked_next["action_mask"], dtype=torch.float32, device=device
            )
            next_goals_all = torch.tensor(
                stacked_next["goal_vector"], dtype=torch.float32, device=device
            )
            state_rep = state_next_t.repeat(N_AGENTS, 1)
            _, _, _, next_values = agent.get_action_and_value(
                next_obs_all.flatten(0, 1),
                next_role_all.flatten(0, 1),
                next_mask_all.flatten(0, 1),
                next_goals_all.flatten(0, 1),
                state_rep,
            )
            next_values = next_values.view(N_AGENTS, NUM_ENVS)
            b_adv = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}

            for i, a in enumerate(AGENTS):
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_values[i]
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[a][t + 1]

                    delta = (
                        b_shaped_rewards[a][t]
                        + GAMMA * nextvalues * nextnonterminal
                        - b_values[a][t]
                    )

                    # 3. Retroactive Causal Trace: temporal credit assignment decay
                    lastgaelam = (
                        delta
                        + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                        + GAMMA_CAUSAL * b_affordances[a][t]
                    )
                    b_adv[a][t] = lastgaelam

        b_returns = {a: b_adv[a] + b_values[a] for a in AGENTS}

        # --- UPDATE PHASE (PPO) ---
        b_states_flat = b_states.reshape(-1, state_dim)
        for _epoch in range(UPDATE_EPOCHS):
            for a in AGENTS:
                obs_flat = b_obs[a].reshape(-1, *OBSERVATION_SIZE)
                role_flat = b_role[a].reshape(-1, N_AGENTS)
                mask_flat = b_mask[a].reshape(-1, ACTION_SPACE_SIZE)
                goal_flat = b_goal[a].reshape(-1, GOAL_VECTOR_DIM)
                action_flat = b_actions[a].reshape(-1)
                logprob_flat = b_logprobs[a].reshape(-1)
                adv_flat = b_adv[a].reshape(-1)
                ret_flat = b_returns[a].reshape(-1)

                adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_flat,
                    role_flat,
                    mask_flat,
                    goal_flat,
                    b_states_flat,
                    action=action_flat,
                )

                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_norm * ratio
                pg_loss2 = -adv_norm * torch.clamp(
                    ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.smooth_l1_loss(newvalue, ret_flat, beta=1.0)
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                last_pg_loss = pg_loss.item()
                last_v_loss = v_loss.item()
                last_ent_loss = entropy.mean().item()
                last_approx_kl = ((ratio - 1.0) - logratio).mean().item()
                last_clip_frac = (torch.abs(ratio - 1.0) > CLIP_COEF).float().mean().item()
                var_y = torch.var(ret_flat)
                last_explained_var = (
                    (1.0 - torch.var(ret_flat - newvalue) / (var_y + 1e-8)).item()
                    if var_y > 1e-8
                    else 0.0
                )
                last_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), 0.5).item()
                optimizer.step()

        if update % 5 == 0:
            win_rate = float(np.mean(completed_wins[-25:])) if completed_wins else 0.0
            mean_episodic_reward = (
                float(np.mean(completed_episode_returns[-25:]))
                if completed_episode_returns
                else 0.0
            )
            max_stage_steps = env_config.get("max_steps", 150)
            total_stage_guards = env_config.get("guard_count", 0)
            stage_alarm_max = env_config.get("alarm_max", 100.0)
            avg_steps = (
                float(np.mean(completed_episode_steps[-25:]))
                if completed_episode_steps
                else 0.0
            )
            avg_alarm = (
                float(np.mean(completed_episode_alarms[-25:]))
                if completed_episode_alarms
                else 0.0
            )
            avg_muscle_guards = (
                float(np.mean(completed_muscle_guards[-25:]))
                if completed_muscle_guards
                else 0.0
            )

            scout_tag_rate = (
                float(np.mean(completed_scout_interact[-25:]))
                if completed_scout_interact
                else 0.0
            )
            scout_avg_pois = (
                float(np.mean(completed_scout_pois[-25:]))
                if completed_scout_pois
                else 0.0
            )
            hacker_hack_rate = (
                float(np.mean(completed_hacker_hack[-25:]))
                if completed_hacker_hack
                else 0.0
            )
            muscle_neutralize_rate = (
                float(np.mean(completed_muscle_neutralize[-25:]))
                if completed_muscle_neutralize
                else 0.0
            )
            extractor_loot_rate = (
                float(np.mean(completed_extractor_loot[-25:]))
                if completed_extractor_loot
                else 0.0
            )
            avg_agents_extract = (
                float(np.mean(completed_agents_at_extract[-25:]))
                if completed_agents_at_extract
                else 0.0
            )

            # Console log (clean & compact)
            console_logger.info(
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f}"
            )
            # Detailed file log
            if file_logger:
                file_logger.info(
                    f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | Scout Tag Rate: {scout_tag_rate:.2f} (Avg POIs: {scout_avg_pois:.1f}) | Hacker Hack Rate: {hacker_hack_rate:.2f} | Muscle Neutralize Rate: {muscle_neutralize_rate:.2f} (Avg Guards: {avg_muscle_guards:.1f}/{total_stage_guards}) | Extractor Loot Rate: {extractor_loot_rate:.2f} | Avg Agents at Extract: {avg_agents_extract:.2f}/4"
                )

            curve_logger.log_update(
                update=update,
                total_updates=num_updates,
                timesteps=update * num_steps * NUM_ENVS,
                elapsed_sec=time.time() - start_time,
                fps=float((update * num_steps * NUM_ENVS) / max(0.001, time.time() - start_time)),
                win_rate=win_rate,
                mean_return=mean_episodic_reward,
                avg_steps=avg_steps,
                avg_alarm=avg_alarm,
                stage_alarm_max=stage_alarm_max,
                scout_tag_rate=scout_tag_rate,
                hacker_hack_rate=hacker_hack_rate,
                muscle_neut_rate=muscle_neutralize_rate,
                extractor_loot_rate=extractor_loot_rate,
                avg_agents_extract=avg_agents_extract,
                active_experts=1,
                effective_experts=1.0,
                total_spawns=0,
                switch_rate=0.0,
                routing_entropy=0.0,
                policy_loss=locals().get("last_pg_loss", 0.0),
                value_loss=locals().get("last_v_loss", 0.0),
                entropy_loss=locals().get("last_ent_loss", 0.0),
                approx_kl=locals().get("last_approx_kl", 0.0),
                clip_fraction=locals().get("last_clip_frac", 0.0),
                explained_variance=locals().get("last_explained_var", 0.0),
                policy_grad_norm=locals().get("last_grad_norm", 0.0),
                vram_mb=(
                    float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                    if torch.cuda.is_available()
                    else 0.0
                ),
            )

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))

    curve_logger.close()
    elapsed_wall_clock = time.time() - start_time
    record_stage_completion(
        algo_name=algo_name,
        stage_idx=stage_idx,
        seed=effective_seed,
        run_id=effective_run_id,
        timesteps=total_timesteps,
        wall_clock_sec=elapsed_wall_clock,
        completed_wins=completed_wins,
        completed_returns=completed_episode_returns,
        completed_steps=completed_episode_steps,
        completed_alarms=completed_episode_alarms,
        completed_scout_interact=completed_scout_interact,
        completed_scout_pois=completed_scout_pois,
        completed_hacker_hack=completed_hacker_hack,
        completed_muscle_neutralize=completed_muscle_neutralize,
        completed_muscle_guards=completed_muscle_guards,
        completed_extractor_loot=completed_extractor_loot,
        completed_agents_at_extract=completed_agents_at_extract,
        stage_max_steps=int(max_stage_steps) if "max_stage_steps" in locals() else 150,
        stage_alarm_max=float(stage_alarm_max) if "stage_alarm_max" in locals() else 100.0,
        total_stage_guards=int(total_stage_guards) if "total_stage_guards" in locals() else 0,
        map_size=env_config.get("map_size", (11, 11))[0] if env_config else 11,
        active_experts=1,
        dormant_experts_count=0,
        total_spawns=0,
        expert_switch_rate=0.0,
        routing_entropy=0.0,
        expert_usage_pct=None,
        role_expert_counts=None,
        ablation="none",
        spawn_history=None,
        save_dir=log_dir or save_ckpt_dir,
        results_root="results",
    )
    print(f"Saved checkpoint and results to {save_ckpt_dir}")

    vec_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARC Training")
    parser.add_argument("--stage", type=int, default=0, help="Stage index")
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total timesteps"
    )
    parser.add_argument(
        "--seeds",
        "--seed",
        dest="seed",
        type=lambda s: int(s.split(",")[0].strip()) if "," in str(s) else int(str(s).strip()),
        default=0,
        help="Random seed (e.g. 5 or 0)",
    )
    parser.add_argument(
        "--save-dir",
        "--save-ckpt-dir",
        dest="save_dir",
        type=str,
        default=None,
        help="Directory to save checkpoint and logs",
    )
    parser.add_argument(
        "--load-ckpt", type=str, default=None, help="Path to checkpoint to load"
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
        "--run-id",
        type=str,
        default=None,
        help="Experiment run ID (defaults to 4-char base62 timestamp hash)",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"results/marc/seed_{args.seed}/stage_{args.stage}"

    load_ckpt = args.load_ckpt
    if load_ckpt is None and args.stage > 0:
        prev_ckpt = f"results/marc/seed_{args.seed}/stage_{args.stage - 1}/model.pt"
        if os.path.exists(prev_ckpt):
            load_ckpt = prev_ckpt

    train(
        algo_name="marc",
        stage_idx=args.stage,
        total_timesteps=args.timesteps,
        load_ckpt_path=load_ckpt,
        save_ckpt_dir=save_dir,
        log_dir=save_dir,
        seed=args.seed,
        use_rust=args.rust,
        run_id=args.run_id,
    )
