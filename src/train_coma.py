"""
Counterfactual Multi-Agent Policy Gradients (COMA) for HEIST.
Based on Foerster et al. (AAAI 2018): "Counterfactual Multi-Agent Policy Gradients".
CleanRL single-file philosophy.
"""

import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical

from constants import (
    ACTION_SPACE_SIZE,
    AGENTS,
    CLIP_COEF,
    CURRICULUM_STAGES,
    ENT_COEF,
    GAE_LAMBDA,
    GAMMA,
    GOAL_VECTOR_DIM,
    LR,
    N_AGENTS,
    NUM_ENVS,
    NUM_STEPS,
    OBSERVATION_SIZE,
    UPDATE_EPOCHS,
    VF_COEF,
)
from thermal_guard import check_thermal_guard
from vec_env import VectorEnv


class ComaNetwork(nn.Module):
    """
    Counterfactual Multi-Agent Policy Gradients (COMA) Network.
    - Decentralized Actor: pi(u^a | obs^a, role^a, goal^a)
    - Centralized Counterfactual Critic: Q(s, (., u^{-a}), role^a, goal^a) -> R^|U|
    """

    def __init__(self, state_dim):
        super().__init__()
        # Actor sees: 11x11 local view (121) + Role One-Hot (4) + Goal Vector (2) = 127
        actor_in_dim = (
            (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1])
            + N_AGENTS
            + GOAL_VECTOR_DIM
        )
        self.actor = nn.Sequential(
            nn.Linear(actor_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE),
        )

        # Critic sees: Global State + (N_AGENTS - 1) other agents' one-hot actions + Role + Goal
        critic_in_dim = (
            state_dim
            + ((N_AGENTS - 1) * ACTION_SPACE_SIZE)
            + N_AGENTS
            + GOAL_VECTOR_DIM
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, ACTION_SPACE_SIZE),  # Outputs Q-values for all |U| actions
        )

    def get_action(self, obs, role, mask, goal, action=None):
        x_actor = torch.cat([obs.flatten(start_dim=1), role, goal], dim=1)
        logits = self.actor(x_actor)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), probs.probs

    def get_q_values(self, state, other_actions_onehot, role, goal):
        x_critic = torch.cat([state, other_actions_onehot, role, goal], dim=1)
        return self.critic(x_critic)


def _build_other_actions_onehot(actions_dict, device):
    """
    Builds the (N_AGENTS - 1) one-hot action representations for each agent.
    Returns: dict mapping agent name -> tensor of shape [NUM_ENVS, (N_AGENTS - 1) * ACTION_SPACE_SIZE]
    """
    other_onehots = {}
    for i, a in enumerate(AGENTS):
        others = []
        for j, other_a in enumerate(AGENTS):
            if i != j:
                onehot = F.one_hot(
                    actions_dict[other_a].long(), num_classes=ACTION_SPACE_SIZE
                ).float()
                others.append(onehot)
        other_onehots[a] = torch.cat(others, dim=-1).to(device)
    return other_onehots


def train(
    algo_name="coma",
    stage_idx=0,
    env_config=None,
    total_timesteps=1000,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    log_dir=None,
    seed=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console_logger = logging.getLogger(f"console_{algo_name}_{stage_idx}")
    console_logger.setLevel(logging.INFO)
    console_logger.handlers = [logging.StreamHandler()]
    console_logger.propagate = False

    file_logger = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_logger = logging.getLogger(f"file_{algo_name}_{stage_idx}")
        file_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "train.log"), mode="w"
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        file_logger.handlers = [file_handler]
        file_logger.propagate = False

    msg = f"Training {algo_name} Stage {stage_idx} on {device}..."
    console_logger.info(msg)
    if file_logger:
        file_logger.info(msg)

    if env_config is None:
        env_config = dict(CURRICULUM_STAGES[stage_idx])
    vec_env = VectorEnv(
        NUM_ENVS,
        config=env_config,
        base_seed=seed * 1000 if seed is not None else 0,
    )
    state_dim = vec_env.state_dim

    agent = ComaNetwork(state_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        agent.load_state_dict(torch.load(load_ckpt_path, map_location=device))
        logging.info(f"Loaded checkpoint from {load_ckpt_path}")  # noqa: LOG015

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)

    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)
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

    other_act_dim = (N_AGENTS - 1) * ACTION_SPACE_SIZE

    for update in range(1, num_updates + 1):
        check_thermal_guard()

        # --- ROLLOUT BUFFER ---
        b_obs = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, *OBSERVATION_SIZE)).to(device)
            for a in AGENTS
        }
        b_role = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, N_AGENTS)).to(device)
            for a in AGENTS
        }
        b_mask = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, ACTION_SPACE_SIZE)).to(device)
            for a in AGENTS
        }
        b_goal = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, GOAL_VECTOR_DIM)).to(device)
            for a in AGENTS
        }
        b_actions = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS
        }
        b_logprobs = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS
        }
        b_probs = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, ACTION_SPACE_SIZE)).to(device)
            for a in AGENTS
        }
        b_other_actions = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, other_act_dim)).to(device)
            for a in AGENTS
        }
        b_rewards = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS
        }
        b_q_taken = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS
        }
        b_dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
        b_states = torch.zeros((NUM_STEPS, NUM_ENVS, state_dim)).to(device)

        for step in range(NUM_STEPS):
            b_states[step] = torch.tensor(next_state, dtype=torch.float32).to(
                device
            )
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

                # 1. Sample actions from decentralized actors
                actions, logprobs, _, probs = agent.get_action(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    goals_all.flatten(0, 1),
                )
                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                probs = probs.view(N_AGENTS, NUM_ENVS, ACTION_SPACE_SIZE)

                step_actions_dict = {}
                for i, a in enumerate(AGENTS):
                    b_obs[a][step] = obs_all[i]
                    b_role[a][step] = role_all[i]
                    b_mask[a][step] = mask_all[i]
                    b_goal[a][step] = goals_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_probs[a][step] = probs[i]
                    actions_dict[a] = actions[i].cpu().numpy()
                    step_actions_dict[a] = actions[i]

                # 2. Build other agents' one-hot actions and evaluate Q(s, u)
                other_onehots = _build_other_actions_onehot(
                    step_actions_dict, device
                )
                for a in AGENTS:
                    b_other_actions[a][step] = other_onehots[a]
                    q_vals = agent.get_q_values(
                        b_states[step],
                        other_onehots[a],
                        b_role[a][step],
                        b_goal[a][step],
                    )
                    b_q_taken[a][step] = q_vals.gather(
                        1, b_actions[a][step].long().unsqueeze(1)
                    ).squeeze(1)

            # Step environment
            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            for e in range(NUM_ENVS):
                is_done = terms["scout"][e] or truncs["scout"][e]
                if is_done:
                    global_episodes += 1
                    scout_info = infos[e]["scout"]
                    is_win = bool(scout_info.get("win", False))
                    if is_win:
                        global_wins += 1
                    completed_wins.append(float(is_win))
                    completed_episode_returns.append(
                        float(current_env_returns[e])
                    )
                    interval_wins.append(float(is_win))
                    interval_episode_returns.append(
                        float(current_env_returns[e])
                    )
                    current_env_returns[e] = 0.0

                    interact_succ = float(
                        scout_info.get("scout_interact_success", False)
                    )
                    pois_tagged = float(scout_info.get("scout_pois_tagged", 0))
                    hack_succ = float(
                        scout_info.get("hacker_hack_success", False)
                    )
                    neutralize_succ = float(
                        scout_info.get("muscle_neutralize_success", False)
                    )
                    guards_neutralized = float(
                        scout_info.get("muscle_guards_neutralized", 0)
                    )
                    loot_succ = float(
                        scout_info.get("extractor_loot_success", False)
                    )
                    agents_extract = float(
                        scout_info.get("agents_at_extract", 0)
                    )
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

            next_state = vec_env.state
            next_done = torch.tensor(
                terms["scout"] | truncs["scout"], dtype=torch.float32
            ).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(
                    rewards[a], dtype=torch.float32
                ).to(device)

        # --- COUNTERFACTUAL ADVANTAGE & TD(lambda) CRITIC TARGETS ---
        with torch.no_grad():
            # Estimate next state Q-values for bootstrap
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
            next_state_t = torch.tensor(
                next_state, dtype=torch.float32, device=device
            )

            next_act, _, _, _ = agent.get_action(
                next_obs_all.flatten(0, 1),
                next_role_all.flatten(0, 1),
                next_mask_all.flatten(0, 1),
                next_goals_all.flatten(0, 1),
            )
            next_act = next_act.view(N_AGENTS, NUM_ENVS)
            next_act_dict = {a: next_act[i] for i, a in enumerate(AGENTS)}
            next_other_onehots = _build_other_actions_onehot(
                next_act_dict, device
            )

            next_q_taken = {}
            for i, a in enumerate(AGENTS):
                q_next_vals = agent.get_q_values(
                    next_state_t,
                    next_other_onehots[a],
                    next_role_all[i],
                    next_goals_all[i],
                )
                next_q_taken[a] = q_next_vals.gather(
                    1, next_act[i].long().unsqueeze(1)
                ).squeeze(1)

            # TD(lambda) returns and counterfactual advantage calculation
            b_returns = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}
            b_adv = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}

            for a in AGENTS:
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextq = next_q_taken[a]
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextq = b_q_taken[a][t + 1]

                    # TD-error: delta = r + gamma * Q(s_{t+1}, u_{t+1}) - Q(s_t, u_t)
                    delta = (
                        b_rewards[a][t]
                        + GAMMA * nextq * nextnonterminal
                        - b_q_taken[a][t]
                    )
                    lastgaelam = (
                        delta
                        + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    )
                    b_returns[a][t] = lastgaelam + b_q_taken[a][t]

                    # Counterfactual baseline: b(s, u^{-a}) = sum_{u'} pi(u'|s) * Q(s, (u', u^{-a}))
                    q_vals_t = agent.get_q_values(
                        b_states[t],
                        b_other_actions[a][t],
                        b_role[a][t],
                        b_goal[a][t],
                    )
                    cf_baseline = (b_probs[a][t] * q_vals_t).sum(dim=-1)
                    # Counterfactual advantage: A^a(s, u) = Q(s, (u^a, u^{-a})) - b(s, u^{-a})
                    b_adv[a][t] = b_q_taken[a][t] - cf_baseline

        # --- UPDATE PHASE (PPO / COMA) ---
        b_states_flat = b_states.reshape(-1, state_dim)
        for _epoch in range(UPDATE_EPOCHS):
            for a in AGENTS:
                obs_flat = b_obs[a].reshape(-1, *OBSERVATION_SIZE)
                role_flat = b_role[a].reshape(-1, N_AGENTS)
                mask_flat = b_mask[a].reshape(-1, ACTION_SPACE_SIZE)
                goal_flat = b_goal[a].reshape(-1, GOAL_VECTOR_DIM)
                action_flat = b_actions[a].reshape(-1)
                logprob_flat = b_logprobs[a].reshape(-1)
                other_act_flat = b_other_actions[a].reshape(-1, other_act_dim)
                adv_flat = b_adv[a].reshape(-1)
                ret_flat = b_returns[a].reshape(-1)

                adv_flat = (adv_flat - adv_flat.mean()) / (
                    adv_flat.std() + 1e-8
                )

                # Actor Forward
                _, newlogprob, entropy, newprobs = agent.get_action(
                    obs_flat,
                    role_flat,
                    mask_flat,
                    goal_flat,
                    action=action_flat,
                )

                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(
                    ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic Forward (Q-value on taken actions)
                q_vals_all = agent.get_q_values(
                    b_states_flat, other_act_flat, role_flat, goal_flat
                )
                q_taken_new = q_vals_all.gather(
                    1, action_flat.long().unsqueeze(1)
                ).squeeze(1)

                v_loss = 0.5 * ((q_taken_new - ret_flat) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # --- LOGGING ---
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

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(np.mean(completed_wins[-100:]))
            if completed_wins
            else 0.0,
            "lifetime_win_rate": float(global_wins / max(1, global_episodes)),
            "mean_reward": float(np.mean(completed_episode_returns[-100:]))
            if completed_episode_returns
            else 0.0,
            "avg_episode_steps": float(np.mean(completed_episode_steps[-100:]))
            if completed_episode_steps
            else 0.0,
            "max_stage_steps": int(max_stage_steps)
            if "max_stage_steps" in locals()
            else 150,
            "avg_alarm": float(np.mean(completed_episode_alarms[-100:]))
            if completed_episode_alarms
            else 0.0,
            "stage_alarm_max": float(stage_alarm_max)
            if "stage_alarm_max" in locals()
            else 100.0,
            "scout_interact_rate": float(np.mean(completed_scout_interact[-100:]))
            if completed_scout_interact
            else 0.0,
            "scout_avg_pois_tagged": float(np.mean(completed_scout_pois[-100:]))
            if completed_scout_pois
            else 0.0,
            "hacker_hack_rate": float(np.mean(completed_hacker_hack[-100:]))
            if completed_hacker_hack
            else 0.0,
            "muscle_neutralize_rate": float(np.mean(completed_muscle_neutralize[-100:]))
            if completed_muscle_neutralize
            else 0.0,
            "muscle_avg_guards_neutralized": float(
                np.mean(completed_muscle_guards[-100:])
            )
            if completed_muscle_guards
            else 0.0,
            "total_stage_guards": int(total_stage_guards)
            if "total_stage_guards" in locals()
            else 0,
            "extractor_loot_rate": float(np.mean(completed_extractor_loot[-100:]))
            if completed_extractor_loot
            else 0.0,
            "avg_agents_at_extract": float(np.mean(completed_agents_at_extract[-100:]))
            if completed_agents_at_extract
            else 0.0,
        }
        with open(os.path.join(save_ckpt_dir, "results.json"), "w") as jf:
            json.dump(results, jf, indent=4)
        msg = f"Saved checkpoint and results to {save_ckpt_dir}"
        console_logger.info(msg)
        if file_logger:
            file_logger.info(msg)

    vec_env.close()


if __name__ == "__main__":
    train()
