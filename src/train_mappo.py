import json
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from constants import (
    ACTION_SPACE_SIZE,
    AGENTS,
    CLIP_COEF,
    ENT_COEF,
    GAE_LAMBDA,
    GAMMA,
    LR,
    N_AGENTS,
    NUM_ENVS,
    NUM_STEPS,
    OBSERVATION_SIZE,
    UPDATE_EPOCHS,
    VF_COEF,
)
from vec_env import VectorEnv


class MappoNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # Actor sees: 7x7 Grid (49) + Role One-Hot (4) = 53 dims
        actor_in_dim = (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS
        self.actor = nn.Sequential(
            nn.Linear(actor_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE),  # Outputs raw logits for 6 actions
        )

        # Critic sees: Global state (everything)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),  # Outputs a single expected reward (Value)
        )

    def get_action_and_value(self, obs, role, mask, state=None, action=None):
        # 1. Prepare Actor Input
        x_actor = torch.cat([obs.flatten(start_dim=1), role], dim=1)
        logits = self.actor(x_actor)

        # 2. THE MASKING TRICK: Make illegal actions infinitely bad (-1e9)
        masked_logits = logits + ((1.0 - mask) * -1e9)

        # 3. Sample an action
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()

        # 4. Get Critic Value
        value = self.critic(state).squeeze(-1) if state is not None else None

        return action, probs.log_prob(action), probs.entropy(), value


def train(
    algo_name="test",
    stage_idx=0,
    env_config=None,
    total_timesteps=1000,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    log_dir=None,
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
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        file_logger.handlers = [file_handler]
        file_logger.propagate = False

    msg = f"Training {algo_name} Stage {stage_idx} on {device}..."
    console_logger.info(msg)
    if file_logger:
        file_logger.info(msg)

    # Stage 0: 11x11, no guards, role spawn
    if env_config is None:
        env_config = {
            "map_size": (11, 11),
            "guard_count": 0,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 100,
            "spawn_mode": "role",
        }
    vec_env = VectorEnv(NUM_ENVS, config=env_config)
    state_dim = vec_env.state_dim

    agent = MappoNetwork(state_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        agent.load_state_dict(torch.load(load_ckpt_path, map_location=device))
        logging.info(  # noqa: LOG015
            f"Loaded checkpoint from {load_ckpt_path}"
        )

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)

    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)
    global_episodes = 0
    global_wins = 0
    current_env_returns = np.zeros(NUM_ENVS)
    completed_episode_returns = []
    completed_wins = []
    completed_scout_interact = []
    completed_scout_pois = []
    completed_hacker_hack = []
    completed_muscle_neutralize = []
    completed_extractor_loot = []
    completed_agents_at_extract = []

    for update in range(1, num_updates + 1):
        # --- ROLLOUT PHASE ---
        # We store transitions here to learn from them later
        b_obs = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, *OBSERVATION_SIZE)).to(device)
            for a in AGENTS
        }
        b_role = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, N_AGENTS)).to(device) for a in AGENTS
        }
        b_mask = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, ACTION_SPACE_SIZE)).to(device)
            for a in AGENTS
        }
        b_actions = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_logprobs = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_rewards = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_values = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
        b_states = torch.zeros((NUM_STEPS, NUM_ENVS, state_dim)).to(device)

        for step in range(NUM_STEPS):
            b_states[step] = torch.tensor(next_state, dtype=torch.float32).to(device)
            b_dones[step] = next_done

            actions_dict = {}
            with torch.no_grad():
                # 1. ONE fast transfer to the GPU using the pre-stacked arrays from vec_env
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

                # 2. ONE massive forward pass for all 4 agents across 8 envs (Batch Size = 32)
                state_rep = b_states[step].repeat(
                    N_AGENTS, 1
                )  # Match state shape to the 32 agents
                actions, logprobs, _, values = agent.get_action_and_value(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    state_rep,
                )

                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                values = values.view(N_AGENTS, NUM_ENVS)

                for i, a in enumerate(AGENTS):
                    b_obs[a][step] = obs_all[i]
                    b_role[a][step] = role_all[i]
                    b_mask[a][step] = mask_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            # Step the environment!
            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            # Accumulate per-agent average return per env
            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            for e in range(NUM_ENVS):
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
                    current_env_returns[e] = 0.0

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
                    completed_extractor_loot.append(
                        float(scout_info.get("extractor_loot_success", False))
                    )
                    completed_agents_at_extract.append(
                        float(scout_info.get("agents_at_extract", 0))
                    )

            next_state = vec_env.state
            next_done = torch.tensor(
                terms["scout"] | truncs["scout"], dtype=torch.float32
            ).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(
                    device
                )

        # --- GAE (ADVANTAGE) CALCULATION ---
        with torch.no_grad():
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
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)

            _, _, _, next_val = agent.get_action_and_value(
                next_obs_all.flatten(0, 1),
                next_role_all.flatten(0, 1),
                next_mask_all.flatten(0, 1),
                next_state_t.repeat(N_AGENTS, 1),
            )
            next_val = next_val.view(N_AGENTS, NUM_ENVS)

            b_adv = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}
            for i, a in enumerate(AGENTS):
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_val[i]
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[a][t + 1]

                    delta = (
                        b_rewards[a][t]
                        + GAMMA * nextvalues * nextnonterminal
                        - b_values[a][t]
                    )
                    b_adv[a][t] = lastgaelam = (
                        delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    )

        b_returns = {a: b_adv[a] + b_values[a] for a in AGENTS}

        # --- UPDATE PHASE (PPO) ---
        b_states_flat = b_states.reshape(-1, state_dim)
        for _epoch in range(UPDATE_EPOCHS):
            for a in AGENTS:
                obs_flat = b_obs[a].reshape(-1, *OBSERVATION_SIZE)
                role_flat = b_role[a].reshape(-1, N_AGENTS)
                mask_flat = b_mask[a].reshape(-1, ACTION_SPACE_SIZE)
                action_flat = b_actions[a].reshape(-1)
                logprob_flat = b_logprobs[a].reshape(-1)
                adv_flat = b_adv[a].reshape(-1)
                ret_flat = b_returns[a].reshape(-1)

                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_flat,
                    role_flat,
                    mask_flat,
                    b_states_flat,
                    action=action_flat,
                )

                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(
                    ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - ret_flat) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # --- LOGGING ---
        if update % 5 == 0:
            avg_reward = sum(b_rewards[a].mean().item() for a in AGENTS) / N_AGENTS
            win_rate = (
                float(np.mean(completed_wins[-100:])) if completed_wins else 0.0
            )
            mean_episodic_reward = (
                float(np.mean(completed_episode_returns[-100:]))
                if completed_episode_returns
                else 0.0
            )
            scout_tag_rate = (
                float(np.mean(completed_scout_interact[-100:]))
                if completed_scout_interact
                else 0.0
            )
            scout_avg_pois = (
                float(np.mean(completed_scout_pois[-100:]))
                if completed_scout_pois
                else 0.0
            )
            hacker_hack_rate = (
                float(np.mean(completed_hacker_hack[-100:]))
                if completed_hacker_hack
                else 0.0
            )
            muscle_neutralize_rate = (
                float(np.mean(completed_muscle_neutralize[-100:]))
                if completed_muscle_neutralize
                else 0.0
            )
            extractor_loot_rate = (
                float(np.mean(completed_extractor_loot[-100:]))
                if completed_extractor_loot
                else 0.0
            )
            avg_agents_extract = (
                float(np.mean(completed_agents_at_extract[-100:]))
                if completed_agents_at_extract
                else 0.0
            )

            # Console log (clean & compact)
            console_logger.info(
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Step Reward: {avg_reward:.3f}"
            )
            # Detailed file log
            if file_logger:
                file_logger.info(
                    f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Step Reward: {avg_reward:.3f} | Scout Tag Rate: {scout_tag_rate:.2f} (Avg POIs: {scout_avg_pois:.1f}) | Hacker Hack Rate: {hacker_hack_rate:.2f} | Muscle Neutralize Rate: {muscle_neutralize_rate:.2f} | Extractor Loot Rate: {extractor_loot_rate:.2f} | Avg Agents at Extract: {avg_agents_extract:.2f}/4"
                )

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(win_rate) if "win_rate" in locals() else 0.0,
            "lifetime_win_rate": float(global_wins / max(1, global_episodes)),
            "mean_reward": float(mean_episodic_reward)
            if "mean_episodic_reward" in locals()
            else 0.0,
            "mean_step_reward": float(avg_reward) if "avg_reward" in locals() else 0.0,
            "scout_interact_rate": float(scout_tag_rate)
            if "scout_tag_rate" in locals()
            else 0.0,
            "scout_avg_pois_tagged": float(scout_avg_pois)
            if "scout_avg_pois" in locals()
            else 0.0,
            "hacker_hack_rate": float(hacker_hack_rate)
            if "hacker_hack_rate" in locals()
            else 0.0,
            "muscle_neutralize_rate": float(muscle_neutralize_rate)
            if "muscle_neutralize_rate" in locals()
            else 0.0,
            "extractor_loot_rate": float(extractor_loot_rate)
            if "extractor_loot_rate" in locals()
            else 0.0,
            "avg_agents_at_extract": float(avg_agents_extract)
            if "avg_agents_extract" in locals()
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
