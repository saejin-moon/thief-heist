import json
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from constants import (
    ACTION_SPACE_SIZE,
    AFFORDANCE_COEF,
    AGENTS,
    ALARM_MAX,
    ALPHA_ALARM,
    CLIP_COEF,
    ENT_COEF,
    GAMMA,
    GAMMA_CAUSAL,
    LR,
    N_AGENTS,
    NUM_ENVS,
    NUM_STEPS,
    OBSERVATION_SIZE,
    UPDATE_EPOCHS,
    VF_COEF,
)
from vec_env import VectorEnv

# Controls how severely the alarm penalizes macro credit


# CleanRL philosophy: Everything in one file.
class MarcNetwork(nn.Module):
    """
    Standard Actor-Critic network. In MARC, the architecture is flat like MAPPO,
    but the credit assignment (GAE calculation) is profoundly different.
    """

    def __init__(self, state_dim):
        super().__init__()
        actor_in_dim = (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS
        self.actor = nn.Sequential(
            nn.Linear(actor_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_action_and_value(self, obs, role, mask, state=None, action=None):
        x_actor = torch.cat([obs.flatten(start_dim=1), role], dim=1)
        logits = self.actor(x_actor)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
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

    handlers = [logging.StreamHandler()]
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "train.log")))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=handlers,
        force=True,
    )

    logging.info(  # noqa: LOG015
        f"Training {algo_name} Stage {stage_idx} on {device}..."
    )

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

    agent = MarcNetwork(state_dim).to(device)
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

    for update in range(1, num_updates + 1):
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
        b_wins = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.bool).to(device)
        b_states = torch.zeros((NUM_STEPS, NUM_ENVS, state_dim)).to(device)

        # MARC specific buffers
        b_alarms = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
        b_affordances = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS
        }

        for step in range(NUM_STEPS):
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

                state_rep = b_states[step].repeat(N_AGENTS, 1)
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
                    b_obs[a][step], b_role[a][step], b_mask[a][step] = (
                        obs_all[i],
                        role_all[i],
                        mask_all[i],
                    )
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            next_obs_new, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            # Accumulate team total return per env
            step_team_reward = sum(rewards[a] for a in AGENTS)
            current_env_returns += step_team_reward

            for e in range(NUM_ENVS):
                if infos[e]["scout"].get("win", False):
                    b_wins[step, e] = True
                # Check for termination to update tracking metrics
                is_done = terms["scout"][e] or truncs["scout"][e]

                if is_done:
                    global_episodes += 1
                    if infos[e]["scout"].get("win", False):
                        global_wins += 1
                    completed_episode_returns.append(float(current_env_returns[e]))
                    current_env_returns[e] = 0.0

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

            next_state = vec_env.state
            next_done = torch.tensor(
                terms["scout"] | truncs["scout"], dtype=torch.float32
            ).to(device)

            # Global alarm used for Macro Weighting
            alarms_list = [
                infos[e].get("scout", {}).get("alarm", 0.0) for e in range(NUM_ENVS)
            ]
            b_alarms[step] = torch.tensor(alarms_list, dtype=torch.float32).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(
                    device
                )

            next_obs = next_obs_new

        # --- MARC ADVANTAGE CALCULATION ---
        with torch.no_grad():
            next_val = agent.critic(
                torch.tensor(next_state, dtype=torch.float32).to(device)
            ).squeeze(-1)
            b_adv = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}

            # Trajectory win states derive directly from the environment signals.

            win_mask = b_wins.any(dim=0)
            # Apply alarm scaling and outcome factors.
            macro_alarm_factor = torch.exp(-ALPHA_ALARM * (b_alarms / ALARM_MAX))
            macro_outcome = torch.where(win_mask, 1.0, 0.5)  # [NUM_ENVS]
            unshielded_omega = (
                macro_outcome.unsqueeze(0) * macro_alarm_factor
            )  # [NUM_STEPS, NUM_ENVS]

            for a in AGENTS:
                # Shielding protects upstream enablers from downstream failures.
                shield_mask = (~win_mask.unsqueeze(0)) & (b_affordances[a] > 0)
                omega_t = torch.where(shield_mask, macro_alarm_factor, unshielded_omega)

                # Base temporal difference
                deltas = torch.zeros_like(b_rewards[a])
                for t in range(NUM_STEPS):
                    nextnonterminal = 1.0 - (
                        next_done if t == NUM_STEPS - 1 else b_dones[t + 1]
                    )
                    nextvalues = next_val if t == NUM_STEPS - 1 else b_values[a][t + 1]
                    deltas[t] = (
                        b_rewards[a][t]
                        + GAMMA * nextvalues * nextnonterminal
                        - b_values[a][t]
                    )

                # Micro Credit: Base TD + Affordance delta
                micro_credit = deltas + (b_affordances[a] * AFFORDANCE_COEF)
                immediate_marc = micro_credit * omega_t

                # Retroactive causal trace propagation.
                retro_trace = torch.zeros(NUM_ENVS).to(device)
                for t in reversed(range(NUM_STEPS)):
                    retro_trace = (
                        immediate_marc[t]
                        + GAMMA_CAUSAL * (1.0 - b_dones[t]) * retro_trace
                    )
                    b_adv[a][t] = retro_trace

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
                    obs_flat, role_flat, mask_flat, b_states_flat, action=action_flat
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

        if update % 5 == 0:
            avg_reward = sum(b_rewards[a].mean().item() for a in AGENTS) / N_AGENTS
            win_rate = global_wins / max(1, global_episodes)
            mean_episodic_reward = (
                float(np.mean(completed_episode_returns[-100:]))
                if completed_episode_returns
                else 0.0
            )
            logging.info(  # noqa: LOG015
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Step Reward: {avg_reward:.3f}"
            )

    # Save checkpoint and results
    if save_ckpt_dir:
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(win_rate) if "win_rate" in locals() else 0.0,
            "mean_reward": float(mean_episodic_reward) if "mean_episodic_reward" in locals() else 0.0,
            "mean_step_reward": float(avg_reward) if "avg_reward" in locals() else 0.0,
        }
        with open(os.path.join(save_ckpt_dir, "results.json"), "w") as jf:
            json.dump(results, jf, indent=4)
        logging.info(  # noqa: LOG015
            f"Saved checkpoint and results to {save_ckpt_dir}"
        )


if __name__ == "__main__":
    train()
