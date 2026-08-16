import json
import logging
import os

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
            nn.Linear(64, ACTION_SPACE_SIZE),  # Outputs raw logits for 7 actions
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
    print(f"Training on {device}...")

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

                # 3. Unflatten the outputs back to [4 Agents, 8 Envs]
                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                values = values.view(N_AGENTS, NUM_ENVS)

                # 4. Store in buffers
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

            # Step the environment!
            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            for e in range(NUM_ENVS):
                # Check for termination to update tracking metrics
                is_done = terms["scout"][e] or truncs["scout"][e]

                if is_done:
                    global_episodes += 1
                    if infos[e]["scout"].get("win", False):
                        global_wins += 1
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
            next_val = agent.critic(
                torch.tensor(next_state, dtype=torch.float32).to(device)
            ).squeeze(-1)
            b_adv = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}

            for a in AGENTS:
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_val
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[a][t + 1]

                    # Here is the core of RL: Delta = Reward + Expected_Next_Value - Current_Value
                    delta = (
                        b_rewards[a][t]
                        + GAMMA * nextvalues * nextnonterminal
                        - b_values[a][t]
                    )
                    b_adv[a][t] = lastgaelam = (
                        delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    )

        # --- UPDATE PHASE (PPO) ---
        b_returns = {a: b_adv[a] + b_values[a] for a in AGENTS}

        # Flatten batches
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

                # Normalize advantage
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_flat, role_flat, mask_flat, b_states_flat, action=action_flat
                )

                # PPO Clipping Math
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
            # A win is when the team gets the +15 reward (which > 5)
            win_rate = global_wins / max(1, global_episodes)
            logging.info(  # noqa: LOG015
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Mean Reward: {avg_reward:.3f}"
            )

    # Save checkpoint and results
    if save_ckpt_dir:
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(win_rate) if "win_rate" in locals() else 0.0,
            "mean_reward": float(avg_reward) if "avg_reward" in locals() else 0.0,
        }
        with open(os.path.join(save_ckpt_dir, "results.json"), "w") as jf:
            json.dump(results, jf, indent=4)
        logging.info(  # noqa: LOG015
            f"Saved checkpoint and results to {save_ckpt_dir}"
        )


if __name__ == "__main__":
    train()
