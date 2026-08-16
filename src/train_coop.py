"""
CO-OP: Confidence-Oriented Option Pool
Decentralized bottom-up routing across specialized experts via Critic confidence bidding.
"""
import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
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
    """Base Actor-Critic network representing an individual Expert."""

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


class CoopNetwork(nn.Module):
    """Pool of experts routed bottom-up via Critic confidence."""

    def __init__(self, state_dim, num_experts=2):
        super().__init__()
        self.experts = nn.ModuleList([MappoNetwork(state_dim) for _ in range(num_experts)])
        self.num_experts = num_experts

    def get_action_and_value(self, obs, role, mask, state, action=None, expert_idx=None):
        batch_size = obs.shape[0]

        # 1. Decentralized Bidding / Routing
        if expert_idx is None:
            expert_values = []
            for expert in self.experts:
                val = expert.critic(state).squeeze(-1)
                expert_values.append(val)
            
            expert_values = torch.stack(expert_values, dim=1)  # [Batch, num_experts]
            chosen_expert = torch.argmax(expert_values, dim=1)  # [Batch]
        else:
            chosen_expert = expert_idx

        # 2. Execution by Chosen Expert
        actions = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        logprobs = torch.zeros(batch_size, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, device=obs.device)

        for k, expert in enumerate(self.experts):
            mask_k = (chosen_expert == k)
            if not mask_k.any():
                continue

            act_k = action[mask_k] if action is not None else None
            a, lp, ent, v = expert.get_action_and_value(
                obs[mask_k],
                role[mask_k],
                mask[mask_k],
                state[mask_k] if state is not None else None,
                action=act_k,
            )

            actions[mask_k] = a
            logprobs[mask_k] = lp
            entropies[mask_k] = ent
            if v is not None:
                values[mask_k] = v

        return actions, logprobs, entropies, values, chosen_expert


def train(
    algo_name="coop",
    stage_idx=0,
    env_config=None,
    total_timesteps=120_000,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    log_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, "train.log"),
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            force=True,
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

    print(f"Training {algo_name} Stage {stage_idx} on {device}...")
    logging.info(f"Training {algo_name} Stage {stage_idx} on {device}...")

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

    agent = CoopNetwork(state_dim, num_experts=2).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        try:
            agent.load_state_dict(torch.load(load_ckpt_path, map_location=device, weights_only=True))
            print(f"Loaded checkpoint from {load_ckpt_path}")
            logging.info(f"Loaded checkpoint from {load_ckpt_path}")
        except Exception as e:
            print(f"Warning: Could not load full checkpoint ({e}). Training from scratch.")

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)

    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)
    global_episodes = 0
    global_wins = 0

    for update in range(1, num_updates + 1):
        b_obs = {a: torch.zeros((NUM_STEPS, NUM_ENVS, *OBSERVATION_SIZE)).to(device) for a in AGENTS}
        b_role = {a: torch.zeros((NUM_STEPS, NUM_ENVS, N_AGENTS)).to(device) for a in AGENTS}
        b_mask = {a: torch.zeros((NUM_STEPS, NUM_ENVS, ACTION_SPACE_SIZE)).to(device) for a in AGENTS}
        b_actions = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_logprobs = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_rewards = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_values = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        b_dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
        b_states = torch.zeros((NUM_STEPS, NUM_ENVS, state_dim)).to(device)
        b_experts = {a: torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long).to(device) for a in AGENTS}

        # --- ROLLOUT PHASE ---
        for step in range(NUM_STEPS):
            b_states[step] = torch.tensor(next_state, dtype=torch.float32).to(device)
            b_dones[step] = next_done

            actions_dict = {}
            with torch.no_grad():
                stacked = next_obs["_stacked"]
                obs_all = torch.tensor(stacked["observation"], dtype=torch.float32, device=device)
                role_all = torch.tensor(stacked["role_id"], dtype=torch.float32, device=device)
                mask_all = torch.tensor(stacked["action_mask"], dtype=torch.float32, device=device)

                state_rep = b_states[step].repeat(N_AGENTS, 1)
                actions, logprobs, _, values, chosen_expert = agent.get_action_and_value(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    state_rep,
                )

                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                values = values.view(N_AGENTS, NUM_ENVS)
                experts_unflattened = chosen_expert.view(N_AGENTS, NUM_ENVS)

                for i, a in enumerate(AGENTS):
                    b_experts[a][step] = experts_unflattened[i]
                    b_obs[a][step] = obs_all[i]
                    b_role[a][step] = role_all[i]
                    b_mask[a][step] = mask_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            for e in range(NUM_ENVS):
                is_done = terms["scout"][e] or truncs["scout"][e]
                if is_done:
                    global_episodes += 1
                    if infos[e]["scout"].get("win", False):
                        global_wins += 1

            next_state = vec_env.state
            next_done = torch.tensor(terms["scout"] | truncs["scout"], dtype=torch.float32).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(device)

        # --- GAE (ADVANTAGE) CALCULATION ---
        with torch.no_grad():
            stacked_next = next_obs["_stacked"]
            next_obs_all = torch.tensor(stacked_next["observation"], dtype=torch.float32, device=device)
            next_role_all = torch.tensor(stacked_next["role_id"], dtype=torch.float32, device=device)
            next_mask_all = torch.tensor(stacked_next["action_mask"], dtype=torch.float32, device=device)
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
            
            _, _, _, next_val, _ = agent.get_action_and_value(
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

                    delta = b_rewards[a][t] + GAMMA * nextvalues * nextnonterminal - b_values[a][t]
                    b_adv[a][t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

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
                experts_flat = b_experts[a].reshape(-1)

                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    obs_flat,
                    role_flat,
                    mask_flat,
                    b_states_flat,
                    action=action_flat,
                    expert_idx=experts_flat,
                )

                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
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
            win_rate = global_wins / max(1, global_episodes)
            msg = f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Mean Reward: {avg_reward:.3f}"
            print(msg)
            logging.info(msg)

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(win_rate) if "win_rate" in locals() else 0.0,
            "mean_reward": float(avg_reward) if "avg_reward" in locals() else 0.0,
        }
        with open(os.path.join(save_ckpt_dir, "results.json"), "w") as jf:
            json.dump(results, jf, indent=4)
        print(f"Saved checkpoint and results to {save_ckpt_dir}")
        logging.info(f"Saved checkpoint and results to {save_ckpt_dir}")

    vec_env.close()


if __name__ == "__main__":
    train()