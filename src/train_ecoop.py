"""
E-COOP: Evolutionary Confidence-Oriented Option Pool
Combines decentralized Critic confidence bidding with FIM-guided genetic mutation and routing hysteresis.
"""
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
    ECOOP_EVOLUTION_INTERVAL,
    ECOOP_EVOLUTION_START,
    ECOOP_MUTATION_NOISE,
    ECOOP_POOL_SIZE,
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
    """Base Expert Actor-Critic module."""

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


class EcoopNetwork(nn.Module):
    """Dynamic pool of experts with confidence routing, hysteresis, and grace periods."""

    def __init__(self, state_dim, max_experts=ECOOP_POOL_SIZE):
        super().__init__()
        self.experts = nn.ModuleList([MappoNetwork(state_dim) for _ in range(max_experts)])
        self.max_experts = max_experts

    def get_action_and_value(
        self,
        obs,
        role,
        mask,
        state,
        active_experts,
        action=None,
        expert_idx=None,
        previous_expert=None,
        grace_expert=None,
        epsilon=0.05,
    ):
        batch_size = obs.shape[0]

        # 1. Routing Decision
        if expert_idx is not None:
            chosen_expert = expert_idx
        elif grace_expert is not None and grace_expert < active_experts:
            # Routing Grace Period: unconditionally route to child during burn-in
            chosen_expert = torch.full((batch_size,), grace_expert, dtype=torch.long, device=obs.device)
        else:
            expert_values = []
            for k in range(active_experts):
                val = self.experts[k].critic(state).squeeze(-1)
                expert_values.append(val)
            expert_values = torch.stack(expert_values, dim=1)  # [Batch, active_experts]

            # Hybrid Routing Hysteresis: prevent chattering at zero-crossings
            if previous_expert is not None:
                chosen_expert = previous_expert.clone()
                for b in range(batch_size):
                    prev_idx = previous_expert[b].item()
                    prev_val = expert_values[b, prev_idx]
                    best_idx = torch.argmax(expert_values[b]).item()
                    best_val = expert_values[b, best_idx]

                    # Switch threshold: V_rival > V_curr + max(eps_abs, eps_rel * |V_curr|)
                    switch_thresh = prev_val + max(epsilon, epsilon * abs(prev_val.item()))
                    if best_idx != prev_idx and best_val > switch_thresh:
                        chosen_expert[b] = best_idx
            else:
                chosen_expert = torch.argmax(expert_values, dim=1)

        # 2. Execution
        actions = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        logprobs = torch.zeros(batch_size, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, device=obs.device)

        for k in range(active_experts):
            mask_k = (chosen_expert == k)
            if not mask_k.any():
                continue

            act_k = action[mask_k] if action is not None else None
            a, lp, ent, v = self.experts[k].get_action_and_value(
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


def compute_fim_diagonal(expert_model, sample_obs, sample_role, sample_mask, sample_actions):
    """Computes empirical Fisher Information Matrix (FIM) diagonal from policy log-likelihood."""
    fim = {name: torch.zeros_like(param) for name, param in expert_model.named_parameters()}
    expert_model.zero_grad()

    n_samples = min(len(sample_obs), 128)
    for i in range(n_samples):
        o = sample_obs[i:i+1]
        r = sample_role[i:i+1]
        m = sample_mask[i:i+1]
        a = sample_actions[i:i+1]

        x_actor = torch.cat([o.flatten(start_dim=1), r], dim=1)
        logits = expert_model.actor(x_actor)
        masked_logits = logits + ((1.0 - m) * -1e9)
        probs = Categorical(logits=masked_logits)
        
        log_prob = probs.log_prob(a)
        log_prob.backward(retain_graph=True)

        for name, param in expert_model.named_parameters():
            if param.grad is not None:
                fim[name] += (param.grad.data.clone() ** 2) / n_samples
        expert_model.zero_grad()

    return fim


def train(
    algo_name="ecoop",
    stage_idx=0,
    env_config=None,
    total_timesteps=120_000,
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

    agent = EcoopNetwork(state_dim, max_experts=ECOOP_POOL_SIZE).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        try:
            agent.load_state_dict(torch.load(load_ckpt_path, map_location=device, weights_only=True))
            print(f"Loaded checkpoint from {load_ckpt_path}")
            logging.info(  # noqa: LOG015
                f"Loaded checkpoint from {load_ckpt_path}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load full checkpoint ({e}). Training from scratch.")

    active_experts = 1
    grace_updates_remaining = 0
    current_grace_expert = None
    env_previous_expert = torch.zeros(N_AGENTS * NUM_ENVS, dtype=torch.long, device=device)

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)

    num_updates = total_timesteps // (NUM_ENVS * NUM_STEPS)
    global_episodes = 0
    global_wins = 0
    current_env_returns = np.zeros(NUM_ENVS)
    completed_episode_returns = []
    completed_scout_interact = []
    completed_scout_pois = []
    completed_hacker_hack = []
    completed_muscle_neutralize = []
    completed_extractor_loot = []
    completed_agents_at_extract = []

    for update in range(1, num_updates + 1):
        if grace_updates_remaining > 0:
            grace_updates_remaining -= 1
            if grace_updates_remaining == 0:
                current_grace_expert = None

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
                    active_experts=active_experts,
                    previous_expert=env_previous_expert,
                    grace_expert=current_grace_expert,
                )
                env_previous_expert = chosen_expert

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

            # Accumulate per-agent average return per env
            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            for e in range(NUM_ENVS):
                is_done = terms["scout"][e] or truncs["scout"][e]
                if is_done:
                    global_episodes += 1
                    scout_info = infos[e]["scout"]
                    if scout_info.get("win", False):
                        global_wins += 1
                    completed_episode_returns.append(float(current_env_returns[e]))
                    current_env_returns[e] = 0.0

                    completed_scout_interact.append(float(scout_info.get("scout_interact_success", False)))
                    completed_scout_pois.append(float(scout_info.get("scout_pois_tagged", 0)))
                    completed_hacker_hack.append(float(scout_info.get("hacker_hack_success", False)))
                    completed_muscle_neutralize.append(float(scout_info.get("muscle_neutralize_success", False)))
                    completed_extractor_loot.append(float(scout_info.get("extractor_loot_success", False)))
                    completed_agents_at_extract.append(float(scout_info.get("agents_at_extract", 0)))

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
                active_experts=active_experts,
                previous_expert=env_previous_expert,
                grace_expert=current_grace_expert,
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
                    active_experts=active_experts,
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

        # --- EVOLUTIONARY CROSSOVER / FIM MUTATION ---
        do_crossover = (
            update == ECOOP_EVOLUTION_START
            or (update > ECOOP_EVOLUTION_START and (update - ECOOP_EVOLUTION_START) % ECOOP_EVOLUTION_INTERVAL == 0)
        )

        if do_crossover and active_experts < ECOOP_POOL_SIZE:
            # Find best expert by average value prediction
            best_expert = 0
            best_val = -1e9
            all_experts_tensor = torch.stack([b_experts[a] for a in AGENTS])
            all_values_tensor = torch.stack([b_values[a] for a in AGENTS])
            for k in range(active_experts):
                expert_vals = all_values_tensor[all_experts_tensor == k]
                if len(expert_vals) > 0:
                    mean_v = expert_vals.mean().item()
                    if mean_v > best_val:
                        best_val = mean_v
                        best_expert = k

            new_expert = active_experts
            msg = f"Evolutionary Crossover: Cloning Expert {best_expert} (val={best_val:.3f}) to Expert {new_expert}"
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

            # Compute empirical FIM from all rollout samples
            sample_obs = torch.cat(
                [b_obs[a].reshape(-1, *OBSERVATION_SIZE) for a in AGENTS], dim=0
            )
            sample_role = torch.cat(
                [b_role[a].reshape(-1, N_AGENTS) for a in AGENTS], dim=0
            )
            sample_mask = torch.cat(
                [b_mask[a].reshape(-1, ACTION_SPACE_SIZE) for a in AGENTS], dim=0
            )
            sample_acts = torch.cat(
                [b_actions[a].reshape(-1).long() for a in AGENTS], dim=0
            )

            fim = compute_fim_diagonal(
                agent.experts[best_expert],
                sample_obs,
                sample_role,
                sample_mask,
                sample_acts,
            )

            # FIM-scaled inverse mutation
            with torch.no_grad():
                for name, param in agent.experts[new_expert].named_parameters():
                    if name in fim:
                        f_diag = fim[name]
                        # Inverse square root scaling with safety clipping
                        scale = torch.clamp(1.0 / (torch.sqrt(f_diag) + 1e-8), max=10.0)
                        noise = torch.randn_like(param) * scale * ECOOP_MUTATION_NOISE
                        param.add_(noise)

            # Adam Cold-Start: clear stale momentum for the new mutant
            for p in agent.experts[new_expert].parameters():
                if p in optimizer.state:
                    del optimizer.state[p]

            active_experts += 1
            current_grace_expert = new_expert
            grace_updates_remaining = 25

        # --- LOGGING ---
        if update % 5 == 0:
            avg_reward = sum(b_rewards[a].mean().item() for a in AGENTS) / N_AGENTS
            win_rate = global_wins / max(1, global_episodes)
            mean_episodic_reward = (
                float(np.mean(completed_episode_returns[-100:]))
                if completed_episode_returns
                else 0.0
            )
            scout_tag_rate = float(np.mean(completed_scout_interact[-100:])) if completed_scout_interact else 0.0
            scout_avg_pois = float(np.mean(completed_scout_pois[-100:])) if completed_scout_pois else 0.0
            hacker_hack_rate = float(np.mean(completed_hacker_hack[-100:])) if completed_hacker_hack else 0.0
            muscle_neutralize_rate = float(np.mean(completed_muscle_neutralize[-100:])) if completed_muscle_neutralize else 0.0
            extractor_loot_rate = float(np.mean(completed_extractor_loot[-100:])) if completed_extractor_loot else 0.0
            avg_agents_extract = float(np.mean(completed_agents_at_extract[-100:])) if completed_agents_at_extract else 0.0

            # Console log (clean & compact)
            console_logger.info(
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Step Reward: {avg_reward:.3f} | Active Experts: {active_experts}"
            )
            # Detailed file log
            if file_logger:
                file_logger.info(
                    f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Step Reward: {avg_reward:.3f} | Scout Tag Rate: {scout_tag_rate:.2f} (Avg POIs: {scout_avg_pois:.1f}) | Hacker Hack Rate: {hacker_hack_rate:.2f} | Muscle Neutralize Rate: {muscle_neutralize_rate:.2f} | Extractor Loot Rate: {extractor_loot_rate:.2f} | Avg Agents at Extract: {avg_agents_extract:.2f}/4 | Active Experts: {active_experts}"
                )

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(win_rate) if "win_rate" in locals() else 0.0,
            "mean_reward": float(mean_episodic_reward) if "mean_episodic_reward" in locals() else 0.0,
            "mean_step_reward": float(avg_reward) if "avg_reward" in locals() else 0.0,
            "scout_interact_rate": float(scout_tag_rate) if "scout_tag_rate" in locals() else 0.0,
            "scout_avg_pois_tagged": float(scout_avg_pois) if "scout_avg_pois" in locals() else 0.0,
            "hacker_hack_rate": float(hacker_hack_rate) if "hacker_hack_rate" in locals() else 0.0,
            "muscle_neutralize_rate": float(muscle_neutralize_rate) if "muscle_neutralize_rate" in locals() else 0.0,
            "extractor_loot_rate": float(extractor_loot_rate) if "extractor_loot_rate" in locals() else 0.0,
            "avg_agents_at_extract": float(avg_agents_extract) if "avg_agents_extract" in locals() else 0.0,
            "active_experts": active_experts,
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