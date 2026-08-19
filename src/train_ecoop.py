"""
E-COOP: Evolutionary Confidence-Oriented Option Pool
Combines decentralized Critic confidence bidding with FIM-guided genetic mutation and routing hysteresis.
"""
import collections
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
    CURRICULUM_STAGES,
    ECOOP_CROSSOVER_DAMPING,
    ECOOP_CULL_WINDOW_UPDATES,
    ECOOP_EVOLUTION_INTERVAL,
    ECOOP_GRACE_UPDATES,
    ECOOP_HYSTERESIS_EPSILON,
    ECOOP_MUTANT_ENVS,
    ECOOP_MUTATION_NOISE,
    ECOOP_RAMP_UPDATES,
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
from thermal_guard import check_thermal_guard
from vec_env import VectorEnv


class MappoNetwork(nn.Module):
    """
    Individual Expert Actor-Critic module.
    
    Each specialist maintains:
    - Actor MLP: maps local egocentric observation + one-hot role vector -> action logits.
    - Critic MLP: maps global centralized environment state + one-hot role vector -> state value V(s).
    - Running Value Statistics (val_mean, val_var): tracks empirical return distribution
      for Relative Normalized Bidding z-score computation.
    """

    def __init__(self, state_dim):
        super().__init__()
        actor_in_dim = (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS
        critic_in_dim = state_dim + N_AGENTS
        
        # Policy Network: maps local 7x7 view + role ID -> unnormalized action logits
        self.actor = nn.Sequential(
            nn.Linear(actor_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE),
        )
        
        # Centralized Critic Network: maps global state + role ID -> expected return
        self.critic = nn.Sequential(
            nn.Linear(critic_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Buffers for Exponential Moving Average (EMA) of return distribution
        self.register_buffer("val_mean", torch.tensor(0.0))
        self.register_buffer("val_var", torch.tensor(1.0))
        self.register_buffer("val_count", torch.tensor(0.0))

    def update_val_stats(self, values_tensor):
        """
        Updates the running mean and variance of returns observed by this expert.
        Used for normalized z-score bidding across heterogeneous task difficulties.
        """
        with torch.no_grad():
            if values_tensor.numel() > 0:
                batch_m = values_tensor.mean()
                batch_v = values_tensor.var(unbiased=False) if values_tensor.numel() > 1 else torch.tensor(1.0, device=values_tensor.device)
                
                # First update initializes buffers directly
                if self.val_count == 0:
                    self.val_mean.copy_(batch_m)
                    self.val_var.copy_(torch.clamp(batch_v, min=1e-4))
                    self.val_count.copy_(torch.tensor(1.0, device=values_tensor.device))
                else:
                    # Exponential moving average with decay rate alpha = 0.05
                    alpha = 0.05
                    self.val_mean.copy_((1.0 - alpha) * self.val_mean + alpha * batch_m)
                    self.val_var.copy_((1.0 - alpha) * self.val_var + alpha * torch.clamp(batch_v, min=1e-4))

    def get_action_and_value(self, obs, role, mask, state=None, action=None):
        """
        Forward pass for a batch of transitions assigned to this expert.
        Applies invalid action masking (-1e9) prior to categorical sampling.
        """
        x_actor = torch.cat([obs.flatten(start_dim=1), role], dim=1)
        logits = self.actor(x_actor)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        if state is not None:
            x_critic = torch.cat([state, role], dim=1)
            value = self.critic(x_critic).squeeze(-1)
        else:
            value = None
        return action, probs.log_prob(action), probs.entropy(), value


class EcoopNetwork(nn.Module):
    """
    Self-regulating pool of heterogeneous expert modules.
    
    Routes actions based on decentralized Critic confidence bidding, enforces temporal
    consistency via hysteresis, isolates exploration during mutant grace periods, and
    smooths admission via progressive parent-child bidding ramps.
    """

    def __init__(self, state_dim, num_initial_experts=1):
        super().__init__()
        self.state_dim = state_dim
        self.experts = nn.ModuleList([MappoNetwork(state_dim) for _ in range(max(1, num_initial_experts))])

    def add_expert(self):
        """Appends a fresh expert module onto the device of existing parameters and returns its index."""
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        new_mod = MappoNetwork(self.state_dim).to(device)
        self.experts.append(new_mod)
        return len(self.experts) - 1

    def prune_experts(self, survivor_indices):
        """Retains only the surviving expert modules in contiguous order."""
        self.experts = nn.ModuleList([self.experts[i] for i in survivor_indices])

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
        ramp_alphas=None,
        parent_map=None,
        epsilon=ECOOP_HYSTERESIS_EPSILON,
        deterministic=False,
    ):
        """
        Executes routing and policy forward passes across the expert pool.
        
        If expert_idx is explicitly passed (e.g. during PPO policy loss updates), actions
        are evaluated using the exact expert that took the step during rollout.
        Otherwise, Relative Normalized Bidding with Hysteresis selects the best expert per agent.
        """
        batch_size = obs.shape[0]
        x_critic = torch.cat([state, role], dim=1)

        # 1. Relative Normalized Routing Decision
        if expert_idx is not None:
            chosen_expert = expert_idx
        else:
            if grace_expert is not None and grace_expert < active_experts:
                candidate_experts = [k for k in range(active_experts) if k != grace_expert]
            else:
                candidate_experts = list(range(active_experts))

            candidate_tensor = torch.tensor(candidate_experts, dtype=torch.long, device=obs.device)

            expert_bids = []
            for k in candidate_experts:
                raw_val = self.experts[k].critic(x_critic).squeeze(-1)
                mean_k = self.experts[k].val_mean
                var_k = self.experts[k].val_var

                # Progressive Bidding Ramp: smoothly admit fresh mutants without over-optimism whiplash
                if ramp_alphas is not None and k in ramp_alphas and parent_map is not None and k in parent_map:
                    alpha = ramp_alphas[k]
                    parent_idx = parent_map[k]
                    if parent_idx < active_experts:
                        parent_raw = self.experts[parent_idx].critic(x_critic).squeeze(-1)
                        raw_val = parent_raw * (1.0 - alpha) + raw_val * alpha
                        mean_k = self.experts[parent_idx].val_mean * (1.0 - alpha) + mean_k * alpha
                        var_k = self.experts[parent_idx].val_var * (1.0 - alpha) + var_k * alpha

                # Relative Normalized Bidding: z-score against expert's blended running distribution
                std_k = torch.sqrt(var_k) + 1e-6
                norm_bid = (raw_val - mean_k) / std_k
                expert_bids.append(norm_bid)

            expert_bids = torch.stack(expert_bids, dim=1)  # [Batch, len(candidate_experts)]

            # Greedy Argmax with Hybrid Hysteresis for temporal policy coherence (100% GPU Vectorized)
            best_rel_idx = torch.argmax(expert_bids, dim=1)
            best_idx = candidate_tensor[best_rel_idx]

            if previous_expert is not None:
                has_prev = (previous_expert >= 0)
                matches = (previous_expert.unsqueeze(1) == candidate_tensor)
                prev_in_candidates = matches.any(dim=1)
                prev_rel_idx = matches.long().argmax(dim=1)

                prev_bid = torch.gather(expert_bids, 1, prev_rel_idx.unsqueeze(1)).squeeze(1)
                best_bid = torch.gather(expert_bids, 1, best_rel_idx.unsqueeze(1)).squeeze(1)

                switch_thresh = prev_bid + torch.clamp(epsilon * torch.abs(prev_bid), min=epsilon)
                should_switch = (best_idx != previous_expert) & (best_bid > switch_thresh)
                chosen_expert = torch.where(has_prev & prev_in_candidates & ~should_switch, previous_expert, best_idx)
            else:
                chosen_expert = best_idx

            # Option B: Environment Sharding Grace Routing (100% GPU Vectorized)
            # Competitive specialists run on (NUM_ENVS - ECOOP_MUTANT_ENVS) envs; mutant runs on ECOOP_MUTANT_ENVS envs
            if grace_expert is not None and grace_expert < active_experts:
                threshold_env = max(0, NUM_ENVS - ECOOP_MUTANT_ENVS)
                env_ids = torch.arange(batch_size, device=obs.device) % NUM_ENVS
                is_grace_env = env_ids >= threshold_env
                chosen_expert = torch.where(is_grace_env, torch.tensor(grace_expert, device=obs.device, dtype=chosen_expert.dtype), chosen_expert)

        # 2. Execution
        actions = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        logprobs = torch.zeros(batch_size, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, device=obs.device)

        for k in range(active_experts):
            mask_k = (chosen_expert == k)
            if not mask_k.any():
                continue

            act_k = action[mask_k].long() if action is not None else None
            a, lp, ent, v = self.experts[k].get_action_and_value(
                obs[mask_k],
                role[mask_k],
                mask[mask_k],
                state[mask_k] if state is not None else None,
                action=act_k,
            )

            actions[mask_k] = a.long()
            logprobs[mask_k] = lp
            entropies[mask_k] = ent
            if v is not None:
                values[mask_k] = v

        return actions, logprobs, entropies, values, chosen_expert


def compute_fim_diagonal(expert_model, sample_obs, sample_role, sample_mask, sample_actions, max_samples_per_role=128):
    """
    Computes the empirical Fisher Information Matrix (FIM) diagonal from policy log-likelihood.
    
    Uses balanced stratified sampling across all 4 agent roles (Scout, Hacker, Muscle, Extractor)
    to ensure that sensitivity data equally protects combat, stealth, and hacking competencies
    during genetic recombination and mutation.
    """
    fim = {name: torch.zeros_like(param) for name, param in expert_model.named_parameters()}
    expert_model.zero_grad()
    n_total = len(sample_obs)
    if n_total == 0:
        return fim

    # Stratified sampling across all roles (equal representation for Scout, Hacker, Muscle, Extractor)
    samples_per_agent = n_total // N_AGENTS
    selected_indices = []
    for ag_i in range(N_AGENTS):
        ag_start = ag_i * samples_per_agent
        perm = ag_start + torch.randperm(samples_per_agent, device=sample_obs.device)[:max_samples_per_role]
        selected_indices.append(perm)
    selected = torch.cat(selected_indices)
    n_eval = len(selected)

    o = sample_obs[selected]
    r = sample_role[selected]
    m = sample_mask[selected]
    a = sample_actions[selected]

    x_actor = torch.cat([o.flatten(start_dim=1), r], dim=1)
    logits = expert_model.actor(x_actor)
    masked_logits = logits + ((1.0 - m) * -1e9)
    probs = Categorical(logits=masked_logits)
    log_prob = probs.log_prob(a)

    for j in range(n_eval):
        expert_model.zero_grad()
        log_prob[j].backward(retain_graph=True)
        for name, param in expert_model.named_parameters():
            if param.grad is not None:
                fim[name] += (param.grad.data.clone() ** 2) / n_eval
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
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        file_logger.handlers = [file_handler]
        file_logger.propagate = False

    msg = f"Training {algo_name} Stage {stage_idx} on {device}..."
    console_logger.info(msg)
    if file_logger:
        file_logger.info(msg)

    if env_config is None:
        env_config = dict(CURRICULUM_STAGES[stage_idx])
    vec_env = VectorEnv(NUM_ENVS, config=env_config)
    state_dim = vec_env.state_dim

    agent = EcoopNetwork(state_dim, num_initial_experts=1).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    active_experts = 1
    total_spawns = 0
    spawn_history = []
    current_grace_expert = None
    grace_updates_remaining = 0
    mutant_parent_map = {}
    mutant_ramp_remaining = {}
    expert_consecutive_zero_usage = collections.defaultdict(int)
    env_previous_expert = None

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        try:
            ckpt = torch.load(load_ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
            expert_indices = {int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")}
            needed_experts = max(expert_indices) + 1 if expert_indices else 1
            while len(agent.experts) < needed_experts:
                agent.add_expert()
            if len(agent.experts) > needed_experts:
                agent.prune_experts(list(range(needed_experts)))
            agent.load_state_dict(state_dict)
            optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                active_experts = int(ckpt.get("active_experts", len(agent.experts)))
                total_spawns = int(ckpt.get("total_spawns", 0))
                spawn_history = list(ckpt.get("spawn_history", []))
            else:
                active_experts = len(agent.experts)
            msg = f"Loaded checkpoint from {load_ckpt_path} (Active specialists: {active_experts}, total spawns: {total_spawns})"
            print(msg)
            logging.info(msg)  # noqa: LOG015
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load full checkpoint ({e}). Training from scratch.")

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
    last_expert_usage = {}
    last_switch_rate = 0.0

    for update in range(1, num_updates + 1):
        check_thermal_guard()
        if grace_updates_remaining > 0:
            grace_updates_remaining -= 1
            if grace_updates_remaining == 0:
                current_grace_expert = None

        # Compute Progressive Bidding Ramp alphas for all graduating mutants
        current_ramp_alphas = {}
        for k in list(mutant_ramp_remaining.keys()):
            if k == current_grace_expert:
                current_ramp_alphas[k] = 0.0
            else:
                rem = mutant_ramp_remaining[k]
                alpha = 1.0 - (rem / ECOOP_RAMP_UPDATES)
                current_ramp_alphas[k] = alpha
                mutant_ramp_remaining[k] -= 1
                if mutant_ramp_remaining[k] <= 0:
                    del mutant_ramp_remaining[k]

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

        total_switches = 0
        total_switch_opportunities = 0

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
                    ramp_alphas=current_ramp_alphas,
                    parent_map=mutant_parent_map,
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
                    is_win = bool(scout_info.get("win", False))
                    if is_win:
                        global_wins += 1
                    completed_wins.append(float(is_win))
                    completed_episode_returns.append(float(current_env_returns[e]))
                    interval_wins.append(float(is_win))
                    interval_episode_returns.append(float(current_env_returns[e]))
                    current_env_returns[e] = 0.0

                    interact_succ = float(scout_info.get("scout_interact_success", False))
                    pois_tagged = float(scout_info.get("scout_pois_tagged", 0))
                    hack_succ = float(scout_info.get("hacker_hack_success", False))
                    neutralize_succ = float(scout_info.get("muscle_neutralize_success", False))
                    guards_neutralized = float(scout_info.get("muscle_guards_neutralized", 0))
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

                    if env_previous_expert is not None:
                        for ag_i in range(N_AGENTS):
                            env_previous_expert[ag_i * NUM_ENVS + e] = -1

            next_state = vec_env.state
            next_done = torch.tensor(terms["scout"] | truncs["scout"], dtype=torch.float32).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(device)

        # Track usage and switch rate (100% Vectorized)
        experts_stacked = torch.stack([b_experts[a] for a in AGENTS])  # [N_AGENTS, NUM_STEPS, NUM_ENVS]
        all_experts_tensor = experts_stacked.view(-1)
        total_decisions = max(1, all_experts_tensor.numel())
        last_expert_usage = {
            str(k): round(float((all_experts_tensor == k).sum().item()) / total_decisions * 100.0, 1)
            for k in range(active_experts)
        }
        if experts_stacked.shape[1] > 1:
            total_switches = (experts_stacked[:, 1:] != experts_stacked[:, :-1]).sum().item()
            total_switch_opps = experts_stacked[:, 1:].numel()
            last_switch_rate = float(total_switches) / max(1, total_switch_opps)
        else:
            last_switch_rate = 0.0

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

                # Per-Expert Advantage Normalization (100% Mathematical Isolation)
                adv_norm = torch.zeros_like(adv_flat)
                for k in range(active_experts):
                    mask_k = (experts_flat == k)
                    if mask_k.any():
                        adv_k = adv_flat[mask_k]
                        if adv_k.numel() > 1:
                            adv_norm[mask_k] = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-8)
                        else:
                            adv_norm[mask_k] = adv_k - adv_k.mean()
                adv_flat = adv_norm

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

        # Update running value stats for Relative Normalized Bidding ONCE per update across full batch
        all_experts_flat = torch.stack([b_experts[a] for a in AGENTS]).view(-1)
        all_returns_flat = torch.cat([b_returns[a].view(-1) for a in AGENTS])
        for k in range(active_experts):
            mask_k = (all_experts_flat == k)
            if mask_k.any():
                agent.experts[k].update_val_stats(all_returns_flat[mask_k])

        # --- EVOLUTIONARY CROSSOVER / FIM MUTATION & ACTIVE PRUNING ---
        min_updates_remaining = round(
            ECOOP_GRACE_UPDATES + ECOOP_RAMP_UPDATES + (ECOOP_EVOLUTION_INTERVAL / 2.0)
        )
        do_crossover = (
            (update % ECOOP_EVOLUTION_INTERVAL == 0)
            and ((num_updates - update) >= min_updates_remaining)
        )

        all_experts_tensor = torch.stack([b_experts[a] for a in AGENTS])
        for k in range(active_experts):
            if (all_experts_tensor == k).any() or (k == current_grace_expert):
                expert_consecutive_zero_usage[k] = 0
            else:
                expert_consecutive_zero_usage[k] += 1

        if do_crossover:
            # Find best expert and evaluate mean values for all active experts
            all_values_tensor = torch.stack([b_values[a] for a in AGENTS])
            expert_mean_vals = {}
            best_expert = 0
            best_val = -1e9
            for k in range(active_experts):
                expert_vals = all_values_tensor[all_experts_tensor == k]
                if len(expert_vals) > 0:
                    mean_v = expert_vals.mean().item()
                else:
                    # Evaluate actual critic value on current states & roles rather than sentinel
                    with torch.no_grad():
                        sample_roles = torch.cat([b_role[ag].flatten(0, 1) for ag in AGENTS], dim=0)
                        sample_states = torch.cat([b_states.flatten(0, 1) for _ in AGENTS], dim=0)
                        x_critic_eval = torch.cat([sample_states, sample_roles], dim=1)
                        mean_v = agent.experts[k].critic(x_critic_eval).mean().item()
                expert_mean_vals[k] = mean_v
                if mean_v > best_val:
                    best_val = mean_v
                    best_expert = k

            # Option A: Cull extinct experts (inactive for >= ECOOP_CULL_WINDOW_UPDATES)
            culled_indices = [
                k for k in range(active_experts)
                if k != best_expert
                and expert_consecutive_zero_usage[k] >= ECOOP_CULL_WINDOW_UPDATES
                and k != current_grace_expert
            ]

            if culled_indices:
                survivors = [k for k in range(active_experts) if k not in culled_indices]
                cull_names = ", ".join(f"E{c}" for c in culled_indices)
                cull_msg = f"[Evolution Event] Update: {update} | Culled {len(culled_indices)} extinct expert(s) ({cull_names}) inactive for >= {ECOOP_CULL_WINDOW_UPDATES} updates"
                console_logger.info(cull_msg)
                if file_logger:
                    file_logger.info(cull_msg)

                # Prune culled parameters from optimizer state without resetting survivor momentum
                for culled_idx in culled_indices:
                    for p in agent.experts[culled_idx].parameters():
                        if p in optimizer.state:
                            del optimizer.state[p]
                agent.prune_experts(survivors)
                optimizer.param_groups = [{"params": expert.parameters(), "lr": LR} for expert in agent.experts]
                env_previous_expert = None

                if current_grace_expert is not None:
                    if current_grace_expert in survivors:
                        current_grace_expert = survivors.index(current_grace_expert)
                    else:
                        current_grace_expert = None
                        grace_updates_remaining = 0

                survivor_mean_vals = [expert_mean_vals[old_idx] for old_idx in survivors]
                new_zero_usage = collections.defaultdict(int)
                new_parent_map = {}
                new_ramp_remaining = {}

                for new_idx, old_idx in enumerate(survivors):
                    new_zero_usage[new_idx] = expert_consecutive_zero_usage[old_idx]
                    if old_idx in mutant_parent_map and mutant_parent_map[old_idx] in survivors:
                        new_parent_map[new_idx] = survivors.index(mutant_parent_map[old_idx])
                    if old_idx in mutant_ramp_remaining:
                        new_ramp_remaining[new_idx] = mutant_ramp_remaining[old_idx]

                expert_consecutive_zero_usage = new_zero_usage
                mutant_parent_map = new_parent_map
                mutant_ramp_remaining = new_ramp_remaining
                best_expert = survivors.index(best_expert)
                active_experts = len(survivors)
                expert_mean_vals = {i: survivor_mean_vals[i] for i in range(active_experts)}

            # Dynamic Pool Growth: Spawn fresh mutant into newly appended slot
            parent_pool_size = active_experts
            new_expert = agent.add_expert()
            agent.experts[new_expert].val_mean.copy_(agent.experts[best_expert].val_mean)
            agent.experts[new_expert].val_var.copy_(agent.experts[best_expert].val_var)
            agent.experts[new_expert].val_count.copy_(agent.experts[best_expert].val_count)
            optimizer.add_param_group({"params": agent.experts[new_expert].parameters(), "lr": LR})
            active_experts = len(agent.experts)

            total_spawns += 1
            event_record = {
                "update": update,
                "parent_expert": best_expert,
                "child_expert": new_expert,
                "parent_value": float(best_val),
                "replaced_expert": None,
                "active_experts": active_experts,
                "total_spawns": total_spawns,
                "crossover_type": "all_pool_fisher_recombination",
            }
            spawn_history.append(event_record)

            msg = f"[Evolution Event] Update: {update} | Spawned E{new_expert} via All-Pool Fisher Recombination (top E{best_expert}, val={best_val:.3f}) | Active Pool: {active_experts} experts | Total Spawns: {total_spawns}"
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

            # 1. Compute Softmax Fitness Weights across ALL parent experts in the pool
            val_tensor = torch.tensor(
                [expert_mean_vals[k] for k in range(parent_pool_size)],
                dtype=torch.float32,
            )
            fitness_weights = torch.softmax(val_tensor, dim=0).numpy()

            # 2. Extract Rollout Batch Samples
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

            # 3. Compute empirical FIM for EVERY parent expert in the pool
            pool_fims = []
            for k in range(parent_pool_size):
                fim_k = compute_fim_diagonal(
                    agent.experts[k],
                    sample_obs,
                    sample_role,
                    sample_mask,
                    sample_acts,
                )
                pool_fims.append(fim_k)

            # 4. Perform All-Pool Fisher-Weighted Parameter Recombination & Mutation
            with torch.no_grad():
                for name, param in agent.experts[new_expert].named_parameters():
                    weighted_param_sum = torch.zeros_like(param)
                    weight_denom = torch.zeros_like(param)
                    f_combined = torch.zeros_like(param)

                    for k in range(parent_pool_size):
                        w_k = float(fitness_weights[k])
                        param_k = dict(agent.experts[k].named_parameters())[name].data
                        fim_k = pool_fims[k].get(name, torch.zeros_like(param))

                        # Effective Fisher weight = w_k * (F_k + ECOOP_CROSSOVER_DAMPING)
                        eff_weight = w_k * (fim_k + ECOOP_CROSSOVER_DAMPING)
                        weighted_param_sum += eff_weight * param_k
                        weight_denom += eff_weight
                        f_combined += w_k * fim_k

                    # Recombined parameter
                    recombined = weighted_param_sum / (weight_denom + 1e-8)

                    # 5. Geometry-Aware Mutation on Recombined Offspring
                    scale = torch.clamp(
                        1.0 / (torch.sqrt(f_combined) + 1e-8), max=10.0
                    )
                    noise = torch.randn_like(param) * scale * ECOOP_MUTATION_NOISE
                    param.copy_(recombined + noise)

            # 6. Adam Cold-Start: clear stale momentum for the new mutant
            for p in agent.experts[new_expert].parameters():
                if p in optimizer.state:
                    del optimizer.state[p]

            mutant_parent_map[new_expert] = best_expert
            mutant_ramp_remaining[new_expert] = ECOOP_RAMP_UPDATES
            current_grace_expert = new_expert
            grace_updates_remaining = ECOOP_GRACE_UPDATES

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
            avg_steps = float(np.mean(completed_episode_steps[-25:])) if completed_episode_steps else 0.0
            avg_alarm = float(np.mean(completed_episode_alarms[-25:])) if completed_episode_alarms else 0.0
            avg_muscle_guards = float(np.mean(completed_muscle_guards[-25:])) if completed_muscle_guards else 0.0

            scout_tag_rate = float(np.mean(completed_scout_interact[-25:])) if completed_scout_interact else 0.0
            scout_avg_pois = float(np.mean(completed_scout_pois[-25:])) if completed_scout_pois else 0.0
            hacker_hack_rate = float(np.mean(completed_hacker_hack[-25:])) if completed_hacker_hack else 0.0
            muscle_neutralize_rate = float(np.mean(completed_muscle_neutralize[-25:])) if completed_muscle_neutralize else 0.0
            extractor_loot_rate = float(np.mean(completed_extractor_loot[-25:])) if completed_extractor_loot else 0.0
            avg_agents_extract = float(np.mean(completed_agents_at_extract[-25:])) if completed_agents_at_extract else 0.0

            expert_usage_str = ", ".join(f"E{k}: {v}%" for k, v in last_expert_usage.items())

            # Console log (clean & compact)
            console_logger.info(
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/100 | Active Experts: {active_experts}"
            )
            # Detailed file log
            if file_logger:
                file_logger.info(
                    f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/100 | Scout Tag Rate: {scout_tag_rate:.2f} (Avg POIs: {scout_avg_pois:.1f}) | Hacker Hack Rate: {hacker_hack_rate:.2f} | Muscle Neutralize Rate: {muscle_neutralize_rate:.2f} (Avg Guards: {avg_muscle_guards:.1f}/{total_stage_guards}) | Extractor Loot Rate: {extractor_loot_rate:.2f} | Avg Agents at Extract: {avg_agents_extract:.2f}/4 | Active: {active_experts} | Total Spawns: {total_spawns} | Switch Rate: {last_switch_rate * 100:.1f}% | Usage: [{expert_usage_str}]"
                )

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        checkpoint_payload = {
            "model_state": agent.state_dict(),
            "active_experts": active_experts,
            "total_spawns": total_spawns,
            "spawn_history": spawn_history,
        }
        torch.save(checkpoint_payload, os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(np.mean(completed_wins[-100:])) if completed_wins else 0.0,
            "lifetime_win_rate": float(global_wins / max(1, global_episodes)),
            "mean_reward": float(np.mean(completed_episode_returns[-100:])) if completed_episode_returns else 0.0,
            "avg_episode_steps": float(np.mean(completed_episode_steps[-100:])) if completed_episode_steps else 0.0,
            "max_stage_steps": int(max_stage_steps) if "max_stage_steps" in locals() else 150,
            "avg_alarm": float(np.mean(completed_episode_alarms[-100:])) if completed_episode_alarms else 0.0,
            "scout_interact_rate": float(np.mean(completed_scout_interact[-100:])) if completed_scout_interact else 0.0,
            "scout_avg_pois_tagged": float(np.mean(completed_scout_pois[-100:])) if completed_scout_pois else 0.0,
            "hacker_hack_rate": float(np.mean(completed_hacker_hack[-100:])) if completed_hacker_hack else 0.0,
            "muscle_neutralize_rate": float(np.mean(completed_muscle_neutralize[-100:])) if completed_muscle_neutralize else 0.0,
            "muscle_avg_guards_neutralized": float(np.mean(completed_muscle_guards[-100:])) if completed_muscle_guards else 0.0,
            "total_stage_guards": int(total_stage_guards) if "total_stage_guards" in locals() else 0,
            "extractor_loot_rate": float(np.mean(completed_extractor_loot[-100:])) if completed_extractor_loot else 0.0,
            "avg_agents_at_extract": float(np.mean(completed_agents_at_extract[-100:])) if completed_agents_at_extract else 0.0,
            "active_experts": active_experts,
            "total_spawns": total_spawns,
            "expert_switch_rate": float(last_switch_rate),
            "expert_usage_pct": last_expert_usage,
            "spawn_history": spawn_history,
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