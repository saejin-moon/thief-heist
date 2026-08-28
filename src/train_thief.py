"""
THIEF: Targeted Hysteresis-routed Incubated Evolution via Fisher-geometry.
CleanRL-style implementation of decentralized value-bidding multi-agent RL with
dynamic 16-environment sharding, deficit-triggered dynamic spawning, failure-targeted
gradient mutation, all-pool Fisher information geometry recombination, FIFO incubation
queue with 50% environment floor guarantee, and strict 50-update post-grace cooldown.
"""

import collections
import json
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from constants import *
from thermal_guard import check_thermal_guard
from vec_env import VectorEnv


class ManagerNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        in_dim = state_dim + N_AGENTS
        self.actor_mean = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, THIEF_HER_GOAL_DIM),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, THIEF_HER_GOAL_DIM))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_action_and_value(self, state, role, action=None):
        x = torch.cat([state, role], dim=1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            action = torch.clamp(action, -1.0, 1.0)

        value = self.critic(state).squeeze(-1)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value


class MappoActor(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = (
            (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS + THIEF_HER_GOAL_DIM
        )
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE),
        )

    def forward(self, x):
        return self.network(x)


class MappoCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        in_dim = state_dim + N_AGENTS + THIEF_HER_GOAL_DIM
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


class MappoNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.actor = MappoActor()
        self.critic = MappoCritic(state_dim)

    def get_action_and_value(
        self, obs, role, mask, goal, state=None, action=None, deterministic=False
    ):
        x_actor = torch.cat([obs.flatten(1), role, goal], dim=1)
        logits = self.actor(x_actor)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = masked_logits.argmax(dim=-1)
            else:
                action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        value = None
        if state is not None:
            x_critic = torch.cat([state, role, goal], dim=1)
            value = self.critic(x_critic).squeeze(-1)

        return action, log_prob, entropy, value


class ThiefNetwork(nn.Module):
    """
    Modular Specialist Pool with HER Sub-Goal Navigation for THIEF.
    Routes actions based on decentralized Critic confidence bidding with dynamic
    temporal hysteresis, supports dynamic 16-environment sharding, and manages
    dynamic addition and culling of specialist sub-networks conditioned on sub-goals.
    """

    def __init__(self, state_dim, num_initial_experts=THIEF_INITIAL_EXPERTS):
        super().__init__()
        self.state_dim = state_dim
        self.manager = ManagerNetwork(state_dim)
        self.experts = nn.ModuleList(
            [MappoNetwork(state_dim) for _ in range(max(1, num_initial_experts))]
        )

    def get_manager_action_and_value(self, state, role, action=None):
        return self.manager.get_action_and_value(state, role, action)

    def add_expert(self):
        """Appends a fresh expert module onto the device of existing parameters and returns its index."""
        device = (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )
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
        goal,
        state,
        active_experts,
        action=None,
        expert_idx=None,
        previous_expert=None,
        grace_experts=None,
        dormant_experts=None,
        epsilon=THIEF_HYSTERESIS_EPSILON,
        deterministic=False,
        num_envs=THIEF_NUM_ENVS,
    ):
        """
        Executes routing and policy forward passes across the expert pool conditioned on sub-goals.
        Implements Dynamic Environment Sharding & Dormancy Freezing:
        - When grace_experts is empty: 100% of environments run competitive bidding across active, non-dormant experts.
        - When M mutants incubate: each occupies 2 dedicated sandbox envs (max 8 envs),
          leaving all remaining (16 - 2M) envs for general competitive bidding.
        - Dormant experts are excluded from candidate bidding without altering expert tensor indices.
        """
        batch_size = obs.shape[0]
        x_critic = torch.cat([state, role, goal], dim=1) if state is not None else None

        # 1. Routing Selection Phase
        if expert_idx is not None:
            chosen_expert = expert_idx
        else:
            # Parse incubating grace experts (supports list/set or single index)
            if grace_experts is not None:
                grace_list = (
                    list(grace_experts)
                    if isinstance(grace_experts, (list, tuple, set))
                    else [grace_experts]
                )
                grace_set = {g for g in grace_list if g < active_experts}
            else:
                grace_list = []
                grace_set = set()

            dormant_set = set(dormant_experts) if dormant_experts is not None else set()

            # Candidate experts for general competitive bidding (active, non-grace, non-dormant)
            candidate_experts = [
                k
                for k in range(active_experts)
                if k not in grace_set and k not in dormant_set
            ]
            if not candidate_experts:
                candidate_experts = [
                    k for k in range(active_experts) if k not in dormant_set
                ] or list(range(active_experts))

            if len(candidate_experts) == 1:
                best_idx = torch.full(
                    (batch_size,),
                    candidate_experts[0],
                    dtype=torch.long,
                    device=obs.device,
                )
            else:
                bids = torch.stack(
                    [self.experts[k].critic(x_critic) for k in candidate_experts],
                    dim=-1,
                ).squeeze(1)

                best_candidate_idx = bids.argmax(dim=-1)
                best_idx = torch.tensor(candidate_experts, device=obs.device)[
                    best_candidate_idx
                ]

            if previous_expert is not None:
                has_prev = (previous_expert >= 0) & (previous_expert < active_experts)
                prev_in_candidates = torch.tensor(
                    [p.item() in candidate_experts for p in previous_expert],
                    device=obs.device,
                    dtype=torch.bool,
                )

                v_bids_all = torch.stack(
                    [self.experts[k].critic(x_critic) for k in range(active_experts)],
                    dim=-1,
                ).squeeze(1)

                v_prev = v_bids_all.gather(
                    1, previous_expert.clamp(0, active_experts - 1).unsqueeze(1)
                ).squeeze(1)
                v_best = v_bids_all.gather(1, best_idx.unsqueeze(1)).squeeze(1)

                dynamic_barrier = torch.max(
                    torch.full_like(v_prev, epsilon),
                    epsilon * torch.abs(v_prev),
                )
                should_switch = v_best > (v_prev + dynamic_barrier)

                chosen_expert = torch.where(
                    has_prev & prev_in_candidates & ~should_switch,
                    previous_expert,
                    best_idx,
                )
            else:
                chosen_expert = best_idx

            # Dynamic Sandbox Incubation Routing across 16 environments
            if grace_list:
                env_ids = torch.arange(batch_size, device=obs.device) % num_envs
                num_active_mutants = min(THIEF_MAX_SANDBOX_EXPERTS, len(grace_list))
                total_sandbox_envs = num_active_mutants * THIEF_ENVS_PER_MUTANT
                start_sandbox_env = num_envs - total_sandbox_envs

                for m_idx, g_exp in enumerate(grace_list[:num_active_mutants]):
                    if g_exp >= active_experts:
                        continue
                    m_start = start_sandbox_env + (m_idx * THIEF_ENVS_PER_MUTANT)
                    m_end = m_start + THIEF_ENVS_PER_MUTANT
                    is_m_env = (env_ids >= m_start) & (env_ids < m_end)
                    chosen_expert = torch.where(
                        is_m_env,
                        torch.tensor(
                            g_exp, device=obs.device, dtype=chosen_expert.dtype
                        ),
                        chosen_expert,
                    )

        # 2. Execution Phase
        actions = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        logprobs = torch.zeros(batch_size, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, device=obs.device)

        for k in range(active_experts):
            mask_k = chosen_expert == k
            if not mask_k.any():
                continue

            act_k = action[mask_k].long() if action is not None else None
            a, lp, ent, v = self.experts[k].get_action_and_value(
                obs[mask_k],
                role[mask_k],
                mask[mask_k],
                goal[mask_k],
                state[mask_k] if state is not None else None,
                action=act_k,
                deterministic=deterministic,
            )
            actions[mask_k] = a
            logprobs[mask_k] = lp
            entropies[mask_k] = ent
            if v is not None:
                values[mask_k] = v

        return actions, logprobs, entropies, values, chosen_expert


def update_optimizer_params(old_optimizer, agent, lr=LR):
    """
    Preserves Adam's 1st moment (exp_avg) and 2nd moment (exp_avg_sq) for all surviving
    expert parameters across evolutionary events, and cleanly zero-initializes new mutant parameters.
    """
    new_optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    if old_optimizer is not None:
        for p in agent.parameters():
            if p in old_optimizer.state:
                new_optimizer.state[p] = old_optimizer.state[p]
    return new_optimizer


def compute_fim_diagonal(
    agent, expert_idx, b_obs, b_role, b_mask, b_actions, b_goal, device
):
    """
    Computes the exact diagonal of the empirical Fisher Information Matrix (FIM)
    strictly for Actor parameters across 100% of the rollout buffer transitions.
    """
    actor = agent.experts[expert_idx].actor
    x_actor = torch.cat([b_obs.flatten(1), b_role, b_goal], dim=1).to(device)
    b_mask_dev = b_mask.to(device)
    b_actions_dev = b_actions.to(device)
    total_samples = x_actor.shape[0]

    fisher_diag = [
        torch.zeros_like(p, device=device)
        for p in actor.parameters()
        if p.requires_grad
    ]

    chunk_size = 256
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        sub_x = x_actor[start:end]
        sub_mask = b_mask_dev[start:end]
        sub_actions = b_actions_dev[start:end]

        logits = actor(sub_x)
        masked_logits = logits + ((1.0 - sub_mask) * -1e9)
        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(sub_actions)

        for i in range(len(sub_actions)):
            actor.zero_grad()
            log_probs[i].backward(retain_graph=(i < len(sub_actions) - 1))
            with torch.no_grad():
                for f_p, p in zip(
                    fisher_diag,
                    [p for p in actor.parameters() if p.requires_grad],
                ):
                    if p.grad is not None:
                        f_p.add_(p.grad.data.pow(2))

    with torch.no_grad():
        for f_p in fisher_diag:
            f_p.div_(total_samples)

    return fisher_diag


def check_deficit_trigger(
    agent,
    active_experts,
    b_obs,
    b_role,
    b_state,
    b_goal,
    update,
    last_grace_end_update,
    active_sandbox_mutants,
    incubation_queue,
    device,
    dormant_experts=None,
):
    """
    Online rollout deficit detection: triggers only when:
    1. Warmup period has completed (update > THIEF_WARMUP_UPDATES).
    2. Zero mutants are in training/grace AND zero mutants in incubation queue.
    3. Exactly THIEF_POST_GRACE_COOLDOWN_UPDATES have elapsed since the last mutant graduated.
    4. Active expert pool capacity has not reached maximum.
    5. >20% of transitions exhibit confidence deficit (max_k V_k < THIEF_DEFICIT_THRESHOLD).
    """
    if update <= THIEF_WARMUP_UPDATES:
        return False, None
    if active_sandbox_mutants or incubation_queue:
        return False, None
    if (update - last_grace_end_update) < THIEF_POST_GRACE_COOLDOWN_UPDATES:
        return False, None

    dormant_set = set(dormant_experts) if dormant_experts is not None else set()
    active_bidding_count = active_experts - len(dormant_set)
    if active_bidding_count >= THIEF_MAX_EXPERTS:
        return False, None

    non_dormant = [k for k in range(active_experts) if k not in dormant_set]
    if not non_dormant:
        non_dormant = list(range(active_experts))

    x_critic = torch.cat([b_state, b_role, b_goal], dim=1).to(device)
    with torch.no_grad():
        bids = torch.stack(
            [agent.experts[k].critic(x_critic) for k in non_dormant],
            dim=-1,
        ).squeeze(1)
        v_max = bids.max(dim=-1).values

    deficit_mask = v_max < THIEF_DEFICIT_THRESHOLD
    deficit_count = deficit_mask.sum().item()
    deficit_ratio = deficit_count / float(len(v_max))

    if deficit_ratio >= THIEF_DEFICIT_RATIO_TRIGGER and deficit_count >= 32:
        return True, deficit_mask

    return False, None


def compute_deficit_gradient(
    recombined_actor,
    b_obs,
    b_role,
    b_mask,
    b_actions,
    b_goal,
    b_returns,
    b_values,
    deficit_mask,
    device,
):
    """
    Computes the normalized policy deficit gradient strictly on the subset of
    transitions where all active experts failed to output confident value bids.
    """
    sub_obs = b_obs[deficit_mask].to(device)
    sub_role = b_role[deficit_mask].to(device)
    sub_mask = b_mask[deficit_mask].to(device)
    sub_actions = b_actions[deficit_mask].to(device)
    sub_goal = b_goal[deficit_mask].to(device)
    sub_returns = b_returns[deficit_mask].to(device)
    sub_values = b_values[deficit_mask].to(device)

    x_actor = torch.cat([sub_obs.flatten(1), sub_role, sub_goal], dim=1)
    logits = recombined_actor(x_actor)
    masked_logits = logits + ((1.0 - sub_mask) * -1e9)
    dist = Categorical(logits=masked_logits)
    log_probs = dist.log_prob(sub_actions)

    deficit_adv = sub_returns - sub_values
    deficit_loss = -(log_probs * deficit_adv).mean()

    recombined_actor.zero_grad()
    deficit_loss.backward()

    deficit_grads = []
    with torch.no_grad():
        for p in recombined_actor.parameters():
            if p.grad is not None:
                g = p.grad.data.clone()
                g_norm = g.norm() + 1e-8
                deficit_grads.append(g / g_norm)
            else:
                deficit_grads.append(torch.zeros_like(p))

    return deficit_grads


def recombine_and_mutate_thief(
    agent,
    active_experts,
    b_obs,
    b_role,
    b_mask,
    b_actions,
    b_goal,
    b_returns,
    b_values,
    b_state,
    deficit_mask,
    device,
    dormant_experts=None,
):
    """
    Executes THIEF Evolutionary Operator:
    1. Multi-parent empirical return fitness weighting across active non-dormant parents.
    2. All-Pool Full-Buffer Fisher Information Matrix evaluation on Actors.
    3. Geometry-aware Fisher crossover across active parents.
    4. Recombined Critic initialization (clean pool baseline, no E0 carbon copy).
    5. Deficit-directed gradient mutation on failure transitions + Riemannian noise.
    6. Structured Orthogonal Antithetic Sampling (capped at THIEF_MAX_SANDBOX_EXPERTS = 4).
    """
    dormant_set = set(dormant_experts) if dormant_experts is not None else set()
    parent_candidates = [k for k in range(active_experts) if k not in dormant_set]
    if not parent_candidates:
        parent_candidates = list(range(active_experts))

    x_critic = torch.cat([b_state, b_role, b_goal], dim=1).to(device)
    with torch.no_grad():
        parent_values = [
            agent.experts[k].critic(x_critic).mean().item() for k in parent_candidates
        ]

    val_tensor = torch.tensor(parent_values, device=device)
    val_norm = val_tensor - val_tensor.max()
    weights = torch.softmax(val_norm / 1.0, dim=0).cpu().numpy()

    # Compute FIM for each active parent Actor
    fims = [
        compute_fim_diagonal(agent, k, b_obs, b_role, b_mask, b_actions, b_goal, device)
        for k in parent_candidates
    ]

    # Recombine Actor Parameters
    actor_params = [
        [p.data for p in agent.experts[k].actor.parameters()] for k in parent_candidates
    ]
    recombined_actor_params = []
    effective_fishers = []
    damping = THIEF_CROSSOVER_DAMPING

    for p_idx in range(len(actor_params[0])):
        weighted_p_sum = torch.zeros_like(actor_params[0][p_idx])
        weight_sum = torch.zeros_like(actor_params[0][p_idx])
        eff_f = torch.zeros_like(actor_params[0][p_idx])

        for i, k in enumerate(parent_candidates):
            w = weights[i]
            f = fims[i][p_idx]
            eff_f += w * f
            inv_var = w * (f + damping)
            weighted_p_sum += inv_var * actor_params[i][p_idx]
            weight_sum += inv_var

        recomb_p = weighted_p_sum / (weight_sum + 1e-8)
        recombined_actor_params.append(recomb_p)
        effective_fishers.append(eff_f)

    # Recombine Critic Parameters (clean weighted average across parent pool)
    critic_params = [
        [p.data for p in agent.experts[k].critic.parameters()]
        for k in parent_candidates
    ]
    recombined_critic_params = []
    for p_idx in range(len(critic_params[0])):
        w_crit = torch.zeros_like(critic_params[0][p_idx])
        for i, k in enumerate(parent_candidates):
            w_crit += weights[i] * critic_params[i][p_idx]
        recombined_critic_params.append(w_crit)

    # Build temporary recombined Actor to compute deficit gradient
    temp_actor = MappoActor().to(device)
    for p, r_p in zip(temp_actor.parameters(), recombined_actor_params):
        p.data.copy_(r_p)

    deficit_grads = None
    if deficit_mask is not None and deficit_mask.sum().item() >= 32:
        deficit_grads = compute_deficit_gradient(
            temp_actor,
            b_obs,
            b_role,
            b_mask,
            b_actions,
            b_goal,
            b_returns,
            b_values,
            deficit_mask,
            device,
        )

    # Structured Orthogonal Antithetic Sampling capped at THIEF_MAX_SANDBOX_EXPERTS (4)
    num_mutants = min(max(2, len(parent_candidates)), THIEF_MAX_SANDBOX_EXPERTS)
    child_indices = []

    # Exploration temperature spectrum per mutant:
    # M0: Gradient follower (+z0, low noise, high grad)
    # M1: Antithetic mirror 1 (-z0, medium noise, medium grad)
    # M2: Orthogonal explorer (+z1, medium noise, medium grad)
    # M3: Broad basin explorer (-z1, high noise, low grad)
    grad_scales = [0.08, 0.05, 0.05, 0.02]
    noise_scales = [0.015, 0.030, 0.030, 0.060]

    all_child_actor_params = [[] for _ in range(num_mutants)]

    for p_idx, (r_p, eff_f) in enumerate(
        zip(recombined_actor_params, effective_fishers)
    ):
        f_safe = torch.clamp(eff_f, min=1e-6)
        inv_f_sqrt = 1.0 / torch.sqrt(f_safe + 1e-4)
        inv_f_scaled = inv_f_sqrt / (inv_f_sqrt.mean() + 1e-8)
        inv_f_damped = torch.clamp(inv_f_scaled, max=10.0)

        # Base orthogonal noise vectors z0, z1
        z0 = torch.randn_like(r_p)
        z0 = z0 / (z0.norm() + 1e-8) * np.sqrt(z0.numel())

        raw_z1 = torch.randn_like(r_p)
        proj = (z0 * raw_z1).sum() / ((z0 * z0).sum() + 1e-8)
        z1 = raw_z1 - proj * z0
        z1 = z1 / (z1.norm() + 1e-8) * np.sqrt(z1.numel())

        dir_vectors = [z0, -z0, z1, -z1]
        g_p = deficit_grads[p_idx] if deficit_grads is not None else None

        for m in range(num_mutants):
            vec = dir_vectors[m % len(dir_vectors)]
            noise = vec * noise_scales[m % len(noise_scales)] * inv_f_damped
            mutated_p = r_p + noise

            if g_p is not None:
                mutated_p = mutated_p - (grad_scales[m % len(grad_scales)] * g_p)

            all_child_actor_params[m].append(mutated_p)

    for m in range(num_mutants):
        c_idx = agent.add_expert()
        for p, c_p in zip(
            agent.experts[c_idx].actor.parameters(), all_child_actor_params[m]
        ):
            p.data.copy_(c_p)
        for p, c_p in zip(
            agent.experts[c_idx].critic.parameters(), recombined_critic_params
        ):
            p.data.copy_(c_p)
        child_indices.append(c_idx)

    best_parent_cand_idx = int(np.argmax(parent_values))
    best_parent_idx = parent_candidates[best_parent_cand_idx]
    best_parent_val = float(parent_values[best_parent_cand_idx])

    return child_indices, weights, best_parent_idx, best_parent_val


def train(
    algo_name="thief",
    stage_idx=0,
    env_config=None,
    total_timesteps=200_000,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    log_dir=None,
    seed=0,
):
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = THIEF_NUM_ENVS

    console_logger = logging.getLogger(f"console_{algo_name}_{stage_idx}_{seed}")
    console_logger.setLevel(logging.INFO)
    console_logger.handlers = [logging.StreamHandler()]
    console_logger.propagate = False

    file_logger = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_logger = logging.getLogger(f"file_{algo_name}_{stage_idx}_{seed}")
        file_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        file_logger.handlers = [file_handler]
        file_logger.propagate = False

    msg = (
        f"Training {algo_name} Stage {stage_idx} (Seed {seed}) on {device} ("
        f"16 Envs | HER Sub-Goal Navigation | Dynamic Sharding | Pool Doubling | FIFO Queue | 30-Update Post-Grace Cooldown)..."
    )
    console_logger.info(msg)
    if file_logger:
        file_logger.info(msg)

    if env_config is None:
        env_config = dict(CURRICULUM_STAGES[stage_idx])
    vec_env = VectorEnv(
        num_envs, config=env_config, base_seed=seed * 1000 if seed is not None else 0
    )
    state_dim = vec_env.state_dim
    map_w, map_h = env_config.get("map_size", (15, 15))
    map_diag = np.sqrt(map_w**2 + map_h**2)

    agent = ThiefNetwork(state_dim, num_initial_experts=THIEF_INITIAL_EXPERTS).to(
        device
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    active_experts = THIEF_INITIAL_EXPERTS
    total_spawns = 0
    spawn_history = []

    # Queue, Dynamic Incubation & Dormancy Tracking
    active_sandbox_mutants = {}  # Dict mapping active mutant_idx -> updates_completed
    incubation_queue = []  # FIFO list of waiting mutant_idx
    last_grace_end_update = 0  # Timestamp of when last mutant finished grace
    expert_consecutive_zero_usage = collections.defaultdict(int)
    dormant_experts = set()  # Set of frozen inactive expert indices
    env_previous_expert = None

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        try:
            ckpt = torch.load(load_ckpt_path, map_location=device, weights_only=False)
            state_dict = (
                ckpt["model_state"]
                if (isinstance(ckpt, dict) and "model_state" in ckpt)
                else ckpt
            )
            expert_indices = {
                int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
            }
            needed_experts = (
                max(expert_indices) + 1 if expert_indices else THIEF_INITIAL_EXPERTS
            )
            while len(agent.experts) < needed_experts:
                agent.add_expert()
            if len(agent.experts) > needed_experts:
                agent.prune_experts(list(range(needed_experts)))
            agent.load_state_dict(state_dict)
            optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                active_experts = int(ckpt.get("active_experts", len(agent.experts)))
                dormant_experts = set(ckpt.get("dormant_experts", []))
                total_spawns = int(ckpt.get("total_spawns", 0))
                spawn_history = list(ckpt.get("spawn_history", []))
            else:
                active_experts = len(agent.experts)

            # If loaded checkpoint has uncompacted dormant experts, prune them immediately
            surviving_indices = [
                k for k in range(active_experts) if k not in dormant_experts
            ]
            if len(surviving_indices) < active_experts and len(surviving_indices) >= 1:
                agent.prune_experts(surviving_indices)
                optimizer = update_optimizer_params(optimizer, agent, lr=LR)
                active_experts = len(surviving_indices)
                dormant_experts.clear()

            active_bidding = active_experts - len(dormant_experts)
            msg = (
                f"Loaded checkpoint from {load_ckpt_path} "
                f"(Active bidding specialists: {active_bidding}/{active_experts}, total spawns: {total_spawns})"
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load full checkpoint ({e}). Initializing fresh.")

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(num_envs).to(device)

    num_updates = total_timesteps // (num_envs * NUM_STEPS)
    num_macro_steps = NUM_STEPS // THIEF_MACRO_HORIZON
    global_episodes = 0
    global_wins = 0
    current_env_returns = np.zeros(num_envs)
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

    last_expert_usage = {}
    last_switch_rate = 0.0
    last_her_reach_rate = 0.0
    last_her_avg_dist = 0.0

    for update in range(1, num_updates + 1):
        check_thermal_guard()

        # Worker Buffers
        b_obs = {
            a: torch.zeros((NUM_STEPS, num_envs, *OBSERVATION_SIZE)).to(device)
            for a in AGENTS
        }
        b_role = {
            a: torch.zeros((NUM_STEPS, num_envs, N_AGENTS)).to(device) for a in AGENTS
        }
        b_mask = {
            a: torch.zeros((NUM_STEPS, num_envs, ACTION_SPACE_SIZE)).to(device)
            for a in AGENTS
        }
        b_goal = {
            a: torch.zeros((NUM_STEPS, num_envs, THIEF_HER_GOAL_DIM)).to(device)
            for a in AGENTS
        }
        b_actions = {a: torch.zeros((NUM_STEPS, num_envs)).to(device) for a in AGENTS}
        b_logprobs = {a: torch.zeros((NUM_STEPS, num_envs)).to(device) for a in AGENTS}
        b_rewards = {a: torch.zeros((NUM_STEPS, num_envs)).to(device) for a in AGENTS}
        b_values = {a: torch.zeros((NUM_STEPS, num_envs)).to(device) for a in AGENTS}
        b_dones = torch.zeros((NUM_STEPS, num_envs)).to(device)
        b_states = torch.zeros((NUM_STEPS, num_envs, state_dim)).to(device)
        b_experts = {
            a: torch.zeros((NUM_STEPS, num_envs), dtype=torch.long).to(device)
            for a in AGENTS
        }

        # Manager Macro Buffers (SMDP)
        m_states = torch.zeros((num_macro_steps, num_envs, state_dim)).to(device)
        m_goals = {
            a: torch.zeros((num_macro_steps, num_envs, THIEF_HER_GOAL_DIM)).to(device)
            for a in AGENTS
        }
        m_logprobs = {
            a: torch.zeros((num_macro_steps, num_envs)).to(device) for a in AGENTS
        }
        m_values = {
            a: torch.zeros((num_macro_steps, num_envs)).to(device) for a in AGENTS
        }
        m_rewards = {
            a: torch.zeros((num_macro_steps, num_envs)).to(device) for a in AGENTS
        }
        m_dones = torch.zeros((num_macro_steps, num_envs)).to(device)

        # HER Tracking State
        current_goals = {
            a: torch.zeros((num_envs, THIEF_HER_GOAL_DIM)).to(device) for a in AGENTS
        }
        prev_poses = {a: np.zeros((num_envs, 2)) for a in AGENTS}
        macro_reward_acc = {a: torch.zeros(num_envs).to(device) for a in AGENTS}

        her_relabeled_obs = []
        her_relabeled_role = []
        her_relabeled_mask = []
        her_relabeled_actions = []
        her_relabeled_goal = []
        her_relabeled_expert = []

        her_goals_assigned = 0
        her_goals_reached = 0
        her_total_dist = 0.0

        total_switches = 0
        total_switch_opportunities = 0
        current_sandbox_list = list(active_sandbox_mutants.keys())

        # --- ROLLOUT PHASE ---
        for step in range(NUM_STEPS):
            b_states[step] = torch.tensor(next_state, dtype=torch.float32).to(device)
            b_dones[step] = next_done

            # Macro Step: Manager assigns sub-goals every THIEF_MACRO_HORIZON steps
            if step % THIEF_MACRO_HORIZON == 0:
                macro_idx = step // THIEF_MACRO_HORIZON
                m_states[macro_idx] = b_states[step]
                m_dones[macro_idx] = next_done

                with torch.no_grad():
                    for a_idx, a in enumerate(AGENTS):
                        role_onehot = torch.zeros(num_envs, N_AGENTS, device=device)
                        role_onehot[:, a_idx] = 1.0
                        g_act, g_lp, _, g_val = agent.get_manager_action_and_value(
                            b_states[step], role_onehot
                        )
                        current_goals[a] = g_act
                        m_goals[a][macro_idx] = g_act
                        m_logprobs[a][macro_idx] = g_lp
                        m_values[a][macro_idx] = g_val
                        macro_reward_acc[a].zero_()

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
                goals_all = torch.cat(
                    [current_goals[a] for a in AGENTS], dim=0
                )  # Shape (N_AGENTS * num_envs, 2)

                state_rep = b_states[step].repeat(N_AGENTS, 1)
                actions, logprobs, _, values, chosen_expert = (
                    agent.get_action_and_value(
                        obs_all.flatten(0, 1),
                        role_all.flatten(0, 1),
                        mask_all.flatten(0, 1),
                        goals_all,
                        state_rep,
                        active_experts=active_experts,
                        previous_expert=env_previous_expert,
                        grace_experts=current_sandbox_list,
                        dormant_experts=dormant_experts,
                        num_envs=num_envs,
                    )
                )

                if env_previous_expert is not None:
                    valid_prev = env_previous_expert >= 0
                    switches = (chosen_expert != env_previous_expert) & valid_prev
                    total_switches += switches.sum().item()
                    total_switch_opportunities += valid_prev.sum().item()

                env_previous_expert = chosen_expert

                actions = actions.view(N_AGENTS, num_envs)
                logprobs = logprobs.view(N_AGENTS, num_envs)
                values = values.view(N_AGENTS, num_envs)
                experts_unflattened = chosen_expert.view(N_AGENTS, num_envs)

                for i, a in enumerate(AGENTS):
                    b_experts[a][step] = experts_unflattened[i]
                    b_obs[a][step] = obs_all[i]
                    b_role[a][step] = role_all[i]
                    b_mask[a][step] = mask_all[i]
                    b_goal[a][step] = current_goals[a]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            # Accumulate per-agent average return per env
            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            for e in range(num_envs):
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

                    if env_previous_expert is not None:
                        for ag_i in range(N_AGENTS):
                            env_previous_expert[ag_i * num_envs + e] = -1

            next_state = vec_env.state
            next_done = torch.tensor(
                terms["scout"] | truncs["scout"], dtype=torch.float32
            ).to(device)

            # Compute Goal Distances & Intrinsic Progress Reward
            for a in AGENTS:
                gx = (current_goals[a][:, 0].cpu().numpy() + 1.0) * 0.5 * map_w
                gy = (current_goals[a][:, 1].cpu().numpy() + 1.0) * 0.5 * map_h
                curr_pos = np.array(
                    [infos[e].get(a, {}).get("pos", (0, 0)) for e in range(num_envs)]
                )
                dist = np.sqrt((curr_pos[:, 0] - gx) ** 2 + (curr_pos[:, 1] - gy) ** 2)

                if step % THIEF_MACRO_HORIZON == 0:
                    prev_poses[a] = curr_pos

                prev_dist = np.sqrt(
                    (prev_poses[a][:, 0] - gx) ** 2 + (prev_poses[a][:, 1] - gy) ** 2
                )
                progress = (prev_dist - dist) / max(1.0, map_diag)
                prev_poses[a] = curr_pos

                ext_r = torch.tensor(rewards[a], dtype=torch.float32, device=device)
                int_r = torch.tensor(
                    progress, dtype=torch.float32, device=device
                ).clamp(-1.0, 1.0)

                b_rewards[a][step] = ext_r + (THIEF_HER_REWARD_COEF * int_r)
                macro_reward_acc[a] += ext_r

                # End of Macro Segment: Record Manager Reward & HER Hindsight Relabeling
                if (step + 1) % THIEF_MACRO_HORIZON == 0:
                    macro_idx = step // THIEF_MACRO_HORIZON
                    m_rewards[a][macro_idx] = macro_reward_acc[a].clone()

                    for e in range(num_envs):
                        her_goals_assigned += 1
                        her_total_dist += float(dist[e])

                        if dist[e] <= THIEF_HER_REACH_DIST:
                            her_goals_reached += 1
                        else:
                            # Hindsight Relabeling: Goal achieved was the actual reached position
                            hgx = float(
                                np.clip(
                                    (curr_pos[e, 0] / (0.5 * map_w)) - 1.0, -1.0, 1.0
                                )
                            )
                            hgy = float(
                                np.clip(
                                    (curr_pos[e, 1] / (0.5 * map_h)) - 1.0, -1.0, 1.0
                                )
                            )
                            hg = torch.tensor(
                                [hgx, hgy], dtype=torch.float32, device=device
                            )

                            seg_start = step - THIEF_MACRO_HORIZON + 1
                            for tau in range(seg_start, step + 1):
                                her_relabeled_obs.append(b_obs[a][tau, e])
                                her_relabeled_role.append(b_role[a][tau, e])
                                her_relabeled_mask.append(b_mask[a][tau, e])
                                her_relabeled_actions.append(b_actions[a][tau, e])
                                her_relabeled_goal.append(hg)
                                her_relabeled_expert.append(b_experts[a][tau, e])

        last_her_reach_rate = (
            (her_goals_reached / max(1, her_goals_assigned)) * 100
            if her_goals_assigned > 0
            else 0.0
        )
        last_her_avg_dist = (
            her_total_dist / max(1, her_goals_assigned)
            if her_goals_assigned > 0
            else 0.0
        )

        # --- GAE CALCULATION (WORKERS) ---
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
            goals_all = torch.cat([current_goals[a] for a in AGENTS], dim=0)
            state_rep = (
                torch.tensor(next_state, dtype=torch.float32)
                .to(device)
                .repeat(N_AGENTS, 1)
            )

            _, _, _, next_values, _ = agent.get_action_and_value(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                goals_all,
                state_rep,
                active_experts=active_experts,
                previous_expert=env_previous_expert,
                grace_experts=current_sandbox_list,
                dormant_experts=dormant_experts,
                num_envs=num_envs,
            )
            next_values = next_values.view(N_AGENTS, num_envs)

            b_advantages = {
                a: torch.zeros_like(b_rewards[a]).to(device) for a in AGENTS
            }
            b_returns = {a: torch.zeros_like(b_rewards[a]).to(device) for a in AGENTS}

            for a_idx, a in enumerate(AGENTS):
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_values[a_idx]
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[a][t + 1]

                    delta = (
                        b_rewards[a][t]
                        + GAMMA * nextvalues * nextnonterminal
                        - b_values[a][t]
                    )
                    b_advantages[a][t] = lastgaelam = (
                        delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    )
                b_returns[a] = b_advantages[a] + b_values[a]

        # --- GAE CALCULATION (MANAGER SMDP) ---
        with torch.no_grad():
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
            m_next_value = agent.manager.critic(next_state_t).squeeze(-1)

            m_advantages = {
                a: torch.zeros_like(m_rewards[a]).to(device) for a in AGENTS
            }
            m_returns = {a: torch.zeros_like(m_rewards[a]).to(device) for a in AGENTS}

            for a in AGENTS:
                lastgaelam = 0
                for t in reversed(range(num_macro_steps)):
                    if t == num_macro_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = m_next_value
                    else:
                        nextnonterminal = 1.0 - m_dones[t + 1]
                        nextvalues = m_values[a][t + 1]

                    delta = (
                        m_rewards[a][t]
                        + (GAMMA**THIEF_MACRO_HORIZON) * nextvalues * nextnonterminal
                        - m_values[a][t]
                    )
                    m_advantages[a][t] = lastgaelam = (
                        delta
                        + (GAMMA**THIEF_MACRO_HORIZON)
                        * GAE_LAMBDA
                        * nextnonterminal
                        * lastgaelam
                    )
                m_returns[a] = m_advantages[a] + m_values[a]

        # Flatten worker transitions across all roles and environments
        flat_obs = torch.cat([b_obs[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_role = torch.cat([b_role[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_mask = torch.cat([b_mask[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_goal = torch.cat([b_goal[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_actions = torch.cat([b_actions[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_logprobs = torch.cat([b_logprobs[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_advantages = torch.cat(
            [b_advantages[a].flatten(0, 1) for a in AGENTS], dim=0
        )
        flat_returns = torch.cat([b_returns[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_values = torch.cat([b_values[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_states = b_states.repeat(N_AGENTS, 1, 1).flatten(0, 1)
        flat_experts = torch.cat([b_experts[a].flatten(0, 1) for a in AGENTS], dim=0)

        # Flatten manager transitions
        flat_m_states = m_states.repeat(N_AGENTS, 1, 1).flatten(0, 1)
        flat_m_role = torch.cat(
            [
                torch.zeros(
                    num_macro_steps * num_envs,
                    N_AGENTS,
                    device=device,
                ).scatter_(
                    1,
                    torch.full(
                        (num_macro_steps * num_envs, 1),
                        i,
                        dtype=torch.long,
                        device=device,
                    ),
                    1.0,
                )
                for i in range(N_AGENTS)
            ],
            dim=0,
        )
        flat_m_goals = torch.cat([m_goals[a].flatten(0, 1) for a in AGENTS], dim=0)
        flat_m_logprobs = torch.cat(
            [m_logprobs[a].flatten(0, 1) for a in AGENTS], dim=0
        )
        flat_m_advantages = torch.cat(
            [m_advantages[a].flatten(0, 1) for a in AGENTS], dim=0
        )
        flat_m_returns = torch.cat([m_returns[a].flatten(0, 1) for a in AGENTS], dim=0)

        total_samples = flat_obs.shape[0]
        usage_counts = torch.bincount(flat_experts, minlength=active_experts)
        last_expert_usage = {
            k: round(float(usage_counts[k].item() / total_samples) * 100, 1)
            for k in range(active_experts)
        }

        last_switch_rate = (
            (total_switches / total_switch_opportunities) * 100
            if total_switch_opportunities > 0
            else 0.0
        )

        for k in range(active_experts):
            if usage_counts[k] == 0:
                expert_consecutive_zero_usage[k] += 1
            else:
                expert_consecutive_zero_usage[k] = 0

        # HER auxiliary tensors
        has_her = len(her_relabeled_actions) > 0
        if has_her:
            her_obs_t = torch.stack(her_relabeled_obs, dim=0)
            her_role_t = torch.stack(her_relabeled_role, dim=0)
            her_mask_t = torch.stack(her_relabeled_mask, dim=0)
            her_act_t = torch.stack(her_relabeled_actions, dim=0)
            her_goal_t = torch.stack(her_relabeled_goal, dim=0)
            her_exp_t = torch.stack(her_relabeled_expert, dim=0)

        # --- PPO OPTIMIZATION LOOP ---
        batch_size = total_samples
        minibatch_size = batch_size // 4
        b_inds = np.arange(batch_size)
        m_batch_size = flat_m_states.shape[0]
        m_minibatch_size = max(1, m_batch_size // 4)
        m_b_inds = np.arange(m_batch_size)

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            np.random.shuffle(m_b_inds)

            # Manager Update Step
            for m_start in range(0, m_batch_size, m_minibatch_size):
                m_end = m_start + m_minibatch_size
                m_mb_inds = m_b_inds[m_start:m_end]

                m_mb_state = flat_m_states[m_mb_inds]
                m_mb_role = flat_m_role[m_mb_inds]
                m_mb_goals = flat_m_goals[m_mb_inds]
                m_mb_logprobs = flat_m_logprobs[m_mb_inds]
                m_mb_adv = flat_m_advantages[m_mb_inds]
                m_mb_returns = flat_m_returns[m_mb_inds]

                m_mb_adv_norm = (m_mb_adv - m_mb_adv.mean()) / (m_mb_adv.std() + 1e-8)

                _, new_m_lp, m_ent, new_m_val = agent.get_manager_action_and_value(
                    m_mb_state, m_mb_role, action=m_mb_goals
                )
                m_ratio = (new_m_lp - m_mb_logprobs).exp()

                m_pg1 = -m_mb_adv_norm * m_ratio
                m_pg2 = -m_mb_adv_norm * torch.clamp(
                    m_ratio, 1 - CLIP_COEF, 1 + CLIP_COEF
                )
                m_pg_loss = torch.max(m_pg1, m_pg2).mean()
                m_v_loss = 0.5 * ((new_m_val - m_mb_returns) ** 2).mean()

                mgr_loss = m_pg_loss - (ENT_COEF * m_ent.mean()) + (VF_COEF * m_v_loss)

                optimizer.zero_grad()
                mgr_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            # Worker Specialists Update Step
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_exp = flat_experts[mb_inds]
                mb_adv = flat_advantages[mb_inds]

                # Per-expert advantage normalization
                mb_adv_norm = torch.zeros_like(mb_adv)
                for k in range(active_experts):
                    k_mask = mb_exp == k
                    if k_mask.sum() > 1:
                        k_adv = mb_adv[k_mask]
                        mb_adv_norm[k_mask] = (k_adv - k_adv.mean()) / (
                            k_adv.std() + 1e-8
                        )
                    elif k_mask.sum() == 1:
                        mb_adv_norm[k_mask] = 0.0

                mb_obs = flat_obs[mb_inds]
                mb_role = flat_role[mb_inds]
                mb_mask = flat_mask[mb_inds]
                mb_goal = flat_goal[mb_inds]
                mb_state = flat_states[mb_inds]
                mb_actions = flat_actions[mb_inds]
                mb_logprobs = flat_logprobs[mb_inds]
                mb_returns = flat_returns[mb_inds]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs,
                    mb_role,
                    mb_mask,
                    mb_goal,
                    mb_state,
                    active_experts=active_experts,
                    action=mb_actions,
                    expert_idx=mb_exp,
                    grace_experts=current_sandbox_list,
                    dormant_experts=dormant_experts,
                )

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                pg_loss1 = -mb_adv_norm * ratio
                pg_loss2 = -mb_adv_norm * torch.clamp(
                    ratio, 1 - CLIP_COEF, 1 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - (ENT_COEF * entropy_loss) + (VF_COEF * v_loss)

                # Auxiliary HER Hindsight Navigation Loss
                if has_her:
                    her_sample_size = min(len(her_act_t), minibatch_size // 2)
                    her_idx = torch.randint(
                        0, len(her_act_t), (her_sample_size,), device=device
                    )

                    _, her_newlogprob, _, _, _ = agent.get_action_and_value(
                        her_obs_t[her_idx],
                        her_role_t[her_idx],
                        her_mask_t[her_idx],
                        her_goal_t[her_idx],
                        state=None,
                        active_experts=active_experts,
                        action=her_act_t[her_idx],
                        expert_idx=her_exp_t[her_idx],
                        grace_experts=current_sandbox_list,
                        dormant_experts=dormant_experts,
                    )
                    her_aux_loss = -her_newlogprob.mean()
                    loss = loss + (THIEF_HER_AUX_COEF * her_aux_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # Lifecycle Step A: Sandbox Incubation Progression & Queue Processing
        graduated_this_update = []
        for m_idx in list(active_sandbox_mutants.keys()):
            active_sandbox_mutants[m_idx] += 1
            if active_sandbox_mutants[m_idx] >= THIEF_ISOLATION_UPDATES:
                graduated_this_update.append(m_idx)
                del active_sandbox_mutants[m_idx]

        if graduated_this_update:
            grad_names = ", ".join(f"E{i}" for i in graduated_this_update)
            msg = (
                f"[Incubation Event] Update: {update} | Mutant(s) ({grad_names}) "
                f"completed {THIEF_ISOLATION_UPDATES} sandbox updates. Calibrated Critics promoted to general bidding pool!"
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

        # Pull from incubation queue into newly opened sandbox slots
        while (
            len(active_sandbox_mutants) < THIEF_MAX_SANDBOX_EXPERTS
            and len(incubation_queue) > 0
        ):
            next_m = incubation_queue.pop(0)
            active_sandbox_mutants[next_m] = 0
            msg = (
                f"[Queue Event] Update: {update} | Mutant E{next_m} entered sandbox incubator "
                f"({len(active_sandbox_mutants)} active, {len(incubation_queue)} waiting in queue)"
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

        # Mark last_grace_end_update timestamp when entire cohort has graduated
        if (
            graduated_this_update
            and len(active_sandbox_mutants) == 0
            and len(incubation_queue) == 0
        ):
            last_grace_end_update = update
            msg = (
                f"[Incubation Complete] Update: {update} | All mutants graduated! "
                f"Enforcing {THIEF_POST_GRACE_COOLDOWN_UPDATES}-update cooldown before next deficit trigger."
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

        # Lifecycle Step B: Dynamic Deficit Spawning Check
        deficit_triggered, deficit_mask = check_deficit_trigger(
            agent,
            active_experts,
            flat_obs,
            flat_role,
            flat_states,
            flat_goal,
            update,
            last_grace_end_update,
            active_sandbox_mutants,
            incubation_queue,
            device,
            dormant_experts=dormant_experts,
        )

        if deficit_triggered:
            child_indices, weights, best_parent, parent_val = (
                recombine_and_mutate_thief(
                    agent,
                    active_experts,
                    flat_obs,
                    flat_role,
                    flat_mask,
                    flat_actions,
                    flat_goal,
                    flat_returns,
                    flat_values,
                    flat_states,
                    deficit_mask,
                    device,
                    dormant_experts=dormant_experts,
                )
            )
            old_count = active_experts
            active_experts += len(child_indices)
            total_spawns += len(child_indices)

            # Preserve Adam momentum and variance state for surviving parents
            optimizer = update_optimizer_params(optimizer, agent, lr=LR)

            for c_idx in child_indices:
                if len(active_sandbox_mutants) < THIEF_MAX_SANDBOX_EXPERTS:
                    active_sandbox_mutants[c_idx] = 0
                else:
                    incubation_queue.append(c_idx)

                spawn_history.append(
                    {
                        "update": update,
                        "parent_expert": best_parent,
                        "child_expert": c_idx,
                        "parent_value": parent_val,
                        "active_experts": active_experts,
                        "total_spawns": total_spawns,
                        "trigger": "critic_deficit_dynamic",
                    }
                )

            spawned_names = ", ".join(f"E{i}" for i in child_indices)
            parent_mix_str = ", ".join(
                f"E{k}: {weights[i] * 100:.1f}%"
                for i, k in enumerate(
                    [idx for idx in range(old_count) if idx not in dormant_experts]
                    or range(old_count)
                )
            )
            msg = (
                f"[Deficit Evolution Event] Update: {update} | Triggered Dynamic Deficit Spawn! "
                f"Spawned {len(child_indices)} Structured Orthogonal Mutants (Pool: {old_count} -> {active_experts} experts) "
                f"({spawned_names} recombined from active parents [{parent_mix_str}]) | "
                f"Active Sandbox Mutants: {list(active_sandbox_mutants.keys())}, Queue: {incubation_queue}"
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

        # Lifecycle Step C: Extinction Dormancy (Zero Culling Shock)
        if (
            len(active_sandbox_mutants) == 0
            and len(incubation_queue) == 0
            and (active_experts - len(dormant_experts)) > 1
        ):
            newly_dormant = []
            for k in range(1, active_experts):
                if (
                    k not in dormant_experts
                    and expert_consecutive_zero_usage[k] >= THIEF_CULL_WINDOW_UPDATES
                ):
                    dormant_experts.add(k)
                    newly_dormant.append(k)

            if newly_dormant:
                dormant_names = ", ".join(f"E{i}" for i in newly_dormant)
                msg = (
                    f"[Dormancy Event] Update: {update} | Expert(s) ({dormant_names}) entered Dormant State "
                    f"(0% usage for >= {THIEF_CULL_WINDOW_UPDATES} updates). Preserved in memory, excluded from bidding."
                )
                console_logger.info(msg)
                if file_logger:
                    file_logger.info(msg)

        # Logging
        if update % 5 == 0:
            avg_win = np.mean(completed_wins[-100:]) if completed_wins else 0.0
            avg_rew = (
                np.mean(completed_episode_returns[-100:])
                if completed_episode_returns
                else 0.0
            )
            avg_steps = (
                np.mean(completed_episode_steps[-100:])
                if completed_episode_steps
                else 0.0
            )
            avg_alarm = (
                np.mean(completed_episode_alarms[-100:])
                if completed_episode_alarms
                else 0.0
            )
            avg_scout_int = (
                np.mean(completed_scout_interact[-100:])
                if completed_scout_interact
                else 0.0
            )
            avg_scout_pois = (
                np.mean(completed_scout_pois[-100:]) if completed_scout_pois else 0.0
            )
            avg_hacker_hack = (
                np.mean(completed_hacker_hack[-100:]) if completed_hacker_hack else 0.0
            )
            avg_muscle_neut = (
                np.mean(completed_muscle_neutralize[-100:])
                if completed_muscle_neutralize
                else 0.0
            )
            avg_muscle_guards = (
                np.mean(completed_muscle_guards[-100:])
                if completed_muscle_guards
                else 0.0
            )
            avg_extractor_loot = (
                np.mean(completed_extractor_loot[-100:])
                if completed_extractor_loot
                else 0.0
            )
            avg_agents_extract = (
                np.mean(completed_agents_at_extract[-100:])
                if completed_agents_at_extract
                else 0.0
            )

            stage_max_steps = env_config.get("max_steps", 300)
            stage_alarm_max = float(env_config.get("alarm_max", 100.0))
            stage_guards = env_config.get("guard_count", 0)

            active_bidding_count = active_experts - len(dormant_experts)
            usage_str = ", ".join(
                f"E{k}{'(dormant)' if k in dormant_experts else ''}: {last_expert_usage.get(k, 0.0)}%"
                for k in range(active_experts)
            )

            # Compact console log (clean standard output with HER reach rate)
            console_str = (
                f"Update: {update}/{num_updates} | "
                f"Win Rate: {avg_win:.2f} | "
                f"Episodic Return: {avg_rew:.3f} | "
                f"Goals Reached: {last_her_reach_rate:.1f}% (Dist: {last_her_avg_dist:.1f}) | "
                f"Steps: {avg_steps:.1f}/{stage_max_steps} | "
                f"Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | "
                f"Active Experts: {active_bidding_count}/{active_experts}"
            )
            console_logger.info(console_str)

            # Comprehensive file log (full sub-task breakdown matching E-COOP + HER)
            if file_logger:
                file_str = (
                    f"Update: {update}/{num_updates} | "
                    f"Win Rate: {avg_win:.2f} | "
                    f"Episodic Return: {avg_rew:.3f} | "
                    f"Steps: {avg_steps:.1f}/{stage_max_steps} | "
                    f"Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | "
                    f"Goals Reached: {last_her_reach_rate:.1f}% (Avg Dist: {last_her_avg_dist:.1f} tiles, Relabeled: {len(her_relabeled_actions)}) | "
                    f"Scout Tag Rate: {avg_scout_int:.2f} (Avg POIs: {avg_scout_pois:.1f}) | "
                    f"Hacker Hack Rate: {avg_hacker_hack:.2f} | "
                    f"Muscle Neutralize Rate: {avg_muscle_neut:.2f} (Avg Guards: {avg_muscle_guards:.1f}/{stage_guards}) | "
                    f"Extractor Loot Rate: {avg_extractor_loot:.2f} | "
                    f"Avg Agents at Extract: {avg_agents_extract:.2f}/4 | "
                    f"Active: {active_bidding_count}/{active_experts} | "
                    f"Total Spawns: {total_spawns} | "
                    f"Switch Rate: {last_switch_rate:.1f}% | "
                    f"Usage: [{usage_str}]"
                )
                file_logger.info(file_str)

    vec_env.close()

    # Final Evaluation & Inter-Stage Compaction (prune dormant experts cleanly for saved model)
    surviving_indices = [k for k in range(active_experts) if k not in dormant_experts]
    if len(surviving_indices) < active_experts and len(surviving_indices) >= 1:
        agent.prune_experts(surviving_indices)
        active_experts = len(surviving_indices)
        dormant_experts.clear()

    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        model_save_path = os.path.join(save_ckpt_dir, "model.pt")
        torch.save(
            {
                "model_state": agent.state_dict(),
                "active_experts": active_experts,
                "dormant_experts": list(dormant_experts),
                "total_spawns": total_spawns,
                "spawn_history": spawn_history,
            },
            model_save_path,
        )
        print(f"Saved final THIEF model to {model_save_path}")

    avg_win = float(np.mean(completed_wins[-100:])) if completed_wins else 0.0
    lifetime_win = float(np.mean(completed_wins)) if completed_wins else 0.0
    avg_rew = (
        float(np.mean(completed_episode_returns[-100:]))
        if completed_episode_returns
        else 0.0
    )
    avg_steps = (
        float(np.mean(completed_episode_steps[-100:]))
        if completed_episode_steps
        else 0.0
    )
    avg_alarm = (
        float(np.mean(completed_episode_alarms[-100:]))
        if completed_episode_alarms
        else 0.0
    )
    avg_scout_int = (
        float(np.mean(completed_scout_interact[-100:]))
        if completed_scout_interact
        else 0.0
    )
    avg_scout_pois = (
        float(np.mean(completed_scout_pois[-100:])) if completed_scout_pois else 0.0
    )
    avg_hacker_hack = (
        float(np.mean(completed_hacker_hack[-100:])) if completed_hacker_hack else 0.0
    )
    avg_muscle_neut = (
        float(np.mean(completed_muscle_neutralize[-100:]))
        if completed_muscle_neutralize
        else 0.0
    )
    avg_muscle_guards = (
        float(np.mean(completed_muscle_guards[-100:]))
        if completed_muscle_guards
        else 0.0
    )
    avg_extractor_loot = (
        float(np.mean(completed_extractor_loot[-100:]))
        if completed_extractor_loot
        else 0.0
    )
    avg_agents_extract = (
        float(np.mean(completed_agents_at_extract[-100:]))
        if completed_agents_at_extract
        else 0.0
    )

    stage_max_steps = env_config.get("max_steps", 300)
    stage_alarm_max = float(env_config.get("alarm_max", 100.0))
    stage_guards = env_config.get("guard_count", 0)

    results_data = {
        "algo": algo_name,
        "stage": stage_idx,
        "win_rate": avg_win,
        "lifetime_win_rate": lifetime_win,
        "mean_reward": avg_rew,
        "avg_episode_steps": avg_steps,
        "max_stage_steps": stage_max_steps,
        "avg_alarm": avg_alarm,
        "stage_alarm_max": stage_alarm_max,
        "her_goal_reach_rate": last_her_reach_rate,
        "her_avg_goal_distance": last_her_avg_dist,
        "scout_interact_rate": avg_scout_int,
        "scout_avg_pois_tagged": avg_scout_pois,
        "hacker_hack_rate": avg_hacker_hack,
        "muscle_neutralize_rate": avg_muscle_neut,
        "muscle_avg_guards_neutralized": avg_muscle_guards,
        "total_stage_guards": stage_guards,
        "extractor_loot_rate": avg_extractor_loot,
        "avg_agents_at_extract": avg_agents_extract,
        "active_experts": active_experts,
        "dormant_experts": list(dormant_experts),
        "total_spawns": total_spawns,
        "expert_switch_rate": last_switch_rate,
        "expert_usage_pct": last_expert_usage,
        "spawn_history": spawn_history,
    }

    if log_dir is not None:
        results_path = os.path.join(log_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Saved results summary to {results_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="THIEF Training")
    parser.add_argument("--stage", type=int, default=0, help="Stage index")
    parser.add_argument(
        "--timesteps", type=int, default=200_000, help="Total timesteps"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
    args = parser.parse_args()

    train(
        algo_name="thief",
        stage_idx=args.stage,
        total_timesteps=args.timesteps,
        load_ckpt_path=args.load_ckpt,
        save_ckpt_dir=args.save_dir,
        log_dir=args.save_dir,
        seed=args.seed,
    )
