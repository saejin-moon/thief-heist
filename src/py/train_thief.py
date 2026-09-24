"""
THIEF: Value-bidding multi-agent RL with dynamic expert spawning,
parameter recombination, warmup queue, and switching hysteresis.
"""

import collections
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
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
    NUM_STEPS,
    OBSERVATION_SIZE,
    THIEF_DEFICIT_MIN_SAMPLES,
    THIEF_ENVS_PER_MUTANT,
    THIEF_GRADIENT_CONFLICT_THRESHOLD,
    THIEF_HER_AUX_COEF,
    THIEF_HER_REACH_DIST,
    THIEF_HYSTERESIS_EPSILON,
    THIEF_INITIAL_EXPERTS,
    THIEF_ISOLATION_UPDATES,
    THIEF_MACRO_HORIZON,
    THIEF_MAX_SANDBOX_EXPERTS,
    THIEF_MUTATION_NOISE,
    THIEF_NUM_ENVS,
    THIEF_POST_GRACE_COOLDOWN_UPDATES,
    THIEF_PROGRESS_COEF,
    THIEF_TARGETED_LR,
    THIEF_WARMUP_UPDATES,
    UPDATE_EPOCHS,
    VF_COEF,
)
from thermal_guard import check_thermal_guard
from torch import nn
from torch.distributions.categorical import Categorical
from vec_env import make_vec_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MappoActor(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = (
            (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS + GOAL_VECTOR_DIM
        )
        self.network = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, ACTION_SPACE_SIZE), std=0.01),
        )
        self.fisher_ema = None

    def forward(self, x):
        return self.network(x)


class MappoCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        in_dim = state_dim + N_AGENTS + GOAL_VECTOR_DIM
        self.network = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        return self.network(x)


class MappoNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.actor = MappoActor()
        self.critic = MappoCritic(state_dim)

    def get_action_and_value(self, obs, role, mask, goal, state=None, action=None):
        x_actor = torch.cat([obs.flatten(1), role, goal], dim=1)
        logits = self.actor(x_actor)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
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
    Modular Specialist Pool with Goal-Conditioned Mission Orientation for THIEF.
    Routes actions based on decentralized Critic confidence bidding with dynamic
    temporal hysteresis, supports dynamic 16-environment sharding, and manages
    dynamic addition and culling of specialist sub-networks conditioned on mission targets.
    """

    def __init__(self, state_dim, num_initial_experts=THIEF_INITIAL_EXPERTS):
        super().__init__()
        self.state_dim = state_dim
        self.experts = nn.ModuleList(
            [MappoNetwork(state_dim) for _ in range(max(1, num_initial_experts))]
        )

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
        num_envs=THIEF_NUM_ENVS,
    ):
        """
        Executes routing and policy forward passes across the expert pool conditioned on mission targets.
        Environment routing:
        - When grace_experts is empty: all environments run competitive bidding across active, non-dormant experts.
        - When new experts are warming up: each occupies dedicated warmup envs,
          leaving remaining envs for general competitive bidding.
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

                if not self.training:
                    best_candidate_idx = bids.argmax(dim=-1)
                else:
                    # 10% epsilon-greedy candidate exploration during training
                    eps = 0.10
                    greedy_idx = bids.argmax(dim=-1)
                    rand_idx = torch.randint(
                        0, len(candidate_experts), (batch_size,), device=obs.device
                    )
                    explore_mask = torch.rand(batch_size, device=obs.device) < eps
                    best_candidate_idx = torch.where(explore_mask, rand_idx, greedy_idx)

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

            # Warmup routing across environments
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
            )
            actions[mask_k] = a
            logprobs[mask_k] = lp
            entropies[mask_k] = ent
            if v is not None:
                values[mask_k] = v

        return actions, logprobs, entropies, values, chosen_expert


def update_optimizer_params(old_optimizer, agent, lr=LR):
    """
    Preserves Adam moments for existing expert parameters across spawn events,
    and initializes optimizer state for new expert parameters.
    """
    new_optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    if old_optimizer is not None:
        for p in agent.parameters():
            if p in old_optimizer.state:
                new_optimizer.state[p] = old_optimizer.state[p]
    return new_optimizer


def check_plateau_trigger(
    completed_wins, window=100, min_episodes=200, threshold=0.03, ceiling=0.90
):
    """Check if win rate has plateaued. Returns True if improvement is below threshold and stage is not already won."""
    if len(completed_wins) < min_episodes:
        return False
    recent = completed_wins[-window:]
    half = len(recent) // 2
    if half < 5:
        return False
    first_half = float(np.mean(recent[:half]))
    second_half = float(np.mean(recent[half:]))
    if second_half >= ceiling:
        return False
    return (second_half - first_half) < threshold


def check_gradient_interference_trigger(
    agent,
    active_experts,
    b_obs,
    b_role,
    b_mask,
    b_actions,
    b_goal,
    b_advantages,
    update,
    last_grace_end_update,
    active_sandbox_mutants,
    incubation_queue,
    device,
    dormant_experts=None,
    num_updates=None,
    b_experts=None,
):
    """
    Evaluates whether the active policy is experiencing catastrophic gradient interference
    between positive-advantage (success) and negative-advantage (failure) sub-tasks.
    """
    warmup = (
        min(THIEF_WARMUP_UPDATES, int(0.15 * num_updates))
        if num_updates
        else THIEF_WARMUP_UPDATES
    )
    if update <= warmup:
        return False, None, 0.0
    if num_updates is not None and update > int(0.85 * num_updates):
        return False, None, 0.0
    if active_sandbox_mutants or incubation_queue:
        return False, None, 0.0
    cooldown = (
        min(THIEF_POST_GRACE_COOLDOWN_UPDATES, int(0.20 * num_updates))
        if num_updates
        else THIEF_POST_GRACE_COOLDOWN_UPDATES
    )
    if (update - last_grace_end_update) < cooldown:
        return False, None, 0.0

    adv = b_advantages.to(device)
    pos_mask = adv > 0.0
    neg_mask = adv < -0.5

    if (
        pos_mask.sum().item() < THIEF_DEFICIT_MIN_SAMPLES
        or neg_mask.sum().item() < THIEF_DEFICIT_MIN_SAMPLES
    ):
        return False, None, 0.0

    dormant_set = set(dormant_experts) if dormant_experts is not None else set()
    parent_candidates = [
        k for k in range(active_experts) if k not in dormant_set
    ] or list(range(active_experts))
    if b_experts is not None:
        # Measure conflict through the routing-dominant (most-used non-dormant)
        # expert, not blindly expert 0 — expert 0's tear may not reflect the pool.
        usage = torch.bincount(
            b_experts[b_experts < active_experts].to(device),
            minlength=active_experts,
        )
        usage_masked = [
            usage[k].item() if k not in dormant_set else -1
            for k in range(active_experts)
        ]
        if max(usage_masked) > 0:
            parent_idx = int(np.argmax(usage_masked))
        else:
            parent_idx = parent_candidates[0]
    else:
        parent_idx = parent_candidates[0]
    actor = agent.experts[parent_idx].actor

    def compute_partition_grad(mask):
        sub_inds = torch.where(mask)[0]
        if len(sub_inds) > 256:
            perm = torch.randperm(len(sub_inds))[:256]
            sub_inds = sub_inds[perm]

        x_act = torch.cat(
            [b_obs[sub_inds].flatten(1), b_role[sub_inds], b_goal[sub_inds]], dim=1
        ).to(device)
        m_act = b_mask[sub_inds].to(device)
        a_act = b_actions[sub_inds].to(device)
        adv_sub = adv[sub_inds]

        logits = actor(x_act)
        masked_logits = logits + ((1.0 - m_act) * -1e9)
        dist = Categorical(logits=masked_logits)
        log_prob = dist.log_prob(a_act)
        loss = -(log_prob * adv_sub).mean()

        actor.zero_grad()
        loss.backward()

        grads = []
        with torch.no_grad():
            for p in actor.parameters():
                if p.grad is not None:
                    grads.append(p.grad.data.flatten())
        return torch.cat(grads) if grads else None

    g_pos = compute_partition_grad(pos_mask)
    g_neg = compute_partition_grad(neg_mask)

    if g_pos is None or g_neg is None:
        return False, None, 0.0

    cos_sim = (
        torch.dot(g_pos, g_neg) / (g_pos.norm() * g_neg.norm() + 1e-8)
    ).item()

    if cos_sim < THIEF_GRADIENT_CONFLICT_THRESHOLD:
        return True, neg_mask, cos_sim
    return False, None, cos_sim


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
    """Normalized gradient on failure transitions."""
    if not deficit_mask.any():
        return [torch.zeros_like(p) for p in recombined_actor.parameters()]

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


def spawn_targeted_specialist(
    agent,
    active_experts,
    b_obs,
    b_role,
    b_mask,
    b_actions,
    b_goal,
    b_returns,
    b_values,
    b_states,
    b_experts,
    device,
    dormant_experts=None,
    ablation="none",
    deficit_mask=None,
):
    """
    Spawn a new expert:
    1. Recombine parameters across parent experts (weighted by value estimates and Fisher diagonal)
    2. Gradient step toward failure transitions (deficit gradient)
    3. Scaled exploration noise
    New critic initialized from scratch.

    Ablation variants:
    - 'uniform_recomb': uniform parameter average over parents (weights w_k = 1/K, isotropic noise).
    - 'clone_best': child actor = best parent actor + noise.
    """
    dormant_set = set(dormant_experts) if dormant_experts is not None else set()
    parent_candidates = [
        k for k in range(active_experts) if k not in dormant_set
    ] or list(range(active_experts))

    # Convert to flattened tensors if passed as dicts
    flat_obs = (
        torch.cat([b_obs[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_obs, dict)
        else b_obs
    )
    flat_role = (
        torch.cat([b_role[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_role, dict)
        else b_role
    )
    flat_mask = (
        torch.cat([b_mask[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_mask, dict)
        else b_mask
    )
    flat_actions = (
        torch.cat([b_actions[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_actions, dict)
        else b_actions
    )
    flat_goal = (
        torch.cat([b_goal[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_goal, dict)
        else b_goal
    )
    flat_returns = (
        torch.cat([b_returns[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_returns, dict)
        else b_returns
    )
    flat_values = (
        torch.cat([b_values[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_values, dict)
        else b_values
    )
    flat_experts = (
        torch.cat([b_experts[a].flatten(0, 1) for a in AGENTS], dim=0)
        if isinstance(b_experts, dict)
        else b_experts
    )

    # --- Pick best parent by mean critic value ---
    expert_mean_vals = {}
    for k in parent_candidates:
        ev = flat_values[flat_experts == k]
        if len(ev) > 0:
            expert_mean_vals[k] = ev.mean().item()
        elif b_states is not None:
            with torch.no_grad():
                sample_states = (
                    torch.cat([b_states.flatten(0, 1) for _ in AGENTS], dim=0)
                    if isinstance(b_states, dict)
                    else (
                        b_states
                        if b_states.shape[0] == flat_role.shape[0]
                        else b_states.repeat(N_AGENTS, 1, 1).flatten(0, 1)
                        if b_states.dim() == 3
                        else b_states
                    )
                )
                x_critic = torch.cat([sample_states, flat_role, flat_goal], dim=1).to(
                    device
                )
                expert_mean_vals[k] = agent.experts[k].critic(x_critic).mean().item()
        else:
            expert_mean_vals[k] = 0.0

    best_parent = max(expert_mean_vals, key=expert_mean_vals.get)
    best_parent_val = expert_mean_vals[best_parent]

    # --- Fitness weights across all parents ---
    val_tensor = torch.tensor(
        [expert_mean_vals[k] for k in parent_candidates], dtype=torch.float32
    )
    fitness_weights = torch.softmax(val_tensor, dim=0).numpy()

    # --- Failure mask ---
    # When provided, use the spawn trigger's negative-advantage deficit mask
    # (the exact transitions that caused the gradient conflict) so the child
    # targets the observed failing behavior. Otherwise fall back to the
    # generic overconfidence mask (returns < values).
    if deficit_mask is not None:
        deficit_mask = deficit_mask.to(flat_returns.device)
    else:
        deficit_mask = (flat_returns - flat_values) < 0
    deficit_sample_count = int(deficit_mask.sum().item())

    # --- Compute targeted deficit gradient on a cloned temp actor of best parent ---
    temp_actor = MappoActor().to(device)
    for p, parent_p in zip(
        temp_actor.parameters(), agent.experts[best_parent].actor.parameters()
    ):
        p.data.copy_(parent_p.data)
    deficit_grads = compute_deficit_gradient(
        temp_actor,
        flat_obs,
        flat_role,
        flat_mask,
        flat_actions,
        flat_goal,
        flat_returns,
        flat_values,
        deficit_mask,
        device,
    )

    # --- Spawn child specialist with independent cold-start critic ---
    c_idx = agent.add_expert()

    # --- Initialize child actor ---
    if ablation == "clone_best":
        # Ablation: child = best parent + isotropic noise (no recombination,
        # no deficit push).
        with torch.no_grad():
            for name, param in agent.experts[c_idx].actor.named_parameters():
                parent_param = dict(agent.experts[best_parent].actor.named_parameters())[name].data
                p_std = float(parent_param.std().item()) if parent_param.numel() > 1 and parent_param.std().item() > 1e-5 else 0.01
                param.copy_(parent_param + torch.randn_like(param) * min(0.15 * p_std, THIEF_MUTATION_NOISE))
        return c_idx, best_parent, best_parent_val

    with torch.no_grad():
        for name, param in agent.experts[c_idx].actor.named_parameters():
            if ablation == "uniform_recomb":
                # Ablation: uniform parameter average over parents (equal weights)
                # + targeted deficit push + isotropic noise.
                param_sum = torch.zeros_like(param)
                for rel_idx, k in enumerate(parent_candidates):
                    param_k = dict(agent.experts[k].actor.named_parameters())[name].data
                    param_sum += param_k / len(parent_candidates)
                recombined = param_sum

                p_path_idx = [
                    i
                    for i, (n, _) in enumerate(
                        agent.experts[best_parent].actor.named_parameters()
                    )
                    if n == name
                ]
                deficit_p = (
                    deficit_grads[p_path_idx[0]] if p_path_idx else torch.zeros_like(param)
                )
                targeted = recombined - (THIEF_TARGETED_LR * deficit_p)
                p_std = float(param.std().item()) if param.numel() > 1 and param.std().item() > 1e-5 else 0.01
                noise = torch.randn_like(param) * min(0.15 * p_std, THIEF_MUTATION_NOISE)
                param.copy_(targeted + noise)
                continue

            # 1. Parameter recombination across parents
            weighted_sum = torch.zeros_like(param)
            weight_denom = torch.zeros_like(param)
            f_combined = torch.zeros_like(param)

            for rel_idx, k in enumerate(parent_candidates):
                w_k = float(fitness_weights[rel_idx])
                param_k = dict(agent.experts[k].actor.named_parameters())[name].data
                fim_k = (
                    agent.experts[k].actor.fisher_ema.get(name, torch.zeros_like(param))
                    if agent.experts[k].actor.fisher_ema is not None
                    else torch.zeros_like(param)
                )
                eff_weight = w_k * (fim_k + 1e-4)
                weighted_sum += eff_weight * param_k
                weight_denom += eff_weight
                f_combined += w_k * fim_k
            recombined = weighted_sum / (weight_denom + 1e-8)

            # 2. Targeted deficit push (relative to best parent)
            p_path_idx = [
                i
                for i, (n, _) in enumerate(
                    agent.experts[best_parent].actor.named_parameters()
                )
                if n == name
            ]
            deficit_p = (
                deficit_grads[p_path_idx[0]] if p_path_idx else torch.zeros_like(param)
            )
            targeted = recombined - (THIEF_TARGETED_LR * deficit_p)

            # 3. Geometry-aware exploration noise calibrated by parameter scale
            scale = torch.clamp(1.0 / (torch.sqrt(f_combined) + 1e-8), max=10.0)
            p_std = float(param.std().item()) if param.numel() > 1 and param.std().item() > 1e-5 else 0.01
            noise_std = torch.clamp(scale * THIEF_MUTATION_NOISE, max=0.15 * p_std)
            noise = torch.randn_like(param) * noise_std
            param.copy_(targeted + noise)

    return c_idx, best_parent, best_parent_val


def train(
    algo_name="thief",
    stage_idx=0,
    env_config=None,
    total_timesteps=200_000,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    log_dir=None,
    seed=0,
    use_rust=False,
    ablation="none",
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
    num_envs = THIEF_NUM_ENVS

    curve_logger = TrainCurveLogger(
        run_id=effective_run_id,
        algo=algo_name,
        seed=effective_seed,
        stage=stage_idx,
        results_root="results",
        local_dir=log_dir,
        flush_interval=10,
        ablation=ablation,
    )

    console_logger = logging.getLogger(f"console_{algo_name}_{stage_idx}_{effective_seed}")
    console_logger.setLevel(logging.INFO)
    console_logger.handlers = [logging.StreamHandler()]
    console_logger.propagate = False

    file_logger = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_logger = logging.getLogger(f"file_{algo_name}_{stage_idx}_{effective_seed}")
        file_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        file_logger.handlers = [file_handler]
        file_logger.propagate = False

    engine_tag = "Native Rust Rayon" if use_rust else "Python Multiprocessing"
    ablation_tag = f" | Ablation: {ablation}" if ablation != "none" else ""
    msg = (
        f"Training {algo_name} Stage {stage_idx} (Seed {seed}) on {device} "
        f"({num_envs} envs, {engine_tag} | HER Sub-Goal Navigation | "
        f"Gradient-Interference Spawning{ablation_tag})..."
    )
    console_logger.info(msg)
    if file_logger:
        file_logger.info(msg)

    if env_config is None:
        env_config = dict(CURRICULUM_STAGES[stage_idx])
    vec_env = make_vec_env(
        num_envs,
        config=env_config,
        base_seed=seed * 1000 if seed is not None else 0,
        use_rust=use_rust,
    )
    state_dim = vec_env.state_dim

    agent = ThiefNetwork(state_dim, num_initial_experts=THIEF_INITIAL_EXPERTS).to(
        device
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    active_experts = THIEF_INITIAL_EXPERTS
    total_spawns = 0
    spawn_history = []
    dormant_experts = set()
    env_previous_expert = None

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        try:
            ckpt = torch.load(load_ckpt_path, map_location=device, weights_only=False)
            state_dict = (
                ckpt["model_state"]
                if isinstance(ckpt, dict) and "model_state" in ckpt
                else ckpt
            )
            expert_indices = {
                int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
            }
            needed_experts = max(expert_indices) + 1 if expert_indices else 1
            while len(agent.experts) < needed_experts:
                agent.add_expert()
            if len(agent.experts) > needed_experts:
                agent.prune_experts(list(range(needed_experts)))

            agent.load_state_dict(state_dict, strict=False)
            optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                active_experts = int(ckpt.get("active_experts", len(agent.experts)))
                dormant_experts = set(ckpt.get("dormant_experts", []))
                total_spawns = int(ckpt.get("total_spawns", 0))
                spawn_history = list(ckpt.get("spawn_history", []))
            else:
                active_experts = len(agent.experts)
                dormant_experts = set()

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
                f"(Active experts: {active_bidding}/{active_experts}, total spawns: {total_spawns})"
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load checkpoint ({e}). Initializing fresh.")

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(num_envs).to(device)

    num_steps = 250 if stage_idx >= 3 else NUM_STEPS
    num_updates = total_timesteps // (num_envs * num_steps)
    dynamic_cull_window = max(60, int(0.40 * num_updates))

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

    active_sandbox_mutants = collections.OrderedDict()
    incubation_queue = []
    last_grace_end_update = 0
    last_routing_entropy = 0.0
    expert_consecutive_zero_usage = collections.defaultdict(int)

    last_expert_usage = {}
    last_switch_rate = 0.0
    last_her_reach_rate = 0.0
    last_her_avg_dist = 0.0
    consecutive_conflicts = 0  # Gradient-conflict hysteresis counter

    prev_poses = {
        a: [np.zeros(2, dtype=np.float32) for _ in range(num_envs)] for a in AGENTS
    }
    macro_start_poses = {
        a: [np.zeros(2, dtype=np.float32) for _ in range(num_envs)] for a in AGENTS
    }

    for update in range(1, num_updates + 1):
        check_thermal_guard()

        # Worker Buffers
        b_obs = {
            a: torch.zeros((num_steps, num_envs, *OBSERVATION_SIZE)).to(device)
            for a in AGENTS
        }
        b_role = {
            a: torch.zeros((num_steps, num_envs, N_AGENTS)).to(device) for a in AGENTS
        }
        b_mask = {
            a: torch.zeros((num_steps, num_envs, ACTION_SPACE_SIZE)).to(device)
            for a in AGENTS
        }
        b_goal = {
            a: torch.zeros((num_steps, num_envs, GOAL_VECTOR_DIM)).to(device)
            for a in AGENTS
        }
        b_actions = {a: torch.zeros((num_steps, num_envs)).to(device) for a in AGENTS}
        b_logprobs = {a: torch.zeros((num_steps, num_envs)).to(device) for a in AGENTS}
        b_rewards = {a: torch.zeros((num_steps, num_envs)).to(device) for a in AGENTS}
        b_values = {a: torch.zeros((num_steps, num_envs)).to(device) for a in AGENTS}
        b_dones = torch.zeros((num_steps, num_envs)).to(device)
        b_states = torch.zeros((num_steps, num_envs, state_dim)).to(device)
        b_experts = {
            a: torch.zeros((num_steps, num_envs), dtype=torch.long).to(device)
            for a in AGENTS
        }

        # HER Buffers
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
                actions, logprobs, _, values, chosen_expert = (
                    agent.get_action_and_value(
                        obs_all.flatten(0, 1),
                        role_all.flatten(0, 1),
                        mask_all.flatten(0, 1),
                        goals_all.flatten(0, 1),
                        state_rep,
                        active_experts=active_experts,
                        previous_expert=env_previous_expert,
                        grace_experts=current_sandbox_list,
                        dormant_experts=dormant_experts,
                        epsilon=(
                            0.0 if ablation == "no_hysteresis" else THIEF_HYSTERESIS_EPSILON
                        ),
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
                    b_goal[a][step] = goals_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            # Sub-Goal Potential-Based Progress Reward Shaping
            for i, a in enumerate(AGENTS):
                for e in range(num_envs):
                    agent_info = infos[e].get(a, {})
                    curr_pos = np.array(agent_info.get("pos", (0, 0)), dtype=np.float32)
                    goal_vec_np = goals_all[i, e].cpu().numpy()

                    if step == 0 or (update == 1 and prev_poses[a][e].sum() == 0):
                        prev_poses[a][e] = curr_pos.copy()
                        macro_start_poses[a][e] = curr_pos.copy()

                    delta_step = curr_pos - prev_poses[a][e]
                    progress_val = float(np.dot(delta_step, goal_vec_np))
                    prev_poses[a][e] = curr_pos.copy()

                    int_r = float(np.clip(progress_val, -1.0, 1.0))
                    ext_r = float(rewards[a][e])
                    b_rewards[a][step, e] = ext_r + (THIEF_PROGRESS_COEF * int_r)

                    # HER Macro Horizon Segmentation & Hindsight Relabeling
                    if (step + 1) % THIEF_MACRO_HORIZON == 0:
                        start_pos = macro_start_poses[a][e]
                        disp = curr_pos - start_pos
                        dist_covered = float(np.linalg.norm(disp))
                        her_goals_assigned += 1
                        her_total_dist += dist_covered

                        if dist_covered >= THIEF_HER_REACH_DIST:
                            her_goals_reached += 1
                        elif dist_covered > 0.1:
                            hg = np.clip(disp / (dist_covered + 1e-5), -1.0, 1.0)
                            hg_t = torch.tensor(
                                hg, dtype=torch.float32, device=device
                            )

                            seg_start = step - THIEF_MACRO_HORIZON + 1
                            for tau in range(seg_start, step + 1):
                                her_relabeled_obs.append(b_obs[a][tau, e])
                                her_relabeled_role.append(b_role[a][tau, e])
                                her_relabeled_mask.append(b_mask[a][tau, e])
                                her_relabeled_actions.append(b_actions[a][tau, e])
                                her_relabeled_goal.append(hg_t)
                                her_relabeled_expert.append(b_experts[a][tau, e])

                        macro_start_poses[a][e] = curr_pos.copy()

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

        # Track usage and switch rate
        experts_stacked = torch.stack([b_experts[a] for a in AGENTS])
        all_experts_tensor = experts_stacked.view(-1)
        usage_counts = torch.bincount(all_experts_tensor, minlength=active_experts)
        total_samples = all_experts_tensor.numel()
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
            goals_all = torch.tensor(
                stacked["goal_vector"], dtype=torch.float32, device=device
            ).flatten(0, 1)
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
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
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

        if her_goals_assigned > 0:
            last_her_reach_rate = (her_goals_reached / her_goals_assigned) * 100.0
            last_her_avg_dist = her_total_dist / her_goals_assigned

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

        # HER auxiliary tensors
        has_her = len(her_relabeled_actions) > 0
        if has_her:
            her_obs_t = torch.stack(her_relabeled_obs, dim=0)
            her_role_t = torch.stack(her_relabeled_role, dim=0)
            her_mask_t = torch.stack(her_relabeled_mask, dim=0)
            her_act_t = torch.stack(her_relabeled_actions, dim=0)
            her_goal_t = torch.stack(her_relabeled_goal, dim=0)
            her_exp_t = torch.stack(her_relabeled_expert, dim=0)
            her_total = her_act_t.shape[0]

        # --- PPO OPTIMIZATION LOOP ---
        batch_size = total_samples
        minibatch_size = batch_size // 4
        b_inds = np.arange(batch_size)

        # Full-Batch Advantage Normalization per expert
        flat_adv_norm = torch.zeros_like(flat_advantages)
        for k in range(active_experts):
            k_mask = flat_experts == k
            if k_mask.sum() > 1:
                k_adv = flat_advantages[k_mask]
                flat_adv_norm[k_mask] = (k_adv - k_adv.mean()) / (k_adv.std() + 1e-8)
            elif k_mask.sum() == 1:
                flat_adv_norm[k_mask] = 0.0
        flat_advantages = flat_adv_norm

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)

            # Worker Specialists Update Step
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_exp = flat_experts[mb_inds]
                mb_adv = flat_advantages[mb_inds]

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

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.smooth_l1_loss(newvalue, mb_returns, beta=1.0)
                entropy_loss = entropy.mean()

                # MoE Load Balancing Loss across candidate experts to prevent monopoly collapse
                active_candidates = [
                    k for k in range(active_experts) if k not in dormant_experts
                ]
                if len(active_candidates) > 1:
                    x_crit_mb = torch.cat([mb_state, mb_role, mb_goal], dim=1)
                    cand_bids = torch.stack(
                        [
                            agent.experts[k].critic(x_crit_mb).squeeze(-1)
                            for k in active_candidates
                        ],
                        dim=1,
                    )
                    routing_probs = F.softmax(cand_bids / 2.0, dim=1)
                    mean_p = routing_probs.mean(dim=0)
                    freqs = torch.zeros_like(mean_p)
                    for idx_c, k in enumerate(active_candidates):
                        freqs[idx_c] = (mb_exp == k).float().mean()
                    load_balance_loss = len(active_candidates) * (mean_p * freqs).sum()
                else:
                    load_balance_loss = torch.tensor(0.0, device=device)

                balance_coef = 0.0 if ablation == "no_balance" else 0.01

                # Auxiliary HER Hindsight Navigation Policy Loss
                her_loss = torch.tensor(0.0, device=device)
                if has_her:
                    her_mb_size = min(minibatch_size, her_total)
                    her_mb_inds = np.random.choice(
                        her_total, size=her_mb_size, replace=False
                    )
                    h_obs = her_obs_t[her_mb_inds]
                    h_role = her_role_t[her_mb_inds]
                    h_mask = her_mask_t[her_mb_inds]
                    h_goal = her_goal_t[her_mb_inds]
                    h_act = her_act_t[her_mb_inds]
                    h_exp = her_exp_t[her_mb_inds]

                    total_her_loss = 0.0
                    n_expert_groups = 0
                    for k in range(active_experts):
                        k_sel = h_exp == k
                        if k_sel.sum() > 0:
                            k_obs = h_obs[k_sel]
                            k_role = h_role[k_sel]
                            k_mask = h_mask[k_sel]
                            k_goal = h_goal[k_sel]
                            k_act = h_act[k_sel]

                            bk = k_obs.shape[0]
                            x_actor = torch.cat(
                                [k_obs.view(bk, -1), k_role, k_goal], dim=1
                            )
                            logits = agent.experts[k].actor(x_actor)
                            logits = logits.masked_fill(k_mask == 0, -1e8)
                            probs = Categorical(logits=logits)
                            total_her_loss = (
                                total_her_loss - probs.log_prob(k_act).mean()
                            )
                            n_expert_groups += 1

                    if n_expert_groups > 0:
                        her_loss = total_her_loss / n_expert_groups

                loss = (
                    pg_loss
                    - (ENT_COEF * entropy_loss)
                    + (VF_COEF * v_loss)
                    + (balance_coef * load_balance_loss)
                    + (THIEF_HER_AUX_COEF * her_loss)
                )

                optimizer.zero_grad()
                loss.backward()

                # Track optimization diagnostics
                last_pg_loss = pg_loss.item()
                last_v_loss = v_loss.item()
                last_ent_loss = entropy_loss.item()
                last_approx_kl = ((ratio - 1.0) - logratio).mean().item()
                last_clip_frac = (torch.abs(ratio - 1.0) > CLIP_COEF).float().mean().item()
                var_y = torch.var(mb_returns)
                last_explained_var = (
                    (1.0 - torch.var(mb_returns - newvalue) / (var_y + 1e-8)).item()
                    if var_y > 1e-8
                    else 0.0
                )

                # Update EMA Fisher diagonal
                for k in range(active_experts):
                    for name, param in agent.experts[k].actor.named_parameters():
                        if param.grad is not None:
                            if agent.experts[k].actor.fisher_ema is None:
                                agent.experts[k].actor.fisher_ema = {}
                            if name not in agent.experts[k].actor.fisher_ema:
                                agent.experts[k].actor.fisher_ema[name] = (
                                    torch.zeros_like(param.grad)
                                )
                            agent.experts[k].actor.fisher_ema[name] = (
                                0.99 * agent.experts[k].actor.fisher_ema[name]
                                + 0.01 * param.grad.data**2
                            )
                last_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), 0.5).item()
                optimizer.step()

        # Step A: Warmup progression & queue processing
        isolation_window = (
            min(THIEF_ISOLATION_UPDATES, max(5, int(0.15 * num_updates)))
            if num_updates
            else THIEF_ISOLATION_UPDATES
        )
        graduated_this_update = []
        for m_idx in list(active_sandbox_mutants.keys()):
            active_sandbox_mutants[m_idx] += 1
            if active_sandbox_mutants[m_idx] >= isolation_window:
                graduated_this_update.append(m_idx)
                del active_sandbox_mutants[m_idx]

        if graduated_this_update:
            grad_names = ", ".join(f"E{i}" for i in graduated_this_update)
            msg = (
                f"[Promotion] Update {update} | Expert(s) ({grad_names}) "
                f"completed warmup ({THIEF_ISOLATION_UPDATES} updates) -> added to bidding pool"
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

        # Pull from warmup queue into newly opened slots
        while (
            len(active_sandbox_mutants) < THIEF_MAX_SANDBOX_EXPERTS
            and len(incubation_queue) > 0
        ):
            next_m = incubation_queue.pop(0)
            active_sandbox_mutants[next_m] = 0
            msg = (
                f"[Warmup Queue] Update {update} | Expert E{next_m} entered warmup "
                f"({len(active_sandbox_mutants)} active, {len(incubation_queue)} waiting in queue)"
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

        # Mark last_grace_end_update timestamp when entire cohort has finished warmup
        if (
            graduated_this_update
            and len(active_sandbox_mutants) == 0
            and len(incubation_queue) == 0
        ):
            last_grace_end_update = update
            msg = (
                f"[Warmup Complete] Update {update} | All experts active. "
                f"Enforcing {THIEF_POST_GRACE_COOLDOWN_UPDATES}-update cooldown before next trigger."
            )
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)

        # Inactive expert check:
        for k in range(active_experts):
            if k in dormant_experts:
                continue
            if expert_consecutive_zero_usage[k] >= dynamic_cull_window:
                dormant_experts.add(k)
                msg = (
                    f"[Pruning] Update {update} | Deactivating expert E{k} "
                    f"(0% usage for >= {dynamic_cull_window} updates)"
                )
                console_logger.info(msg)
                if file_logger:
                    file_logger.info(msg)

        # Step B: Dynamic Gradient Interference Spawning Check
        if not active_sandbox_mutants and not incubation_queue:
            if ablation == "fixed_schedule":
                # Ablation: bypass gradient-conflict detection; spawn on a fixed cadence instead.
                FIXED_SPAWN_INTERVAL = 60
                conflict_triggered = update % FIXED_SPAWN_INTERVAL == 0
                deficit_mask, cos_sim = None, 0.0
            else:
                conflict_triggered, deficit_mask, cos_sim = (
                    check_gradient_interference_trigger(
                        agent,
                        active_experts,
                        flat_obs,
                        flat_role,
                        flat_mask,
                        flat_actions,
                        flat_goal,
                        flat_advantages,
                        update,
                        last_grace_end_update,
                        active_sandbox_mutants,
                        incubation_queue,
                        device,
                        dormant_experts=dormant_experts,
                        num_updates=num_updates,
                        b_experts=flat_experts,
                    )
                )
                # Hysteresis: require the conflict to persist on 2 consecutive
                # updates before committing to a spawn (damps single-check noise).
                consecutive_conflicts = consecutive_conflicts + 1 if conflict_triggered else 0
                conflict_triggered = consecutive_conflicts >= 2
            min_updates_needed = (
                THIEF_ISOLATION_UPDATES
                + THIEF_WARMUP_UPDATES
                + THIEF_POST_GRACE_COOLDOWN_UPDATES
            )
            has_enough_time = (num_updates - update) >= min_updates_needed

            if conflict_triggered and has_enough_time:
                c_idx, best_parent, parent_val = spawn_targeted_specialist(
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
                    flat_experts,
                    device,
                    dormant_experts=dormant_experts,
                    ablation=ablation,
                    deficit_mask=deficit_mask,
                )
                old_count = active_experts
                active_experts += 1
                total_spawns += 1
                consecutive_conflicts = 0

                # Preserve Adam momentum and variance state for surviving parents
                optimizer = update_optimizer_params(optimizer, agent, lr=LR)

                if ablation == "no_incubation":
                    # Ablation: child skips warmup entirely and joins the
                    # global bidding pool with its cold-start critic immediately.
                    last_grace_end_update = update
                elif len(active_sandbox_mutants) < THIEF_MAX_SANDBOX_EXPERTS:
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
                        "trigger": (
                            "fixed_schedule" if ablation == "fixed_schedule" else "gradient_interference"
                        ),
                        "cos_sim": cos_sim,
                        "deficit_samples": int(deficit_mask.sum().item()) if deficit_mask is not None else 0,
                    }
                )

                trigger_name = (
                    "Fixed-Schedule Event" if ablation == "fixed_schedule" else "Gradient Conflict Event"
                )
                msg = (
                    f"[{trigger_name}] Update {update} | "
                    f"Spawned expert E{c_idx} from parent E{best_parent} (Pool: {old_count} -> {active_experts} experts) | "
                    + (
                        f"(cos_sim = {cos_sim:.3f} < {THIEF_GRADIENT_CONFLICT_THRESHOLD}) | "
                        if ablation != "fixed_schedule"
                        else ""
                    )
                    + f"In warmup: {list(active_sandbox_mutants.keys())}, Queue: {incubation_queue}"
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

            # Routing entropy H(f) in nats over non-dormant experts (0 = full monopoly)
            usage_probs = [
                last_expert_usage.get(k, 0.0) / 100.0
                for k in range(active_experts)
                if k not in dormant_experts
            ]
            if usage_probs and sum(usage_probs) > 0:
                total_p = sum(usage_probs)
                routing_entropy = abs(
                    float(
                        -sum(
                            (p / total_p) * np.log(p / total_p)
                            for p in usage_probs
                            if p > 0
                        )
                    )
                )
            else:
                routing_entropy = 0.0
            last_routing_entropy = routing_entropy

            # Compact console log
            console_str = (
                f"Update: {update}/{num_updates} | "
                f"Win Rate: {avg_win:.2f} | "
                f"Episodic Return: {avg_rew:.3f} | "
                f"Steps: {avg_steps:.1f}/{stage_max_steps} | "
                f"Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | "
                f"Active Experts: {active_bidding_count}/{active_experts} | "
                f"Goals Reached: {last_her_reach_rate:.1f}% (Dist: {last_her_avg_dist:.1f})"
            )
            console_logger.info(console_str)

            # Comprehensive file log
            if file_logger:
                file_str = (
                    f"Update: {update}/{num_updates} | "
                    f"Win Rate: {avg_win:.2f} | "
                    f"Episodic Return: {avg_rew:.3f} | "
                    f"Steps: {avg_steps:.1f}/{stage_max_steps} | "
                    f"Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | "
                    f"Scout Tag Rate: {avg_scout_int:.2f} (Avg POIs: {avg_scout_pois:.1f}) | "
                    f"Hacker Hack Rate: {avg_hacker_hack:.2f} | "
                    f"Muscle Neutralize Rate: {avg_muscle_neut:.2f} (Avg Guards: {avg_muscle_guards:.1f}/{stage_guards}) | "
                    f"Extractor Loot Rate: {avg_extractor_loot:.2f} | "
                    f"Avg Agents at Extract: {avg_agents_extract:.2f}/4 | "
                    f"Active: {active_bidding_count}/{active_experts} | "
                    f"Total Spawns: {total_spawns} | "
                    f"Switch Rate: {last_switch_rate:.1f}% | "
                    f"Routing Entropy: {routing_entropy:.3f} nats | "
                    f"Goals Reached: {last_her_reach_rate:.1f}% (Avg Dist: {last_her_avg_dist:.1f} tiles, Relabeled: {len(her_relabeled_actions)}) | "
                    f"Usage: [{usage_str}]"
                )
                file_logger.info(file_str)

            curve_logger.log_update(
                update=update,
                total_updates=num_updates,
                timesteps=update * num_steps * num_envs,
                elapsed_sec=time.time() - start_time,
                fps=float((update * num_steps * num_envs) / max(0.001, time.time() - start_time)),
                win_rate=avg_win,
                mean_return=avg_rew,
                avg_steps=avg_steps,
                avg_alarm=avg_alarm,
                stage_alarm_max=stage_alarm_max,
                scout_tag_rate=avg_scout_int,
                hacker_hack_rate=avg_hacker_hack,
                muscle_neut_rate=avg_muscle_neut,
                extractor_loot_rate=avg_extractor_loot,
                avg_agents_extract=avg_agents_extract,
                active_experts=active_bidding_count,
                effective_experts=float(np.exp(routing_entropy)) if routing_entropy > 0 else 1.0,
                total_spawns=total_spawns,
                switch_rate=last_switch_rate,
                routing_entropy=routing_entropy,
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

    # Ensure curve logger is closed/flushed
    curve_logger.close()

    # Build role-expert contingency matrix for Mutual Information
    role_expert_counts = np.zeros((N_AGENTS, active_experts), dtype=np.float64)
    if "flat_role" in locals() and "flat_experts" in locals():
        role_idx = flat_role.argmax(dim=-1) if flat_role.ndim > 1 else flat_role
        for r in range(N_AGENTS):
            for k in range(active_experts):
                role_expert_counts[r, k] = float(
                    ((role_idx == r) & (flat_experts == k)).sum().item()
                )

    stage_max_steps = env_config.get("max_steps", 300)
    stage_alarm_max = float(env_config.get("alarm_max", 100.0))
    stage_guards = env_config.get("guard_count", 0)
    elapsed_wall_clock = time.time() - start_time

    results_data = record_stage_completion(
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
        stage_max_steps=stage_max_steps,
        stage_alarm_max=stage_alarm_max,
        total_stage_guards=stage_guards,
        map_size=env_config.get("map_size", (11, 11))[0] if env_config else 11,
        active_experts=active_experts,
        dormant_experts_count=len(dormant_experts),
        total_spawns=total_spawns,
        expert_switch_rate=last_switch_rate,
        routing_entropy=last_routing_entropy,
        expert_usage_pct=last_expert_usage,
        role_expert_counts=role_expert_counts,
        ablation=ablation,
        spawn_history=spawn_history,
        save_dir=log_dir or save_ckpt_dir,
        results_root="results",
    )
    print(f"Recorded stage results to Parquet and JSON in {log_dir or save_ckpt_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="THIEF Training")
    parser.add_argument("--stage", type=int, default=0, help="Stage index")
    parser.add_argument(
        "--timesteps", type=int, default=200_000, help="Total timesteps"
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
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "no_incubation",
            "uniform_recomb",
            "clone_best",
            "no_balance",
            "no_hysteresis",
            "fixed_schedule",
        ],
        help="THIEF micro-ablation variant (default: none = full THIEF)",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"results/thief/seed_{args.seed}/stage_{args.stage}"

    load_ckpt = args.load_ckpt
    if load_ckpt is None and args.stage > 0:
        prev_ckpt = f"results/thief/seed_{args.seed}/stage_{args.stage - 1}/model.pt"
        if os.path.exists(prev_ckpt):
            load_ckpt = prev_ckpt

    train(
        algo_name="thief",
        stage_idx=args.stage,
        total_timesteps=args.timesteps,
        load_ckpt_path=load_ckpt,
        save_ckpt_dir=save_dir,
        log_dir=save_dir,
        seed=args.seed,
        use_rust=args.rust,
        ablation=args.ablation,
        run_id=args.run_id,
    )
