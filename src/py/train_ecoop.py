"""
E-COOP: Mixture of experts with value-bidding routing, expert spawning, and switching hysteresis.
"""

import collections
import json
import logging
import os

import numpy as np
import torch
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
    ECOOP_MUTATION_NOISE,
    ECOOP_NUM_ENVS,
    ECOOP_PROGRESS_COEF,
    ENT_COEF,
    GAE_LAMBDA,
    GAMMA,
    GOAL_VECTOR_DIM,
    LR,
    N_AGENTS,
    NUM_STEPS,
    OBSERVATION_SIZE,
    UPDATE_EPOCHS,
    VF_COEF,
)
from thermal_guard import check_thermal_guard
from torch.nn import functional as F
from torch import nn
from torch.distributions.categorical import Categorical
from vec_env import make_vec_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MappoNetwork(nn.Module):
    """
    Individual Expert Actor-Critic module.

    Each specialist maintains:
    - Actor MLP: maps local egocentric observation + one-hot role vector + goal vector -> action logits.
    - Critic MLP: maps global centralized environment state + one-hot role vector + goal vector -> state value V(s, role).
    """

    def __init__(self, state_dim):
        super().__init__()
        actor_in_dim = (
            (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS + GOAL_VECTOR_DIM
        )
        critic_in_dim = state_dim + N_AGENTS + GOAL_VECTOR_DIM

        # Policy Network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(actor_in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, ACTION_SPACE_SIZE), std=0.01),
        )

        # Centralized Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Fisher diagonal EMA (initialized on first backward)
        self.fisher_ema = None

    def get_action_and_value(self, obs, role, mask, goal, state=None, action=None):
        """
        Forward pass for a batch of transitions assigned to this expert.
        Applies invalid action masking (-1e9) prior to categorical sampling.
        """
        x_actor = torch.cat([obs.flatten(start_dim=1), role, goal], dim=1)
        logits = self.actor(x_actor)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        if state is not None:
            x_critic = torch.cat([state, role, goal], dim=1)
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
        grace_expert=None,
        epsilon=ECOOP_HYSTERESIS_EPSILON,
    ):
        """
        Executes routing and policy forward passes across the expert pool.
        """
        batch_size = obs.shape[0]
        x_critic = torch.cat([state, role, goal], dim=1)

        # 1. Routing Selection Phase
        if expert_idx is not None:
            chosen_expert = expert_idx
        else:
            # Mask out grace expert from general competitive bidding pool
            if grace_expert is not None and grace_expert < active_experts:
                candidate_experts = [
                    k for k in range(active_experts) if k != grace_expert
                ]
            else:
                candidate_experts = list(range(active_experts))

            candidate_tensor = torch.tensor(
                candidate_experts, dtype=torch.long, device=obs.device
            )

            # Direct Role-Conditioned Critic Bids: V_k(s, role, goal)
            expert_bids = torch.stack(
                [
                    self.experts[k].critic(x_critic).squeeze(-1)
                    for k in candidate_experts
                ],
                dim=1,
            )

            # Greedy Argmax with Vectorized Hybrid Hysteresis for temporal policy coherence
            best_rel_idx = torch.argmax(expert_bids, dim=1)
            best_idx = candidate_tensor[best_rel_idx]

            if previous_expert is not None:
                has_prev = previous_expert >= 0
                matches = previous_expert.unsqueeze(1) == candidate_tensor
                prev_in_candidates = matches.any(dim=1)
                prev_rel_idx = matches.long().argmax(dim=1)

                prev_bid = torch.gather(
                    expert_bids, 1, prev_rel_idx.unsqueeze(1)
                ).squeeze(1)
                best_bid = torch.gather(
                    expert_bids, 1, best_rel_idx.unsqueeze(1)
                ).squeeze(1)

                # Dynamic switching threshold
                switch_thresh = prev_bid + torch.clamp(
                    epsilon * torch.abs(prev_bid), min=epsilon
                )
                should_switch = (best_idx != previous_expert) & (
                    best_bid > switch_thresh
                )
                chosen_expert = torch.where(
                    has_prev & prev_in_candidates & ~should_switch,
                    previous_expert,
                    best_idx,
                )
            else:
                chosen_expert = best_idx

            # Environment Sharding Grace Routing
            if grace_expert is not None and grace_expert < active_experts:
                threshold_env = (
                    ECOOP_NUM_ENVS // 2
                )  # 50% of environments for the mutant
                env_ids = torch.arange(batch_size, device=obs.device) % ECOOP_NUM_ENVS
                is_grace_env = env_ids >= threshold_env
                chosen_expert = torch.where(
                    is_grace_env,
                    torch.tensor(
                        grace_expert,
                        device=obs.device,
                        dtype=chosen_expert.dtype,
                    ),
                    chosen_expert,
                )

        # 2. Execution
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

            actions[mask_k] = a.long()
            logprobs[mask_k] = lp
            entropies[mask_k] = ent
            if v is not None:
                values[mask_k] = v

        return actions, logprobs, entropies, values, chosen_expert


def train(
    algo_name="ecoop",
    stage_idx=0,
    env_config=None,
    total_timesteps=120_000,
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
    num_envs = ECOOP_NUM_ENVS
    vec_env = make_vec_env(
        num_envs,
        config=env_config,
        base_seed=seed * 1000 if seed is not None else 0,
        use_rust=use_rust,
    )
    state_dim = vec_env.state_dim

    agent = EcoopNetwork(state_dim, num_initial_experts=1).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    active_experts = 1
    total_spawns = 0
    spawn_history = []
    current_grace_expert = None
    grace_updates_remaining = 0
    expert_consecutive_zero_usage = collections.defaultdict(int)
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
            needed_experts = max(expert_indices) + 1 if expert_indices else 1
            while len(agent.experts) < needed_experts:
                agent.add_expert()
            if len(agent.experts) > needed_experts:
                agent.prune_experts(list(range(needed_experts)))

            agent.load_state_dict(state_dict)
            optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

            if isinstance(ckpt, dict) and "active_experts" in ckpt:
                active_experts = ckpt["active_experts"]
            else:
                active_experts = len(agent.experts)

            if isinstance(ckpt, dict) and "total_spawns" in ckpt:
                total_spawns = ckpt["total_spawns"]
            if isinstance(ckpt, dict) and "spawn_history" in ckpt:
                spawn_history = list(ckpt["spawn_history"])

            msg = f"Successfully loaded checkpoint from {load_ckpt_path} (Active experts: {active_experts})"
            console_logger.info(msg)
            if file_logger:
                file_logger.info(msg)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not load full checkpoint ({e}). Initializing fresh.")

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(num_envs).to(device)

    num_steps = 250 if stage_idx >= 3 else NUM_STEPS
    num_updates = total_timesteps // (num_envs * num_steps)
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

    prev_poses = {
        a: [np.zeros(2, dtype=np.float32) for _ in range(num_envs)] for a in AGENTS
    }

    for update in range(1, num_updates + 1):
        check_thermal_guard()
        if grace_updates_remaining > 0:
            grace_updates_remaining -= 1
            if grace_updates_remaining == 0:
                current_grace_expert = None

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

        total_switches = 0
        total_switch_opportunities = 0

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
                        grace_expert=current_grace_expert,
                    )
                )

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

                    if step == 0:
                        prev_poses[a][e] = curr_pos

                    delta_step = curr_pos - prev_poses[a][e]
                    progress_val = float(np.dot(delta_step, goal_vec_np))
                    prev_poses[a][e] = curr_pos

                    int_r = float(np.clip(progress_val, -1.0, 1.0))
                    ext_r = float(rewards[a][e])
                    b_rewards[a][step, e] = ext_r + (ECOOP_PROGRESS_COEF * int_r)

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
        total_decisions = max(1, all_experts_tensor.numel())
        last_expert_usage = {
            str(k): round(
                float((all_experts_tensor == k).sum().item()) / total_decisions * 100.0,
                1,
            )
            for k in range(active_experts)
        }
        if experts_stacked.shape[1] > 1:
            total_switches = (
                (experts_stacked[:, 1:] != experts_stacked[:, :-1]).sum().item()
            )
            total_switch_opps = experts_stacked[:, 1:].numel()
            last_switch_rate = float(total_switches) / max(1, total_switch_opps)
        else:
            last_switch_rate = 0.0

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
            next_goals_all = torch.tensor(
                stacked_next["goal_vector"], dtype=torch.float32, device=device
            )
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)

            _, _, _, next_val, _ = agent.get_action_and_value(
                next_obs_all.flatten(0, 1),
                next_role_all.flatten(0, 1),
                next_mask_all.flatten(0, 1),
                next_goals_all.flatten(0, 1),
                next_state_t.repeat(N_AGENTS, 1),
                active_experts=active_experts,
                previous_expert=env_previous_expert,
                grace_expert=current_grace_expert,
            )
            next_val = next_val.view(N_AGENTS, num_envs)

            b_adv = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}
            for i, a in enumerate(AGENTS):
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
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
                goal_flat = b_goal[a].reshape(-1, GOAL_VECTOR_DIM)
                action_flat = b_actions[a].reshape(-1)
                logprob_flat = b_logprobs[a].reshape(-1)
                adv_flat = b_adv[a].reshape(-1)
                ret_flat = b_returns[a].reshape(-1)
                experts_flat = b_experts[a].reshape(-1)

                # Per-Expert Advantage Normalization
                adv_norm = torch.zeros_like(adv_flat)
                for k in range(active_experts):
                    mask_k = experts_flat == k
                    if mask_k.any():
                        adv_k = adv_flat[mask_k]
                        if adv_k.numel() > 1:
                            adv_norm[mask_k] = (adv_k - adv_k.mean()) / (
                                adv_k.std() + 1e-8
                            )
                        else:
                            adv_norm[mask_k] = adv_k - adv_k.mean()
                adv_flat = adv_norm

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    obs_flat,
                    role_flat,
                    mask_flat,
                    goal_flat,
                    b_states_flat,
                    active_experts=active_experts,
                    action=action_flat,
                    expert_idx=experts_flat,
                )

                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(
                    ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.smooth_l1_loss(newvalue, ret_flat, beta=1.0)
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()

                # Track optimization diagnostics
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

                # Update EMA Fisher diagonal
                for k in range(active_experts):
                    for name, param in agent.experts[k].actor.named_parameters():
                        if param.grad is not None:
                            if agent.experts[k].fisher_ema is None:
                                agent.experts[k].fisher_ema = {}
                            if name not in agent.experts[k].fisher_ema:
                                agent.experts[k].fisher_ema[name] = torch.zeros_like(
                                    param.grad
                                )
                            agent.experts[k].fisher_ema[name] = (
                                0.99 * agent.experts[k].fisher_ema[name]
                                + 0.01 * param.grad.data**2
                            )
                last_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), 0.5).item()
                optimizer.step()

        # --- EVOLUTIONARY CROSSOVER / FIM MUTATION & ACTIVE PRUNING ---
        min_updates_remaining = round(
            ECOOP_GRACE_UPDATES + (ECOOP_EVOLUTION_INTERVAL / 2.0)
        )
        # Only spawn if the champion is plateauing
        do_crossover = (update % ECOOP_EVOLUTION_INTERVAL == 0) and (
            (num_updates - update) >= min_updates_remaining
        )

        all_experts_tensor = torch.stack([b_experts[a] for a in AGENTS])
        for k in range(active_experts):
            if (all_experts_tensor == k).any() or (k == current_grace_expert):
                expert_consecutive_zero_usage[k] = 0
            else:
                expert_consecutive_zero_usage[k] += 1

        if do_crossover:
            # Check win-rate plateau and mastery ceiling
            if len(completed_wins) >= 200:
                recent = completed_wins[-100:]
                half = len(recent) // 2
                first_half = float(np.mean(recent[:half]))
                second_half = float(np.mean(recent[half:]))
                improvement = second_half - first_half
                if second_half >= 0.90 or improvement >= 0.03:
                    # Already mastered or still improving — skip spawning, but still do culling
                    do_crossover = False

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
                        sample_roles = torch.cat(
                            [b_role[ag].flatten(0, 1) for ag in AGENTS], dim=0
                        )
                        sample_goals = torch.cat(
                            [b_goal[ag].flatten(0, 1) for ag in AGENTS], dim=0
                        )
                        sample_states = torch.cat(
                            [b_states.flatten(0, 1) for _ in AGENTS], dim=0
                        )
                        x_critic_eval = torch.cat(
                            [sample_states, sample_roles, sample_goals], dim=1
                        )
                        mean_v = agent.experts[k].critic(x_critic_eval).mean().item()
                expert_mean_vals[k] = mean_v
                if mean_v > best_val:
                    best_val = mean_v
                    best_expert = k

            # Option A: Prune inactive experts (>= ECOOP_CULL_WINDOW_UPDATES updates without usage)
            culled_indices = [
                k
                for k in range(active_experts)
                if k != best_expert
                and expert_consecutive_zero_usage[k] >= ECOOP_CULL_WINDOW_UPDATES
                and k != current_grace_expert
            ]

            if culled_indices:
                survivors = [
                    k for k in range(active_experts) if k not in culled_indices
                ]
                cull_names = ", ".join(f"E{c}" for c in culled_indices)
                cull_msg = f"[Pruning] Update {update} | Pruned {len(culled_indices)} inactive expert(s) ({cull_names}) (>= {ECOOP_CULL_WINDOW_UPDATES} updates without usage)"
                console_logger.info(cull_msg)
                if file_logger:
                    file_logger.info(cull_msg)

                # Prune culled parameters from optimizer state and rebuild optimizer param_groups
                for culled_idx in culled_indices:
                    for p in agent.experts[culled_idx].parameters():
                        if p in optimizer.state:
                            del optimizer.state[p]
                agent.prune_experts(survivors)
                optimizer.param_groups.clear()
                for exp in agent.experts:
                    optimizer.add_param_group({"params": exp.parameters(), "lr": LR})
                env_previous_expert = None

                if current_grace_expert is not None:
                    if current_grace_expert in survivors:
                        current_grace_expert = survivors.index(current_grace_expert)
                    else:
                        current_grace_expert = None
                        grace_updates_remaining = 0

                survivor_mean_vals = [
                    expert_mean_vals[old_idx] for old_idx in survivors
                ]
                new_zero_usage = collections.defaultdict(int)

                for new_idx, old_idx in enumerate(survivors):
                    new_zero_usage[new_idx] = expert_consecutive_zero_usage[old_idx]

                expert_consecutive_zero_usage = new_zero_usage
                best_expert = survivors.index(best_expert)
                active_experts = len(survivors)
                expert_mean_vals = {
                    i: survivor_mean_vals[i] for i in range(active_experts)
                }

            # ... rest of spawn block only runs if do_crossover is still True
            if do_crossover:
                # Dynamic Pool Growth: Spawn new expert into newly appended slot
                parent_pool_size = active_experts
                new_expert = agent.add_expert()
                optimizer.add_param_group(
                    {"params": agent.experts[new_expert].parameters(), "lr": LR}
                )
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
                    "crossover_type": "all_pool_recombination",
                }
                spawn_history.append(event_record)

                msg = f"[Spawn] Update {update} | Spawned E{new_expert} from E{best_expert} (val={best_val:.3f}) | Active pool: {active_experts} experts | Total spawns: {total_spawns}"
                console_logger.info(msg)
                if file_logger:
                    file_logger.info(msg)

                # 1. Compute Softmax Fitness Weights across ALL parent experts in the pool
                val_tensor = torch.tensor(
                    [expert_mean_vals[k] for k in range(parent_pool_size)],
                    dtype=torch.float32,
                )
                fitness_weights = torch.softmax(val_tensor, dim=0).numpy()

                # 2. Copy best critic and recombine actor parameters
                agent.experts[new_expert].critic.load_state_dict(
                    agent.experts[best_expert].critic.state_dict()
                )

                with torch.no_grad():
                    for name, param in agent.experts[
                        new_expert
                    ].actor.named_parameters():
                        weighted_param_sum = torch.zeros_like(param)
                        weight_denom = torch.zeros_like(param)
                        f_combined = torch.zeros_like(param)

                        for k in range(parent_pool_size):
                            w_k = float(fitness_weights[k])
                            param_k = dict(agent.experts[k].actor.named_parameters())[
                                name
                            ].data
                            fim_k = (
                                agent.experts[k].fisher_ema.get(
                                    name, torch.zeros_like(param)
                                )
                                if agent.experts[k].fisher_ema is not None
                                else torch.zeros_like(param)
                            )

                            # Effective Fisher weight = w_k * (F_k + ECOOP_CROSSOVER_DAMPING)
                            eff_weight = w_k * (fim_k + ECOOP_CROSSOVER_DAMPING)
                            weighted_param_sum += eff_weight * param_k
                            weight_denom += eff_weight
                            f_combined += w_k * fim_k

                        # Recombined parameter
                        recombined = weighted_param_sum / (weight_denom + 1e-8)

                        # Parameter perturbation
                        scale = torch.clamp(
                            1.0 / (torch.sqrt(f_combined) + 1e-8), max=10.0
                        )
                        noise = torch.randn_like(param) * scale * ECOOP_MUTATION_NOISE
                        param.copy_(recombined + noise)

                # Initialize optimizer state for new expert
                for p in agent.experts[new_expert].parameters():
                    if p in optimizer.state:
                        del optimizer.state[p]

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

            expert_usage_str = ", ".join(
                f"E{k}: {v}%" for k, v in last_expert_usage.items()
            )

            # Console log
            console_logger.info(
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | Active Experts: {active_experts}"
            )
            # File log
            if file_logger:
                file_logger.info(
                    f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | Scout Tag Rate: {scout_tag_rate:.2f} (Avg POIs: {scout_avg_pois:.1f}) | Hacker Hack Rate: {hacker_hack_rate:.2f} | Muscle Neutralize Rate: {muscle_neutralize_rate:.2f} (Avg Guards: {avg_muscle_guards:.1f}/{total_stage_guards}) | Extractor Loot Rate: {extractor_loot_rate:.2f} | Avg Agents at Extract: {avg_agents_extract:.2f}/4 | Active: {active_experts} | Total Spawns: {total_spawns} | Switch Rate: {last_switch_rate * 100:.1f}% | Usage: [{expert_usage_str}]"
                )

            curve_logger.log_update(
                update=update,
                total_updates=num_updates,
                timesteps=update * NUM_STEPS * num_envs,
                elapsed_sec=time.time() - start_time,
                fps=float((update * NUM_STEPS * num_envs) / max(0.001, time.time() - start_time)),
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
                active_experts=active_experts,
                effective_experts=float(np.exp(0.0)),
                total_spawns=total_spawns,
                switch_rate=last_switch_rate,
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
        checkpoint_payload = {
            "model_state": agent.state_dict(),
            "active_experts": active_experts,
            "total_spawns": total_spawns,
            "spawn_history": spawn_history,
        }
        torch.save(checkpoint_payload, os.path.join(save_ckpt_dir, "model.pt"))

    curve_logger.close()

    role_expert_counts = np.zeros((N_AGENTS, active_experts), dtype=np.float64)
    if "role_flat" in locals() and "experts_flat" in locals():
        role_idx = role_flat.argmax(dim=-1) if role_flat.ndim > 1 else role_flat
        for r in range(N_AGENTS):
            for k in range(active_experts):
                role_expert_counts[r, k] = float(
                    ((role_idx == r) & (experts_flat == k)).sum().item()
                )

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
        stage_max_steps=int(max_stage_steps) if "max_stage_steps" in locals() else 150,
        stage_alarm_max=float(stage_alarm_max) if "stage_alarm_max" in locals() else 100.0,
        total_stage_guards=int(total_stage_guards) if "total_stage_guards" in locals() else 0,
        map_size=env_config.get("map_size", (11, 11))[0] if env_config else 11,
        active_experts=active_experts,
        dormant_experts_count=0,
        total_spawns=total_spawns,
        expert_switch_rate=float(last_switch_rate),
        routing_entropy=0.0,
        expert_usage_pct=last_expert_usage,
        role_expert_counts=role_expert_counts,
        ablation="none",
        spawn_history=spawn_history,
        save_dir=log_dir or save_ckpt_dir,
        results_root="results",
    )
    print(f"Saved checkpoint and results to {save_ckpt_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E-COOP Training")
    parser.add_argument("--stage", type=int, default=0, help="Stage index")
    parser.add_argument(
        "--timesteps", type=int, default=120_000, help="Total timesteps"
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
        save_dir = f"results/ecoop/seed_{args.seed}/stage_{args.stage}"

    load_ckpt = args.load_ckpt
    if load_ckpt is None and args.stage > 0:
        prev_ckpt = f"results/ecoop/seed_{args.seed}/stage_{args.stage - 1}/model.pt"
        if os.path.exists(prev_ckpt):
            load_ckpt = prev_ckpt

    train(
        algo_name="ecoop",
        stage_idx=args.stage,
        total_timesteps=args.timesteps,
        load_ckpt_path=load_ckpt,
        save_ckpt_dir=save_dir,
        log_dir=save_dir,
        seed=args.seed,
        use_rust=args.rust,
        run_id=args.run_id,
    )
