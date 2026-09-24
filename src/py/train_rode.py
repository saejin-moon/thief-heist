"""
RODE: Role-Oriented Action Decomposition for Multi-Agent Reinforcement Learning (Wang et al., ICLR 2021).

Key Algorithmic Components:
1. Action Representation Learning:
   Learns discrete action embedding e(a) in R^{d_a} via forward transition dynamics:
   f_psi(o_i, e(a)) approx (o_i' - o_i).
2. Action Space Clustering:
   Partitions discrete actions into K_r role subsets based on learned action representations:
   - Locomotion / Movement actions
   - Standby / Recon actions
   - Mission / Specialized Interaction actions
3. Hierarchical Execution:
   - High-level role selector mu_omega(r_i | o_i) selects role r_i every c macro-steps.
   - Low-level policy pi_theta(a_i | o_i, r_i) executes actions restricted to the role subset via masking.
4. Centralized Critic V_psi(s, role_id, goal, r_i).
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

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
from dataset import TrainCurveLogger, get_effective_run_id, record_stage_completion
from thermal_guard import check_thermal_guard
from vec_env import make_vec_env

NUM_ROLES = 3
ACTION_EMBED_DIM = 8
MACRO_STEP = 5
REP_LOSS_COEF = 0.1


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActionRepresentationPredictor(nn.Module):
    """Predicts observation transition delta (o_{t+1} - o_t) from observation features and action embedding."""

    def __init__(self, obs_dim, embed_dim=ACTION_EMBED_DIM):
        super().__init__()
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(obs_dim + embed_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, obs_dim), std=0.1),
        )

    def forward(self, obs_feat, action_embed):
        return self.predictor(torch.cat([obs_feat, action_embed], dim=-1))


class RodeNetwork(nn.Module):
    """RODE Architecture with Action Representation Learning and Hierarchical Role Decomposition."""

    def __init__(self, state_dim, num_roles=NUM_ROLES, embed_dim=ACTION_EMBED_DIM):
        super().__init__()
        self.num_roles = num_roles
        self.embed_dim = embed_dim
        obs_flat = OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]
        self.obs_flat = obs_flat
        self.feat_dim = obs_flat + N_AGENTS + GOAL_VECTOR_DIM

        # Action representations
        self.action_embeddings = nn.Embedding(ACTION_SPACE_SIZE, embed_dim)
        self.action_predictor = ActionRepresentationPredictor(obs_flat, embed_dim)

        # High-level Role Selector (chooses role r in {0, ..., num_roles - 1})
        self.role_selector = nn.Sequential(
            layer_init(nn.Linear(self.feat_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_roles), std=0.01),
        )

        # Low-level Action Policy conditioned on observation features and chosen role
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.feat_dim + num_roles, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, ACTION_SPACE_SIZE), std=0.01),
        )

        # Centralized Critic conditioned on global state, role_id, goal, and active role
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + N_AGENTS + GOAL_VECTOR_DIM + num_roles, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Canonical semantic role-action clustering matrix:
        # Role 0: Movement (UP=0, DOWN=1, LEFT=2, RIGHT=3, WAIT=4)
        # Role 1: Specialist Interaction (INTERACT=5, WAIT=4)
        # Role 2: Full Action Space / Recon (All actions)
        default_masks = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("role_action_masks", default_masks)

    def select_role(self, obs, role_id, goal):
        feat = torch.cat([obs.flatten(start_dim=1), role_id, goal], dim=1)
        role_logits = self.role_selector(feat)
        role_dist = Categorical(logits=role_logits)
        role = role_dist.sample()
        return role, role_dist.log_prob(role), role_dist.entropy()

    def get_action_and_value(self, obs, role_id, mask, goal, current_role, state=None, action=None):
        feat = torch.cat([obs.flatten(start_dim=1), role_id, goal], dim=1)
        role_onehot = F.one_hot(current_role, num_classes=self.num_roles).float()

        # Actor conditioned on features + role onehot
        x_actor = torch.cat([feat, role_onehot], dim=1)
        logits = self.actor(x_actor)

        # RODE action masking: restrict actions to active role subset AND environmental legal mask
        role_masks = self.role_action_masks[current_role]
        effective_mask = mask * role_masks
        # Safety fallback: if no action is valid in role, fall back to full env mask
        no_valid = effective_mask.sum(dim=-1, keepdim=True) == 0
        effective_mask = torch.where(no_valid, mask, effective_mask)

        masked_logits = logits + ((1.0 - effective_mask) * -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        value = None
        if state is not None:
            x_critic = torch.cat([state, role_id, goal, role_onehot], dim=1)
            value = self.critic(x_critic).squeeze(-1)

        return action, probs.log_prob(action), probs.entropy(), value


def train(
    algo_name="rode",
    stage_idx=0,
    env_config=None,
    total_timesteps=1000,
    load_ckpt_path=None,
    save_ckpt_dir=None,
    log_dir=None,
    seed=None,
    use_rust=False,
    run_id=None,
):
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
    vec_env = make_vec_env(
        NUM_ENVS,
        config=env_config,
        base_seed=seed * 1000 if seed is not None else 0,
        use_rust=use_rust,
    )
    state_dim = vec_env.state_dim

    agent = RodeNetwork(state_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        ckpt = torch.load(load_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        agent.load_state_dict(state_dict)
        msg_ckpt = f"Loaded checkpoint from {load_ckpt_path}"
        console_logger.info(msg_ckpt)
        if file_logger:
            file_logger.info(msg_ckpt)

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)

    num_steps = 250 if stage_idx >= 3 else NUM_STEPS
    num_updates = total_timesteps // (NUM_ENVS * num_steps)
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

    # Active roles for all agents across all envs: (N_AGENTS, NUM_ENVS)
    current_roles = torch.zeros((N_AGENTS, NUM_ENVS), dtype=torch.long, device=device)

    for update in range(1, num_updates + 1):
        check_thermal_guard()
        b_obs = {a: torch.zeros((num_steps, NUM_ENVS, *OBSERVATION_SIZE)).to(device) for a in AGENTS}
        b_next_obs = {a: torch.zeros((num_steps, NUM_ENVS, *OBSERVATION_SIZE)).to(device) for a in AGENTS}
        b_role = {a: torch.zeros((num_steps, NUM_ENVS, N_AGENTS)).to(device) for a in AGENTS}
        b_mask = {a: torch.zeros((num_steps, NUM_ENVS, ACTION_SPACE_SIZE)).to(device) for a in AGENTS}
        b_goal = {a: torch.zeros((num_steps, NUM_ENVS, GOAL_VECTOR_DIM)).to(device) for a in AGENTS}
        b_actions = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_logprobs = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_rewards = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_values = {a: torch.zeros((num_steps, NUM_ENVS)).to(device) for a in AGENTS}
        b_active_roles = {a: torch.zeros((num_steps, NUM_ENVS), dtype=torch.long).to(device) for a in AGENTS}
        b_dones = torch.zeros((num_steps, NUM_ENVS)).to(device)
        b_states = torch.zeros((num_steps, NUM_ENVS, state_dim)).to(device)

        for step in range(num_steps):
            b_states[step] = torch.tensor(next_state, dtype=torch.float32).to(device)
            b_dones[step] = next_done

            actions_dict = {}
            with torch.no_grad():
                stacked = next_obs["_stacked"]
                obs_all = torch.tensor(stacked["observation"], dtype=torch.float32, device=device)
                role_all = torch.tensor(stacked["role_id"], dtype=torch.float32, device=device)
                mask_all = torch.tensor(stacked["action_mask"], dtype=torch.float32, device=device)
                goals_all = torch.tensor(stacked["goal_vector"], dtype=torch.float32, device=device)

                # Macro-step role selection every MACRO_STEP steps
                if step % MACRO_STEP == 0:
                    for i in range(N_AGENTS):
                        sampled_role, _, _ = agent.select_role(obs_all[i], role_all[i], goals_all[i])
                        current_roles[i] = sampled_role

                state_rep = b_states[step].repeat(N_AGENTS, 1)
                flat_roles = current_roles.flatten()
                actions, logprobs, _, values = agent.get_action_and_value(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    goals_all.flatten(0, 1),
                    flat_roles,
                    state_rep,
                )

                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                values = values.view(N_AGENTS, NUM_ENVS)

                for i, a in enumerate(AGENTS):
                    b_obs[a][step] = obs_all[i]
                    b_role[a][step] = role_all[i]
                    b_mask[a][step] = mask_all[i]
                    b_goal[a][step] = goals_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    b_active_roles[a][step] = current_roles[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)
            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            next_stacked = next_obs["_stacked"]
            next_obs_all = torch.tensor(next_stacked["observation"], dtype=torch.float32, device=device)
            for i, a in enumerate(AGENTS):
                b_next_obs[a][step] = next_obs_all[i]
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(device)

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

            dones_scout = [terms["scout"][e] or truncs["scout"][e] for e in range(NUM_ENVS)]
            next_done = torch.tensor(dones_scout, dtype=torch.float32).to(device)

        # GAE Calculation
        with torch.no_grad():
            state_next_t = torch.tensor(next_state, dtype=torch.float32).to(device)
            stacked = next_obs["_stacked"]
            obs_all = torch.tensor(stacked["observation"], dtype=torch.float32, device=device)
            role_all = torch.tensor(stacked["role_id"], dtype=torch.float32, device=device)
            mask_all = torch.tensor(stacked["action_mask"], dtype=torch.float32, device=device)
            goals_all = torch.tensor(stacked["goal_vector"], dtype=torch.float32, device=device)
            state_rep = state_next_t.repeat(N_AGENTS, 1)

            flat_roles = current_roles.flatten()
            _, _, _, next_values = agent.get_action_and_value(
                obs_all.flatten(0, 1),
                role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                goals_all.flatten(0, 1),
                flat_roles,
                state_rep,
            )
            next_values = next_values.view(N_AGENTS, NUM_ENVS)

            b_advantages = {a: torch.zeros_like(b_rewards[a]).to(device) for a in AGENTS}
            for i, a in enumerate(AGENTS):
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_values[i]
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[a][t + 1]

                    delta = b_rewards[a][t] + GAMMA * nextvalues * nextnonterminal - b_values[a][t]
                    b_advantages[a][t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

        b_returns = {a: b_advantages[a] + b_values[a] for a in AGENTS}

        # UPDATE PHASE (RODE Hierarchical PPO + Action Representation Forward Model)
        b_states_flat = b_states.reshape(-1, state_dim)
        for _epoch in range(UPDATE_EPOCHS):
            for a in AGENTS:
                obs_flat = b_obs[a].reshape(-1, *OBSERVATION_SIZE)
                next_obs_flat = b_next_obs[a].reshape(-1, *OBSERVATION_SIZE)
                role_flat = b_role[a].reshape(-1, N_AGENTS)
                mask_flat = b_mask[a].reshape(-1, ACTION_SPACE_SIZE)
                goal_flat = b_goal[a].reshape(-1, GOAL_VECTOR_DIM)
                action_flat = b_actions[a].reshape(-1).long()
                active_role_flat = b_active_roles[a].reshape(-1).long()
                logprob_flat = b_logprobs[a].reshape(-1)
                adv_flat = b_advantages[a].reshape(-1)
                ret_flat = b_returns[a].reshape(-1)

                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                # Low-level PPO update
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_flat,
                    role_flat,
                    mask_flat,
                    goal_flat,
                    active_role_flat,
                    b_states_flat,
                    action=action_flat,
                )

                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.smooth_l1_loss(newvalue, ret_flat, beta=1.0)

                # Action Representation Learning: forward model predicting Delta o
                obs_flat_dim = obs_flat.flatten(start_dim=1)
                next_obs_flat_dim = next_obs_flat.flatten(start_dim=1)
                target_delta_obs = next_obs_flat_dim - obs_flat_dim

                act_embeds = agent.action_embeddings(action_flat)
                predicted_delta = agent.action_predictor(obs_flat_dim, act_embeds)
                rep_loss = F.mse_loss(predicted_delta, target_delta_obs)

                # High-level Role Selector regularizer (entropy bonus)
                feat_flat = torch.cat([obs_flat_dim, role_flat, goal_flat], dim=1)
                role_logits = agent.role_selector(feat_flat)
                role_entropy = Categorical(logits=role_logits).entropy().mean()

                loss = (
                    pg_loss
                    - ENT_COEF * entropy.mean()
                    + VF_COEF * v_loss
                    + REP_LOSS_COEF * rep_loss
                    - 0.005 * role_entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # LOGGING
        if update % 5 == 0:
            win_rate = float(np.mean(completed_wins[-25:])) if completed_wins else 0.0
            mean_episodic_reward = float(np.mean(completed_episode_returns[-25:])) if completed_episode_returns else 0.0
            avg_steps = float(np.mean(completed_episode_steps[-25:])) if completed_episode_steps else 0.0
            avg_alarm = float(np.mean(completed_episode_alarms[-25:])) if completed_episode_alarms else 0.0

            log_str = (
                f"Update {update}/{num_updates} | Win Rate: {win_rate * 100:.1f}% | "
                f"Ep Return: {mean_episodic_reward:.2f} | Steps: {avg_steps:.1f} | Alarm: {avg_alarm:.1f}"
            )
            console_logger.info(log_str)
            if file_logger:
                file_logger.info(log_str)

        step_num = update * NUM_ENVS * num_steps
        curve_logger.log_update(
            update=update,
            total_updates=num_updates,
            timesteps=step_num,
            elapsed_sec=time.time() - start_time,
            fps=float(step_num / max(0.001, time.time() - start_time)),
            win_rate=float(np.mean(interval_wins)) if interval_wins else 0.0,
            mean_return=float(np.mean(interval_episode_returns)) if interval_episode_returns else 0.0,
            avg_steps=float(np.mean(interval_episode_steps)) if interval_episode_steps else 0.0,
            avg_alarm=float(np.mean(interval_episode_alarms)) if interval_episode_alarms else 0.0,
            stage_alarm_max=float(env_config.get("alarm_max", 100.0)) if env_config else 100.0,
            hacker_hack_rate=float(np.mean(interval_hacker_hack)) if interval_hacker_hack else 0.0,
            muscle_neut_rate=float(np.mean(interval_muscle_neutralize)) if interval_muscle_neutralize else 0.0,
            extractor_loot_rate=float(np.mean(interval_extractor_loot)) if interval_extractor_loot else 0.0,
            avg_agents_extract=float(np.mean(interval_agents_at_extract)) if interval_agents_at_extract else 0.0,
            active_experts=1,
            effective_experts=1.0,
            total_spawns=0,
            switch_rate=0.0,
            routing_entropy=0.0,
            policy_loss=pg_loss.item() if "pg_loss" in locals() else 0.0,
            value_loss=v_loss.item() if "v_loss" in locals() else 0.0,
            entropy_loss=entropy.mean().item() if "entropy" in locals() else 0.0,
            explained_variance=0.0,
        )
        interval_wins.clear()
        interval_episode_returns.clear()
        interval_episode_steps.clear()
        interval_episode_alarms.clear()
        interval_scout_interact.clear()
        interval_scout_pois.clear()
        interval_hacker_hack.clear()
        interval_muscle_neutralize.clear()
        interval_muscle_guards.clear()
        interval_extractor_loot.clear()
        interval_agents_at_extract.clear()

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        torch.save({"model_state": agent.state_dict()}, os.path.join(save_ckpt_dir, "model.pt"))

    curve_logger.close()
    elapsed_wall_clock = time.time() - start_time
    record_stage_completion(
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
        stage_max_steps=env_config.get("max_steps", 150) if env_config else 150,
        stage_alarm_max=float(env_config.get("alarm_max", 100.0)) if env_config else 100.0,
        total_stage_guards=int(env_config.get("guard_count", 0)) if env_config else 0,
        map_size=env_config.get("map_size", (11, 11))[0] if env_config else 11,
        active_experts=1,
        dormant_experts_count=0,
        total_spawns=0,
        expert_switch_rate=0.0,
        routing_entropy=0.0,
        expert_usage_pct=None,
        role_expert_counts=None,
        ablation="none",
        spawn_history=None,
        save_dir=log_dir or save_ckpt_dir,
        results_root="results",
    )
    print(f"Saved checkpoint and results to {save_ckpt_dir}")
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RODE Training")
    parser.add_argument("--stage", type=int, default=0, help="Stage index")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps")
    parser.add_argument(
        "--seeds",
        "--seed",
        dest="seed",
        type=lambda s: int(s.split(",")[0].strip()) if "," in str(s) else int(str(s).strip()),
        default=0,
        help="Random seed",
    )
    parser.add_argument("--save-dir", "--save-ckpt-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--load-ckpt", type=str, default=None)
    parser.add_argument("--rust", "--use-rust", dest="rust", action="store_true", default=False)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"results/rode/seed_{args.seed}/stage_{args.stage}"

    load_ckpt = args.load_ckpt
    if load_ckpt is None and args.stage > 0:
        prev_ckpt = f"results/rode/seed_{args.seed}/stage_{args.stage - 1}/model.pt"
        if os.path.exists(prev_ckpt):
            load_ckpt = prev_ckpt

    train(
        algo_name="rode",
        stage_idx=args.stage,
        total_timesteps=args.timesteps,
        load_ckpt_path=load_ckpt,
        save_ckpt_dir=save_dir,
        log_dir=save_dir,
        seed=args.seed,
        use_rust=args.rust,
        run_id=args.run_id,
    )
