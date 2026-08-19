import json
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from constants import (
    ACTION_SPACE_SIZE,
    AGENTS,
    CLIP_COEF,
    CURRICULUM_STAGES,
    ENT_COEF,
    GAE_LAMBDA,
    GAMMA,
    LR,
    MACRO_STEP,
    MAHIRO_INTRINSIC_REWARD_COEF,
    N_AGENTS,
    NUM_ENVS,
    NUM_STEPS,
    OBSERVATION_SIZE,
    UPDATE_EPOCHS,
    VF_COEF,
)
from thermal_guard import check_thermal_guard
from vec_env import VectorEnv

# Manager makes a decision every 5 steps
total_timesteps = 300_000


class HierarchicalNetwork(nn.Module):
    """
    MAHIRO Architecture.

    The manager outputs a spatial goal. The worker outputs a discrete action based on the observation and the manager goal.
    """

    def __init__(self, state_dim):
        super().__init__()
        # Manager architecture (Centralized logic)
        self.manager_actor = nn.Sequential(
            nn.Linear(state_dim + N_AGENTS, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # Continuous spatial goal (x, y)
        )
        self.manager_actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.manager_critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Worker architecture (Decentralized execution)
        worker_in_dim = (
            (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS + 2
        )  # +2 for goal
        self.worker_actor = nn.Sequential(
            nn.Linear(worker_in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE),
        )
        self.worker_critic = nn.Sequential(
            nn.Linear(state_dim + 2, 64),
            nn.Tanh(),  # Worker critic knows the goal too
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_manager_action_and_value(self, state, role, action=None):
        x = torch.cat([state, role], dim=1)
        action_mean = self.manager_actor(x)
        action_logstd = self.manager_actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        value = self.manager_critic(state).squeeze(-1)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def get_worker_action_and_value(
        self, obs, role, mask, goal, state=None, action=None
    ):
        x = torch.cat([obs.flatten(start_dim=1), role, goal], dim=1)
        logits = self.worker_actor(x)
        masked_logits = logits + ((1.0 - mask) * -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        if state is not None:
            critic_x = torch.cat([state, goal], dim=1)
            value = self.worker_critic(critic_x).squeeze(-1)
        else:
            value = None

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
    map_w, map_h = env_config["map_size"]
    map_diag = np.sqrt(map_w**2 + map_h**2)

    agent = HierarchicalNetwork(state_dim).to(device)
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
    last_mean_goal_dist = 0.0
    last_w_loss = 0.0
    last_m_loss = 0.0

    for update in range(1, num_updates + 1):
        check_thermal_guard()
        # Worker Buffers
        w_obs = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, *OBSERVATION_SIZE)).to(device)
            for a in AGENTS
        }
        w_role = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, N_AGENTS)).to(device) for a in AGENTS
        }
        w_mask = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS, ACTION_SPACE_SIZE)).to(device)
            for a in AGENTS
        }
        w_goals = {a: torch.zeros((NUM_STEPS, NUM_ENVS, 2)).to(device) for a in AGENTS}
        w_actions = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        w_logprobs = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        w_rewards = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        w_values = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}
        w_dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
        w_states = torch.zeros((NUM_STEPS, NUM_ENVS, state_dim)).to(device)

        # Manager Buffers
        m_steps = NUM_STEPS // MACRO_STEP
        m_states_buf = torch.zeros((m_steps, NUM_ENVS, state_dim)).to(device)
        m_actions_buf = {
            a: torch.zeros((m_steps, NUM_ENVS, 2)).to(device) for a in AGENTS
        }
        m_logprobs_buf = {
            a: torch.zeros((m_steps, NUM_ENVS)).to(device) for a in AGENTS
        }
        m_rewards_buf = {a: torch.zeros((m_steps, NUM_ENVS)).to(device) for a in AGENTS}
        m_values_buf = {a: torch.zeros((m_steps, NUM_ENVS)).to(device) for a in AGENTS}
        m_dones_buf = torch.zeros((m_steps, NUM_ENVS)).to(device)

        current_goals = {a: torch.zeros((NUM_ENVS, 2)).to(device) for a in AGENTS}
        macro_reward_acc = {a: torch.zeros(NUM_ENVS).to(device) for a in AGENTS}
        step_goal_distances = []

        for step in range(NUM_STEPS):
            w_states[step] = torch.tensor(next_state, dtype=torch.float32).to(device)
            w_dones[step] = next_done

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
            state_t = torch.tensor(next_state, dtype=torch.float32).to(device)

            # Manager steps evaluate global goals.
            if step % MACRO_STEP == 0:
                m_step = step // MACRO_STEP
                m_states_buf[m_step] = state_t
                m_dones_buf[m_step] = next_done

                with torch.no_grad():
                    state_rep = state_t.repeat(N_AGENTS, 1)
                    m_acts, m_logp, _, m_vals = agent.get_manager_action_and_value(
                        state_rep, role_all.flatten(0, 1)
                    )

                    m_acts = m_acts.view(N_AGENTS, NUM_ENVS, 2)
                    m_logp = m_logp.view(N_AGENTS, NUM_ENVS)
                    m_vals = m_vals.view(N_AGENTS, NUM_ENVS)

                    for i, a in enumerate(AGENTS):
                        current_goals[a] = m_acts[i]
                        m_actions_buf[a][m_step] = m_acts[i]
                        m_logprobs_buf[a][m_step] = m_logp[i]
                        m_values_buf[a][m_step] = m_vals[i]
                        if m_step > 0:
                            m_rewards_buf[a][m_step - 1] = macro_reward_acc[a].clone()
                        macro_reward_acc[a].zero_()

            # --- WORKER LOGIC ---
            actions_dict = {}
            with torch.no_grad():
                for i, a in enumerate(AGENTS):
                    w_obs[a][step] = obs_all[i]
                    w_role[a][step] = role_all[i]
                    w_mask[a][step] = mask_all[i]
                    w_goals[a][step] = current_goals[a]

                    w_act, w_logp, _, w_val = agent.get_worker_action_and_value(
                        obs_all[i],
                        role_all[i],
                        mask_all[i],
                        current_goals[a],
                        state=state_t,
                    )

                    w_actions[a][step] = w_act
                    w_logprobs[a][step] = w_logp
                    w_values[a][step] = w_val
                    actions_dict[a] = w_act.cpu().numpy()

            next_obs, rewards, terms, truncs, infos = vec_env.step(actions_dict)

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

                    interval_scout_interact.append(interact_succ)
                    interval_scout_pois.append(pois_tagged)
                    interval_hacker_hack.append(hack_succ)
                    interval_muscle_neutralize.append(neutralize_succ)
                    interval_muscle_guards.append(guards_neutralized)
                    interval_extractor_loot.append(loot_succ)
                    interval_agents_at_extract.append(agents_extract)
                    interval_episode_steps.append(ep_steps)
                    interval_episode_alarms.append(ep_alarm)

            # Assign Intrinsic Worker Rewards
            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            for a in AGENTS:
                base_reward = rewards[a]

                # Calculate the intrinsic reward using the L2 distance between the agent position and the spatial goal.
                gx = np.clip(
                    (current_goals[a][:, 0].cpu().numpy() + 1.0) * 0.5 * map_w,
                    0.0,
                    float(map_w),
                )
                gy = np.clip(
                    (current_goals[a][:, 1].cpu().numpy() + 1.0) * 0.5 * map_h,
                    0.0,
                    float(map_h),
                )

                # Get actual agent positions from info
                poses = np.array(
                    [infos[e].get(a, {}).get("pos", (0, 0)) for e in range(NUM_ENVS)]
                )
                dist = np.sqrt((poses[:, 0] - gx) ** 2 + (poses[:, 1] - gy) ** 2)
                step_goal_distances.extend(dist.tolist())
                intrinsic_reward = -(dist / max(1.0, map_diag))

                w_rewards[a][step] = torch.tensor(
                    base_reward + MAHIRO_INTRINSIC_REWARD_COEF * intrinsic_reward,
                    dtype=torch.float32,
                ).to(device)

                # Manager only gets the extrinsic reward
                macro_reward_acc[a] += torch.tensor(
                    base_reward, dtype=torch.float32
                ).to(device)

        if step_goal_distances:
            last_mean_goal_dist = float(np.mean(step_goal_distances))

        # Finalize last macro reward
        for a in AGENTS:
            m_rewards_buf[a][-1] = macro_reward_acc[a].clone()

        # --- WORKER ADVANTAGE COMPUTATION (GAE) ---
        with torch.no_grad():
            state_next_t = torch.tensor(next_state, dtype=torch.float32).to(device)
            w_next_vals = {}
            for a in AGENTS:
                obs_t = torch.tensor(
                    next_obs[a]["observation"], dtype=torch.float32, device=device
                )
                role_t = torch.tensor(
                    next_obs[a]["role_id"], dtype=torch.float32, device=device
                )
                mask_t = torch.tensor(
                    next_obs[a]["action_mask"], dtype=torch.float32, device=device
                )

                _, _, _, val = agent.get_worker_action_and_value(
                    obs_t,
                    role_t,
                    mask_t,
                    current_goals[a],
                    state=state_next_t,
                )
                w_next_vals[a] = val

            w_adv = {a: torch.zeros_like(w_rewards[a]) for a in AGENTS}
            for a in AGENTS:
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = w_next_vals[a]
                    else:
                        nextnonterminal = 1.0 - w_dones[t + 1]
                        nextvalues = w_values[a][t + 1]

                    delta = (
                        w_rewards[a][t]
                        + GAMMA * nextvalues * nextnonterminal
                        - w_values[a][t]
                    )
                    w_adv[a][t] = lastgaelam = (
                        delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                    )

            w_ret = {a: w_adv[a] + w_values[a] for a in AGENTS}

        # --- MANAGER ADVANTAGE COMPUTATION (GAE) ---
        with torch.no_grad():
            m_adv = {a: torch.zeros_like(m_rewards_buf[a]) for a in AGENTS}
            for a in AGENTS:
                lastgaelam = 0
                for t in reversed(range(m_steps)):
                    if t == m_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = 0
                    else:
                        nextnonterminal = 1.0 - m_dones_buf[t + 1]
                        nextvalues = m_values_buf[a][t + 1]

                    delta = (
                        m_rewards_buf[a][t]
                        + (GAMMA**MACRO_STEP) * nextvalues * nextnonterminal
                        - m_values_buf[a][t]
                    )
                    m_adv[a][t] = lastgaelam = (
                        delta
                        + (GAMMA**MACRO_STEP)
                        * GAE_LAMBDA
                        * nextnonterminal
                        * lastgaelam
                    )

            m_ret = {a: m_adv[a] + m_values_buf[a] for a in AGENTS}

        # --- UPDATE PHASE (WORKER & MANAGER OPTIMIZATION) ---
        w_states_flat = w_states.reshape(-1, state_dim)
        m_state_flat = m_states_buf.reshape(-1, state_dim)

        for _epoch in range(UPDATE_EPOCHS):
            for a in AGENTS:
                # 1. Update Worker
                w_obs_flat = w_obs[a].reshape(-1, *OBSERVATION_SIZE)
                w_role_flat = w_role[a].reshape(-1, N_AGENTS)
                w_mask_flat = w_mask[a].reshape(-1, ACTION_SPACE_SIZE)
                w_goals_flat = w_goals[a].reshape(-1, 2)
                w_act_flat = w_actions[a].reshape(-1)
                w_logp_flat = w_logprobs[a].reshape(-1)
                w_adv_flat = w_adv[a].reshape(-1)
                w_ret_flat = w_ret[a].reshape(-1)

                w_adv_norm = (w_adv_flat - w_adv_flat.mean()) / (
                    w_adv_flat.std() + 1e-8
                )

                _, w_newlogp, w_entropy, w_newvalue = agent.get_worker_action_and_value(
                    w_obs_flat,
                    w_role_flat,
                    w_mask_flat,
                    w_goals_flat,
                    state=w_states_flat,
                    action=w_act_flat,
                )

                w_ratio = (w_newlogp - w_logp_flat).exp()
                w_pg1 = -w_adv_norm * w_ratio
                w_pg2 = -w_adv_norm * torch.clamp(
                    w_ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF
                )
                w_pg_loss = torch.max(w_pg1, w_pg2).mean()
                w_v_loss = 0.5 * ((w_newvalue - w_ret_flat) ** 2).mean()
                w_loss = w_pg_loss - ENT_COEF * w_entropy.mean() + VF_COEF * w_v_loss
                last_w_loss = float(w_loss.item())

                # 2. Update Manager
                m_role_flat = (
                    w_role[a][0]
                    .unsqueeze(0)
                    .repeat(m_steps, 1, 1)
                    .reshape(-1, N_AGENTS)
                )
                m_act_flat = m_actions_buf[a].reshape(-1, 2)
                m_logp_flat = m_logprobs_buf[a].reshape(-1)
                m_adv_flat = m_adv[a].reshape(-1)
                m_ret_flat = m_ret[a].reshape(-1)

                m_adv_norm = (m_adv_flat - m_adv_flat.mean()) / (
                    m_adv_flat.std() + 1e-8
                )

                _, m_newlogp, m_entropy, m_newvalue = (
                    agent.get_manager_action_and_value(
                        m_state_flat, m_role_flat, m_act_flat
                    )
                )

                m_ratio = (m_newlogp - m_logp_flat).exp()
                m_pg1 = -m_adv_norm * m_ratio
                m_pg2 = -m_adv_norm * torch.clamp(
                    m_ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF
                )
                m_pg_loss = torch.max(m_pg1, m_pg2).mean()
                m_v_loss = 0.5 * ((m_newvalue - m_ret_flat) ** 2).mean()
                m_loss = m_pg_loss - ENT_COEF * m_entropy.mean() + VF_COEF * m_v_loss
                last_m_loss = float(m_loss.item())

                total_loss = w_loss + m_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
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

            # Console log (clean & compact)
            console_logger.info(
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/100"
            )
            # Detailed file log
            if file_logger:
                file_logger.info(
                    f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/100 | Scout Tag Rate: {scout_tag_rate:.2f} (Avg POIs: {scout_avg_pois:.1f}) | Hacker Hack Rate: {hacker_hack_rate:.2f} | Muscle Neutralize Rate: {muscle_neutralize_rate:.2f} (Avg Guards: {avg_muscle_guards:.1f}/{total_stage_guards}) | Extractor Loot Rate: {extractor_loot_rate:.2f} | Avg Agents at Extract: {avg_agents_extract:.2f}/4 | Avg Goal Dist: {last_mean_goal_dist:.2f} | W-Loss: {last_w_loss:.4f} | M-Loss: {last_m_loss:.4f}"
                )

    # Save checkpoint and results
    if save_ckpt_dir:
        os.makedirs(save_ckpt_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(np.mean(completed_wins[-100:]))
            if completed_wins
            else 0.0,
            "lifetime_win_rate": float(global_wins / max(1, global_episodes)),
            "mean_reward": float(np.mean(completed_episode_returns[-100:]))
            if completed_episode_returns
            else 0.0,
            "avg_episode_steps": float(np.mean(completed_episode_steps[-100:]))
            if completed_episode_steps
            else 0.0,
            "max_stage_steps": int(max_stage_steps)
            if "max_stage_steps" in locals()
            else 150,
            "avg_alarm": float(np.mean(completed_episode_alarms[-100:]))
            if completed_episode_alarms
            else 0.0,
            "scout_interact_rate": float(np.mean(completed_scout_interact[-100:]))
            if completed_scout_interact
            else 0.0,
            "scout_avg_pois_tagged": float(np.mean(completed_scout_pois[-100:]))
            if completed_scout_pois
            else 0.0,
            "hacker_hack_rate": float(np.mean(completed_hacker_hack[-100:]))
            if completed_hacker_hack
            else 0.0,
            "muscle_neutralize_rate": float(np.mean(completed_muscle_neutralize[-100:]))
            if completed_muscle_neutralize
            else 0.0,
            "muscle_avg_guards_neutralized": float(
                np.mean(completed_muscle_guards[-100:])
            )
            if completed_muscle_guards
            else 0.0,
            "total_stage_guards": int(total_stage_guards)
            if "total_stage_guards" in locals()
            else 0,
            "extractor_loot_rate": float(np.mean(completed_extractor_loot[-100:]))
            if completed_extractor_loot
            else 0.0,
            "avg_agents_at_extract": float(np.mean(completed_agents_at_extract[-100:]))
            if completed_agents_at_extract
            else 0.0,
            "avg_goal_distance": float(last_mean_goal_dist),
            "worker_loss": float(last_w_loss),
            "manager_loss": float(last_m_loss),
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
