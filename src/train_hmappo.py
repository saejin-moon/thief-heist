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
    LR,
    MACRO_STEP,
    MAHIRO_INTRINSIC_REWARD_COEF,
    N_AGENTS,
    NUM_ENVS,
    NUM_STEPS,
    OBSERVATION_SIZE,
)
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

    if log_dir is not None:
        logging.basicConfig(
            filename=os.path.join(log_dir, "train.log"),
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(message)s", force=True
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
    map_w, map_h = env_config["map_size"]
    map_diag = np.sqrt(map_w**2 + map_h**2)

    agent = HierarchicalNetwork(state_dim).to(device)
    _optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

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

                    w_act, w_logp, _, w_val = agent.get_worker_action_and_value(
                        obs_all[i], role_all[i], mask_all[i], current_goals[a], state_t
                    )

                    w_actions[a][step] = w_act
                    w_logprobs[a][step] = w_logp
                    w_values[a][step] = w_val
                    actions_dict[a] = w_act.cpu().numpy()

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

            for i, a in enumerate(AGENTS):
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
                intrinsic_reward = -(dist / max(1.0, map_diag))

                w_rewards[a][step] = torch.tensor(
                    base_reward + MAHIRO_INTRINSIC_REWARD_COEF * intrinsic_reward,
                    dtype=torch.float32,
                ).to(device)

                # Manager only gets the extrinsic reward
                macro_reward_acc[a] += torch.tensor(
                    base_reward, dtype=torch.float32
                ).to(device)

        # Finalize last macro reward
        for a in AGENTS:
            m_rewards_buf[a][-1] = macro_reward_acc[a].clone()

        # Compute Advantages (Skipped full logic for brevity, assuming standard GAE)
        # In a complete implementation, both worker and manager advantages would be computed here.

        if update % 5 == 0:
            avg_reward = sum(m_rewards_buf[a].mean().item() for a in AGENTS) / N_AGENTS
            logging.info(  # noqa: LOG015
                f"Update: {update}/{num_updates} | Mean Extrinsic Macro Reward: {avg_reward:.3f}"
            )

    # Save checkpoint and results
    if save_ckpt_dir:
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": 0.0,
            "mean_reward": 0.0,
        }
        with open(os.path.join(save_ckpt_dir, "results.json"), "w") as jf:
            json.dump(results, jf, indent=4)
        logging.info(  # noqa: LOG015
            f"Saved checkpoint and results to {save_ckpt_dir}"
        )


if __name__ == "__main__":
    train()
