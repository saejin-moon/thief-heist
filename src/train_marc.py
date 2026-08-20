import json
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from constants import (
    ACTION_SPACE_SIZE,
    AFFORDANCE_COEF,
    AGENTS,
    ALPHA_ALARM,
    CLIP_COEF,
    CURRICULUM_STAGES,
    ENT_COEF,
    GAE_LAMBDA,
    GAMMA,
    GAMMA_CAUSAL,
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

# Controls how severely the alarm penalizes macro credit


# CleanRL philosophy: Everything in one file.
class MarcNetwork(nn.Module):
    """
    Standard Actor-Critic network. In MARC, the architecture is flat like MAPPO,
    but the credit assignment (GAE calculation) is profoundly different.
    """

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

    agent = MarcNetwork(state_dim).to(device)
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

    for update in range(1, num_updates + 1):
        check_thermal_guard()
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

        # MARC specialized buffers
        b_affordances = {
            a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS
        }
        b_alarms = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
        b_wins = np.zeros((NUM_STEPS, NUM_ENVS), dtype=bool)

        # --- ROLLOUT PHASE ---
        for step in range(NUM_STEPS):
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

                state_rep = b_states[step].repeat(N_AGENTS, 1)
                actions, logprobs, _, values = agent.get_action_and_value(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    state_rep,
                )

                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                values = values.view(N_AGENTS, NUM_ENVS)

                for i, a in enumerate(AGENTS):
                    b_obs[a][step] = obs_all[i]
                    b_role[a][step] = role_all[i]
                    b_mask[a][step] = mask_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            # Step the environment!
            next_obs_new, rewards, terms, truncs, infos = vec_env.step(actions_dict)

            # Accumulate per-agent average return per env
            step_agent_reward = sum(rewards[a] for a in AGENTS) / N_AGENTS
            current_env_returns += step_agent_reward

            for e in range(NUM_ENVS):
                if infos[e]["scout"].get("win", False):
                    b_wins[step, e] = True
                # Check for termination to update tracking metrics
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

            # The team unlocked an affordance if an agent interacted and the mask volume increased.
            mask_new = next_obs_new["_stacked"]["action_mask"]
            mask_vol_new = mask_new.sum(axis=(0, 2))  # sum over agents and action dims
            mask_vol_old = stacked["action_mask"].sum(axis=(0, 2))
            unlocked = mask_vol_new > mask_vol_old

            for i, a in enumerate(AGENTS):
                # 5 is INTERACT
                interacted = actions_dict[a] == 5
                # Assign affordance delta (1.0) if they interacted and unlocked something
                b_affordances[a][step] = torch.tensor(
                    interacted & unlocked, dtype=torch.float32
                ).to(device)

            b_alarms[step] = torch.tensor(
                [infos[e]["scout"]["alarm"] for e in range(NUM_ENVS)]
            ).to(device)

            next_state = vec_env.state
            next_done = torch.tensor(
                terms["scout"] | truncs["scout"], dtype=torch.float32
            ).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(
                    device
                )

            next_obs = next_obs_new

        # --- MARC CAUSAL REWARD & ADVANTAGE RESTRUCTURING ---
        # 1. Macro-Outcome Multiplier: Omega_t = (1.0 + 2.0 * Win_t) * exp(-alpha * Alarm_t / 100)
        alarm_weights = torch.exp(-ALPHA_ALARM * (b_alarms / 100.0))
        # Win tensor shape [NUM_STEPS, NUM_ENVS]
        win_tensor = torch.tensor(
            b_wins, dtype=torch.float32, device=device
        )  # 1 for win step, 0 otherwise
        macro_weights = (1.0 + 2.0 * win_tensor) * alarm_weights

        # 2. Re-assign rewards with affordance bonus and macro weight
        b_shaped_rewards = {}
        for a in AGENTS:
            # Immediate Credit: r_shaped = r_env + c_affordance * A_t
            r_imm = b_rewards[a] + AFFORDANCE_COEF * b_affordances[a]
            # Macro-weighting: Omega_t * r_shaped
            b_shaped_rewards[a] = macro_weights * r_imm

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

                    delta = (
                        b_shaped_rewards[a][t]
                        + GAMMA * nextvalues * nextnonterminal
                        - b_values[a][t]
                    )

                    # 3. Retroactive Causal Trace: temporal credit assignment decay
                    # Combines standard GAE with causal decay gamma_causal
                    lastgaelam = (
                        delta
                        + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                        + GAMMA_CAUSAL * b_affordances[a][t]
                    )
                    b_adv[a][t] = lastgaelam

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

                adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_flat, role_flat, mask_flat, b_states_flat, action=action_flat
                )

                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_norm * ratio
                pg_loss2 = -adv_norm * torch.clamp(
                    ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - ret_flat) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

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

            # Console log (clean & compact)
            console_logger.info(
                f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f}"
            )
            # Detailed file log
            if file_logger:
                file_logger.info(
                    f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Episodic Return: {mean_episodic_reward:.3f} | Steps: {avg_steps:.1f}/{max_stage_steps} | Alarm: {avg_alarm:.1f}/{stage_alarm_max:.0f} | Scout Tag Rate: {scout_tag_rate:.2f} (Avg POIs: {scout_avg_pois:.1f}) | Hacker Hack Rate: {hacker_hack_rate:.2f} | Muscle Neutralize Rate: {muscle_neutralize_rate:.2f} (Avg Guards: {avg_muscle_guards:.1f}/{total_stage_guards}) | Extractor Loot Rate: {extractor_loot_rate:.2f} | Avg Agents at Extract: {avg_agents_extract:.2f}/4"
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
            "stage_alarm_max": float(stage_alarm_max)
            if "stage_alarm_max" in locals()
            else 100.0,
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
