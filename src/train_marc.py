import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from constants import ACTION_SPACE_SIZE, N_AGENTS, OBSERVATION_SIZE, AGENTS, ALARM_MAX
from vec_env import VectorEnv

# Hyperparameters
LR = 2.5e-4
NUM_ENVS = 8
NUM_STEPS = 128
TOTAL_TIMESTEPS = 300_000
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
CLIP_COEF = 0.2

# MARC Specific Hyperparameters
ALPHA_ALARM = 1.5           # Controls how severely the alarm penalizes macro credit
GAMMA_CAUSAL = 0.95         # Retroactive discount factor for causal trace
AFFORDANCE_COEF = 0.5       # Reward weight for unlocking affordances for the team

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
            nn.Linear(actor_in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MARC on {device}...")

    env_config = {
        "map_size": (11, 11), "guard_count": 0, "camera_count": 0, 
        "door_count": 0, "max_steps": 100, "spawn_mode": "role"
    }
    vec_env = VectorEnv(NUM_ENVS, config=env_config)
    state_dim = vec_env.state_dim
    
    agent = MarcNetwork(state_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

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
        
        # MARC specific buffers
        b_alarms = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
        b_affordances = {a: torch.zeros((NUM_STEPS, NUM_ENVS)).to(device) for a in AGENTS}

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
                actions, logprobs, _, values = agent.get_action_and_value(
                    obs_all.flatten(0, 1), role_all.flatten(0, 1), mask_all.flatten(0, 1), state_rep
                )
                
                actions = actions.view(N_AGENTS, NUM_ENVS)
                logprobs = logprobs.view(N_AGENTS, NUM_ENVS)
                values = values.view(N_AGENTS, NUM_ENVS)
                
                for i, a in enumerate(AGENTS):
                    b_obs[a][step], b_role[a][step], b_mask[a][step] = obs_all[i], role_all[i], mask_all[i]
                    b_actions[a][step] = actions[i]
                    b_logprobs[a][step] = logprobs[i]
                    b_values[a][step] = values[i]
                    actions_dict[a] = actions[i].cpu().numpy()

            next_obs_new, rewards, terms, truncs, infos = vec_env.step(actions_dict)
            
            # Structural Affordance Detection for MARC Micro Credit
            # If an agent interacted and the total mask volume increased, they unlocked an affordance!
            mask_new = next_obs_new["_stacked"]["action_mask"]
            mask_vol_new = mask_new.sum(axis=(0, 2)) # sum over agents and action dims
            mask_vol_old = stacked["action_mask"].sum(axis=(0, 2))
            unlocked = mask_vol_new > mask_vol_old
            
            for i, a in enumerate(AGENTS):
                # 5 is INTERACT
                interacted = (actions_dict[a] == 5)
                # Assign affordance delta (1.0) if they interacted and unlocked something
                b_affordances[a][step] = torch.tensor(interacted & unlocked, dtype=torch.float32).to(device)

            next_state = vec_env.state
            next_done = torch.tensor(terms["scout"] | truncs["scout"], dtype=torch.float32).to(device)
            
            # Global alarm used for Macro Weighting
            alarms_list = [infos[e].get("scout", {}).get("alarm", 0.0) for e in range(NUM_ENVS)]
            b_alarms[step] = torch.tensor(alarms_list, dtype=torch.float32).to(device)

            for a in AGENTS:
                b_rewards[a][step] = torch.tensor(rewards[a], dtype=torch.float32).to(device)
            
            next_obs = next_obs_new

        # --- MARC ADVANTAGE CALCULATION ---
        with torch.no_grad():
            next_val = agent.critic(torch.tensor(next_state, dtype=torch.float32).to(device)).squeeze(-1)
            b_adv = {a: torch.zeros_like(b_rewards[a]) for a in AGENTS}
            
            # Determine trajectory win states (any agent got >5 reward at any step)
            win_mask = torch.zeros(NUM_ENVS, dtype=torch.bool).to(device)
            for a in AGENTS:
                win_mask |= (b_rewards[a] > 5.0).any(dim=0)
            
            # Macro Weighting (Omega_t): Alarm scaling and Outcome factor
            macro_alarm_factor = torch.exp(-ALPHA_ALARM * (b_alarms / ALARM_MAX))
            macro_outcome = torch.where(win_mask, 1.0, 0.5) # [NUM_ENVS]
            unshielded_omega = macro_outcome.unsqueeze(0) * macro_alarm_factor # [NUM_STEPS, NUM_ENVS]
            
            for a in AGENTS:
                # Binary Success Masking (Shielding):
                # Protect upstream enablers (affordance > 0) from downstream incompetence (loss)
                shield_mask = (~win_mask.unsqueeze(0)) & (b_affordances[a] > 0)
                omega_t = torch.where(shield_mask, macro_alarm_factor, unshielded_omega)
                
                # Base temporal difference
                deltas = torch.zeros_like(b_rewards[a])
                for t in range(NUM_STEPS):
                    nextnonterminal = 1.0 - (next_done if t == NUM_STEPS - 1 else b_dones[t + 1])
                    nextvalues = next_val if t == NUM_STEPS - 1 else b_values[a][t + 1]
                    deltas[t] = b_rewards[a][t] + GAMMA * nextvalues * nextnonterminal - b_values[a][t]
                
                # Micro Credit: Base TD + Affordance delta
                micro_credit = deltas + (b_affordances[a] * AFFORDANCE_COEF)
                immediate_marc = micro_credit * omega_t
                
                # Retroactive Causal Trace Propagation
                retro_trace = torch.zeros(NUM_ENVS).to(device)
                for t in reversed(range(NUM_STEPS)):
                    retro_trace = immediate_marc[t] + GAMMA_CAUSAL * (1.0 - b_dones[t]) * retro_trace
                    b_adv[a][t] = retro_trace

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
                
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs_flat, role_flat, mask_flat, b_states_flat, action=action_flat)
                
                logratio = newlogprob - logprob_flat
                ratio = logratio.exp()
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((newvalue - ret_flat) ** 2).mean()
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if update % 5 == 0:
            avg_reward = sum(b_rewards[a].mean().item() for a in AGENTS) / N_AGENTS
            win_rate = (b_rewards["scout"] > 5.0).float().max(dim=0).values.mean().item()
            print(f"Update: {update}/{num_updates} | Win Rate: {win_rate:.2f} | Mean Reward: {avg_reward:.3f}")

if __name__ == "__main__":
    train()
