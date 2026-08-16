import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from constants import ACTION_SPACE_SIZE, N_AGENTS, OBSERVATION_SIZE, AGENTS
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
ECOOP_POOL_SIZE = 8

class MappoNetwork(nn.Module):
    """Base network that acts as an 'Expert' in the E-COOP pool."""
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

class EcoopNetwork(nn.Module):
    """
    E-COOP: Evolutionary Confidence-Oriented Option Pool.
    Maintains a pool of MappoNetworks (Experts). 
    Agents route to the expert with the highest value (confidence) prediction.
    """
    def __init__(self, state_dim, max_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([MappoNetwork(state_dim) for _ in range(max_experts)])
        self.max_experts = max_experts

    def get_action_and_value(self, obs, role, mask, state, active_experts, action=None, expert_idx=None, previous_expert=None, epsilon=0.05):
        batch_size = obs.shape[0]
        
        # 1. Routing Logic
        if expert_idx is None:
            expert_values = []
            for k in range(active_experts):
                val = self.experts[k].critic(state).squeeze(-1)
                expert_values.append(val)
            expert_values = torch.stack(expert_values, dim=1)
            
            # Hybrid Routing Hysteresis: prevent routing chatter
            if previous_expert is not None:
                chosen_expert = previous_expert.clone()
                for b in range(batch_size):
                    prev_idx = previous_expert[b]
                    prev_val = expert_values[b, prev_idx]
                    best_idx = torch.argmax(expert_values[b])
                    best_val = expert_values[b, best_idx]
                    
                    if best_idx != prev_idx and best_val > prev_val + max(epsilon, epsilon * abs(prev_val.item())):
                        chosen_expert[b] = best_idx
            else:
                chosen_expert = torch.argmax(expert_values, dim=1)
        else:
            chosen_expert = expert_idx

        # 2. Execution
        actions = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        logprobs = torch.zeros(batch_size, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, device=obs.device)

        for k in range(active_experts):
            mask_k = (chosen_expert == k)
            if not mask_k.any(): 
                continue
            
            a, lp, ent, v = self.experts[k].get_action_and_value(
                obs[mask_k], role[mask_k], mask[mask_k], state[mask_k], 
                action[mask_k] if action is not None else None
            )
            
            actions[mask_k] = a
            logprobs[mask_k] = lp
            entropies[mask_k] = ent
            values[mask_k] = v

        return actions, logprobs, entropies, values, chosen_expert

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training E-COOP on {device}...")

    env_config = {
        "map_size": (11, 11), "guard_count": 0, "camera_count": 0, 
        "door_count": 0, "max_steps": 100, "spawn_mode": "role"
    }
    vec_env = VectorEnv(NUM_ENVS, config=env_config)
    state_dim = vec_env.state_dim
    
    agent = EcoopNetwork(state_dim, max_experts=ECOOP_POOL_SIZE).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    active_experts = 1
    next_obs, next_state = vec_env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)
    
    env_expert_idx = torch.zeros(N_AGENTS * NUM_ENVS, dtype=torch.long, device=device)

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

    for update in range(1, num_updates + 1):
        # Data collection buffers omitted for brevity...
        # In a full implementation, we run rollouts and store transitions just like MAPPO.

        # Evolutionary Crossover Event (E-COOP specific mechanic)
        # At rigid generational intervals, E-COOP duplicates the best expert.
        do_crossover = False
        if update == 500 or (update > 500 and (update - 500) % 200 == 0):
            do_crossover = True

        if do_crossover and active_experts < ECOOP_POOL_SIZE:
            best_expert = 0 # Normally found by inspecting value averages
            new_expert = active_experts
            
            print(f"Evolutionary Crossover: Cloning Expert {best_expert} to {new_expert}")
            agent.experts[new_expert].load_state_dict(agent.experts[best_expert].state_dict())
            
            # FIM-Scaled Asexual Mutation
            # E-COOP uses Fisher Information Matrix scaling to inject noise,
            # protecting critical neural pathways while mutating flat ones.
            with torch.no_grad():
                for param in agent.experts[new_expert].parameters():
                    # Simplified uniform noise proxy for demonstration
                    param.add_(torch.randn_like(param) * 0.05)
            
            active_experts += 1

        if update % 5 == 0:
            print(f"Update: {update}/{num_updates} | Active Experts: {active_experts}")

if __name__ == "__main__":
    train()
