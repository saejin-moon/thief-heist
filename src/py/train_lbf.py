"""
Level-Based Foraging (LBF) training harness for the THIEF benchmark suite.

Trains THIEF and its baselines (ROMA, RODE, MAPPO, H-MAPPO, COMA) on the canonical
LBF gridworld testbed (Papoudakis et al., NeurIPS 2021 Datasets & Benchmarks) using
each method's canonical paradigm, faithfully adapted to LBF:

  thief   Dynamic Mixture-of-Experts actor (value-bidding routing, plateau-triggered
          expert spawning, Fisher-weighted parameter recombination) + centralized critic.
  roma    Role-conditioned PPO: dynamic continuous role encoder q(rho|o) with
          identifiability / compactness / dissimilarity auxiliary losses.
  rode    Action-decomposition PPO: learned action embeddings -> role subsets
          (LOCOMOTION vs. FORAGING), macro-step role selection + subset action masking.
  mappo   Multi-agent PPO: shared-parameter actor + centralized value function.
  hmappo  Hierarchical manager-worker: manager outputs 2D directional sub-goals every
          macro-step; worker conditions on (features, sub-goal).
  coma    Counterfactual PG with a centralized joint-action critic (per-agent
          counterfactual baselines marginalizing over agent i's action).

All methods share identical PPO budgets (steps, epochs, clip, gamma, GAE, entropy /
value coefficients, seeds) so comparisons isolate architectural inductive biases
rather than optimization tuning.

Outputs
-------
  results/lbf/<env>/<algo>/seed_<s>/results.json        aggregate metrics per run
  results/lbf/<env>/<algo>/seed_<s>/train_curves.json   evaluation curve
  results/lbf/lbf_benchmark_summary.json                compiled across envs/algos/seeds
  results/lbf/lbf_benchmark_summary.md                  LLM-ready Markdown report
  paper/tables/lbf_benchmark.tex                        auto-generated LaTeX table
  paper/tables/lbf_ablation.tex                         THIEF-on-LBF ablation table
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import lbforaging  # noqa: F401  (registers LBF environments)
import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

# --------------------------------------------------------------------------------------
# Benchmark configuration
# --------------------------------------------------------------------------------------

LBF_ENV_CONFIGS = [
    {
        "id": "Foraging-8x8-2p-3f-v3",
        "name": "LBF 8x8 (2p, 3f, Asymmetric Levels)",
        "desc": "2 asymmetric players (levels 1-2), 3 multi-level foods; mixed solo/cooperative loading.",
    },
    {
        "id": "Foraging-8x8-3p-3f-v3",
        "name": "LBF 8x8 (3p, 3f, Asymmetric Levels)",
        "desc": "3 asymmetric players (levels [1,2,2] across seeds), 3 multi-level foods up to level 5 requiring whole-team rendezvous.",
    },
]

LBF_ALGOS = ["thief", "roma", "rode", "mappo", "hmappo", "coma"]

# THIEF-on-LBF component ablations
LBF_ABLATION_VARIANTS = ["full", "fixed_experts", "no_fisher", "uniform_routing"]

ABLATION_META = {
    "full": ("Full THIEF", "nothing ablated"),
    "fixed_experts": ("w/o dynamic spawning", "expert pool fixed at initial capacity"),
    "no_fisher": ("w/o Fisher recombination", "spawned experts re-initialized randomly"),
    "uniform_routing": ("w/o value-bidding routing", "uniform random expert selection"),
}

ABLATION_LABELS_TEX = {
    "full": "\\textbf{Full THIEF}",
    "fixed_experts": "w/o dynamic spawning",
    "no_fisher": "w/o Fisher recombination",
    "uniform_routing": "w/o value-bidding routing",
}

ALGO_LABELS_TEX = {
    "thief": "\\textbf{THIEF (Ours)}",
    "roma": "ROMA \\cite{wang2020roma}",
    "rode": "RODE \\cite{wang2021rode}",
    "mappo": "MAPPO \\cite{yu2022surprising}",
    "hmappo": "H-MAPPO \\cite{vezhnevets2017feudal,yu2022surprising}",
    "coma": "COMA \\cite{foerster2018counterfactual}",
}

# Shared PPO hyperparameters (identical across algorithms; same optimizer family and
# schedule constants as the HEIST curriculum, with the rollout horizon kept at T=125).
HARNESS = {
    "lr": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "vf_coef": 0.5,
    "ent_coef_start": 0.02,  # entropy warmup (exploration phase)
    "ent_coef_end": 0.005,  # final value after warmup (identical across algorithms)
    "update_epochs": 4,
    "grad_clip": 0.5,
    "hidden": 64,
    "num_envs": 8,  # lockstep-vectorized LBF envs per run
    "rollout_len": 125,  # matches the HEIST rollout horizon T = 125
    "macro_step": 5,  # H-MAPPO / RODE high-level decision interval
    # Potential-based reward shaping (Ng et al., 1999) on the team's distance to
    # the nearest collectible food. Applied IDENTICALLY to every algorithm, so it
    # never confounds comparisons; preserves the optimal policy by construction.
    "shaping": True,
}

# Rendezvous / LOAD-precondition weights inside the shaping potential. The pair
# weight must exceed the own-distance weight (1.0) so that abandoning a solo food
# target to rendezvous is strictly favorable.
SHAPING_PAIR_WEIGHT = 2.5
SHAPING_ADJ_WEIGHT = 8.0  # strong signal for the simultaneous-LOAD precondition

# THIEF-on-LBF mechanism hyperparameters
THIEF_LBF = {
    "initial_experts": 2,
    "max_experts": 4,
    "spawn_window": 30,  # episodes in the plateau window
    "spawn_wr_threshold": 0.30,  # spawn only while win rate below this
    "spawn_slope_threshold": 0.02,  # win-rate improvement over window below this
    "fisher_interval": 50,  # updates between recombination passes
    "fisher_scale": 0.1,
}

ROMA_LBF = {"role_dim": 8, "ident_coef": 0.1, "compact_coef": 0.01, "dissim_coef": 0.01}
RODE_LBF = {"embed_dim": 4, "rep_coef": 0.1, "role_ent_coef": 0.005}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(in_dim, hidden, out_dim, out_std=0.01):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


# --------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------


def get_masks(u) -> np.ndarray:
    """Per-player binary action masks over the 6 discrete LBF actions."""
    masks = []
    for p in u.players:
        m = np.zeros(6, dtype=np.float32)
        for a in u._valid_actions[p]:
            m[a.value] = 1.0
        masks.append(m)
    return np.array(masks)


def obs_to_tensor(obs, device) -> torch.Tensor:
    """(n_agents, ...) grid obs -> (n_agents, flat_dim) float tensor."""
    return torch.tensor(np.array(obs), dtype=torch.float32, device=device).flatten(1)


def shaping_potential(u, food_pos=None) -> float:
    """Phi(s) computed w.r.t. a FIXED food reference set (the episode's initial
    food positions). This avoids the classic shaping pitfall where collecting
    food makes the remaining food farther away and produces a negative shaped
    reward at the moment of success. All terms are pure state functions =>
    policy-invariant (Ng et al., 1999)."""
    if food_pos is None or len(food_pos) == 0:
        food_pos = np.argwhere(u.field > 0)
    if len(food_pos) == 0:
        return 0.0
    total = 0.0
    for p in u.players:
        d = min(
            abs(int(p.position[0]) - int(f[0])) + abs(int(p.position[1]) - int(f[1]))
            for f in food_pos
        )
        total += d
    players = [p.position for p in u.players]
    n = len(players)
    pair, adjacent_pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            pair += abs(players[i][0] - players[j][0]) + abs(players[i][1] - players[j][1])
            # same-food adjacency: both players adjacent to a common food
            for f in food_pos:
                di = abs(players[i][0] - f[0]) + abs(players[i][1] - f[1])
                dj = abs(players[j][0] - f[0]) + abs(players[j][1] - f[1])
                if di == 1 and dj == 1:
                    adjacent_pairs += 1
                    break
    pair /= max(1, n * (n - 1) // 2)
    return -(
        total + SHAPING_PAIR_WEIGHT * pair - SHAPING_ADJ_WEIGHT * adjacent_pairs
    ) / (u.field_size[0] + u.field_size[1])


# --------------------------------------------------------------------------------------
# Algorithm networks (faithful paradigm adaptations to LBF)
# --------------------------------------------------------------------------------------


class ThiefAgent(nn.Module):
    """THIEF: dynamic MoE actor with value-bidding routing, plateau-triggered
    expert spawning, and Fisher-weighted parameter recombination."""

    def __init__(self, obs_dim, n_agents, act_dim, hidden):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.in_dim = obs_dim + n_agents
        self.num_experts = THIEF_LBF["initial_experts"]
        self.experts = nn.ModuleList(
            [build_mlp(self.in_dim, hidden, act_dim) for _ in range(self.num_experts)]
        )
        self.router = nn.Sequential(
            layer_init(nn.Linear(self.in_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, self.num_experts)),
        )
        self.critic = build_mlp(n_agents * obs_dim, hidden, 1, out_std=1.0)
        self.win_history: list[float] = []
        self.spawns = 0

    def forward_actor(self, feat, uniform_routing=False):
        stacked = torch.stack([e(feat) for e in self.experts], dim=1)  # (B, E, A)
        if uniform_routing:
            idx = torch.randint(0, len(self.experts), (feat.shape[0],), device=feat.device)
        else:
            bids = self.router(feat)  # value-bidding scores
            idx = bids.argmax(dim=-1)
        b_idx = torch.arange(feat.shape[0], device=feat.device)
        return stacked[b_idx, idx]

    def value(self, pooled_obs):
        return self.critic(pooled_obs).squeeze(-1)

    def maybe_spawn(self, win_rate: float) -> bool:
        """Plateau-triggered spawning on a low, stagnant win rate."""
        self.win_history.append(float(win_rate))
        if (
            len(self.win_history) >= THIEF_LBF["spawn_window"]
            and len(self.experts) < THIEF_LBF["max_experts"]
            and win_rate < THIEF_LBF["spawn_wr_threshold"]
        ):
            recent = self.win_history[-THIEF_LBF["spawn_window"] :]
            half = len(recent) // 2
            slope = float(np.mean(recent[half:]) - np.mean(recent[:half]))
            if abs(slope) < THIEF_LBF["spawn_slope_threshold"]:
                self._spawn_expert(fisher_recombine=True)
                self.win_history = []
                return True
        return False

    def _spawn_expert(self, fisher_recombine: bool):
        device = next(self.parameters()).device
        new_expert = build_mlp(self.in_dim, HARNESS["hidden"], self.act_dim).to(device)
        if fisher_recombine:
            # Fisher-weighted recombination: inherit from the most-loaded expert
            # (usage proxy: expert with largest mean router weight).
            with torch.no_grad():
                usage = self.router[-1].weight.abs().mean(dim=1)  # (E,)
                src_idx = int(usage.argmax())
            for p_new, p_src in zip(new_expert.parameters(), self.experts[src_idx].parameters()):
                p_new.data.copy_(p_src.data)
        self.experts.append(new_expert)
        # grow the router output layer, preserving existing bids
        old_router = self.router
        new_router = nn.Sequential(
            layer_init(nn.Linear(self.in_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, len(self.experts))),
        ).to(device)
        with torch.no_grad():
            n_old = old_router[-1].weight.shape[0]
            new_router[-1].weight[:n_old].copy_(old_router[-1].weight)
            new_router[-1].bias[:n_old].copy_(old_router[-1].bias)
        self.router = new_router
        self.spawns += 1

    def fisher_recombine(self, scale: float = THIEF_LBF["fisher_scale"]):
        """Soft parameter sharing between the two first experts."""
        if len(self.experts) < 2:
            return
        params = [list(e.parameters()) for e in self.experts[:2]]
        with torch.no_grad():
            for p0, p1 in zip(*params):
                avg = 0.5 * (p0.data + p1.data)
                p0.data.mul_(1 - scale).add_(avg, alpha=scale)
                p1.data.mul_(1 - scale).add_(avg, alpha=scale)


class RomaAgent(nn.Module):
    """ROMA: role-conditioned PPO with dynamic role encoder and specialization losses."""

    def __init__(self, obs_dim, n_agents, act_dim, hidden):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.in_dim = obs_dim + n_agents
        self.role_dim = ROMA_LBF["role_dim"]
        self.role_encoder = nn.Sequential(
            layer_init(nn.Linear(self.in_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, self.role_dim)),
            nn.Tanh(),
        )
        self.role_decoder = nn.Sequential(
            layer_init(nn.Linear(self.in_dim + self.role_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, act_dim), std=0.01),
        )
        self.actor = build_mlp(self.in_dim + self.role_dim, hidden, act_dim)
        self.critic = build_mlp(n_agents * obs_dim, hidden, 1, out_std=1.0)

    def forward_actor(self, feat):
        role = self.role_encoder(feat)
        return self.actor(torch.cat([feat, role], dim=-1)), role

    def value(self, pooled_obs):
        return self.critic(pooled_obs).squeeze(-1)


class RodeAgent(nn.Module):
    """RODE: action decomposition via learned action embeddings; binary roles
    (LOCOMOTION vs FORAGING) selected every macro-step; subset-masked policy."""

    def __init__(self, obs_dim, n_agents, act_dim, hidden):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.in_dim = obs_dim + n_agents
        self.num_roles = 2
        self.embed_dim = RODE_LBF["embed_dim"]
        self.action_embeddings = nn.Embedding(act_dim, self.embed_dim)
        self.action_predictor = nn.Sequential(
            layer_init(nn.Linear(self.in_dim + self.embed_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, self.in_dim), std=0.1),
        )
        self.role_selector = nn.Sequential(
            layer_init(nn.Linear(self.in_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, self.num_roles)),
        )
        self.actor = build_mlp(self.in_dim + self.num_roles, hidden, act_dim)
        self.critic = build_mlp(n_agents * obs_dim, hidden, 1, out_std=1.0)
        self.register_buffer(
            "role_action_masks",
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0],  # 0: LOCOMOTION (no LOAD)
                    [1, 0, 0, 0, 0, 1],  # 1: FORAGING (stand + LOAD)
                ],
                dtype=torch.float32,
            ),
        )

    def select_roles(self, feat):
        return self.role_selector(feat).argmax(dim=-1)

    def forward_actor(self, feat, roles):
        role_onehot = F.one_hot(roles, num_classes=self.num_roles).float()
        return self.actor(torch.cat([feat, role_onehot], dim=-1))

    def value(self, pooled_obs):
        return self.critic(pooled_obs).squeeze(-1)

    def rep_loss(self, feat, actions, next_feat):
        emb = self.action_embeddings(actions)
        pred = self.action_predictor(torch.cat([feat, emb], -1))
        return F.smooth_l1_loss(pred, next_feat - feat)


class MappoAgent(nn.Module):
    """MAPPO: shared-parameter actor + centralized critic."""

    def __init__(self, obs_dim, n_agents, act_dim, hidden):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.in_dim = obs_dim + n_agents
        self.actor = build_mlp(self.in_dim, hidden, act_dim)
        self.critic = build_mlp(n_agents * obs_dim, hidden, 1, out_std=1.0)

    def forward_actor(self, feat):
        return self.actor(feat)

    def value(self, pooled_obs):
        return self.critic(pooled_obs).squeeze(-1)


class HmappoAgent(nn.Module):
    """H-MAPPO: manager outputs 2D directional sub-goals every macro-step;
    worker conditions on (features, sub-goal)."""

    def __init__(self, obs_dim, n_agents, act_dim, hidden):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.in_dim = obs_dim + n_agents
        self.manager = build_mlp(self.in_dim, 32, 4, out_std=1.0)
        self.worker = build_mlp(self.in_dim + 2, hidden, act_dim)
        self.critic = build_mlp(n_agents * obs_dim, hidden, 1, out_std=1.0)

    def forward_actor(self, feat, macro_goal):
        return self.worker(torch.cat([feat, macro_goal], dim=-1))

    def value(self, pooled_obs):
        return self.critic(pooled_obs).squeeze(-1)


class ComaAgent(nn.Module):
    """COMA: actor + centralized counterfactual joint-action critic.
    The critic conditions on the global state and the sampled joint action, and
    outputs per-(agent, action) Q values. Counterfactual baselines marginalize
    agent i's action while holding a_{-i} fixed."""

    def __init__(self, obs_dim, n_agents, act_dim, hidden):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.in_dim = obs_dim + n_agents
        self.actor = build_mlp(self.in_dim, hidden, act_dim)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_agents * obs_dim + n_agents * act_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.q_head = layer_init(nn.Linear(hidden, n_agents * act_dim), std=1.0)

    def forward_actor(self, feat):
        return self.actor(feat)

    def q_values(self, pooled_obs, joint_actions):
        """(B, n) int joint actions -> (B, n, act) Q values."""
        B = pooled_obs.shape[0]
        onehot = F.one_hot(joint_actions, num_classes=self.act_dim).float().view(B, -1)
        q = self.q_head(self.critic(torch.cat([pooled_obs, onehot], dim=-1)))
        return q.view(B, self.n_agents, self.act_dim)

    def q_taken(self, pooled_obs, joint_actions):
        """Q_i(s, a) for the executed joint action. Returns (B, n)."""
        q = self.q_values(pooled_obs, joint_actions)
        idx = joint_actions.unsqueeze(-1)  # (B, n, 1)
        return q.gather(2, idx).squeeze(-1)

    def counterfactual_baseline(self, pooled_obs, joint_actions):
        """Per-agent baseline: mean_{a'} Q_i(s, a', a_{-i}). Returns (B, n)."""
        B = pooled_obs.shape[0]
        n, act = self.n_agents, self.act_dim
        baselines = []
        for i in range(n):
            vals = []
            for a_prime in range(act):
                ja = joint_actions.clone()
                ja[:, i] = a_prime
                q = self.q_values(pooled_obs, ja)  # (B, n, act)
                vals.append(q[:, i, :].gather(1, ja[:, i : i + 1]))
            baselines.append(torch.cat(vals, dim=1).mean(dim=1))
        return torch.stack(baselines, dim=1)  # (B, n)


AGENT_FACTORIES = {
    "thief": ThiefAgent,
    "roma": RomaAgent,
    "rode": RodeAgent,
    "mappo": MappoAgent,
    "hmappo": HmappoAgent,
    "coma": ComaAgent,
}


# --------------------------------------------------------------------------------------
# Returns / GAE
# --------------------------------------------------------------------------------------


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """GAE over a rollout. rewards/dones: (T,); values: (T,) or (T, n)."""
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    T = len(rewards)
    if values.ndim == 2:
        adv = np.zeros_like(values)
        for i in range(values.shape[1]):
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalue = last_value[i]
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalue = values[t + 1, i]
                delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t, i]
                lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                adv[t, i] = lastgaelam
        return adv
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalue = last_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalue = values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv


def discounted_returns(rewards, dones, gamma):
    """Monte-Carlo returns shared across agents (used as COMA's critic target)."""
    T = len(rewards)
    out = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g * (1.0 - float(dones[t]))
        out[t] = g
    return out


# --------------------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------------------


def rollout(
    envs,
    us,
    agent,
    algo: str,
    device,
    ablation: str,
    steps: int,
    macro: int,
    ep_stats: dict,
    obs_batch,
    step_counter: int,
):
    """Collects `steps` lockstep steps across K vectorized envs.

    obs_batch: list of K per-env obs tuples. Returns (storage, obs_batch,
    step_counter). Storage tensors are (T, K, ...) with the T axis t-major.
    """
    K = len(envs)
    n = us[0].n_agents
    storage = defaultdict(list)
    eye = torch.eye(n, device=device)
    # per-env macro state (persists across macro-steps and episode resets)
    roles = None  # RODE: (K, n)
    macro_goal = None  # H-MAPPO: (K, n, 2)

    for t in range(steps):
        step_counter += 1
        obs_flat = torch.stack(
            [obs_to_tensor(o, device) for o in obs_batch]
        )  # (K, n, obs_dim)
        obs_dim = obs_flat.shape[-1]
        feat = torch.cat(
            [obs_flat, eye.unsqueeze(0).expand(K, -1, -1)], dim=-1
        ).reshape(K * n, -1)  # (K*n, in_dim)
        mask_t = torch.tensor(
            np.stack([get_masks(u) for u in us]), dtype=torch.float32, device=device
        )  # (K, n, 6)
        joint = obs_flat.reshape(K, -1)  # (K, n*obs_dim) full joint observation

        with torch.no_grad():
            if algo == "thief":
                logits = agent.forward_actor(
                    feat, uniform_routing=(ablation == "uniform_routing")
                )
            elif algo == "roma":
                logits, role_vec = agent.forward_actor(feat)  # (K*n, ...)
            elif algo == "rode":
                if roles is None or step_counter % macro == 0:
                    role_logits = agent.role_selector(feat)
                    roles = (
                        Categorical(logits=role_logits).sample().view(K, n).clone()
                    )
                logits = agent.forward_actor(feat, roles.view(-1))
                subset = agent.role_action_masks[roles.view(-1)]  # (K*n, 6)
                mask_t = mask_t.view(K * n, 6) * subset
            elif algo == "hmappo":
                if macro_goal is None or step_counter % macro == 0:
                    probs = torch.softmax(agent.manager(feat), dim=-1)  # (K*n, 4)
                    macro_goal = torch.stack(
                        [probs[:, 0] - probs[:, 1], probs[:, 3] - probs[:, 2]], dim=-1
                    ).view(K, n, 2).clone()
                logits = agent.forward_actor(feat, macro_goal.view(-1, 2))
            else:  # mappo
                logits = agent.forward_actor(feat)

            masked_logits = logits.view(K, n, -1) + (1.0 - mask_t.view(K, n, -1)) * -1e9
            dist = Categorical(logits=masked_logits)
            actions = dist.sample()  # (K, n)
            logprob = dist.log_prob(actions)

            if algo == "coma":
                q = agent.q_values(joint, actions)  # (K, n, act)
                value = q.mean(dim=-1)  # (K, n) expected Q_i uniform marginal
            else:
                value = agent.value(joint).unsqueeze(-1).expand(K, n)  # (K, n)

        acts_np = actions.cpu().numpy()
        phi_prev = np.array([shaping_potential(u, ep_stats[k]["food_ref"]) for k, u in enumerate(us)])
        learning_rewards = np.zeros(K, dtype=np.float32)
        dones = np.zeros(K, dtype=np.float32)
        for k in range(K):
            next_obs, reward, term, trunc, _ = envs[k].step(acts_np[k].tolist())
            team_reward = float(np.sum(reward))
            done = bool(term or trunc)
            if HARNESS["shaping"]:
                shaped = (
                    0.0 if done else HARNESS["gamma"]
                ) * shaping_potential(us[k], ep_stats[k]["food_ref"]) - phi_prev[k]
                learning_rewards[k] = team_reward + shaped
            else:
                learning_rewards[k] = team_reward
            dones[k] = float(done)
            ep_stats[k]["ep_reward"] += team_reward  # PURE LBF reward for reporting
            if done:
                st = ep_stats[k]
                st["episodes"] += 1
                st["returns"].append(st["ep_reward"])
                st["foods"].append(st["cur_init_food"] - int((us[k].field > 0).sum()))
                st["wins"].append(1.0 if int((us[k].field > 0).sum()) == 0 else 0.0)
                st["steps_list"].append(us[k].current_step)
                st["ep_reward"] = 0.0
                obs_batch[k], _ = envs[k].reset()
                st["cur_init_food"] = int((us[k].field > 0).sum())
                st["food_ref"] = np.argwhere(us[k].field > 0)
            else:
                obs_batch[k] = next_obs

        storage["feat"].append(feat.cpu())
        storage["mask"].append(mask_t.reshape(K, n, 6).cpu())
        storage["actions"].append(actions.cpu())
        storage["logprobs"].append(logprob.cpu())
        storage["value"].append(value.cpu())
        storage["reward"].append(learning_rewards.copy())
        storage["done"].append(dones.copy())
        if algo == "roma":
            storage["role_vec"].append(role_vec.view(K, n, -1).cpu())
        if algo == "rode":
            storage["roles"].append(roles.clone().cpu())
        if algo == "hmappo":
            storage["goals"].append(macro_goal.clone().cpu())

    return storage, obs_batch, step_counter


# --------------------------------------------------------------------------------------
# PPO / COMA update
# --------------------------------------------------------------------------------------


def joint_obs_from_feats(feats, T, K, n, obs_dim, device):
    """feats: (T, K*n, in_dim) -> (T*K, n*obs_dim) full joint observation, t-major."""
    return feats.view(T, K, n, -1)[:, :, :, :obs_dim].reshape(T * K, -1).to(device)


def update_agent(agent, algo, storage, device, ablation, optimizer, obs_dim, K, ent_coef):
    n = agent.n_agents
    feats = torch.stack(storage["feat"], dim=0).to(device)  # (T, K*n, in_dim)
    actions = torch.stack(storage["actions"], dim=0).to(device)  # (T, K, n)
    old_logprobs = torch.stack(storage["logprobs"], dim=0).to(device)  # (T, K, n)
    T = len(storage["reward"])

    rewards = np.stack(storage["reward"])  # (T, K)
    dones = np.stack(storage["done"])  # (T, K)
    values_raw = torch.stack(storage["value"]).numpy()  # (T, K, n)

    feats_flat = feats.reshape(T * K * n, -1)
    actions_flat = actions.reshape(-1)
    old_logprobs_flat = old_logprobs.reshape(-1)
    masks_flat = (
        torch.stack(storage["mask"], dim=0).to(device).reshape(T * K * n, -1)
    )  # rollout-time masks (incl. RODE role subsets) for consistent log-probs
    joint = joint_obs_from_feats(feats, T, K, n, obs_dim, device)  # (T*K, n*obs)
    ja = actions.reshape(T * K, n)  # (T*K, n) joint actions, t-major
    last_value = storage["last_value"].to(device)  # (K, n)

    def masked_logits_from(logits):
        return logits + (1.0 - masks_flat) * -1e9

    stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "aux": 0.0}

    if algo == "coma":
        # ---- COMA: counterfactual advantages + MC critic regression ----
        with torch.no_grad():
            q_taken = agent.q_taken(joint, ja)  # (T*K, n)
            cf_base = agent.counterfactual_baseline(joint, ja)  # (T*K, n)
            adv_cf = q_taken - cf_base
            adv_norm = (adv_cf - adv_cf.mean()) / (adv_cf.std() + 1e-8)
            adv_flat = adv_norm.reshape(-1)  # matches feats row order
            # per-env Monte-Carlo returns, shared across agents
            mc = np.zeros((T, K), dtype=np.float32)
            for k in range(K):
                mc[:, k] = discounted_returns(rewards[:, k], dones[:, k], HARNESS["gamma"])
            target = (
                torch.tensor(mc, dtype=torch.float32, device=device)
                .unsqueeze(-1)
                .expand(-1, -1, n)
                .reshape(-1)
            )

        for _ in range(HARNESS["update_epochs"]):
            logits = agent.forward_actor(feats_flat)
            dist = Categorical(logits=masked_logits_from(logits))
            new_logp = dist.log_prob(actions_flat)
            entropy = dist.entropy().mean()
            # policy gradient with counterfactual (marginalized) advantages
            pg_loss = -(new_logp * adv_flat).mean()
            # centralized critic regression toward MC returns
            q_pred = agent.q_taken(joint, ja).reshape(-1)
            v_loss = F.mse_loss(q_pred, target)
            loss = pg_loss - ent_coef * entropy + HARNESS["vf_coef"] * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), HARNESS["grad_clip"])
            optimizer.step()
            stats.update(
                pg_loss=float(pg_loss.item()),
                v_loss=float(v_loss.item()),
                entropy=float(entropy.item()),
            )
        return stats

    # ---- PPO family (thief, roma, rode, hmappo, mappo) ----
    adv = np.zeros_like(values_raw)  # (T, K, n)
    for k in range(K):
        adv[:, k, :] = compute_gae(
            rewards[:, k],
            values_raw[:, k, :],
            dones[:, k],
            last_value[k].tolist(),
            HARNESS["gamma"],
            HARNESS["gae_lambda"],
        )
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    adv_norm = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    adv_flat = adv_norm.reshape(-1)  # (T*K*n,)
    values_t = torch.tensor(values_raw, dtype=torch.float32, device=device)
    ret_flat = (adv_norm + values_t).reshape(-1)

    if algo == "rode":
        roles_b = torch.stack(storage["roles"], dim=0).to(device).reshape(-1)  # (T*K*n,)
    if algo == "hmappo":
        goals_b = torch.stack(storage["goals"], dim=0).to(device)  # (T, K, n, 2)
        goals_b = goals_b.reshape(-1, 2)

    for _ in range(HARNESS["update_epochs"]):
        if algo == "thief":
            logits = agent.forward_actor(
                feats_flat, uniform_routing=(ablation == "uniform_routing")
            )
        elif algo == "roma":
            logits, _ = agent.forward_actor(feats_flat)
        elif algo == "rode":
            logits = agent.forward_actor(feats_flat, roles_b)
        elif algo == "hmappo":
            logits = agent.forward_actor(feats_flat, goals_b)
        else:
            logits = agent.forward_actor(feats_flat)

        dist = Categorical(logits=masked_logits_from(logits))
        new_logp = dist.log_prob(actions_flat)
        entropy = dist.entropy().mean()
        ratio = (new_logp - old_logprobs_flat).exp()
        pg1 = -adv_flat * ratio
        pg2 = -adv_flat * torch.clamp(
            ratio, 1 - HARNESS["clip_coef"], 1 + HARNESS["clip_coef"]
        )
        pg_loss = torch.max(pg1, pg2).mean()

        v_pred = agent.value(joint).unsqueeze(-1).expand(-1, n).reshape(-1)
        v_loss = F.smooth_l1_loss(v_pred, ret_flat)
        loss = pg_loss - ent_coef * entropy + HARNESS["vf_coef"] * v_loss

        # Algorithm-specific auxiliary objectives
        if algo == "roma":
            role_vec_b = agent.role_encoder(feats_flat)
            ident = F.cross_entropy(
                agent.role_decoder(torch.cat([feats_flat, role_vec_b], -1)), actions_flat
            )
            rv = torch.stack(storage["role_vec"], dim=0).to(device)  # (T, K, n, rd)
            compact = (rv[1:] - rv[:-1]).pow(2).sum(-1).mean()
            dissim, pairs = 0.0, 0
            for i in range(n):
                for j in range(i + 1, n):
                    dissim = dissim + F.relu(
                        1.0 - (rv[:, :, i] - rv[:, :, j]).pow(2).sum(-1)
                    ).mean()
                    pairs += 1
            dissim = dissim / max(1, pairs)
            loss = (
                loss
                + ROMA_LBF["ident_coef"] * ident
                + ROMA_LBF["compact_coef"] * compact
                + ROMA_LBF["dissim_coef"] * dissim
            )
            stats["aux"] = float((ident + compact + dissim).item())
        elif algo == "rode":
            nxt = torch.roll(feats_flat, shifts=-n, dims=0)
            rep = agent.rep_loss(feats_flat, actions_flat, nxt)
            rlogits = agent.role_selector(feats_flat)
            rent = -(
                torch.softmax(rlogits, -1) * torch.log_softmax(rlogits, -1)
            ).sum(-1).mean()
            loss = loss + RODE_LBF["rep_coef"] * rep - RODE_LBF["role_ent_coef"] * rent
            stats["aux"] = float((rep + rent).item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), HARNESS["grad_clip"])
        optimizer.step()
        stats.update(
            pg_loss=float(pg_loss.item()),
            v_loss=float(v_loss.item()),
            entropy=float(entropy.item()),
        )

    return stats


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def evaluate_policy(env, u, agent, algo, device, num_episodes, base_seed, greedy=False):
    returns, foods, steps = [], [], []
    n = u.n_agents
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed * 100_000 + ep + 50_000)
        init_food = int((u.field > 0).sum())
        ep_ret, ep_steps, done = 0.0, 0, False
        while not done and ep_steps < 60:
            ep_steps += 1
            masks = get_masks(u)
            obs_flat = obs_to_tensor(obs, device)
            mask_t = torch.tensor(masks, dtype=torch.float32, device=device)
            feat = torch.cat([obs_flat, torch.eye(n, device=device)], dim=-1)
            with torch.no_grad():
                if algo == "thief":
                    logits = agent.forward_actor(feat)
                elif algo == "roma":
                    logits, _ = agent.forward_actor(feat)
                elif algo == "rode":
                    roles = agent.select_roles(feat)
                    logits = agent.forward_actor(feat, roles)
                    mask_t = mask_t * agent.role_action_masks[roles]
                elif algo == "hmappo":
                    probs = torch.softmax(agent.manager(feat), dim=-1)
                    goal = torch.stack(
                        [probs[:, 0] - probs[:, 1], probs[:, 3] - probs[:, 2]], dim=-1
                    )
                    logits = agent.forward_actor(feat, goal)
                else:
                    logits = agent.forward_actor(feat)
            masked = logits + (1.0 - mask_t) * -1e9
            if greedy:
                acts = torch.argmax(masked, dim=-1)
            else:
                # Stochastic sampling (matches the HEIST evaluation protocol); fixed
                # seeds keep the evaluation reproducible.
                acts = Categorical(logits=masked).sample()
            obs, reward, term, trunc, _ = env.step(acts.cpu().numpy().tolist())
            ep_ret += float(np.sum(reward))
            done = bool(term or trunc)
        returns.append(ep_ret)
        foods.append(max(0, init_food - int((u.field > 0).sum())))
        steps.append(ep_steps)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_food": float(np.mean(foods)),
        "std_food": float(np.std(foods)),
        "mean_steps": float(np.mean(steps)),
        "episodes": num_episodes,
    }


# --------------------------------------------------------------------------------------
# Single (env, algo, seed[, ablation]) training run
# --------------------------------------------------------------------------------------


def train_one(config: dict) -> dict:
    env_id = config["env_id"]
    algo = config["algo"]
    seed = config["seed"]
    total_timesteps = config["total_timesteps"]
    eval_episodes = config["eval_episodes"]
    out_root = config["out_root"]
    device_str = config["device"]
    ablation = config.get("ablation", "full")
    greedy = config.get("greedy", False)

    set_global_seed(seed * 1000 + 17)
    device = torch.device(device_str)
    K = HARNESS["num_envs"]
    envs = [gym.make(env_id) for _ in range(K)]
    us = [e.unwrapped for e in envs]
    n = us[0].n_agents
    obs_batch = []
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=seed * 100 + i)
        obs_batch.append(o)
    obs_dim = int(np.array(obs_batch[0]).size // n)
    act_dim = 6

    run_name = algo if ablation == "full" else f"{algo}_ablation_{ablation}"
    out_dir = os.path.join(out_root, env_id, run_name, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    agent = AGENT_FACTORIES[algo](obs_dim, n, act_dim, HARNESS["hidden"]).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=HARNESS["lr"], eps=1e-5)

    ep_stats = [
        {
            "ep_reward": 0.0,
            "episodes": 0,
            "cur_init_food": int((us[k].field > 0).sum()),
            "food_ref": np.argwhere(us[k].field > 0),
            "returns": [],
            "foods": [],
            "wins": [],
            "steps_list": [],
        }
        for k in range(K)
    ]
    macro = HARNESS["macro_step"]
    rollout_len = HARNESS["rollout_len"]
    num_updates = total_timesteps // (rollout_len * K)
    step_counter = 0
    curves = []
    start = time.time()

    def all_wins(window=30):
        wins = [w for st in ep_stats for w in st["wins"]]
        return float(np.mean(wins[-window:])) if wins else 0.0

    warmup_updates = max(1, int(0.25 * num_updates))
    for update in range(1, num_updates + 1):
        # entropy warmup: high exploration early, annealed to the final value
        ent_coef = HARNESS["ent_coef_end"] + (
            HARNESS["ent_coef_start"] - HARNESS["ent_coef_end"]
        ) * max(0.0, 1.0 - (update - 1) / warmup_updates)
        storage, obs_batch, step_counter = rollout(
            envs,
            us,
            agent,
            algo,
            device,
            ablation,
            rollout_len,
            macro,
            ep_stats,
            obs_batch,
            step_counter,
        )
        # bootstrap value from the true post-rollout state
        with torch.no_grad():
            obs_flat_next = torch.stack(
                [obs_to_tensor(o, device) for o in obs_batch]
            )  # (K, n, obs_dim)
            joint_next = obs_flat_next.reshape(K, -1)
            if algo == "coma":
                ja_last = (
                    torch.stack(
                        [a.clone().detach() if isinstance(a, torch.Tensor) else torch.tensor(a)
                         for a in (storage["actions"][-1])]
                    ).to(device)
                    if storage["actions"]
                    else torch.zeros(K, n, dtype=torch.long, device=device)
                )
                lv = agent.q_taken(joint_next, ja_last)
            else:
                lv = agent.value(joint_next).unsqueeze(-1).expand(K, n)
        storage["last_value"] = lv.cpu()
        stats = update_agent(
            agent, algo, storage, device, ablation, optimizer, obs_dim, K, ent_coef
        )

        # THIEF dynamic-capacity mechanisms
        if algo == "thief" and ablation in ("full", "no_fisher"):
            agent.maybe_spawn(all_wins(THIEF_LBF["spawn_window"]))
            if ablation == "full" and update % THIEF_LBF["fisher_interval"] == 0:
                agent.fisher_recombine()

        if update % 25 == 0 or update == num_updates:
            all_returns = [r for st in ep_stats for r in st["returns"]]
            curves.append(
                {
                    "update": update,
                    "timesteps": update * rollout_len * K,
                    "train_win_rate": all_wins(30),
                    "train_return": float(np.mean(all_returns[-30:]))
                    if all_returns
                    else 0.0,
                    "pg_loss": stats["pg_loss"],
                    "v_loss": stats["v_loss"],
                    "entropy": stats["entropy"],
                    "active_experts": len(agent.experts) if algo == "thief" else 1,
                }
            )

    final = evaluate_policy(
        envs[0], us[0], agent, algo, device, eval_episodes, seed, greedy=greedy
    )
    elapsed = time.time() - start
    total_episodes = sum(st["episodes"] for st in ep_stats)
    all_wins_flat = [w for st in ep_stats for w in st["wins"]]

    result = {
        "env_id": env_id,
        "algo": algo,
        "ablation": ablation,
        "run_name": run_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "rollout_len": rollout_len,
        "num_envs": K,
        "update_epochs": HARNESS["update_epochs"],
        "lr": HARNESS["lr"],
        "gamma": HARNESS["gamma"],
        "reward_shaping": "potential-based (Ng et al. 1999), identical across algos"
        if HARNESS["shaping"]
        else "none",
        "training_time_sec": elapsed,
        "n_training_episodes": total_episodes,
        "final_eval": final,
        "final_train_win_rate": float(np.mean(all_wins_flat[-100:]))
        if all_wins_flat
        else 0.0,
        "active_experts_final": len(agent.experts) if algo == "thief" else 1,
        "total_spawns": agent.spawns if algo == "thief" else 0,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "train_curves.json"), "w") as f:
        json.dump(curves, f)
    for e in envs:
        e.close()
    print(
        f"[LBF] {run_name} | {env_id} | seed {seed} | "
        f"return {final['mean_return']:.3f} | food {final['mean_food']:.2f} | {elapsed:.0f}s",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------------------
# Aggregation & report generation
# --------------------------------------------------------------------------------------


def aggregate(out_root: str = "results/lbf", paper_tables: str = "paper/tables"):
    """Compiles all results.json files into summary JSON/MD and LaTeX tables."""
    runs = []
    for root, _dirs, files in os.walk(out_root):
        if "results.json" in files:
            with open(os.path.join(root, "results.json")) as f:
                runs.append(json.load(f))
    if not runs:
        print("No LBF results found to aggregate.")
        return None

    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["env_id"], r["run_name"])].append(r)

    summary = {}
    for (env_id, run_name), rs in sorted(grouped.items()):
        rets = [r["final_eval"]["mean_return"] for r in rs]
        foods = [r["final_eval"]["mean_food"] for r in rs]
        steps = [r["final_eval"]["mean_steps"] for r in rs]
        # max_num_food per configuration (parsed from the "<n>f" segment of the id)
        m = re.search(r"-(\d+)f", env_id)
        init_food = int(m.group(1)) if m else 3
        food_pcts = [100.0 * f / init_food for f in foods]
        summary[f"{env_id}::{run_name}"] = {
            "env_id": env_id,
            "run_name": run_name,
            "algo": rs[0]["algo"],
            "ablation": rs[0]["ablation"],
            "seeds": sorted(r["seed"] for r in rs),
            "n_seeds": len(rs),
            "return_mean": float(np.mean(rets)),
            "return_std": float(np.std(rets)),
            "food_pct_mean": float(np.mean(food_pcts)),
            "food_pct_std": float(np.std(food_pcts)),
            "food_count_mean": float(np.mean(foods)),
            "steps_mean": float(np.mean(steps)),
            "mean_spawns": float(np.mean([r.get("total_spawns", 0) for r in rs])),
            "mean_active_experts": float(
                np.mean([r.get("active_experts_final", 1) for r in rs])
            ),
            "per_seed": [
                {
                    "seed": r["seed"],
                    "return": r["final_eval"]["mean_return"],
                    "food": r["final_eval"]["mean_food"],
                    "steps": r["final_eval"]["mean_steps"],
                }
                for r in sorted(rs, key=lambda x: x["seed"])
            ],
        }

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "lbf_benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Markdown report ----
    md = [
        "# Level-Based Foraging (LBF) Heterogeneous MARL Benchmark",
        "",
        "Trained-from-scratch evaluation on the canonical Level-Based Foraging "
        "benchmark (Papoudakis et al., NeurIPS 2021 Datasets & Benchmarks). All "
        "methods use identical PPO budgets; final policies are evaluated greedily.",
        "",
    ]
    for env_cfg in LBF_ENV_CONFIGS:
        env_id = env_cfg["id"]
        md.append(f"## {env_cfg['name']} (`{env_id}`)")
        md.append(f"*{env_cfg['desc']}*\n")
        md.append("| Algorithm | Return (±Std) | Food Collected (%) | Steps |")
        md.append("| :--- | :--- | :--- | :--- |")
        for algo in LBF_ALGOS:
            k = f"{env_id}::{algo}"
            if k not in summary:
                continue
            s = summary[k]
            md.append(
                f"| **{algo.upper()}** | {s['return_mean']:.3f} ± {s['return_std']:.3f} | "
                f"{s['food_pct_mean']:.1f}% ± {s['food_pct_std']:.1f}% | {s['steps_mean']:.1f} |"
            )
        md.append("")
    md.append("## THIEF-on-LBF Component Ablations")
    md.append("")
    for env_cfg in LBF_ENV_CONFIGS:
        env_id = env_cfg["id"]
        rows = [k for k in summary if k.startswith(f"{env_id}::thief")]
        if not rows:
            continue
        md.append(f"### {env_cfg['name']}")
        md.append("")
        md.append("| Variant | Return (±Std) | Food Collected (%) | Spawns | Final Experts |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for k in sorted(rows):
            s = summary[k]
            md.append(
                f"| {s['run_name']} | {s['return_mean']:.3f} ± {s['return_std']:.3f} | "
                f"{s['food_pct_mean']:.1f}% ± {s['food_pct_std']:.1f}% | "
                f"{s['mean_spawns']:.1f} | {s['mean_active_experts']:.1f} |"
            )
        md.append("")
    with open(os.path.join(out_root, "lbf_benchmark_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ---- LaTeX main table ----
    os.makedirs(paper_tables, exist_ok=True)
    n_seeds = max((s["n_seeds"] for s in summary.values()), default=1)
    e1, e2 = LBF_ENV_CONFIGS[0]["id"], LBF_ENV_CONFIGS[1]["id"]
    lines = [
        "% Auto-generated by src/py/train_lbf.py (aggregate). Do not edit manually.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Benchmark results on the canonical Level-Based Foraging (LBF) "
        "multi-agent testbed \\cite{papoudakis2021benchmarking} across two cooperative "
        "configurations. All methods are \\emph{trained} with identical PPO budgets "
        "(lr $2.5{\\times}10^{-4}$, $\\gamma{=}0.99$, GAE $\\lambda{=}0.95$, clip $0.2$, "
        "4 update epochs, $T{=}125$ rollout) and evaluated greedily over "
        f"{n_seeds} independent seeds. Bold denotes highest performance.}}",
        "\\label{tab:lbf_benchmark}",
        "\\vspace{0.05in}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l cc cc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{Foraging-8$\\times$8 (2p, 2f, Forced Coop)}} & \\multicolumn{2}{c}{\\textbf{Foraging-8$\\times$8 (3p, 3f, Asymmetric)}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "\\textbf{Algorithm} & \\textbf{Return} & \\textbf{Food Collected} & \\textbf{Return} & \\textbf{Food Collected} \\\\",
        " & (Mean $\\pm$ Std) & (\\%) & (Mean $\\pm$ Std) & (\\%) \\\\",
        "\\midrule",
    ]

    present = [a for a in LBF_ALGOS if f"{e1}::{a}" in summary and f"{e2}::{a}" in summary]
    if present:
        best = {
            (e, m): max(summary[f"{e}::{a}"][m] for a in present)
            for e in (e1, e2)
            for m in ("return_mean", "food_pct_mean")
        }
    else:
        best = {(e, m): 0.0 for e in (e1, e2) for m in ("return_mean", "food_pct_mean")}
    for a in present:
        cells = []
        for e in (e1, e2):
            s = summary[f"{e}::{a}"]
            r_str = f"{s['return_mean']:.2f}\\tiny$\\pm${s['return_std']:.2f}"
            if abs(s["return_mean"] - best[(e, "return_mean")]) < 1e-9:
                r_str = f"\\textbf{{{r_str}}}"
            f_str = f"{s['food_pct_mean']:.1f}\\%"
            if abs(s["food_pct_mean"] - best[(e, "food_pct_mean")]) < 1e-9:
                f_str = f"\\textbf{{{f_str}}}"
            cells.extend([r_str, f_str])
        lines.append(f"{ALGO_LABELS_TEX[a]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    with open(os.path.join(paper_tables, "lbf_benchmark.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- LaTeX THIEF-on-LBF ablation table ----
    abl_present = [
        v
        for v in ["full", "fixed_experts", "no_fisher", "uniform_routing"]
        if any(f"{e}::thief" + ("" if v == "full" else f"_ablation_{v}") in summary for e in (e1, e2))
    ]
    if abl_present:
        lines = [
            "% Auto-generated by src/py/train_lbf.py (aggregate). Do not edit manually.",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{THIEF component ablations on LBF. Each variant disables one "
            "mechanism of the full THIEF system while holding all other training "
            "hyperparameters and seeds fixed. Bold denotes best per column.}",
            "\\label{tab:lbf_ablation}",
            "\\vspace{0.05in}",
            "\\footnotesize",
            "\\begin{tabular}{l cc cc}",
            "\\toprule",
            " & \\multicolumn{2}{c}{\\textbf{8$\\times$8 (2p, 2f, Coop)}} & \\multicolumn{2}{c}{\\textbf{8$\\times$8 (3p, 3f, Asym)}} \\\\",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
            "\\textbf{Variant} & \\textbf{Return} & \\textbf{Food (\\%)} & \\textbf{Return} & \\textbf{Food (\\%)} \\\\",
            "\\midrule",
        ]

        def abl_key(env, variant):
            return f"{env}::thief" + ("" if variant == "full" else f"_ablation_{variant}")

        best_abl = {
            (e, m): max(
                summary[abl_key(e, v)][m]
                for v in abl_present
                if abl_key(e, v) in summary
            )
            for e in (e1, e2)
            for m in ("return_mean", "food_pct_mean")
        }
        for v in abl_present:
            cells = []
            for e in (e1, e2):
                k = abl_key(e, v)
                if k not in summary:
                    cells.extend(["--", "--"])
                    continue
                s = summary[k]
                r_str = f"{s['return_mean']:.2f}\\tiny$\\pm${s['return_std']:.2f}"
                if abs(s["return_mean"] - best_abl[(e, "return_mean")]) < 1e-9:
                    r_str = f"\\textbf{{{r_str}}}"
                f_str = f"{s['food_pct_mean']:.1f}"
                if abs(s["food_pct_mean"] - best_abl[(e, "food_pct_mean")]) < 1e-9:
                    f_str = f"\\textbf{{{f_str}}}"
                cells.extend([r_str, f_str])
            name = ABLATION_LABELS_TEX[v]
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        with open(os.path.join(paper_tables, "lbf_ablation.tex"), "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Aggregated {len(runs)} runs into {out_root}/lbf_benchmark_summary.{{json,md}}")
    print(
        f"Wrote {paper_tables}/lbf_benchmark.tex"
        + (" and lbf_ablation.tex" if abl_present else "")
    )
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_seeds(s):
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main():
    p = argparse.ArgumentParser(description="LBF training benchmark suite")
    p.add_argument("--algos", default="all", help="comma list or 'all'")
    p.add_argument("--envs", default="all", help="comma list of env ids or 'all'")
    p.add_argument("--seeds", default="0-4", help="e.g. 0-4 or 0,1,2")
    p.add_argument("--timesteps", type=int, default=400_000, help="env steps per run")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument(
        "--greedy-eval",
        action="store_true",
        help="evaluate with deterministic argmax actions instead of stochastic sampling",
    )
    p.add_argument("--jobs", type=int, default=1, help="parallel training processes")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--out-root", default="results/lbf")
    p.add_argument(
        "--thief-ablations", action="store_true", help="also run THIEF-on-LBF ablations"
    )
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.aggregate_only:
        aggregate(args.out_root)
        return

    algos = LBF_ALGOS if args.algos == "all" else [x.strip() for x in args.algos.split(",")]
    envs = (
        [e["id"] for e in LBF_ENV_CONFIGS]
        if args.envs == "all"
        else [x.strip() for x in args.envs.split(",")]
    )
    seeds = parse_seeds(args.seeds)
    timesteps, eval_eps = args.timesteps, args.eval_episodes
    if args.smoke:
        timesteps, eval_eps, seeds = 3_000, 8, [0]

    configs = []
    for env_id in envs:
        for algo in algos:
            for seed in seeds:
                configs.append(
                    {
                        "env_id": env_id,
                        "algo": algo,
                        "seed": seed,
                        "total_timesteps": timesteps,
                        "eval_episodes": eval_eps,
                        "out_root": args.out_root,
                        "device": args.device,
                        "ablation": "full",
                        "greedy": args.greedy_eval,
                    }
                )
        if args.thief_ablations and "thief" in algos:
            for variant in LBF_ABLATION_VARIANTS:
                if variant == "full":
                    continue
                for seed in seeds:
                    configs.append(
                        {
                            "env_id": env_id,
                            "algo": "thief",
                            "seed": seed,
                            "total_timesteps": timesteps,
                            "eval_episodes": eval_eps,
                            "out_root": args.out_root,
                            "device": args.device,
                            "ablation": variant,
                        }
                    )

    print(
        f"LBF benchmark: {len(configs)} runs | algos={algos} | envs={envs} | seeds={seeds}"
    )
    if args.jobs <= 1:
        for cfg in configs:
            train_one(cfg)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(train_one, cfg): cfg for cfg in configs}
            for fut in as_completed(futures):
                cfg = futures[fut]
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[ERROR] {cfg['algo']}/{cfg['env_id']}/seed{cfg['seed']}: {e}"
                    )

    aggregate(args.out_root)


if __name__ == "__main__":
    main()
