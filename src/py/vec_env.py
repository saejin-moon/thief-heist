"""
High-throughput native Rust VectorEnv for HEIST.
Executes multi-agent environments in multi-threaded parallel Rust memory without Python IPC or pipes.
"""

import json
import os
import sys

import numpy as np

# Ensure local compiled extension is importable
rust_target_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rs", "target", "release")
)
if os.path.exists(rust_target_dir) and rust_target_dir not in sys.path:
    sys.path.insert(0, rust_target_dir)

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import heist_core_rs  # type: ignore
from constants import (
    ACTION_SPACE_SIZE,
    AGENTS,
    GOAL_VECTOR_DIM,
    N_AGENTS,
    OBSERVATION_SIZE,
)


class RustVectorEnv:
    """
    High-throughput native Rust VectorEnv for HEIST.
    Drop-in vectorized environment for CleanRL-style MARL training loops.
    """

    def __init__(self, num_envs=16, config=None, base_seed=0):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.config = config or {}

        config_json = json.dumps(self.config)
        self._native_batch = heist_core_rs.PyHeistBatchEnv(
            num_envs, config_json, base_seed
        )
        self.state_dim = self._native_batch.state_dim()

        # Pre-allocate contiguous NumPy array buffers for zero-copy native memory writes
        self._obs_buf = np.zeros(
            (N_AGENTS, num_envs, OBSERVATION_SIZE[0], OBSERVATION_SIZE[1]),
            dtype=np.int32,
        )
        self._mask_buf = np.zeros(
            (N_AGENTS, num_envs, ACTION_SPACE_SIZE), dtype=np.int8
        )
        self._role_buf = np.zeros((N_AGENTS, num_envs, N_AGENTS), dtype=np.int8)
        self._goal_buf = np.zeros(
            (N_AGENTS, num_envs, GOAL_VECTOR_DIM), dtype=np.float32
        )
        self._state_buf = np.zeros((num_envs, self.state_dim), dtype=np.float32)

        self._rewards_buf = np.zeros((N_AGENTS, num_envs), dtype=np.float32)
        self._terms_buf = np.zeros((N_AGENTS, num_envs), dtype=bool)
        self._truncs_buf = np.zeros((N_AGENTS, num_envs), dtype=bool)

        self.obs = None
        self.state = None

    def reset(self, seed=None):
        base = seed if seed is not None else self.base_seed
        self._native_batch.reset(
            base,
            self._obs_buf,
            self._mask_buf,
            self._role_buf,
            self._goal_buf,
            self._state_buf,
        )
        self.obs = self._format_obs()
        self.state = self._state_buf.copy()
        return self.obs, self.state

    def step(self, actions):
        # Format actions into contiguous 2D array [N_AGENTS, NUM_ENVS]
        act_array = np.stack([actions[a] for a in AGENTS], axis=0).astype(np.int32)

        infos = self._native_batch.step(
            act_array,
            self._obs_buf,
            self._mask_buf,
            self._role_buf,
            self._goal_buf,
            self._rewards_buf,
            self._terms_buf,
            self._truncs_buf,
            self._state_buf,
        )

        rewards = {a: self._rewards_buf[i].copy() for i, a in enumerate(AGENTS)}
        terms = {a: self._terms_buf[i].copy() for i, a in enumerate(AGENTS)}
        truncs = {a: self._truncs_buf[i].copy() for i, a in enumerate(AGENTS)}

        self.obs = self._format_obs()
        self.state = self._state_buf.copy()

        return self.obs, rewards, terms, truncs, infos

    def _format_obs(self):
        packed = {}
        for i, a in enumerate(AGENTS):
            packed[a] = {
                "observation": self._obs_buf[i],
                "action_mask": self._mask_buf[i],
                "role_id": self._role_buf[i],
                "goal_vector": self._goal_buf[i],
            }
        packed["_stacked"] = {
            "observation": self._obs_buf,
            "action_mask": self._mask_buf,
            "role_id": self._role_buf,
            "goal_vector": self._goal_buf,
        }
        return packed

    def close(self):
        pass


VectorEnv = RustVectorEnv


def make_vec_env(num_envs=16, config=None, base_seed=0, use_rust=True):
    """
    Factory function returning the high-throughput native RustVectorEnv.
    """
    return RustVectorEnv(num_envs=num_envs, config=config, base_seed=base_seed)
