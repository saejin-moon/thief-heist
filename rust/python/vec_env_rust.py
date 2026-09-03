import json
import os
import sys

# Ensure local compiled extension is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import heist_core_rs
import numpy as np

AGENTS = ["scout", "hacker", "muscle", "extractor"]
N_AGENTS = len(AGENTS)


class RustVectorEnv:
    """
    High-throughput native Rust VectorEnv for HEIST.
    Drop-in replacement for src/vec_env.py:VectorEnv.
    Executes in multi-threaded parallel C/Rust memory without Python IPC or pipes.
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
        self._obs_buf = np.zeros((N_AGENTS, num_envs, 7, 7), dtype=np.int32)
        self._mask_buf = np.zeros((N_AGENTS, num_envs, 6), dtype=np.int8)
        self._role_buf = np.zeros((N_AGENTS, num_envs, N_AGENTS), dtype=np.int8)
        self._goal_buf = np.zeros((N_AGENTS, num_envs, 2), dtype=np.float32)
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
        self.state = self._state_buf
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
        self.state = self._state_buf

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
