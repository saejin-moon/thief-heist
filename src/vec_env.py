"""
Vectorized environment wrapper for HEIST.

Wraps `num_envs` independent HeistEnv instances so the PPO training loops
can collect rollouts in a simple numpy array layout (one buffer per agent),
mirroring the CleanRL vector-env convention.

Each env is stepped with a dict of per-agent actions.  When an episode
terminates or truncates, that env is immediately reset so the rollout buffer
can be kept dense.  REV-4 (REVISION_PLAN.md §3a) fixes the contract so the
TRUE terminal observation is not lost: ``step`` stashes it (packed per agent)
in ``infos[env]["terminal_observation"]`` (and the terminal env-state in
``infos[env]["terminal_state"]``) BEFORE the internal reset, and returns
terminations and truncations separately so the trainer can bootstrap value
estimates correctly on truncation (REV-3, REVISION_PLAN.md §3b).
"""

import multiprocessing as mp

import numpy as np

from env import AGENTS, HeistEnv


def _worker(remote, parent_remote, config):
    parent_remote.close()
    env = HeistEnv(config)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                if not env.agents:
                    env.reset()
                o, r, t, tr, inf = env.step(data)
                done = bool(any(t.values()) or any(tr.values()))
                term_obs = None
                term_state = None
                if done:
                    # pack terminal observation
                    n_agents = len(AGENTS)
                    obs_shape = o["scout"]["observation"].shape
                    mask_shape = o["scout"]["action_mask"].shape
                    role_shape = o["scout"]["role_id"].shape
                    obs_arr = np.empty((n_agents, 1) + obs_shape, dtype=np.int32)
                    mask_arr = np.empty((n_agents, 1) + mask_shape, dtype=np.int8)
                    role_arr = np.empty((n_agents, 1) + role_shape, dtype=np.int8)
                    for i, a in enumerate(AGENTS):
                        obs_arr[i, 0] = o[a]["observation"]
                        mask_arr[i, 0] = o[a]["action_mask"]
                        role_arr[i, 0] = o[a]["role_id"]
                    term_obs = {}
                    for i, a in enumerate(AGENTS):
                        term_obs[a] = {
                            "observation": obs_arr[i],
                            "action_mask": mask_arr[i],
                            "role_id": role_arr[i],
                        }
                    term_state = env.state()
                    o, _ = env.reset()
                remote.send((o, r, t, tr, inf, done, term_obs, term_state, env.state()))
            elif cmd == "reset":
                o, _ = env.reset(seed=data)
                remote.send((o, env.state()))
            elif cmd == "close":
                remote.close()
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        remote.send(e)
    finally:
        env.close()


class VectorEnv:
    def __init__(self, num_envs, config=None, base_seed=0):
        self.num_envs = num_envs
        self.base_seed = base_seed
        # Prototype env for metadata/inspectability (workers spawn their own)
        self.envs = [HeistEnv(config)]

        # Use multiprocessing context
        ctx = mp.get_context(
            "fork"
            if hasattr(mp, "get_context") and "fork" in mp.get_all_start_methods()
            else "spawn"
        )
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(num_envs)], strict=True
        )
        self.ps = [
            ctx.Process(target=_worker, args=(work_remote, remote, config))
            for work_remote, remote in zip(self.work_remotes, self.remotes, strict=True)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # inspect initial state_dim
        self.remotes[0].send(("reset", base_seed))
        _, initial_state = self.remotes[0].recv()
        self.state_dim = initial_state.shape[0]
        self.obs = None
        self.state = None

    # ------------------------------------------------------------------
    def _pack(self, obs_list):
        n_envs = len(obs_list)
        obs_shape = obs_list[0]["scout"]["observation"].shape
        mask_shape = obs_list[0]["scout"]["action_mask"].shape
        role_shape = obs_list[0]["scout"]["role_id"].shape

        n_agents = len(AGENTS)
        obs_array = np.empty((n_agents, n_envs) + obs_shape, dtype=np.int32)
        mask_array = np.empty((n_agents, n_envs) + mask_shape, dtype=np.int8)
        role_array = np.empty((n_agents, n_envs) + role_shape, dtype=np.int8)

        for i, a in enumerate(AGENTS):
            for j, o in enumerate(obs_list):
                obs_array[i, j] = o[a]["observation"]
                mask_array[i, j] = o[a]["action_mask"]
                role_array[i, j] = o[a]["role_id"]

        packed = {}
        for i, a in enumerate(AGENTS):
            packed[a] = {
                "observation": obs_array[i],
                "action_mask": mask_array[i],
                "role_id": role_array[i],
            }
        packed["_stacked"] = {
            "observation": obs_array,
            "action_mask": mask_array,
            "role_id": role_array,
        }
        return packed

    def reset(self, seed=None):
        base = seed if seed is not None else self.base_seed
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", base + i))

        obs_list, states = [], []
        for remote in self.remotes:
            o, s = remote.recv()
            if isinstance(o, Exception):
                raise o
            obs_list.append(o)
            states.append(s)

        self.obs = self._pack(obs_list)
        self.state = np.stack(states).astype(np.float32)
        self.state_dim = self.state.shape[1]
        return self.obs, self.state

    def step(self, actions):
        for i, remote in enumerate(self.remotes):
            acts = {a: int(actions[a][i]) for a in AGENTS}
            remote.send(("step", acts))

        rewards = {a: np.zeros(self.num_envs, dtype=np.float32) for a in AGENTS}
        terminations = {a: np.zeros(self.num_envs, dtype=bool) for a in AGENTS}
        truncations = {a: np.zeros(self.num_envs, dtype=bool) for a in AGENTS}
        next_obs_list = []
        next_states = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        infos = [{} for _ in range(self.num_envs)]

        for i, remote in enumerate(self.remotes):
            res = remote.recv()
            if isinstance(res, Exception):
                raise res
            o, r, t, tr, inf, done, term_obs, term_state, st = res
            infos[i] = inf
            for a in AGENTS:
                rewards[a][i] = r[a]
                terminations[a][i] = bool(t[a])
                truncations[a][i] = bool(tr[a])
            if done:
                infos[i]["terminal_observation"] = term_obs
                infos[i]["terminal_state"] = term_state
            next_obs_list.append(o)
            next_states[i] = st

        self.obs = self._pack(next_obs_list)
        self.state = next_states
        return self.obs, rewards, terminations, truncations, infos

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)
