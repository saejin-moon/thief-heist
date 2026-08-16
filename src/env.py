"""
HEIST: Hierarchical Environment for Interdependent Sequential Tasks.
Clean Version.
"""

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import ParallelEnv

from constants import *
from map_gen import generate_procedural_map
from vision import (
    bfs_next_step,
    calculate_fov,
    camera_exposure,
    distance_to_nearest_target,
    get_valid_moves,
    line_is_clear,
    next_step_from_distance,
    pick_search_tile,
)


def manhattan(a, b):
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))


class HeistEnv(ParallelEnv):
    def __init__(self, config=None):
        self.config = {
            "map_size": MAP_SIZE,
            "num_rooms_range": (8, 12),
            "guard_count": 4,
            "camera_count": 3,
            "door_count": 4,
            "max_steps": 300,
            "spawn_mode": "role",
        }
        if config:
            self.config.update(config)

        self.agents = AGENTS[:]
        self.possible_agents = AGENTS[:]
        self.map_h, self.map_w = self.config["map_size"]
        self.grid = np.zeros((self.map_h, self.map_w), dtype=np.int32)

        self.action_spaces = {
            a: Discrete(ACTION_SPACE_SIZE) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: Dict(
                {
                    "observation": Box(
                        low=FOG, high=255, shape=OBSERVATION_SIZE, dtype=np.int32
                    ),
                    "action_mask": Box(
                        low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8
                    ),
                    "role_id": Box(low=0, high=1, shape=(N_AGENTS,), dtype=np.int8),
                }
            )
            for a in self.possible_agents
        }

        self.rng = np.random.default_rng()

        # Pre-allocated arrays for O(1) BFS in guards logic
        max_cells = self.map_h * self.map_w
        self._distance_map = np.full((self.map_h, self.map_w), -1, dtype=np.int32)
        self._bfs_queue = np.empty(max_cells, dtype=np.int32)
        self._bfs_previous = np.full(max_cells, -1, dtype=np.int32)
        self._bfs_reset = np.empty(max_cells, dtype=np.int32)

        self._pad = OBSERVATION_SIZE[0] // 2
        self._pad_grid_cache = np.full(
            (self.map_h + 2 * self._pad, self.map_w + 2 * self._pad),
            WALL,
            dtype=np.int32,
        )
        self._pad_explored_cache = np.full(
            (self.map_h + 2 * self._pad, self.map_w + 2 * self._pad), False, dtype=bool
        )
        self._state_buffer = np.zeros(
            6 + (MAP_SIZE[0] * MAP_SIZE[1]) + (N_AGENTS * 2) + (12 * 2) + 12,
            dtype=np.float32,
        )

    def reset(self, seed=None, _options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.alarm = 0.0
        self.terminal_disabled = False
        self.loot_acquired = False
        self.extraction_triggered = False
        self.extraction_countdown = EXTRACTION_COUNTDOWN
        self.hack_progress = 0
        self.tagged_pois = set()
        self._prev_extract_dist = {}
        self.agents = self.possible_agents[:]
        self.explored_map = np.zeros((self.map_h, self.map_w), dtype=bool)
        self._rewarded_breaches = set()
        self._rewarded_guards = set()

        map_data = generate_procedural_map(
            self.rng,
            map_size=(self.map_h, self.map_w),
            num_rooms_range=self.config["num_rooms_range"],
            camera_count=self.config["camera_count"],
            door_count=self.config["door_count"],
        )
        self.grid = map_data["grid"]
        self.terminal_pos = map_data["terminal_pos"]
        self.loot_pos = map_data["loot_pos"]
        self.extract_pos = map_data["extract_pos"]
        self.camera_positions = map_data["camera_positions"]
        self.door_positions = map_data["door_positions"]

        empty = [tuple(int(x) for x in c) for c in np.argwhere(self.grid == EMPTY)]
        self.rng.shuffle(empty)

        self.guard_positions = [
            empty.pop() for _ in range(min(self.config["guard_count"], len(empty)))
        ]
        self.neutralized = np.zeros(len(self.guard_positions), dtype=np.int32)
        self.guard_states = ["patrol"] * len(self.guard_positions)
        self._guard_search_target = [None] * len(self.guard_positions)
        self._guard_search_turns = [0] * len(self.guard_positions)

        self.agent_positions = self._spawn_agents(set(self.guard_positions), empty)

        self._refresh_scout_fov()
        for a in self.agents:
            calculate_fov(
                self.grid,
                self.explored_map,
                self.agent_positions[a][0],
                self.agent_positions[a][1],
                AGENT_VISION_RADIUS,
                self.map_h,
                self.map_w,
                WALL,
            )

        return self._get_all_obs(), {a: {} for a in self.agents}

    def _spawn_agents(self, used, empty):
        pos = {}
        if self.config["spawn_mode"] == "role":
            targets = [
                ("scout", self.terminal_pos),
                ("hacker", self.terminal_pos),
                ("muscle", self.loot_pos),
                ("extractor", self.loot_pos),
            ]
            for agent, target in targets:
                best, best_d = None, 1e9
                for r in range(self.map_h):
                    for c in range(self.map_w):
                        if self.grid[r, c] == EMPTY and (r, c) not in used:
                            d = manhattan((r, c), target)
                            if d < best_d:
                                best_d, best = d, (r, c)
                used.add(best)
                pos[agent] = best
        else:
            for a in self.possible_agents:
                p = empty.pop()
                used.add(p)
                pos[a] = p
        return pos

    def step(self, actions):
        self.current_step += 1
        rewards = {a: REWARD_TIME_BLEED for a in self.agents}

        # 1. Agent Actions
        for agent in list(self.agents):
            action = actions[agent]
            if action == WAIT:
                continue
            elif action == INTERACT:
                self._special_action(agent, rewards)
            elif action == BREACH and agent == "muscle":
                self._muscle_breach(rewards)
            else:
                self._move_agent(agent, action)

        # Hacking interruption penalty
        if (
            self.hack_progress > 0
            and not self.terminal_disabled
            and manhattan(self.agent_positions["hacker"], self.terminal_pos) > 1
        ):
            self.hack_progress = 0
            self._add_alarm(ALARM_HACK_TURN, rewards)

        # 2. Guard Movement & Spotting
        self._move_guards()
        if self._check_caught():
            self._add_alarm(ALARM_GUARD_SPOT, rewards)

        # 3. Camera Exposure
        if not self.terminal_disabled and self.camera_positions:
            exposure = camera_exposure(
                self.grid,
                self.camera_positions,
                list(self.agent_positions.values()),
                WALL,
                DOOR,
                CAMERA_RANGE,
            )
            self._add_alarm(ALARM_CAMERA * exposure.sum(), rewards)

        # 4. Extraction Countdown
        if self.extraction_triggered:
            self.extraction_countdown -= 1
            if self.extraction_countdown == 0:
                self._add_alarm(ALARM_EXTRACTION_TIMEOUT, rewards)

        # 5. Potential-Based Reward Shaping (Convergence)
        if self.loot_acquired or self.extraction_triggered:
            for a in self.agents:
                d_cur = manhattan(self.agent_positions[a], self.extract_pos)
                d_prev = self._prev_extract_dist.get(a, d_cur)
                rewards[a] += CONVERGE_BONUS * (d_prev - d_cur)
                if d_cur <= CONVERGE_RADIUS:
                    rewards[a] += 0.05
                self._prev_extract_dist[a] = d_cur

        # 6. Win / Loss States
        win = (
            self.loot_acquired
            and self.extraction_triggered
            and self.agent_positions["extractor"] == self.extract_pos
            and all(
                manhattan(self.agent_positions[a], self.extract_pos)
                <= WIN_CONVERGE_RADIUS
                for a in self.agents
            )
        )
        lose = self.alarm >= ALARM_MAX

        if win:
            rewards = {a: REWARD_WIN for a in self.agents}
        elif lose:
            rewards = {a: REWARD_LOSE for a in self.agents}

        terms = {a: bool(win or lose) for a in self.agents}
        truncs = {
            a: bool(self.current_step >= self.config["max_steps"]) for a in self.agents
        }
        infos = {a: {"win": bool(win), "alarm": self.alarm} for a in self.agents}

        observations = self._get_all_obs()
        if any(terms.values()) or any(truncs.values()):
            self.agents = []
        return observations, rewards, terms, truncs, infos

    def _move_agent(self, agent, action):
        r, c = self.agent_positions[agent]
        dr, dc = ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        if (
            0 <= nr < self.map_h
            and 0 <= nc < self.map_w
            and self.grid[nr, nc] not in (WALL, DOOR)
        ):
            self.agent_positions[agent] = (nr, nc)
            calculate_fov(
                self.grid,
                self.explored_map,
                nr,
                nc,
                AGENT_VISION_RADIUS,
                self.map_h,
                self.map_w,
                WALL,
            )
            if agent == "scout":
                self._refresh_scout_fov()

    def _refresh_scout_fov(self):
        calculate_fov(
            self.grid,
            self.explored_map,
            self.agent_positions["scout"][0],
            self.agent_positions["scout"][1],
            SCOUT_VISION_RADIUS,
            self.map_h,
            self.map_w,
            WALL,
        )

    def _special_action(self, agent, rewards):
        pos = self.agent_positions[agent]
        if agent == "scout":
            for p in (
                [self.terminal_pos, self.loot_pos, self.extract_pos]
                + self.camera_positions
                + self.door_positions
            ):
                if p not in self.tagged_pois and manhattan(pos, p) <= 3:
                    self.tagged_pois.add(p)
                    rewards["scout"] += REWARD_TAG
                    return
        elif agent == "hacker":
            if (
                not self.terminal_disabled
                and self.terminal_pos in self.tagged_pois
                and manhattan(pos, self.terminal_pos) <= 1
            ):
                self.hack_progress += 1
                self._add_alarm(ALARM_HACK_TURN, rewards)
                if self.hack_progress >= HACK_TURNS:
                    self.terminal_disabled = True
                    rewards["hacker"] += REWARD_TASK
                else:
                    rewards["hacker"] += 0.5
                return
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = pos[0] + dr, pos[1] + dc
                if (
                    0 <= nr < self.map_h
                    and 0 <= nc < self.map_w
                    and self.grid[nr, nc] == DOOR
                ):
                    self.grid[nr, nc] = EMPTY
                    self._add_alarm(ALARM_BYPASS, rewards)
                    rewards["hacker"] += 0.2
                    return
        elif agent == "muscle":
            for gi, gpos in enumerate(self.guard_positions):
                if self.neutralized[gi] == 0 and manhattan(pos, gpos) <= 2:
                    self.neutralized[gi] = NEUTRALIZE_TURNS
                    self._add_alarm(
                        ALARM_NEUTRALIZE, rewards
                    )  # Instant alarm, no delayed queue bloat
                    if gi not in self._rewarded_guards:
                        self._rewarded_guards.add(gi)
                        rewards["muscle"] += REWARD_TASK
                    return
        elif agent == "extractor":
            if (
                not self.loot_acquired
                and self.terminal_disabled
                and manhattan(pos, self.loot_pos) <= 1
            ):
                self.loot_acquired = True
                rewards["extractor"] += REWARD_TASK
            elif self.loot_acquired and not self.extraction_triggered:
                self.extraction_triggered = True
                rewards["extractor"] += 0.5

    def _muscle_breach(self, rewards):
        pos = self.agent_positions["muscle"]
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = pos[0] + dr, pos[1] + dc
            if (
                0 <= nr < self.map_h
                and 0 <= nc < self.map_w
                and self.grid[nr, nc] == WALL
            ):
                self.grid[nr, nc] = EMPTY
                self._add_alarm(ALARM_BREACH, rewards)
                if (nr, nc) not in self._rewarded_breaches:
                    self._rewarded_breaches.add((nr, nc))
                    rewards["muscle"] += REWARD_TASK
                # Alerts nearby guards
                for gi, gpos in enumerate(self.guard_positions):
                    if (
                        self.neutralized[gi] == 0
                        and manhattan((nr, nc), gpos) <= BREACH_RADIUS
                    ):
                        self.guard_states[gi] = "search"
                        self._guard_search_target[gi] = (nr, nc)
                        self._guard_search_turns[gi] = SEARCH_TURNS
                return

    def _move_guards(self):
        converge = self.alarm >= CONVERGE_ALARM
        converge_dist = None
        if converge:
            targets = np.asarray(list(self.agent_positions.values()), dtype=np.int32)
            converge_dist = distance_to_nearest_target(
                self.grid, targets, WALL, DOOR, self._distance_map, self._bfs_queue
            )

        for gi, (gr, gc) in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                self.neutralized[gi] -= 1
                continue

            # LOS Check
            spotted = None
            for apos in self.agent_positions.values():
                if manhattan((gr, gc), apos) <= GUARD_LOS_RANGE and line_is_clear(
                    self.grid, gr, gc, apos[0], apos[1], WALL, DOOR
                ):
                    spotted = apos
                    break

            if spotted:
                self.guard_states[gi] = "search"
                self._guard_search_target[gi] = spotted
                self._guard_search_turns[gi] = SEARCH_TURNS

            if converge:
                self.guard_states[gi] = "converge"
            elif (
                self.guard_states[gi] == "search" and self._guard_search_turns[gi] <= 0
            ):
                self.guard_states[gi] = "patrol"

            valid = [
                (int(m[0]), int(m[1]))
                for m in get_valid_moves(self.grid, gr, gc, WALL, DOOR)
            ]
            if not valid:
                continue

            if self.guard_states[gi] == "converge":
                nr, nc = next_step_from_distance(converge_dist, gr, gc)
                if nr >= 0:
                    self.guard_positions[gi] = (nr, nc)
                else:
                    self.guard_positions[gi] = valid[int(self.rng.integers(len(valid)))]
            elif self.guard_states[gi] == "search":
                target = self._guard_search_target[gi]
                if (gr, gc) == target:
                    tr, tc = pick_search_tile(
                        self.grid,
                        target[0],
                        target[1],
                        SEARCH_RADIUS,
                        WALL,
                        DOOR,
                        float(self.rng.random()),
                    )
                    self._guard_search_target[gi] = target = (int(tr), int(tc))
                nr, nc = bfs_next_step(
                    self.grid,
                    gr,
                    gc,
                    target[0],
                    target[1],
                    WALL,
                    DOOR,
                    self._bfs_queue,
                    self._bfs_previous,
                    self._bfs_reset,
                )
                if nr >= 0:
                    self.guard_positions[gi] = (nr, nc)
                else:
                    self.guard_positions[gi] = valid[int(self.rng.integers(len(valid)))]
                self._guard_search_turns[gi] -= 1
            else:
                self.guard_positions[gi] = valid[int(self.rng.integers(len(valid)))]

    def _check_caught(self):
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                continue
            for apos in self.agent_positions.values():
                if manhattan(gpos, apos) <= CATCH_DISTANCE:
                    return True
        return False

    def _add_alarm(self, amount, rewards):
        prev = self.alarm
        self.alarm = min(self.alarm + amount, ALARM_MAX)
        if self.alarm - prev > 0:
            for a in self.agents:
                rewards[a] -= 0.01 * (self.alarm - prev)

    def _action_mask(self, agent):
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.int8)
        mask[INTERACT] = 0
        mask[BREACH] = 0
        pos = self.agent_positions[agent]

        for a in range(4):
            nr, nc = pos[0] + ACTION_DELTAS[a][0], pos[1] + ACTION_DELTAS[a][1]
            if not (0 <= nr < self.map_h and 0 <= nc < self.map_w) or self.grid[
                nr, nc
            ] in (WALL, DOOR):
                mask[a] = 0

        if agent == "scout":
            for p in (
                [self.terminal_pos, self.loot_pos, self.extract_pos]
                + self.camera_positions
                + self.door_positions
            ):
                if manhattan(pos, p) <= 1:
                    mask[INTERACT] = 1
                    break
        elif agent == "hacker":
            if (
                not self.terminal_disabled
                and self.terminal_pos in self.tagged_pois
                and manhattan(pos, self.terminal_pos) <= 1
            ):
                mask[INTERACT] = 1
            for dr, dc in ACTION_DELTAS.values():
                if (
                    0 <= pos[0] + dr < self.map_h
                    and 0 <= pos[1] + dc < self.map_w
                    and self.grid[pos[0] + dr, pos[1] + dc] == DOOR
                ):
                    mask[INTERACT] = 1
                    break
        elif agent == "muscle":
            if any(
                self.neutralized[gi] == 0 and manhattan(pos, gpos) <= 2
                for gi, gpos in enumerate(self.guard_positions)
            ):
                mask[INTERACT] = 1
            for dr, dc in ACTION_DELTAS.values():
                if (
                    0 <= pos[0] + dr < self.map_h
                    and 0 <= pos[1] + dc < self.map_w
                    and self.grid[pos[0] + dr, pos[1] + dc] == WALL
                ):
                    mask[BREACH] = 1
                    break
        elif agent == "extractor":  # noqa: SIM102
            if (
                not self.loot_acquired
                and self.terminal_disabled
                and manhattan(pos, self.loot_pos) <= 1
            ) or (self.loot_acquired and not self.extraction_triggered):
                mask[INTERACT] = 1
        return mask

    def _get_all_obs(self):
        composite = self.grid.copy()
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] == 0:
                composite[gpos] = GUARD
        for opos in self.agent_positions.values():
            composite[opos] = ALLY

        pad = self._pad
        self._pad_grid_cache[pad:-pad, pad:-pad] = composite
        self._pad_explored_cache[pad:-pad, pad:-pad] = self.explored_map

        obs_dict = {}
        for agent in self.agents:
            r, c = self.agent_positions[agent]
            vr, vc = r + pad, c + pad
            obs = self._pad_grid_cache[
                vr - pad : vr + pad + 1, vc - pad : vc + pad + 1
            ].copy()
            obs[pad, pad] = self.grid[r, c]  # Make self transparent
            explored = self._pad_explored_cache[
                vr - pad : vr + pad + 1, vc - pad : vc + pad + 1
            ]

            for poi in self.tagged_pois:
                if (
                    (poi == self.terminal_pos and not self.terminal_disabled)
                    or (
                        poi == self.loot_pos
                        and self.terminal_disabled
                        and not self.loot_acquired
                    )
                    or (
                        poi == self.extract_pos
                        and (self.loot_acquired or self.extraction_triggered)
                    )
                ):
                    dr, dc = poi[0] - r, poi[1] - c
                    if abs(dr) > pad or abs(dc) > pad:
                        obs[
                            pad + int(np.clip(dr, -pad, pad)),
                            pad + int(np.clip(dc, -pad, pad)),
                        ] = WAYPOINT

            obs_dict[agent] = {
                "observation": np.where(explored, obs, FOG),
                "action_mask": self._action_mask(agent),
                "role_id": ROLE_ONEHOT_ARRAYS[agent],
            }
        return obs_dict

    def state(self):
        self._state_buffer.fill(0)
        self._state_buffer[0] = self.current_step
        self._state_buffer[1] = self.alarm
        self._state_buffer[2] = int(self.terminal_disabled)
        self._state_buffer[3] = int(self.loot_acquired)
        self._state_buffer[4] = int(self.extraction_triggered)
        self._state_buffer[5] = self.extraction_countdown
        idx = 6
        max_h, max_w = MAP_SIZE
        grid_len = max_h * max_w
        padded_grid = np.full((max_h, max_w), WALL, dtype=np.int32)
        padded_grid[:self.map_h, :self.map_w] = self.grid
        self._state_buffer[idx : idx + grid_len] = padded_grid.ravel()
        idx += grid_len
        for a in self.possible_agents:
            self._state_buffer[idx : idx + 2] = self.agent_positions[a]
            idx += 2
        for i in range(12):
            if i < len(self.guard_positions):
                self._state_buffer[idx : idx + 2] = self.guard_positions[i]
            else:
                self._state_buffer[idx : idx + 2] = (-1, -1)
            idx += 2
        for i in range(12):
            if i < len(self.neutralized):
                self._state_buffer[idx] = self.neutralized[i]
            else:
                self._state_buffer[idx] = 0
            idx += 1
        return self._state_buffer
