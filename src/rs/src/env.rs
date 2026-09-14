use std::collections::HashSet;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::constants::*;
use crate::grid::{generate_procedural_map, RoomRect};
use crate::pathfinding::BfsPathfinder;
use crate::types::{manhattan, EnvConfig, GuardState};
use crate::vision::{
    calculate_fov, camera_exposure_sum, get_valid_moves, line_is_clear, pick_search_tile,
};

#[derive(Clone, Debug)]
pub struct StepInfo {
    pub win: bool,
    pub alarm: f32,
    pub steps: usize,
    pub scout_pois_tagged: usize,
    pub scout_interact_success: bool,
    pub hacker_hack_success: bool,
    pub muscle_neutralize_success: bool,
    pub muscle_guards_neutralized: usize,
    pub extractor_loot_success: bool,
    pub agents_at_extract: usize,
    pub agent_positions: [(i32, i32); N_AGENTS],
    pub terminal_pos: (i32, i32),
    pub loot_pos: (i32, i32),
    pub extract_pos: (i32, i32),
}

pub struct HeistEnv {
    pub config: EnvConfig,
    pub map_h: usize,
    pub map_w: usize,
    pub grid: Vec<i32>,
    pub explored_map: Vec<bool>,
    pub pathfinder: BfsPathfinder,
    pub rng: SmallRng,

    pub current_step: usize,
    pub alarm: f32,
    pub alarm_max: f32,
    pub time_bleed: f32,

    pub terminal_disabled: bool,
    pub loot_acquired: bool,
    pub extraction_triggered: bool,
    pub extraction_countdown: usize,
    pub hack_progress: i32,

    pub agent_positions: [(i32, i32); N_AGENTS],
    pub guard_positions: Vec<(i32, i32)>,
    pub neutralized: Vec<i32>,
    pub guard_states: Vec<GuardState>,
    pub guard_search_targets: Vec<(i32, i32)>,
    pub guard_search_turns: Vec<i32>,
    pub guard_contact: Vec<bool>,

    pub terminal_pos: (i32, i32),
    pub loot_pos: (i32, i32),
    pub extract_pos: (i32, i32),
    pub camera_positions: Vec<(i32, i32)>,
    pub door_positions: Vec<(i32, i32)>,
    pub room_rects: Vec<RoomRect>,

    pub tagged_pois: HashSet<(i32, i32)>,
    pub rewarded_guards: HashSet<usize>,
    pub prev_extract_dist: [i32; N_AGENTS],
}

impl HeistEnv {
    pub fn new(config: EnvConfig, seed: u64) -> Self {
        let (map_h, map_w) = config.map_size;
        let total_cells = map_h * map_w;
        Self {
            config: config.clone(),
            map_h,
            map_w,
            grid: vec![WALL; total_cells],
            explored_map: vec![false; total_cells],
            pathfinder: BfsPathfinder::new(total_cells),
            rng: SmallRng::seed_from_u64(seed),

            current_step: 0,
            alarm: 0.0,
            alarm_max: config.alarm_max,
            time_bleed: TOTAL_TIME_BLEED / (config.max_steps as f32),

            terminal_disabled: false,
            loot_acquired: false,
            extraction_triggered: false,
            extraction_countdown: (config.max_steps as f32 * EXTRACTION_COUNTDOWN_RATIO) as usize,
            hack_progress: 0,

            agent_positions: [(0, 0); N_AGENTS],
            guard_positions: Vec::new(),
            neutralized: Vec::new(),
            guard_states: Vec::new(),
            guard_search_targets: Vec::new(),
            guard_search_turns: Vec::new(),
            guard_contact: Vec::new(),

            terminal_pos: (0, 0),
            loot_pos: (0, 0),
            extract_pos: (0, 0),
            camera_positions: Vec::new(),
            door_positions: Vec::new(),
            room_rects: Vec::new(),

            tagged_pois: HashSet::new(),
            rewarded_guards: HashSet::new(),
            prev_extract_dist: [0; N_AGENTS],
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        if let Some(s) = seed {
            self.rng = SmallRng::seed_from_u64(s);
        }

        self.current_step = 0;
        self.alarm = 0.0;
        self.alarm_max = self.config.alarm_max;
        self.time_bleed = TOTAL_TIME_BLEED / (self.config.max_steps as f32);
        self.terminal_disabled = false;
        self.loot_acquired = false;
        self.extraction_triggered = false;
        self.extraction_countdown = (self.config.max_steps as f32 * EXTRACTION_COUNTDOWN_RATIO) as usize;
        self.hack_progress = 0;
        self.tagged_pois.clear();
        self.rewarded_guards.clear();

        let total_cells = self.map_h * self.map_w;
        self.explored_map.iter_mut().take(total_cells).for_each(|e| *e = false);

        let map_data = generate_procedural_map(
            &mut self.rng,
            self.map_h,
            self.map_w,
            self.config.num_rooms_range,
            self.config.camera_count,
            self.config.door_count,
        );

        self.grid = map_data.grid;
        self.terminal_pos = map_data.terminal_pos;
        self.loot_pos = map_data.loot_pos;
        self.extract_pos = map_data.extract_pos;
        self.camera_positions = map_data.camera_positions;
        self.door_positions = map_data.door_positions;
        self.room_rects = map_data.room_rects;

        // 1. Spawn agents clustered in a dedicated room
        let mut used = HashSet::new();
        self.agent_positions = self.spawn_agents(&mut used);

        // 2. Spawn guards at a safe distance
        let mut empty_tiles: Vec<(i32, i32)> = Vec::new();
        for r in 0..self.map_h {
            for c in 0..self.map_w {
                let pos = (r as i32, c as i32);
                if self.grid[r * self.map_w + c] == EMPTY && !used.contains(&pos) {
                    empty_tiles.push(pos);
                }
            }
        }

        let max_dim = self.map_h.max(self.map_w) as i32;
        let target_min_dist = (MIN_GUARD_SPAWN_DIST).min(2.max(max_dim / 3));

        self.guard_positions.clear();
        for dist in (1..=target_min_dist).rev() {
            let mut filtered: Vec<(i32, i32)> = empty_tiles
                .iter()
                .copied()
                .filter(|&p| self.agent_positions.iter().all(|&apos| manhattan(p, apos) >= dist))
                .collect();
            if filtered.len() >= self.config.guard_count {
                filtered.shuffle(&mut self.rng);
                self.guard_positions = filtered.into_iter().take(self.config.guard_count).collect();
                break;
            }
        }

        if self.guard_positions.len() < self.config.guard_count {
            empty_tiles.shuffle(&mut self.rng);
            self.guard_positions = empty_tiles.into_iter().take(self.config.guard_count).collect();
        }

        let num_guards = self.guard_positions.len();
        self.neutralized = vec![0; num_guards];
        self.guard_states = vec![GuardState::Patrol; num_guards];
        self.guard_search_targets = vec![(0, 0); num_guards];
        self.guard_search_turns = vec![0; num_guards];
        self.guard_contact = vec![false; num_guards];

        // Initial vision exploration
        calculate_fov(
            &self.grid,
            &mut self.explored_map,
            self.agent_positions[0].0,
            self.agent_positions[0].1,
            SCOUT_VISION_RADIUS,
            self.map_h,
            self.map_w,
        );

        for i in 1..N_AGENTS {
            calculate_fov(
                &self.grid,
                &mut self.explored_map,
                self.agent_positions[i].0,
                self.agent_positions[i].1,
                AGENT_VISION_RADIUS,
                self.map_h,
                self.map_w,
            );
        }

        for i in 0..N_AGENTS {
            self.prev_extract_dist[i] = manhattan(self.agent_positions[i], self.extract_pos);
        }
    }

    fn spawn_agents(&mut self, used: &mut HashSet<(i32, i32)>) -> [(i32, i32); N_AGENTS] {
        let mut spawn_tiles = Vec::new();

        if !self.room_rects.is_empty() {
            let mut shuffled_rooms = self.room_rects.clone();
            shuffled_rooms.shuffle(&mut self.rng);

            // Prefer room with no POIs
            for rm in &shuffled_rooms {
                let mut room_coords = HashSet::new();
                for r in rm.r..(rm.r + rm.h) {
                    for c in rm.c..(rm.c + rm.w) {
                        room_coords.insert((r as i32, c as i32));
                    }
                }

                let non_empty = [self.terminal_pos, self.loot_pos, self.extract_pos];
                let has_static_poi = non_empty.iter().any(|poi| room_coords.contains(poi))
                    || self.camera_positions.iter().any(|cam| room_coords.contains(cam));

                if !has_static_poi {
                    let mut room_empty: Vec<(i32, i32)> = room_coords
                        .into_iter()
                        .filter(|p| self.grid[(p.0 as usize) * self.map_w + (p.1 as usize)] == EMPTY && !used.contains(p))
                        .collect();
                    if room_empty.len() >= N_AGENTS {
                        room_empty.shuffle(&mut self.rng);
                        spawn_tiles = room_empty.into_iter().take(N_AGENTS).collect();
                        break;
                    }
                }
            }

            if spawn_tiles.len() < N_AGENTS {
                for rm in &shuffled_rooms {
                    let mut room_empty = Vec::new();
                    for r in rm.r..(rm.r + rm.h) {
                        for c in rm.c..(rm.c + rm.w) {
                            let p = (r as i32, c as i32);
                            if self.grid[r * self.map_w + c] == EMPTY && !used.contains(&p) {
                                room_empty.push(p);
                            }
                        }
                    }
                    if room_empty.len() >= N_AGENTS {
                        room_empty.shuffle(&mut self.rng);
                        spawn_tiles = room_empty.into_iter().take(N_AGENTS).collect();
                        break;
                    }
                }
            }
        }

        if spawn_tiles.len() < N_AGENTS {
            let mut empty_tiles = Vec::new();
            for r in 0..self.map_h {
                for c in 0..self.map_w {
                    let p = (r as i32, c as i32);
                    if self.grid[r * self.map_w + c] == EMPTY && !used.contains(&p) {
                        empty_tiles.push(p);
                    }
                }
            }
            if !empty_tiles.is_empty() {
                let center = empty_tiles[0];
                empty_tiles.sort_by_key(|p| manhattan(*p, center));
                spawn_tiles = empty_tiles.into_iter().take(N_AGENTS).collect();
            }
        }

        while spawn_tiles.len() < N_AGENTS {
            spawn_tiles.push((1, 1));
        }

        let mut res = [(0, 0); N_AGENTS];
        for i in 0..N_AGENTS {
            let p = spawn_tiles[i];
            used.insert(p);
            res[i] = p;
        }
        res
    }

    pub fn step(
        &mut self,
        actions: [i32; N_AGENTS],
        out_rewards: &mut [f32; N_AGENTS],
        out_terms: &mut [bool; N_AGENTS],
        out_truncs: &mut [bool; N_AGENTS],
    ) -> StepInfo {
        self.current_step += 1;
        out_rewards.fill(self.time_bleed);

        // 1. Agent Actions
        for i in 0..N_AGENTS {
            let act = actions[i];
            if act == WAIT {
                continue;
            } else if act == INTERACT {
                self.special_action(i, out_rewards);
            } else if act >= 0 && act < 4 {
                self.move_agent(i, act);
            }
        }

        // Hacking interruption penalty
        if self.hack_progress > 0
            && !self.terminal_disabled
            && manhattan(self.agent_positions[1], self.terminal_pos) > 1
        {
            self.hack_progress = 0;
            self.add_alarm(ALARM_HACK_TURN, out_rewards);
        }

        // 2. Guard Movement & Spotting
        self.move_guards();
        self.handle_guard_spotting(out_rewards);

        // 3. Camera Exposure
        if !self.terminal_disabled && !self.camera_positions.is_empty() {
            let exposure_sum = camera_exposure_sum(
                &self.grid,
                &self.camera_positions,
                &self.agent_positions,
                CAMERA_RANGE,
                self.map_h,
                self.map_w,
            );
            self.add_alarm(ALARM_CAMERA * exposure_sum, out_rewards);
        }

        // 4. Extraction Countdown
        if self.extraction_triggered {
            self.extraction_countdown = self.extraction_countdown.saturating_sub(1);
            if self.extraction_countdown == 0 {
                self.add_alarm(ALARM_EXTRACTION_TIMEOUT, out_rewards);
            }
        }

        // 5. Potential-Based Reward Shaping (Convergence)
        let active_guards_count = self
            .guard_positions
            .iter()
            .enumerate()
            .filter(|(gi, _)| self.neutralized[*gi] == 0)
            .count();
        let total_pois = (if self.terminal_pos != (0, 0) { 1 } else { 0 })
            + (if self.loot_pos != (0, 0) { 1 } else { 0 })
            + self.camera_positions.len();

        for i in 0..N_AGENTS {
            let ready = if self.loot_acquired || self.extraction_triggered {
                true
            } else {
                match i {
                    0 => self.tagged_pois.len() >= total_pois,
                    1 => self.terminal_disabled,
                    2 => active_guards_count == 0,
                    _ => false,
                }
            };

            if ready {
                let d_cur = manhattan(self.agent_positions[i], self.extract_pos);
                let d_prev = self.prev_extract_dist[i];
                out_rewards[i] += CONVERGE_BONUS * ((d_prev - d_cur) as f32);
                self.prev_extract_dist[i] = d_cur;
            }
        }

        // 6. Win / Loss States
        let win = self.loot_acquired
            && self.extraction_triggered
            && self.agent_positions[3] == self.extract_pos
            && self.agent_positions.iter().all(|&apos| manhattan(apos, self.extract_pos) <= WIN_CONVERGE_RADIUS);

        let lose = self.alarm >= self.alarm_max;

        if win {
            out_rewards.fill(REWARD_WIN);
        } else if lose {
            out_rewards.fill(REWARD_LOSE);
        }

        let term_flag = win || lose;
        let trunc_flag = self.current_step >= self.config.max_steps;

        out_terms.fill(term_flag);
        out_truncs.fill(trunc_flag);

        let agents_at_extract = self
            .agent_positions
            .iter()
            .filter(|&&p| manhattan(p, self.extract_pos) <= WIN_CONVERGE_RADIUS)
            .count();

        StepInfo {
            win,
            alarm: self.alarm,
            steps: self.current_step,
            scout_pois_tagged: self.tagged_pois.len(),
            scout_interact_success: !self.tagged_pois.is_empty(),
            hacker_hack_success: self.terminal_disabled,
            muscle_neutralize_success: !self.rewarded_guards.is_empty(),
            muscle_guards_neutralized: self.rewarded_guards.len(),
            extractor_loot_success: self.loot_acquired,
            agents_at_extract,
            agent_positions: self.agent_positions,
            terminal_pos: self.terminal_pos,
            loot_pos: self.loot_pos,
            extract_pos: self.extract_pos,
        }
    }

    fn move_agent(&mut self, agent_idx: usize, action: i32) {
        let (r, c) = self.agent_positions[agent_idx];
        let (dr, dc) = ACTION_DELTAS[action as usize];
        let nr = r + dr;
        let nc = c + dc;

        if nr >= 0 && nr < self.map_h as i32 && nc >= 0 && nc < self.map_w as i32 {
            let tile = self.grid[(nr as usize) * self.map_w + (nc as usize)];
            if tile != WALL && tile != DOOR {
                self.agent_positions[agent_idx] = (nr, nc);
                let radius = if agent_idx == 0 { SCOUT_VISION_RADIUS } else { AGENT_VISION_RADIUS };
                calculate_fov(&self.grid, &mut self.explored_map, nr, nc, radius, self.map_h, self.map_w);
            }
        }
    }

    fn special_action(&mut self, agent_idx: usize, rewards: &mut [f32; N_AGENTS]) {
        let pos = self.agent_positions[agent_idx];
        match agent_idx {
            0 => {
                // Scout
                let mut candidates = vec![self.terminal_pos, self.loot_pos, self.extract_pos];
                candidates.extend_from_slice(&self.camera_positions);
                candidates.extend_from_slice(&self.door_positions);

                for p in candidates {
                    if !self.tagged_pois.contains(&p) && manhattan(pos, p) <= SCOUT_TAG_DISTANCE {
                        self.tagged_pois.insert(p);
                        rewards[0] += REWARD_TAG;
                        return;
                    }
                }
            }
            1 => {
                // Hacker
                if !self.terminal_disabled && manhattan(pos, self.terminal_pos) <= 1 {
                    self.hack_progress += 1;
                    self.add_alarm(ALARM_HACK_TURN, rewards);
                    if self.hack_progress >= HACK_TURNS {
                        self.terminal_disabled = true;
                        rewards[1] += REWARD_TASK;
                    } else {
                        rewards[1] += REWARD_HACK_PROGRESS;
                    }
                    return;
                }

                // Door bypass
                for &(dr, dc) in &ACTION_DELTAS[..4] {
                    let nr = pos.0 + dr;
                    let nc = pos.1 + dc;
                    if nr >= 0 && nr < self.map_h as i32 && nc >= 0 && nc < self.map_w as i32 {
                        let idx = (nr as usize) * self.map_w + (nc as usize);
                        if self.grid[idx] == DOOR {
                            self.grid[idx] = EMPTY;
                            self.add_alarm(ALARM_BYPASS, rewards);
                            rewards[1] += REWARD_BYPASS;
                            return;
                        }
                    }
                }
            }
            2 => {
                // Muscle
                for gi in 0..self.guard_positions.len() {
                    if self.neutralized[gi] == 0 && manhattan(pos, self.guard_positions[gi]) <= 2 {
                        self.neutralized[gi] = 1;
                        self.add_alarm(ALARM_NEUTRALIZE, rewards);
                        if !self.rewarded_guards.contains(&gi) {
                            self.rewarded_guards.insert(gi);
                            rewards[2] += REWARD_TASK;
                        }
                        return;
                    }
                }
            }
            3 => {
                // Extractor
                if !self.loot_acquired && self.terminal_disabled && manhattan(pos, self.loot_pos) <= 1 {
                    self.loot_acquired = true;
                    self.extraction_triggered = true;
                    rewards[3] += REWARD_TASK;
                }
            }
            _ => {}
        }
    }

    fn move_guards(&mut self) {
        let converge = self.alarm >= CONVERGE_ALARM;
        if converge {
            self.pathfinder.compute_multi_target_distances(
                &self.grid,
                &self.agent_positions,
                self.map_h,
                self.map_w,
            );
        }

        for gi in 0..self.guard_positions.len() {
            if self.neutralized[gi] > 0 {
                continue;
            }

            let (gr, gc) = self.guard_positions[gi];

            // Line of Sight check
            let mut spotted = None;
            for &apos in &self.agent_positions {
                if manhattan((gr, gc), apos) <= GUARD_LOS_RANGE
                    && line_is_clear(&self.grid, gr, gc, apos.0, apos.1, self.map_h, self.map_w)
                {
                    spotted = Some(apos);
                    break;
                }
            }

            if let Some(target) = spotted {
                self.guard_states[gi] = GuardState::Search;
                self.guard_search_targets[gi] = target;
                self.guard_search_turns[gi] = SEARCH_TURNS;
            }

            if converge {
                self.guard_states[gi] = GuardState::Converge;
            } else if self.guard_states[gi] == GuardState::Search && self.guard_search_turns[gi] <= 0 {
                self.guard_states[gi] = GuardState::Patrol;
            }

            let valid_moves = get_valid_moves(&self.grid, gr, gc, self.map_h, self.map_w);
            let valid_list: Vec<(i32, i32)> = valid_moves.iter().copied().filter(|&p| p != (-1, -1)).collect();
            if valid_list.is_empty() {
                continue;
            }

            match self.guard_states[gi] {
                GuardState::Converge => {
                    let next = self.pathfinder.next_step_from_distance_map(gr, gc, self.map_h, self.map_w);
                    if next != (-1, -1) {
                        self.guard_positions[gi] = next;
                    } else {
                        let rand_idx = self.rng.gen_range(0..valid_list.len());
                        self.guard_positions[gi] = valid_list[rand_idx];
                    }
                }
                GuardState::Search => {
                    let mut target = self.guard_search_targets[gi];
                    if (gr, gc) == target {
                        target = pick_search_tile(
                            &self.grid,
                            target.0,
                            target.1,
                            SEARCH_RADIUS,
                            self.rng.gen::<f64>(),
                            self.map_h,
                            self.map_w,
                        );
                        self.guard_search_targets[gi] = target;
                    }

                    let next = self.pathfinder.bfs_next_step(
                        &self.grid,
                        gr,
                        gc,
                        target.0,
                        target.1,
                        self.map_h,
                        self.map_w,
                    );

                    if next != (-1, -1) {
                        self.guard_positions[gi] = next;
                    } else {
                        let rand_idx = self.rng.gen_range(0..valid_list.len());
                        self.guard_positions[gi] = valid_list[rand_idx];
                    }
                    self.guard_search_turns[gi] -= 1;
                }
                GuardState::Patrol => {
                    let rand_idx = self.rng.gen_range(0..valid_list.len());
                    self.guard_positions[gi] = valid_list[rand_idx];
                }
            }
        }
    }

    fn handle_guard_spotting(&mut self, rewards: &mut [f32; N_AGENTS]) {
        for gi in 0..self.guard_positions.len() {
            if self.neutralized[gi] > 0 {
                if gi < self.guard_contact.len() {
                    self.guard_contact[gi] = false;
                }
                continue;
            }

            let gpos = self.guard_positions[gi];
            let in_contact = self.agent_positions.iter().any(|&apos| manhattan(gpos, apos) <= CATCH_DISTANCE);

            if in_contact {
                if gi < self.guard_contact.len() && !self.guard_contact[gi] {
                    self.add_alarm(ALARM_GUARD_SPOT, rewards);
                    self.guard_contact[gi] = true;
                } else {
                    self.add_alarm(ALARM_GUARD_CONTINUOUS, rewards);
                }
            } else if gi < self.guard_contact.len() {
                self.guard_contact[gi] = false;
            }
        }
    }

    #[allow(dead_code)]
    fn check_caught(&self) -> bool {
        for gi in 0..self.guard_positions.len() {
            if self.neutralized[gi] > 0 {
                continue;
            }
            let gpos = self.guard_positions[gi];
            for &apos in &self.agent_positions {
                if manhattan(gpos, apos) <= CATCH_DISTANCE {
                    return true;
                }
            }
        }
        false
    }

    fn add_alarm(&mut self, amount: f32, rewards: &mut [f32; N_AGENTS]) {
        let max_dim = self.map_h.max(self.map_w) as f32;
        let scale = 1.0f32.min(17.0 / max_dim);
        let scaled_amount = amount * scale;
        let prev = self.alarm;
        self.alarm = (self.alarm + scaled_amount).min(self.alarm_max);
        let diff = self.alarm - prev;
        if diff > 0.0 {
            for r in rewards.iter_mut() {
                *r -= 0.01 * diff;
            }
        }
    }

    pub fn compute_action_masks(&self, out_masks: &mut [i8]) {
        // out_masks shape: [N_AGENTS, 6]
        for ag_idx in 0..N_AGENTS {
            let offset = ag_idx * ACTION_SPACE_SIZE;
            let mask = &mut out_masks[offset..offset + ACTION_SPACE_SIZE];
            mask.fill(1);
            mask[INTERACT as usize] = 0;

            let (r, c) = self.agent_positions[ag_idx];

            for a in 0..4 {
                let (dr, dc) = ACTION_DELTAS[a];
                let nr = r + dr;
                let nc = c + dc;
                if !(nr >= 0 && nr < self.map_h as i32 && nc >= 0 && nc < self.map_w as i32) {
                    mask[a] = 0;
                } else {
                    let tile = self.grid[(nr as usize) * self.map_w + (nc as usize)];
                    if tile == WALL || tile == DOOR {
                        mask[a] = 0;
                    }
                }
            }

            match ag_idx {
                0 => {
                    // Scout interact
                    let mut candidates = vec![self.terminal_pos, self.loot_pos, self.extract_pos];
                    candidates.extend_from_slice(&self.camera_positions);
                    candidates.extend_from_slice(&self.door_positions);
                    for p in candidates {
                        if !self.tagged_pois.contains(&p) && manhattan((r, c), p) <= SCOUT_TAG_DISTANCE {
                            mask[INTERACT as usize] = 1;
                            break;
                        }
                    }
                }
                1 => {
                    // Hacker interact
                    if !self.terminal_disabled && manhattan((r, c), self.terminal_pos) <= 1 {
                        mask[INTERACT as usize] = 1;
                    }
                    for &(dr, dc) in &ACTION_DELTAS[..4] {
                        let nr = r + dr;
                        let nc = c + dc;
                        if nr >= 0 && nr < self.map_h as i32 && nc >= 0 && nc < self.map_w as i32 {
                            if self.grid[(nr as usize) * self.map_w + (nc as usize)] == DOOR {
                                mask[INTERACT as usize] = 1;
                                break;
                            }
                        }
                    }
                }
                2 => {
                    // Muscle interact
                    if self.guard_positions.iter().enumerate().any(|(gi, &gp)| self.neutralized[gi] == 0 && manhattan((r, c), gp) <= 2) {
                        mask[INTERACT as usize] = 1;
                    }
                }
                3 => {
                    // Extractor interact
                    if !self.loot_acquired && self.terminal_disabled && manhattan((r, c), self.loot_pos) <= 1 {
                        mask[INTERACT as usize] = 1;
                    }
                }
                _ => {}
            }
        }
    }

    pub fn compute_observations_and_goals(
        &self,
        out_obs: &mut [i32],  // Shape: [N_AGENTS, 7, 7]
        out_goals: &mut [f32], // Shape: [N_AGENTS, 2]
    ) {
        let pad = OBSERVATION_PAD as i32;

        // Build composite grid
        let mut composite = self.grid.clone();
        for (gi, &gp) in self.guard_positions.iter().enumerate() {
            if self.neutralized[gi] == 0 {
                composite[(gp.0 as usize) * self.map_w + (gp.1 as usize)] = GUARD;
            }
        }
        for &apos in &self.agent_positions {
            composite[(apos.0 as usize) * self.map_w + (apos.1 as usize)] = ALLY;
        }

        for ag_idx in 0..N_AGENTS {
            let (r, c) = self.agent_positions[ag_idx];
            let obs_offset = ag_idx * OBSERVATION_DIM * OBSERVATION_DIM;
            let obs_slice = &mut out_obs[obs_offset..obs_offset + OBSERVATION_DIM * OBSERVATION_DIM];

            for (wr_i, dr) in (-pad..=pad).enumerate() {
                for (wc_i, dc) in (-pad..=pad).enumerate() {
                    let map_r = r + dr;
                    let map_c = c + dc;
                    let out_idx = wr_i * OBSERVATION_DIM + wc_i;

                    if map_r < 0 || map_r >= self.map_h as i32 || map_c < 0 || map_c >= self.map_w as i32 {
                        obs_slice[out_idx] = WALL;
                    } else {
                        let idx = (map_r as usize) * self.map_w + (map_c as usize);
                        if !self.explored_map[idx] {
                            obs_slice[out_idx] = FOG;
                        } else if dr == 0 && dc == 0 {
                            obs_slice[out_idx] = self.grid[idx]; // Self transparent
                        } else {
                            obs_slice[out_idx] = composite[idx];
                        }
                    }
                }
            }

            // HUD Waypoint perimeter compass projection
            let mut target_pois = Vec::new();
            if ag_idx == 1 && !self.terminal_disabled {
                if self.tagged_pois.contains(&self.terminal_pos) {
                    target_pois.push(self.terminal_pos);
                }
            } else if ag_idx == 3 {
                if !self.loot_acquired && self.terminal_disabled {
                    if self.tagged_pois.contains(&self.loot_pos) {
                        target_pois.push(self.loot_pos);
                    }
                } else if self.loot_acquired {
                    target_pois.push(self.extract_pos);
                }
            } else if self.loot_acquired {
                target_pois.push(self.extract_pos);
            }

            for poi in target_pois {
                let dr = poi.0 - r;
                let dc = poi.1 - c;
                if dr.abs() > pad || dc.abs() > pad {
                    let clamped_r = dr.clamp(-pad, pad);
                    let clamped_c = dc.clamp(-pad, pad);
                    let wr = (pad + clamped_r) as usize;
                    let wc = (pad + clamped_c) as usize;
                    obs_slice[wr * OBSERVATION_DIM + wc] = WAYPOINT;
                }
            }

            // 3. 2D Tactical Role Mission Orientation Unit Vector
            let tgt = match ag_idx {
                1 => {
                    // Hacker
                    if !self.terminal_disabled {
                        self.terminal_pos
                    } else {
                        self.extract_pos
                    }
                }
                3 => {
                    // Extractor
                    if !self.loot_acquired {
                        self.loot_pos
                    } else {
                        self.extract_pos
                    }
                }
                2 => {
                    // Muscle
                    let mut active_guards = Vec::new();
                    for (gi, &gp) in self.guard_positions.iter().enumerate() {
                        if self.neutralized[gi] == 0 {
                            active_guards.push(gp);
                        }
                    }
                    if !active_guards.is_empty() {
                        active_guards.sort_by_key(|gp| manhattan((r, c), *gp));
                        active_guards[0]
                    } else {
                        self.extract_pos
                    }
                }
                _ => {
                    // Scout (0)
                    let mut untagged = Vec::new();
                    if self.terminal_pos != (0, 0) && !self.tagged_pois.contains(&self.terminal_pos) {
                        untagged.push(self.terminal_pos);
                    }
                    if self.loot_pos != (0, 0) && !self.tagged_pois.contains(&self.loot_pos) {
                        untagged.push(self.loot_pos);
                    }
                    for &cam_pos in &self.camera_positions {
                        if !self.tagged_pois.contains(&cam_pos) {
                            untagged.push(cam_pos);
                        }
                    }
                    if !untagged.is_empty() {
                        untagged.sort_by_key(|p| manhattan((r, c), *p));
                        untagged[0]
                    } else {
                        self.extract_pos
                    }
                }
            };

            let dr = (tgt.0 - r) as f32;
            let dc = (tgt.1 - c) as f32;
            let norm = (dr * dr + dc * dc).sqrt() + 1e-5;

            let goal_offset = ag_idx * 2;
            out_goals[goal_offset] = (dr / norm).clamp(-1.0, 1.0);
            out_goals[goal_offset + 1] = (dc / norm).clamp(-1.0, 1.0);
        }
    }

    pub fn write_state(&self, out_state: &mut [f32]) {
        out_state.fill(0.0);
        out_state[0] = self.current_step as f32;
        out_state[1] = self.alarm;
        out_state[2] = if self.terminal_disabled { 1.0 } else { 0.0 };
        out_state[3] = if self.loot_acquired { 1.0 } else { 0.0 };
        out_state[4] = if self.extraction_triggered { 1.0 } else { 0.0 };
        out_state[5] = self.extraction_countdown as f32;

        let mut idx = 6;
        let max_h = MAX_MAP_H;
        let max_w = MAX_MAP_W;
        let grid_len = max_h * max_w;

        for r in 0..max_h {
            for c in 0..max_w {
                if r < self.map_h && c < self.map_w {
                    out_state[idx + r * max_w + c] = self.grid[r * self.map_w + c] as f32;
                } else {
                    out_state[idx + r * max_w + c] = WALL as f32;
                }
            }
        }
        idx += grid_len;

        for i in 0..N_AGENTS {
            out_state[idx] = self.agent_positions[i].0 as f32;
            out_state[idx + 1] = self.agent_positions[i].1 as f32;
            idx += 2;
        }

        for i in 0..12 {
            if i < self.guard_positions.len() {
                out_state[idx] = self.guard_positions[i].0 as f32;
                out_state[idx + 1] = self.guard_positions[i].1 as f32;
            } else {
                out_state[idx] = -1.0;
                out_state[idx + 1] = -1.0;
            }
            idx += 2;
        }

        for i in 0..12 {
            if i < self.neutralized.len() {
                out_state[idx] = self.neutralized[i] as f32;
            } else {
                out_state[idx] = 0.0;
            }
            idx += 1;
        }
    }
}
