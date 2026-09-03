use std::collections::HashSet;
use rand::seq::SliceRandom;
use rand::Rng;
use crate::constants::{CAMERA, DOOR, EMPTY, EXTRACT, LOOT, TERMINAL, WALL};

#[derive(Clone, Debug)]
pub struct RoomRect {
    pub r: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

#[derive(Clone, Debug)]
pub struct MapData {
    pub grid: Vec<i32>,
    pub terminal_pos: (i32, i32),
    pub loot_pos: (i32, i32),
    pub extract_pos: (i32, i32),
    pub camera_positions: Vec<(i32, i32)>,
    pub door_positions: Vec<(i32, i32)>,
    pub room_rects: Vec<RoomRect>,
}

pub fn generate_procedural_map<R: Rng + ?Sized>(
    rng: &mut R,
    map_h: usize,
    map_w: usize,
    num_rooms_range: (usize, usize),
    camera_count: usize,
    door_count: usize,
) -> MapData {
    let mut grid = vec![WALL; map_h * map_w];

    // 1. Carve rooms
    let num_rooms = rng.gen_range(num_rooms_range.0..=num_rooms_range.1);
    let mut room_rects = Vec::with_capacity(num_rooms);

    for _ in 0..num_rooms {
        let max_rh = 9.min(map_h.saturating_sub(2)).max(5);
        let max_rw = 9.min(map_w.saturating_sub(2)).max(5);

        let rh = rng.gen_range(4..max_rh);
        let rw = rng.gen_range(4..max_rw);

        let max_rr = map_h.saturating_sub(rh + 1).max(2);
        let max_rc = map_w.saturating_sub(rw + 1).max(2);

        let rr = rng.gen_range(1..max_rr);
        let rc = rng.gen_range(1..max_rc);

        for r in rr..(rr + rh) {
            for c in rc..(rc + rw) {
                grid[r * map_w + c] = EMPTY;
            }
        }
        room_rects.push(RoomRect { r: rr, c: rc, h: rh, w: rw });
    }

    // 2. Connect rooms with L-shaped corridors
    let room_centers: Vec<(usize, usize)> = room_rects
        .iter()
        .map(|rm| (rm.r + rm.h / 2, rm.c + rm.w / 2))
        .collect();

    let mut corridor_coords = HashSet::new();

    if room_centers.len() > 1 {
        for i in 0..(room_centers.len() - 1) {
            let (r1, c1) = room_centers[i];
            let (r2, c2) = room_centers[i + 1];

            let cmin = c1.min(c2);
            let cmax = c1.max(c2);
            let rmin = r1.min(r2);
            let rmax = r1.max(r2);

            // Horizontal sweep on r1
            for c in cmin..=cmax {
                let idx = r1 * map_w + c;
                if grid[idx] == WALL {
                    corridor_coords.insert((r1 as i32, c as i32));
                }
                grid[idx] = EMPTY;
            }

            // Vertical sweep on c2
            for r in rmin..=rmax {
                let idx = r * map_w + c2;
                if grid[idx] == WALL {
                    corridor_coords.insert((r as i32, c2 as i32));
                }
                grid[idx] = EMPTY;
            }
        }
    }

    // Identify room tiles vs corridor tiles
    let mut room_set = HashSet::new();
    for rm in &room_rects {
        for r in rm.r..(rm.r + rm.h) {
            for c in rm.c..(rm.c + rm.w) {
                room_set.insert((r as i32, c as i32));
            }
        }
    }

    // Filter corridor coordinates that are strictly outside any room
    corridor_coords.retain(|pos| !room_set.contains(pos));

    // 3. Place cameras inside random rooms
    let mut room_tiles: Vec<(i32, i32)> = room_set
        .into_iter()
        .filter(|&(r, c)| grid[(r as usize) * map_w + (c as usize)] == EMPTY)
        .collect();
    room_tiles.shuffle(rng);

    let mut camera_positions = Vec::new();
    if camera_count > 0 && !room_tiles.is_empty() {
        let n_cams = camera_count.min(room_tiles.len());
        for pos in &room_tiles[..n_cams] {
            grid[(pos.0 as usize) * map_w + (pos.1 as usize)] = CAMERA;
            camera_positions.push(*pos);
        }
    }

    // 4. Place doors on corridor tiles (between rooms)
    let mut door_candidates: Vec<(i32, i32)> = corridor_coords.into_iter().collect();
    door_candidates.shuffle(rng);

    let mut door_positions = Vec::new();
    let n_doors = door_count.min(door_candidates.len());
    for pos in &door_candidates[..n_doors] {
        grid[(pos.0 as usize) * map_w + (pos.1 as usize)] = DOOR;
        door_positions.push(*pos);
    }

    // 5. Place terminal, loot, extraction on remaining empty space
    let mut empty_coords = Vec::new();
    for r in 0..map_h {
        for c in 0..map_w {
            if grid[r * map_w + c] == EMPTY {
                empty_coords.push((r as i32, c as i32));
            }
        }
    }

    if empty_coords.len() < 3 {
        for r in 0..map_h {
            for c in 0..map_w {
                if grid[r * map_w + c] != WALL && !empty_coords.contains(&(r as i32, c as i32)) {
                    empty_coords.push((r as i32, c as i32));
                }
            }
        }
    }

    empty_coords.shuffle(rng);

    let terminal_pos = empty_coords.pop().unwrap_or((1, 1));
    let loot_pos = empty_coords.pop().unwrap_or((1, 2));
    let extract_pos = empty_coords.pop().unwrap_or((1, 3));

    grid[(terminal_pos.0 as usize) * map_w + (terminal_pos.1 as usize)] = TERMINAL;
    grid[(loot_pos.0 as usize) * map_w + (loot_pos.1 as usize)] = LOOT;
    grid[(extract_pos.0 as usize) * map_w + (extract_pos.1 as usize)] = EXTRACT;

    MapData {
        grid,
        terminal_pos,
        loot_pos,
        extract_pos,
        camera_positions,
        door_positions,
        room_rects,
    }
}
