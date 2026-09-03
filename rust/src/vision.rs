use crate::constants::{DOOR, WALL};
use crate::types::manhattan;

#[inline]
pub fn raycast(
    grid: &[i32],
    explored: &mut [bool],
    start_r: i32,
    start_c: i32,
    target_r: i32,
    target_c: i32,
    map_h: usize,
    map_w: usize,
) {
    let mut r0 = start_r;
    let mut c0 = start_c;
    let r1 = target_r;
    let c1 = target_c;

    let dr = (r1 - r0).abs();
    let dc = (c1 - c0).abs();
    let sr = if r0 < r1 { 1 } else { -1 };
    let sc = if c0 < c1 { 1 } else { -1 };
    let mut err = dr - dc;

    loop {
        if r0 >= 0 && r0 < map_h as i32 && c0 >= 0 && c0 < map_w as i32 {
            let idx = (r0 as usize) * map_w + (c0 as usize);
            explored[idx] = true;
            if grid[idx] == WALL {
                break;
            }
        } else {
            break;
        }

        if r0 == r1 && c0 == c1 {
            break;
        }

        let e2 = 2 * err;
        if e2 > -dc {
            err -= dc;
            r0 += sr;
        }
        if e2 < dr {
            err += dr;
            c0 += sc;
        }
    }
}

pub fn calculate_fov(
    grid: &[i32],
    explored: &mut [bool],
    start_r: i32,
    start_c: i32,
    radius: i32,
    map_h: usize,
    map_w: usize,
) {
    let min_row = (start_r - radius).max(0);
    let max_row = (start_r + radius).min(map_h as i32 - 1);
    let min_col = (start_c - radius).max(0);
    let max_col = (start_c + radius).min(map_w as i32 - 1);

    for r in min_row..=max_row {
        for c in min_col..=max_col {
            raycast(grid, explored, start_r, start_c, r, c, map_h, map_w);
        }
    }
}

#[inline]
pub fn line_is_clear(
    grid: &[i32],
    r0: i32,
    c0: i32,
    r1: i32,
    c1: i32,
    map_h: usize,
    map_w: usize,
) -> bool {
    let dr = (r1 - r0).abs();
    let dc = (c1 - c0).abs();
    let sr = if r0 < r1 { 1 } else { -1 };
    let sc = if c0 < c1 { 1 } else { -1 };
    let mut err = dr - dc;

    let mut r = r0;
    let mut c = c0;

    loop {
        if r == r1 && c == c1 {
            return true;
        }

        if r != r0 || c != c0 {
            if !(r >= 0 && r < map_h as i32 && c >= 0 && c < map_w as i32) {
                return false;
            }
            let tile = grid[(r as usize) * map_w + (c as usize)];
            if tile == WALL || tile == DOOR {
                return false;
            }
        }

        let e2 = 2 * err;
        if e2 > -dc {
            err -= dc;
            r += sr;
        }
        if e2 < dr {
            err += dr;
            c += sc;
        }
    }
}

pub fn camera_exposure_sum(
    grid: &[i32],
    camera_positions: &[(i32, i32)],
    agent_positions: &[(i32, i32)],
    max_range: i32,
    map_h: usize,
    map_w: usize,
) -> f32 {
    let mut exposure_count = 0;
    for &cam in camera_positions {
        for &ag in agent_positions {
            if manhattan(cam, ag) <= max_range {
                if line_is_clear(grid, cam.0, cam.1, ag.0, ag.1, map_h, map_w) {
                    exposure_count += 1;
                }
            }
        }
    }
    exposure_count as f32
}

pub fn pick_search_tile(
    grid: &[i32],
    center_r: i32,
    center_c: i32,
    radius: i32,
    rand_val: f64,
    map_h: usize,
    map_w: usize,
) -> (i32, i32) {
    let mut valid_tiles = Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);
    let min_r = (center_r - radius).max(0);
    let max_r = (center_r + radius).min(map_h as i32 - 1);
    let min_c = (center_c - radius).max(0);
    let max_c = (center_c + radius).min(map_w as i32 - 1);

    for r in min_r..=max_r {
        for c in min_c..=max_c {
            let tile = grid[(r as usize) * map_w + (c as usize)];
            if tile != WALL && tile != DOOR {
                valid_tiles.push((r, c));
            }
        }
    }

    if valid_tiles.is_empty() {
        return (center_r, center_c);
    }

    let idx = ((rand_val * valid_tiles.len() as f64) as usize).min(valid_tiles.len() - 1);
    valid_tiles[idx]
}

pub fn get_valid_moves(
    grid: &[i32],
    r: i32,
    c: i32,
    map_h: usize,
    map_w: usize,
) -> [(i32, i32); 4] {
    let mut moves = [(-1, -1); 4];
    let deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // UP, DOWN, LEFT, RIGHT

    for (i, &(dr, dc)) in deltas.iter().enumerate() {
        let nr = r + dr;
        let nc = c + dc;
        if nr >= 0 && nr < map_h as i32 && nc >= 0 && nc < map_w as i32 {
            let tile = grid[(nr as usize) * map_w + (nc as usize)];
            if tile != WALL && tile != DOOR {
                moves[i] = (nr, nc);
            }
        }
    }
    moves
}
