use crate::constants::{DOOR, WALL};

pub struct BfsPathfinder {
    queue: Vec<i32>,
    previous: Vec<i32>,
    reset_stack: Vec<i32>,
    distance_map: Vec<i32>,
}

impl BfsPathfinder {
    pub fn new(max_cells: usize) -> Self {
        Self {
            queue: vec![0; max_cells],
            previous: vec![-2; max_cells],
            reset_stack: vec![0; max_cells],
            distance_map: vec![-1; max_cells],
        }
    }

    pub fn bfs_next_step(
        &mut self,
        grid: &[i32],
        start_r: i32,
        start_c: i32,
        target_r: i32,
        target_c: i32,
        map_h: usize,
        map_w: usize,
    ) -> (i32, i32) {
        if start_r == target_r && start_c == target_c {
            return (-1, -1);
        }

        let width = map_w as i32;
        let height = map_h as i32;
        let mut head = 0;
        let mut tail = 1;
        let mut reset_ptr = 0;

        let start = start_r * width + start_c;
        let target = target_r * width + target_c;

        self.queue[0] = start;
        self.previous[start as usize] = -1;
        self.reset_stack[reset_ptr] = start;
        reset_ptr += 1;

        while head < tail {
            let current = self.queue[head];
            head += 1;

            if current == target {
                break;
            }

            let row = current / width;
            let col = current % width;

            for direction in 0..4 {
                let (nr, nc) = match direction {
                    0 => (row - 1, col), // UP
                    1 => (row + 1, col), // DOWN
                    2 => (row, col - 1), // LEFT
                    _ => (row, col + 1), // RIGHT
                };

                if nr < 0 || nr >= height || nc < 0 || nc >= width {
                    continue;
                }

                let neighbor = nr * width + nc;
                if self.previous[neighbor as usize] != -2 {
                    continue;
                }

                let tile = grid[neighbor as usize];
                if tile == WALL || tile == DOOR {
                    continue;
                }

                self.previous[neighbor as usize] = current;
                self.queue[tail] = neighbor;
                tail += 1;
                self.reset_stack[reset_ptr] = neighbor;
                reset_ptr += 1;
            }
        }

        let mut res = (-1, -1);
        if self.previous[target as usize] != -2 {
            let mut current = target;
            while self.previous[current as usize] != start {
                current = self.previous[current as usize];
                if current < 0 {
                    res = (-1, -1);
                    break;
                }
            }
            if current >= 0 {
                res = (current / width, current % width);
            }
        }

        for i in 0..reset_ptr {
            self.previous[self.reset_stack[i] as usize] = -2;
        }

        res
    }

    pub fn compute_multi_target_distances(
        &mut self,
        grid: &[i32],
        targets: &[(i32, i32)],
        map_h: usize,
        map_w: usize,
    ) {
        let width = map_w as i32;
        let height = map_h as i32;
        let total_cells = map_h * map_w;

        self.distance_map.iter_mut().take(total_cells).for_each(|d| *d = -1);

        let mut head = 0;
        let mut tail = 0;

        for &(r, c) in targets {
            if r >= 0 && r < height && c >= 0 && c < width {
                let idx = (r * width + c) as usize;
                if self.distance_map[idx] == -1 {
                    self.distance_map[idx] = 0;
                    self.queue[tail] = idx as i32;
                    tail += 1;
                }
            }
        }

        while head < tail {
            let current = self.queue[head] as usize;
            head += 1;

            let row = (current as i32) / width;
            let col = (current as i32) % width;
            let cur_dist = self.distance_map[current];

            for direction in 0..4 {
                let (nr, nc) = match direction {
                    0 => (row - 1, col), // UP
                    1 => (row + 1, col), // DOWN
                    2 => (row, col - 1), // LEFT
                    _ => (row, col + 1), // RIGHT
                };

                if nr < 0 || nr >= height || nc < 0 || nc >= width {
                    continue;
                }

                let neighbor = (nr * width + nc) as usize;
                if self.distance_map[neighbor] != -1 {
                    continue;
                }

                let tile = grid[neighbor];
                if tile == WALL || tile == DOOR {
                    continue;
                }

                self.distance_map[neighbor] = cur_dist + 1;
                self.queue[tail] = neighbor as i32;
                tail += 1;
            }
        }
    }

    pub fn next_step_from_distance_map(
        &self,
        r: i32,
        c: i32,
        map_h: usize,
        map_w: usize,
    ) -> (i32, i32) {
        let width = map_w as i32;
        let height = map_h as i32;
        let idx = (r * width + c) as usize;
        let current_distance = self.distance_map[idx];

        if current_distance <= 0 {
            return (-1, -1);
        }

        for direction in 0..4 {
            let (nr, nc) = match direction {
                0 => (r - 1, c), // UP
                1 => (r + 1, c), // DOWN
                2 => (r, c - 1), // LEFT
                _ => (r, c + 1), // RIGHT
            };

            if nr >= 0 && nr < height && nc >= 0 && nc < width {
                let neighbor_idx = (nr * width + nc) as usize;
                if self.distance_map[neighbor_idx] == current_distance - 1 {
                    return (nr, nc);
                }
            }
        }
        (-1, -1)
    }
}
