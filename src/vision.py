"""
Vision and line-of-sight primitives for HEIST.

All heavy loops are JIT-compiled with Numba so that high-FPS headless
training rollouts are not throttled by the Python interpreter.  This
addresses the "Performance Horizon (Raycasting Bottleneck)" section of
PLAN.md: raycasting is now fully numba-vectorized and callable from plain
NumPy code.

Functions
---------
raycast : fill a Bresenham line into explored_map until a wall is hit
calculate_fov : reveal a radius around a position using raycast
line_is_clear : True when the Bresenham line between two tiles is unobstructed
camera_exposure : which cameras see which agents (vectorized over agents)
"""

import numpy as np
from numba import njit

from constants import DOOR, WALL


@njit(cache=True)
def raycast(grid, explored_map, start_r, start_c, target_r, target_c, wall_val):
    """Walk a Bresenham line from (start_r, start_c) to (target_r, target_c),
    marking every tile traversed as explored until a wall blocks the line."""
    r0, c0 = start_r, start_c
    r1, c1 = target_r, target_c

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        explored_map[r0, c0] = True

        if grid[r0, c0] == wall_val:
            break

        if r0 == r1 and c0 == c1:
            break

        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


@njit(cache=True)
def calculate_fov(grid, explored_map, start_r, start_c, radius, map_h, map_w, wall_val):
    """Reveal every tile within `radius` of (start_r, start_c), clipped to the
    map bounds.  Walls terminate each ray, which is what creates the fog."""
    min_row = max(start_r - radius, 0)
    max_row = min(start_r + radius, map_h - 1)
    min_col = max(start_c - radius, 0)
    max_col = min(start_c + radius, map_w - 1)

    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            raycast(grid, explored_map, start_r, start_c, r, c, wall_val)


@njit(cache=True)
def line_is_clear(grid, r0, c0, r1, c1, wall_val, door_val):
    """Return True when the tile at (r0, c0) can 'see' the tile at (r1, c1):
    the Bresenham line between them is not blocked by a wall or a locked door.

    The starting tile itself never blocks (a camera on the same tile as an
    agent would be a trivial edge case, and we treat the start as transparent).
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        if r == r1 and c == c1:
            return True
        if (r != r0 or c != c0) and (grid[r, c] == wall_val or grid[r, c] == door_val):
            return False
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False


@njit(cache=True)
def bfs_next_step(
    grid,
    start_r,
    start_c,
    target_r,
    target_c,
    wall_val,
    door_val,
    queue,
    previous,
    reset_stack,
):
    """Return the first shortest-path step as ``(row, col)`` or ``(-1, -1)``.

    This is deliberately BFS, rather than A*: movement has uniform cost and
    the action ordering matches ``ACTION_DELTAS`` (up, down, left, right).
    Fixed-size NumPy arrays avoid the per-search Python ``deque``/``dict``
    allocations that made guard movement expensive on large maps.
    """
    if start_r == target_r and start_c == target_c:
        return -1, -1

    height, width = grid.shape
    head = 0
    tail = 1
    reset_ptr = 0
    start = start_r * width + start_c
    target = target_r * width + target_c
    queue[0] = start
    previous[start] = -1
    reset_stack[reset_ptr] = start
    reset_ptr += 1

    while head < tail:
        current = queue[head]
        head += 1
        if current == target:
            break
        row = current // width
        col = current % width
        for direction in range(4):
            if direction == 0:
                nr, nc = row - 1, col
            elif direction == 1:
                nr, nc = row + 1, col
            elif direction == 2:
                nr, nc = row, col - 1
            else:
                nr, nc = row, col + 1
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            neighbor = nr * width + nc
            if previous[neighbor] != -2:
                continue
            tile = grid[nr, nc]
            if tile in (wall_val, door_val):
                continue
            previous[neighbor] = current
            queue[tail] = neighbor
            tail += 1
            reset_stack[reset_ptr] = neighbor
            reset_ptr += 1

    res_r, res_c = -1, -1
    if previous[target] != -2:
        current = target
        while previous[current] != start:
            current = previous[current]
            if current < 0:
                res_r, res_c = -1, -1
                break
        else:
            res_r, res_c = current // width, current % width

    for i in range(reset_ptr):
        previous[reset_stack[i]] = -2

    return res_r, res_c


@njit(cache=True)
def bfs_shortest_path_distance(
    grid,
    start_r,
    start_c,
    target_r,
    target_c,
    wall_val,
    door_val,
    queue,
    distance,
    reset_stack,
):
    """Return shortest-path distance between (start_r, start_c) and (target_r, target_c) or 999999."""
    if start_r == target_r and start_c == target_c:
        return 0
    height, width = grid.shape
    head = 0
    tail = 1
    reset_ptr = 0
    start = start_r * width + start_c
    target = target_r * width + target_c
    queue[0] = start
    distance[start_r, start_c] = 0
    reset_stack[reset_ptr] = start
    reset_ptr += 1

    res_d = 999999
    while head < tail:
        current = queue[head]
        head += 1
        row = current // width
        col = current % width
        d = distance[row, col]
        if current == target:
            res_d = int(d)
            break
        for direction in range(4):
            if direction == 0:
                nr, nc = row - 1, col
            elif direction == 1:
                nr, nc = row + 1, col
            elif direction == 2:
                nr, nc = row, col - 1
            else:
                nr, nc = row, col + 1
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            neighbor = nr * width + nc
            if distance[nr, nc] != -1:
                continue
            tile = grid[nr, nc]
            if tile in (wall_val, door_val):
                continue
            distance[nr, nc] = d + 1
            queue[tail] = neighbor
            tail += 1
            reset_stack[reset_ptr] = neighbor
            reset_ptr += 1

    for i in range(reset_ptr):
        idx = reset_stack[i]
        distance[idx // width, idx % width] = -1

    return res_d


@njit(cache=True)
def distance_to_nearest_target(grid, targets, wall_val, door_val, distance, queue):
    """Return a walkable-cell distance field seeded by every target.

    A single multi-source BFS lets all converging guards follow an optimal
    route to their nearest reachable agent rather than running one BFS per
    guard.  ``-1`` indicates walls, doors, and unreachable cells.
    """
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            distance[r, c] = -1

    head = 0
    tail = 0
    for i in range(targets.shape[0]):
        row = targets[i, 0]
        col = targets[i, 1]
        if row < 0 or row >= height or col < 0 or col >= width:
            continue
        if distance[row, col] == -1:
            distance[row, col] = 0
            queue[tail] = row * width + col
            tail += 1

    while head < tail:
        current = queue[head]
        head += 1
        row = current // width
        col = current % width
        for direction in range(4):
            if direction == 0:
                nr, nc = row - 1, col
            elif direction == 1:
                nr, nc = row + 1, col
            elif direction == 2:
                nr, nc = row, col - 1
            else:
                nr, nc = row, col + 1
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if distance[nr, nc] != -1:
                continue
            tile = grid[nr, nc]
            if tile in (wall_val, door_val):
                continue
            distance[nr, nc] = distance[row, col] + 1
            queue[tail] = nr * width + nc
            tail += 1
    return distance


@njit(cache=True)
def next_step_from_distance(distance, row, col):
    """Pick the first adjacent cell that reduces a BFS distance field."""
    current_distance = distance[row, col]
    if current_distance <= 0:
        return -1, -1
    height, width = distance.shape
    for direction in range(4):
        if direction == 0:
            nr, nc = row - 1, col
        elif direction == 1:
            nr, nc = row + 1, col
        elif direction == 2:
            nr, nc = row, col - 1
        else:
            nr, nc = row, col + 1
        if (
            0 <= nr < height
            and 0 <= nc < width
            and distance[nr, nc] == current_distance - 1
        ):
            return nr, nc
    return -1, -1


def camera_exposure(
    grid, camera_positions, agent_positions, wall_val=WALL, door_val=DOOR, max_range=12
):
    """Return a boolean matrix [n_cameras, n_agents] indicating which cameras
    currently have a clear line of sight to which agents, restricted to
    `max_range` Manhattan distance.  Thin Python wrapper around the
    numba-compiled `line_is_clear`; camera counts are small (typically <= 4)
    so the overhead is negligible."""
    n_cam, n_agent = len(camera_positions), len(agent_positions)
    if n_cam == 0 or n_agent == 0:
        return np.zeros((n_cam, n_agent), dtype=bool)

    c_pos = np.array(camera_positions)
    a_pos = np.array(agent_positions)
    dist = np.abs(c_pos[:, None, 0] - a_pos[None, :, 0]) + np.abs(
        c_pos[:, None, 1] - a_pos[None, :, 1]
    )

    exposure = np.zeros((n_cam, n_agent), dtype=bool)
    for i in range(n_cam):
        for j in range(n_agent):
            if dist[i, j] > max_range:
                continue
            exposure[i, j] = line_is_clear(
                grid,
                c_pos[i, 0],
                c_pos[i, 1],
                a_pos[j, 0],
                a_pos[j, 1],
                wall_val,
                door_val,
            )
    return exposure


@njit(cache=True)
def pick_search_tile(grid, center_r, center_c, radius, wall_val, door_val, rand_val):
    height, width = grid.shape
    max_cells = (2 * radius + 1) * (2 * radius + 1)
    stack_r = np.empty(max_cells, dtype=np.int32)
    stack_c = np.empty(max_cells, dtype=np.int32)
    count = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = center_r + dr, center_c + dc
            if 0 <= nr < height and 0 <= nc < width:
                tile = grid[nr, nc]
                if tile != wall_val and tile != door_val:
                    stack_r[count] = nr
                    stack_c[count] = nc
                    count += 1
    if count == 0:
        return center_r, center_c
    idx = int(rand_val * count)
    return stack_r[idx], stack_c[idx]


@njit(cache=True)
def get_valid_moves(grid, r, c, wall_val, door_val):
    height, width = grid.shape
    moves = np.empty((4, 2), dtype=np.int32)
    count = 0
    for direction in range(4):
        if direction == 0:
            nr, nc = r - 1, c
        elif direction == 1:
            nr, nc = r + 1, c
        elif direction == 2:
            nr, nc = r, c - 1
        else:
            nr, nc = r, c + 1
        if 0 <= nr < height and 0 <= nc < width:
            tile = grid[nr, nc]
            if tile != wall_val and tile != door_val:
                moves[count, 0] = nr
                moves[count, 1] = nc
                count += 1
    return moves[:count]
