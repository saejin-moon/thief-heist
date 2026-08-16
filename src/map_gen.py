"""
Procedural map generator for HEIST.

Generates a connected, random-dungeon-style map with:
  - Rooms carved out of a wall-filled grid
  - L-shaped corridors connecting successive rooms (guaranteeing connectivity)
  - Security cameras placed inside random rooms
  - Doors placed on corridor tiles that sit between rooms
  - Terminal, loot, and extraction-point spawns spread across empty space

The map dimensions and feature counts are parameterised so that
`curriculum.py` can start training on tiny, guard-free maps and ramp
up to the full 50x50, guard/camera-rich benchmark.
"""

import numpy as np

from constants import (
    CAMERA,
    DOOR,
    EMPTY,
    EXTRACT,
    LOOT,
    TERMINAL,
    WALL,
)


def generate_procedural_map(
    rng: np.random.Generator,
    map_size: tuple[int, int] = (50, 50),
    num_rooms_range: tuple[int, int] = (8, 12),
    camera_count: int = 3,
    door_count: int = 4,
) -> dict:
    """Return a dict describing the generated map.

    Returns
    -------
    dict with keys:
        grid          : np.ndarray[map_size]  — base tile grid
        terminal_pos  : (r, c) of the security terminal
        loot_pos      : (r, c) of the loot
        extract_pos   : (r, c) of the extraction point
        camera_positions : list of (r, c) for each camera tile
        room_rects   : list of (r_start, c_start, h, w) for each room
        corridor_coords : set of (r, c) tiles that are corridor (not in any room)
    """
    h, w = map_size
    grid = np.full(map_size, WALL, dtype=np.int32)

    # --- carve rooms --------------------------------------------------------
    num_rooms = rng.integers(num_rooms_range[0], num_rooms_range[1] + 1)
    room_rects: list[tuple[int, int, int, int]] = []
    for _ in range(num_rooms):
        rh = rng.integers(4, min(9, h - 2))
        rw = rng.integers(4, min(9, w - 2))
        rr = rng.integers(1, h - rh - 1)
        rc = rng.integers(1, w - rw - 1)
        grid[rr : rr + rh, rc : rc + rw] = EMPTY
        room_rects.append((rr, rc, rh, rw))

    # --- connect rooms with L-shaped corridors ------------------------------
    room_centers = [(r + rh // 2, c + rw // 2) for (r, c, rh, rw) in room_rects]
    corridor_coords: set[tuple[int, int]] = set()
    for i in range(len(room_centers) - 1):
        r1, c1 = room_centers[i]
        r2, c2 = room_centers[i + 1]
        cmin, cmax = min(c1, c2), max(c1, c2)
        rmin, rmax = min(r1, r2), max(r1, r2)
        # horizontal sweep on c1
        for c in range(cmin, cmax + 1):
            if grid[r1, c] == WALL:
                corridor_coords.add((r1, c))
            grid[r1, c] = EMPTY
        # vertical sweep on c2
        for r in range(rmin, rmax + 1):
            if grid[r, c2] == WALL:
                corridor_coords.add((r, c2))
            grid[r, c2] = EMPTY

    # --- identify all empty tiles (rooms + corridors) ----------------------
    all_empty = np.argwhere(grid == EMPTY)
    # classify which empty tiles belong to a room vs corridor
    room_set: set[tuple[int, int]] = set()
    for rr, rc, rh, rw in room_rects:
        for r in range(rr, rr + rh):
            for c in range(rc, rc + rw):
                room_set.add((r, c))
    corridor_coords = {(r, c) for (r, c) in corridor_coords if (r, c) not in room_set}

    # --- place cameras inside random rooms ----------------------------------
    room_tiles: list[tuple[int, int]] = list(room_set & set(map(tuple, all_empty)))
    rng.shuffle(room_tiles)
    camera_positions: list[tuple[int, int]] = []
    if camera_count > 0 and room_tiles:
        cam_tiles = room_tiles[:camera_count]
        for pos in cam_tiles:
            p = (int(pos[0]), int(pos[1]))
            grid[p[0], p[1]] = CAMERA
            camera_positions.append(p)

    # --- place doors on corridor tiles (between rooms) ---------------------
    door_coords = list(corridor_coords)
    rng.shuffle(door_coords)
    door_positions: list[tuple[int, int]] = []
    for pos in door_coords[:door_count]:
        p = (int(pos[0]), int(pos[1]))
        grid[p[0], p[1]] = DOOR
        door_positions.append(p)

    # --- place terminal, loot, extraction on remaining empty space ----------
    empty_coords = np.argwhere(grid == EMPTY)
    if len(empty_coords) < 3:
        # fallback: use all non-wall tiles for spawning
        empty_coords = np.argwhere(grid != WALL)
    spawns = rng.choice(empty_coords, size=3, replace=False)
    terminal_pos = tuple(spawns[0].tolist())
    loot_pos = tuple(spawns[1].tolist())
    extract_pos = tuple(spawns[2].tolist())
    grid[terminal_pos[0], terminal_pos[1]] = TERMINAL
    grid[loot_pos[0], loot_pos[1]] = LOOT
    grid[extract_pos[0], extract_pos[1]] = EXTRACT

    return {
        "grid": grid,
        "terminal_pos": terminal_pos,
        "loot_pos": loot_pos,
        "extract_pos": extract_pos,
        "camera_positions": camera_positions,
        "door_positions": door_positions,
        "room_rects": room_rects,
        "corridor_coords": corridor_coords,
    }
