import numpy as np
import pybullet as p
from config import (
    OBSTACLE_CONFIG, OBSTACLE_MIN_SPACING,
    OBSTACLE_WALL_MARGIN, OBSTACLE_SPAWN_MARGIN, MAX_PLACEMENT_TRIES,
    AREA_SIZE, CELL_SIZE, GRID_SIZE, SPAWN_X, SPAWN_Y, SAFE_ZONE_RADIUS,
)


def _footprint(cfg):
    if cfg["type"] == "box":
        return np.hypot(cfg["size"][0], cfg["size"][1])
    return cfg["size"][0]


def _too_close_to_spawn(x, y, cfg):

    t    = cfg["type"]
    size = cfg["size"]
    margin = OBSTACLE_SPAWN_MARGIN

    if t == "box":
        hx, hy = size[0], size[1]
        closest_x = np.clip(SPAWN_X, x - hx, x + hx)
        closest_y = np.clip(SPAWN_Y, y - hy, y + hy)
        return np.hypot(closest_x - SPAWN_X, closest_y - SPAWN_Y) < margin
    else:
        radius = size[0]
        return np.hypot(x - SPAWN_X, y - SPAWN_Y) < margin + radius


def generate_obstacle_positions(rng=None):
    if rng is None:
        rng = np.random.default_rng()

    half = AREA_SIZE / 2.0
    lo   = -half + OBSTACLE_WALL_MARGIN
    hi   =  half - OBSTACLE_WALL_MARGIN

    cfg_list = list(OBSTACLE_CONFIG)
    rng.shuffle(cfg_list)

    placed    = []
    positions = []

    for cfg in cfg_list:
        fp          = _footprint(cfg)
        placed_this = False

        for _ in range(MAX_PLACEMENT_TRIES):
            x = rng.uniform(lo + fp, hi - fp)
            y = rng.uniform(lo + fp, hi - fp)

            # Spawn bölgesine mesafe
            if _too_close_to_spawn(x, y, cfg):
                continue

            too_close = any(
                np.hypot(x - px, y - py) - fp - pr < OBSTACLE_MIN_SPACING
                for px, py, pr in placed
            )
            if too_close:
                continue

            placed.append((x, y, fp))
            positions.append((x, y))
            placed_this = True
            break

        if not placed_this:
            positions.append(None)

    valid_cfg = [c for c, pos in zip(cfg_list, positions) if pos is not None]
    valid_pos = [pos for pos in positions if pos is not None]
    return valid_pos, valid_cfg


def spawn_obstacles(positions, cfg_list, physics_client):
    ids = []
    for cfg, (ox, oy) in zip(cfg_list, positions):
        t     = cfg["type"]
        color = cfg["color"]
        size  = cfg["size"]

        if t == "box":
            hx, hy, hz = size
            col_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                physicsClientId=physics_client)
            vis_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz],
                rgbaColor=color, physicsClientId=physics_client)
            z = hz

        elif t == "cylinder":
            radius, height = size
            col_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=radius, height=height,
                physicsClientId=physics_client)
            vis_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=height,
                rgbaColor=color, physicsClientId=physics_client)
            z = height / 2.0
        else:
            continue

        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[ox, oy, z],
            physicsClientId=physics_client)
        ids.append(bid)

    return ids


def rasterize_obstacles_to_grid(positions, cfg_list):
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    half = AREA_SIZE / 2.0

    for cfg, (ox, oy) in zip(cfg_list, positions):
        t    = cfg["type"]
        size = cfg["size"]

        if t == "box":
            hx, hy = size[0], size[1]
            xi_min = max(0, int((ox - hx + half) / CELL_SIZE))
            xi_max = min(GRID_SIZE - 1, int((ox + hx + half) / CELL_SIZE))
            yi_min = max(0, int((oy - hy + half) / CELL_SIZE))
            yi_max = min(GRID_SIZE - 1, int((oy + hy + half) / CELL_SIZE))
            grid[xi_min:xi_max+1, yi_min:yi_max+1] = True

        elif t == "cylinder":
            radius  = size[0]
            cx      = int((ox + half) / CELL_SIZE)
            cy      = int((oy + half) / CELL_SIZE)
            r_cells = int(np.ceil(radius / CELL_SIZE))
            for di in range(-r_cells, r_cells + 1):
                for dj in range(-r_cells, r_cells + 1):
                    wx = (cx + di) * CELL_SIZE - half + CELL_SIZE / 2
                    wy = (cy + dj) * CELL_SIZE - half + CELL_SIZE / 2
                    if np.hypot(wx - ox, wy - oy) <= radius:
                        gi, gj = cx + di, cy + dj
                        if 0 <= gi < GRID_SIZE and 0 <= gj < GRID_SIZE:
                            grid[gi, gj] = True

    spawn_gi = int((SPAWN_X + half) / CELL_SIZE)
    spawn_gj = int((SPAWN_Y + half) / CELL_SIZE)
    clear_r  = int(np.ceil(SAFE_ZONE_RADIUS / CELL_SIZE))
    for di in range(-clear_r, clear_r + 1):
        for dj in range(-clear_r, clear_r + 1):
            if np.hypot(di * CELL_SIZE, dj * CELL_SIZE) <= SAFE_ZONE_RADIUS:
                gi, gj = spawn_gi + di, spawn_gj + dj
                if 0 <= gi < GRID_SIZE and 0 <= gj < GRID_SIZE:
                    grid[gi, gj] = False

    return grid
