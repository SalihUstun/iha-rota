import numpy as np
import pybullet as p
from config import (
    NUM_ENEMIES, ENEMY_RADIUS, DRONE_HEIGHT,
    AREA_SIZE, SPAWN_X, SPAWN_Y, SAFE_ZONE_RADIUS,
)

_ENEMY_COLOR = [1.0, 0.15, 0.15, 1.0]


def spawn_enemies(obstacle_grid, physics_client, rng):
    half      = AREA_SIZE / 2.0
    positions = np.zeros((NUM_ENEMIES, 3))
    ids       = []

    col_id = p.createCollisionShape(
        p.GEOM_SPHERE, radius=ENEMY_RADIUS,
        physicsClientId=physics_client)
    vis_id = p.createVisualShape(
        p.GEOM_SPHERE, radius=ENEMY_RADIUS,
        rgbaColor=_ENEMY_COLOR,
        physicsClientId=physics_client)

    for i in range(NUM_ENEMIES):
        for _ in range(1000):
            x = rng.uniform(-half + ENEMY_RADIUS, half - ENEMY_RADIUS)
            y = rng.uniform(-half + ENEMY_RADIUS, half - ENEMY_RADIUS)

            if np.hypot(x - SPAWN_X, y - SPAWN_Y) < SAFE_ZONE_RADIUS:
                continue

            gi = int((x + half) / (AREA_SIZE / len(obstacle_grid)))
            gj = int((y + half) / (AREA_SIZE / len(obstacle_grid[0])))
            if 0 <= gi < len(obstacle_grid) and 0 <= gj < len(obstacle_grid[0]):
                if obstacle_grid[gi, gj]:
                    continue

            positions[i] = [x, y, DRONE_HEIGHT]
            break

        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=positions[i].tolist(),
            physicsClientId=physics_client)
        ids.append(bid)

    return positions, ids
