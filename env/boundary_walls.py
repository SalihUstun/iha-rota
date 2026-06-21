import pybullet as p
import numpy as np
from config import AREA_SIZE, WALL_THICKNESS, WALL_HEIGHT, WALL_SENSOR_DIST

_WALL_COLOR = [0.7, 0.7, 0.9, 0.35]


def spawn_boundary_walls(physics_client):
    half     = AREA_SIZE / 2.0
    wt       = WALL_THICKNESS / 2.0
    wh       = WALL_HEIGHT    / 2.0
    wall_len = (AREA_SIZE + WALL_THICKNESS * 2) / 2.0

    walls = [
        ([ half + wt,  0.0, wh], [wt, wall_len, wh]),
        ([-half - wt,  0.0, wh], [wt, wall_len, wh]),
        ([0.0,  half + wt, wh], [wall_len, wt, wh]),
        ([0.0, -half - wt, wh], [wall_len, wt, wh]),
    ]

    wall_ids = []
    for pos, half_ext in walls:
        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_ext,
            physicsClientId=physics_client)
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_ext,
            rgbaColor=_WALL_COLOR,
            physicsClientId=physics_client)
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            physicsClientId=physics_client)
        wall_ids.append(bid)

    return wall_ids


def near_wall_penalty(position):
    half = AREA_SIZE / 2.0
    x, y = position[0], position[1]
    min_dist = min(half - x, half + x, half - y, half + y)
    if min_dist < WALL_SENSOR_DIST:
        return -(1.0 - min_dist / WALL_SENSOR_DIST) ** 2
    return 0.0
