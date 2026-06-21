import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from config import (
    AREA_SIZE, CELL_SIZE, GRID_SIZE, SIM_FREQ, CTRL_FREQ, MAX_STEPS,
    DRONE_HEIGHT, DRONE_RADIUS, SPAWN_X, SPAWN_Y, SAFE_ZONE_RADIUS,
    MAX_SPEED, ACTION_DIM, OBS_DIM,
    LIDAR_RAYS, LIDAR_RANGE,
    NUM_ENEMIES, ENEMY_RADIUS, ENEMY_SPEED, ENEMY_NEAR_DIST,
    NUM_TARGETS, TARGET_RADIUS, TARGET_MIN_FROM_SPAWN,
    REWARD_STEP, REWARD_ENEMY_HIT, REWARD_OBSTACLE_HIT,
    REWARD_NEAR_ENEMY, REWARD_TARGET_MID, REWARD_TARGET_FINAL, REWARD_DIST_SCALE,
)
from env.obstacles import generate_obstacle_positions, spawn_obstacles, rasterize_obstacles_to_grid
from env.boundary_walls import spawn_boundary_walls
from env.enemies import spawn_enemies

_ACTION_TO_VEL = np.array([
    [ 0.000,  0.000],
    [ 0.000,  1.000],
    [ 0.707,  0.707],
    [ 1.000,  0.000],
    [ 0.707, -0.707],
    [ 0.000, -1.000],
    [-0.707, -0.707],
    [-1.000,  0.000],
    [-0.707,  0.707],
], dtype=np.float32)

_DRONE_COLOR          = [0.0, 0.6, 1.0, 1.0]
_TARGET_COLOR_ACTIVE  = [0.0, 0.9, 0.0, 0.9]
_TARGET_COLOR_PASSIVE = [0.4, 0.9, 0.4, 0.5]


class IhaEnv(gym.Env):

    metadata = {"render_modes": ["human", "headless"]}

    def __init__(self, render_mode="headless", seed=None):
        super().__init__()
        self.render_mode     = render_mode
        self.rng             = np.random.default_rng(seed)
        self._half           = AREA_SIZE / 2.0
        self._steps_per_ctrl = int(SIM_FREQ / CTRL_FREQ)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_DIM)

        self._pc              = None
        self._drone_id        = None
        self._obstacle_ids    = []
        self._wall_ids        = []
        self._enemy_ids       = []
        self._target_ids      = []
        self._obstacle_grid   = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self._drone_pos       = np.zeros(3)
        self._drone_vel       = np.zeros(3)
        self._enemy_positions = np.zeros((NUM_ENEMIES, 3))
        self._target_positions = np.zeros((NUM_TARGETS, 3))
        self._active_target   = 0
        self._step_count      = 0
        self._lidar_cache     = np.ones(LIDAR_RAYS, dtype=np.float32)


    def reset(self, seed=None, options=None):


        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._pc is None:
            self._full_init()
        else:
            self._soft_reset()
        return self._get_obs(), {}

    def reset_map(self, seed=None):


        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._close_physics()
        self._full_init()
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        prev_pos = self._drone_pos.copy()
        hedef    = self._target_positions[self._active_target]

        self._apply_action(action)
        p.stepSimulation(physicsClientId=self._pc)
        self._update_drone_state()

        reward = REWARD_STEP
        done   = False

        if self._check_obstacle_collision():
            self._reset_drone_to(prev_pos)
            reward += REWARD_OBSTACLE_HIT

        if self._min_enemy_dist() < DRONE_RADIUS + ENEMY_RADIUS:
            self._reset_drone_to(prev_pos)
            reward += REWARD_ENEMY_HIT

        if self._step_count % 2 == 0:
            self._move_enemies()

        if self._min_enemy_dist() < DRONE_RADIUS + ENEMY_RADIUS:
            self._reset_drone_to(prev_pos)
            reward += REWARD_ENEMY_HIT

        if self._min_enemy_dist() < ENEMY_NEAR_DIST:
            reward += REWARD_NEAR_ENEMY

        dist_to_target = np.linalg.norm(self._drone_pos[:2] - hedef[:2])
        if dist_to_target < TARGET_RADIUS:
            if self._active_target < NUM_TARGETS - 1:
                reward += REWARD_TARGET_MID
                self._active_target += 1
                self._refresh_target_colors()
            else:
                reward += REWARD_TARGET_FINAL
                done = True

        prev_dist = np.linalg.norm(prev_pos[:2] - hedef[:2])
        new_dist  = np.linalg.norm(self._drone_pos[:2] - hedef[:2])
        reward += (prev_dist - new_dist) * REWARD_DIST_SCALE

        if self._step_count >= MAX_STEPS:
            done = True

        return self._get_obs(), float(reward), done, False, {}

    def render(self):
        pass

    def close(self):
        self._close_physics()

    def _full_init(self):
        """PyBullet başlat, tüm nesneleri oluştur."""
        self._init_physics()
        self._spawn_ground()
        self._wall_ids = spawn_boundary_walls(self._pc)

        obs_pos, obs_cfg       = generate_obstacle_positions(self.rng)
        self._obstacle_ids     = spawn_obstacles(obs_pos, obs_cfg, self._pc)
        self._obstacle_grid    = rasterize_obstacles_to_grid(obs_pos, obs_cfg)

        self._spawn_drone()
        self._enemy_positions, self._enemy_ids = spawn_enemies(
            self._obstacle_grid, self._pc, self.rng)
        self._target_positions = self._place_targets()
        self._spawn_target_visuals()

        self._active_target  = 0
        self._step_count     = 0
        self._drone_vel[:]   = 0.0
        self._lidar_cache[:] = 1.0

    def _soft_reset(self):

        p.resetBasePositionAndOrientation(
            self._drone_id,
            [SPAWN_X, SPAWN_Y, DRONE_HEIGHT], [0, 0, 0, 1],
            physicsClientId=self._pc)
        self._drone_pos = np.array([SPAWN_X, SPAWN_Y, DRONE_HEIGHT], dtype=float)
        self._drone_vel[:] = 0.0

        self._reposition_enemies()

        self._target_positions = self._place_targets()
        for i, (tid, pos) in enumerate(zip(self._target_ids, self._target_positions)):
            p.resetBasePositionAndOrientation(
                tid, pos.tolist(), [0, 0, 0, 1],
                physicsClientId=self._pc)

        self._active_target  = 0
        self._step_count     = 0
        self._lidar_cache[:] = 1.0
        self._refresh_target_colors()

    def _reposition_enemies(self):
        half = self._half
        for i in range(NUM_ENEMIES):
            for _ in range(1000):
                x = self.rng.uniform(-half + ENEMY_RADIUS, half - ENEMY_RADIUS)
                y = self.rng.uniform(-half + ENEMY_RADIUS, half - ENEMY_RADIUS)
                if np.hypot(x - SPAWN_X, y - SPAWN_Y) < SAFE_ZONE_RADIUS:
                    continue
                gi = int((x + half) / CELL_SIZE)
                gj = int((y + half) / CELL_SIZE)
                if 0 <= gi < GRID_SIZE and 0 <= gj < GRID_SIZE and self._obstacle_grid[gi, gj]:
                    continue
                self._enemy_positions[i] = [x, y, DRONE_HEIGHT]
                break
            p.resetBasePositionAndOrientation(
                self._enemy_ids[i],
                self._enemy_positions[i].tolist(), [0, 0, 0, 1],
                physicsClientId=self._pc)


    def _get_obs(self):
        half  = self._half
        hedef = self._target_positions[self._active_target]

        if NUM_ENEMIES > 0:
            dists   = np.linalg.norm(
                self._enemy_positions[:, :2] - self._drone_pos[:2], axis=1)
            nearest = self._enemy_positions[np.argmin(dists)]
        else:
            nearest = np.array([half, half, DRONE_HEIGHT])

        if self._step_count % 3 == 0:
            self._lidar_cache = self._cast_lidar()

        return np.concatenate([
            self._drone_pos / half,
            self._drone_vel / MAX_SPEED,
            (hedef - self._drone_pos) / half,
            (nearest - self._drone_pos) / half,
            self._lidar_cache,
        ]).astype(np.float32)

    def _init_physics(self):
        if self.render_mode == "human":
            self._pc = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=25, cameraYaw=0, cameraPitch=-55,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self._pc)
        else:
            self._pc = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._pc)
        p.setGravity(0, 0, -9.81, physicsClientId=self._pc)
        p.setTimeStep(1.0 / SIM_FREQ, physicsClientId=self._pc)

    def _close_physics(self):
        if self._pc is not None:
            try:
                p.disconnect(self._pc)
            except Exception:
                pass
            self._pc = None
        self._drone_id     = None
        self._obstacle_ids = []
        self._wall_ids     = []
        self._enemy_ids    = []
        self._target_ids   = []

    def _spawn_ground(self):
        p.loadURDF("plane.urdf", physicsClientId=self._pc)

    def _spawn_drone(self):
        col_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=DRONE_RADIUS, physicsClientId=self._pc)
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=DRONE_RADIUS,
            rgbaColor=_DRONE_COLOR, physicsClientId=self._pc)
        self._drone_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[SPAWN_X, SPAWN_Y, DRONE_HEIGHT],
            physicsClientId=self._pc)
        p.changeDynamics(
            self._drone_id, -1,
            linearDamping=0.9, angularDamping=0.9,
            physicsClientId=self._pc)
        self._drone_pos = np.array([SPAWN_X, SPAWN_Y, DRONE_HEIGHT], dtype=float)

    def _place_targets(self):
        half      = self._half
        positions = np.zeros((NUM_TARGETS, 3))
        r_cells   = int(np.ceil(1.5 / CELL_SIZE))  # 1.5m engel tamponu (3 hücre)
        placed    = []
        for i in range(NUM_TARGETS):
            for _ in range(2000):
                x = self.rng.uniform(-half + 1.0, half - 1.0)
                y = self.rng.uniform(-half + 1.0, half - 1.0)
                if np.hypot(x - SPAWN_X, y - SPAWN_Y) < TARGET_MIN_FROM_SPAWN:
                    continue
                gi = int((x + half) / CELL_SIZE)
                gj = int((y + half) / CELL_SIZE)
                blocked = any(
                    0 <= gi + di < GRID_SIZE and 0 <= gj + dj < GRID_SIZE
                    and self._obstacle_grid[gi + di, gj + dj]
                    for di in range(-r_cells, r_cells + 1)
                    for dj in range(-r_cells, r_cells + 1)
                )
                if blocked:
                    continue
                # Hedefler arası minimum 3.0m
                if any(np.hypot(x - px, y - py) < 3.0 for px, py in placed):
                    continue
                placed.append((x, y))
                positions[i] = [x, y, DRONE_HEIGHT]
                break
        return positions

    def _spawn_target_visuals(self):
        self._target_ids = []
        for i, pos in enumerate(self._target_positions):
            color  = _TARGET_COLOR_ACTIVE if i == 0 else _TARGET_COLOR_PASSIVE
            vis_id = p.createVisualShape(
                p.GEOM_SPHERE, radius=TARGET_RADIUS,
                rgbaColor=color, physicsClientId=self._pc)
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis_id,
                basePosition=pos.tolist(),
                physicsClientId=self._pc)
            self._target_ids.append(bid)

    def _refresh_target_colors(self):
        for i, bid in enumerate(self._target_ids):
            if i < self._active_target:
                continue
            color = _TARGET_COLOR_ACTIVE if i == self._active_target else _TARGET_COLOR_PASSIVE
            p.changeVisualShape(
                bid, -1, rgbaColor=color, physicsClientId=self._pc)

    def _apply_action(self, action):
        vel = _ACTION_TO_VEL[int(action)]
        dt  = 1.0 / CTRL_FREQ
        vx  = float(vel[0]) * MAX_SPEED
        vy  = float(vel[1]) * MAX_SPEED

        new_x = np.clip(
            self._drone_pos[0] + vx * dt,
            -self._half + DRONE_RADIUS, self._half - DRONE_RADIUS)
        new_y = np.clip(
            self._drone_pos[1] + vy * dt,
            -self._half + DRONE_RADIUS, self._half - DRONE_RADIUS)

        p.resetBasePositionAndOrientation(
            self._drone_id, [new_x, new_y, DRONE_HEIGHT], [0, 0, 0, 1],
            physicsClientId=self._pc)
        self._drone_vel = np.array([vx, vy, 0.0])

    def _update_drone_state(self):
        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._pc)
        self._drone_pos = np.array(pos)

    def _reset_drone_to(self, pos):
        p.resetBasePositionAndOrientation(
            self._drone_id, pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self._pc)
        self._drone_pos = pos.copy()
        self._drone_vel[:] = 0.0

    def _move_enemies(self):
        dirs = np.array([[0,1],[0,-1],[1,0],[-1,0],[0,0]], dtype=float)
        dt   = 2.0 / CTRL_FREQ
        for i in range(NUM_ENEMIES):
            ex, ey = self._enemy_positions[i, :2]
            d  = dirs[self.rng.integers(len(dirs))]
            nx = np.clip(ex + d[0] * ENEMY_SPEED * dt,
                         -self._half + ENEMY_RADIUS, self._half - ENEMY_RADIUS)
            ny = np.clip(ey + d[1] * ENEMY_SPEED * dt,
                         -self._half + ENEMY_RADIUS, self._half - ENEMY_RADIUS)
            gi = int((nx + self._half) / CELL_SIZE)
            gj = int((ny + self._half) / CELL_SIZE)
            if 0 <= gi < GRID_SIZE and 0 <= gj < GRID_SIZE and not self._obstacle_grid[gi, gj]:
                self._enemy_positions[i, 0] = nx
                self._enemy_positions[i, 1] = ny
            p.resetBasePositionAndOrientation(
                self._enemy_ids[i],
                [self._enemy_positions[i, 0], self._enemy_positions[i, 1], DRONE_HEIGHT],
                [0, 0, 0, 1], physicsClientId=self._pc)


    def _check_obstacle_collision(self):
        x, y = self._drone_pos[0], self._drone_pos[1]
        half = self._half
        for dx, dy in [(0, 0), (DRONE_RADIUS, 0), (-DRONE_RADIUS, 0),
                       (0, DRONE_RADIUS), (0, -DRONE_RADIUS)]:
            gi = int((x + dx + half) / CELL_SIZE)
            gj = int((y + dy + half) / CELL_SIZE)
            if 0 <= gi < GRID_SIZE and 0 <= gj < GRID_SIZE and self._obstacle_grid[gi, gj]:
                return True
        return False

    def _min_enemy_dist(self):
        if NUM_ENEMIES == 0:
            return float('inf')
        dists = np.linalg.norm(
            self._enemy_positions[:, :2] - self._drone_pos[:2], axis=1)
        return float(np.min(dists))

    def _cast_lidar(self):
        readings = np.ones(LIDAR_RAYS, dtype=np.float32)
        angles   = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False)
        ox, oy, oz = self._drone_pos
        ray_from = [[ox, oy, oz]] * LIDAR_RAYS
        ray_to   = [
            [ox + np.cos(a) * LIDAR_RANGE,
             oy + np.sin(a) * LIDAR_RANGE, oz]
            for a in angles
        ]
        for i, res in enumerate(
                p.rayTestBatch(ray_from, ray_to, physicsClientId=self._pc)):
            readings[i] = res[2]
        return readings


    @property
    def drone_pos(self):
        return self._drone_pos.copy()

    @property
    def enemy_positions(self):
        return self._enemy_positions.copy()

    @property
    def target_positions(self):
        return self._target_positions.copy()

    @property
    def active_target(self):
        return self._active_target

    @property
    def obstacle_grid(self):
        return self._obstacle_grid.copy()
