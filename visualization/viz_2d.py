import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from config import AREA_SIZE, GRID_SIZE, VIZ_WINDOW_SIZE, VIZ_UPDATE_FREQ, NUM_ENEMIES, NUM_TARGETS


class TopDownViz:

    def __init__(self):
        plt.ion()
        fig_size = VIZ_WINDOW_SIZE / 100
        self.fig, self.ax = plt.subplots(figsize=(fig_size, fig_size))
        self.fig.patch.set_facecolor("#1a1a2e")
        self.ax.set_facecolor("#1a1a2e")
        self.fig.canvas.manager.set_window_title("IHA Rota — Kuşbakışı Harita")

        self._map_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        cmap = ListedColormap(["#2d2d44", "#7a5c3a"])

        half = AREA_SIZE / 2.0
        self.img = self.ax.imshow(
            self._map_data.T,
            origin="lower",
            extent=[-half, half, -half, half],
            cmap=cmap, vmin=0, vmax=1,
            interpolation="nearest",
        )

        # Drone 
        self._drone_sc = self.ax.scatter(
            [], [], s=180, marker="*", zorder=6,
            color="#00aaff", edgecolors="white", linewidths=1.2, label="İHA")

        # Düşmanlar 
        self._enemy_sc = self.ax.scatter(
            [], [], s=100, marker="^", zorder=5,
            color="#ff2222", edgecolors="white", linewidths=0.8, label="Düşman")

        # Aktif hedef 
        self._target_active_sc = self.ax.scatter(
            [], [], s=140, marker="s", zorder=5,
            color="#00dd00", edgecolors="white", linewidths=1.0, label="Aktif Hedef")

        # Bekleyen hedefler 
        self._target_passive_sc = self.ax.scatter(
            [], [], s=80, marker="s", zorder=4,
            color="#559955", edgecolors="white", linewidths=0.6, label="Bekleyen Hedef")

        self.ax.set_xlim(-half, half)
        self.ax.set_ylim(-half, half)
        self.ax.set_xlabel("X (m)", color="white", fontsize=9)
        self.ax.set_ylabel("Y (m)", color="white", fontsize=9)
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#555577")

        border = plt.Rectangle(
            (-half, -half), AREA_SIZE, AREA_SIZE,
            linewidth=2, edgecolor="#8888ff", facecolor="none", linestyle="--")
        self.ax.add_patch(border)

        self.ax.legend(
            loc="upper right", fontsize=7,
            facecolor="#2a2a3e", labelcolor="white", framealpha=0.85)

        self.title_text = self.ax.set_title(
            "Sorti: 0  |  Adım: 0  |  Puan: 0.0",
            color="white", fontsize=10, pad=8)

        self.fig.tight_layout()

    def update(self, obstacle_grid, drone_pos, enemy_positions,
               target_positions, active_target, episode, step, score, force=False):

        if not force and step % VIZ_UPDATE_FREQ != 0:
            return

        # Engel haritası
        self._map_data[:] = 0.0
        self._map_data[obstacle_grid] = 1.0
        self.img.set_data(self._map_data.T)

        # Drone
        self._drone_sc.set_offsets([[drone_pos[0], drone_pos[1]]])

        # Düşmanlar
        if len(enemy_positions) > 0:
            self._enemy_sc.set_offsets(enemy_positions[:, :2])
        else:
            self._enemy_sc.set_offsets(np.empty((0, 2)))

        # Hedefler
        if active_target < len(target_positions):
            ap = target_positions[active_target]
            self._target_active_sc.set_offsets([[ap[0], ap[1]]])
        else:
            self._target_active_sc.set_offsets(np.empty((0, 2)))

        passive = [target_positions[i] for i in range(len(target_positions))
                   if i != active_target]
        if passive:
            pp = np.array(passive)
            self._target_passive_sc.set_offsets(pp[:, :2])
        else:
            self._target_passive_sc.set_offsets(np.empty((0, 2)))

        self.title_text.set_text(
            f"Sorti: {episode}  |  Adım: {step}  |  Puan: {score:.1f}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
