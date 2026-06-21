<p align="right">
  🇺🇸 English &nbsp;|&nbsp; <a href="README.tr.md">🇹🇷 Türkçe</a>
</p>

# IHA Rota — 3D Autonomous UAV Navigation with Reinforcement Learning

REINFORCE (Monte Carlo Policy Gradient) algorithm trained in PyBullet physics simulation to navigate a 3D environment, evade enemy agents, and reach sequential targets.

---

## Demo

![Simülasyon Ortamı](assets/simulation.png)

20×20 meter 3D arena with:
- 10 randomly placed **obstacles** (boxes + cylinders)
- 2 **enemy agents** (active pursuit)
- 3 **sequential targets** (must be reached in order)
- 16-ray **LiDAR** sensor

---

## Results

### Training Records (v1 → v6)

![Training Records](assets/training_records.png)

### Success Rate Comparison

![Success Rate](assets/success_rates.png)

### Final Model (v6) — 100,000 Episode Test

| Metric | Value |
|---|---|
| Average Reward | 1834.09 |
| Max Reward | 2298.95 |
| Best Training Score | +1826.63 |
| **Success Rate** | **98.1%** |
| Timeout Rate | 1.9% |
| Total Episodes Evaluated | 100,000 |

The v6 model was trained for ~13,700 episodes on top of v5, reaching a new all-time best of **+1826.63** at episode 8,300, then plateaued. Two further attempts (v7 with lower learning rates) failed to surpass this score, confirming v6 as the final model.

### Reward Distribution (100K episodes)

![Test Distribution](assets/test_distribution.png)

### Example Navigation Path

![Navigation Path](assets/navigation_path.png)

---

## Policy Network Architecture

![Policy Network](assets/policy_network.png)

- **Input:** 28-dim (drone pos/vel, relative target vector, relative enemy vector, 16× LiDAR)
- **Hidden layers:** 256 → 256 (ReLU)
- **Output:** 9 discrete actions (Softmax)
- **Total parameters:** ~75,500

---

## Installation

```bash
conda create -n iha_env python=3.10
conda activate iha_env
pip install pybullet gymnasium torch numpy matplotlib
```

---

## Usage

### Train

```bash
python train.py
```

Version management is done via `VERSION` and `BASE_VERSION` in `config.py`.  
Model is saved to `sonuc/vX_iha_beyni.pth` when a new best score is reached.

### Test (visual, 5 episodes)

```bash
python test.py
```

### Long test (100 episodes, with stats)

```bash
python uzun_test.py
```

### Stress test (30 obstacles, hard scenario)

```bash
python test_zor.py
```

### Train on Kaggle

Open `kaggle_train.ipynb` as a Kaggle notebook. CPU mode is recommended (faster than GPU for this setup).

---

## Project Structure

```
iha_rota/
├── config.py              # All hyperparameters and environment settings
├── train.py               # REINFORCE training loop
├── test.py                # 5-episode visual test
├── uzun_test.py           # 100-episode long test + statistics
├── test_zor.py            # 30-obstacle stress test
├── kaggle_train.ipynb     # Kaggle training notebook
├── env/
│   ├── iha_env.py         # Main Gymnasium environment
│   ├── obstacles.py       # Obstacle placement
│   ├── enemies.py         # Enemy agent behavior
│   └── boundary_walls.py  # Arena boundary walls
├── network/
│   └── policy.py          # IhaPolicy network (28→256→256→9)
├── visualization/
│   └── viz_2d.py          # Real-time 2D top-down visualization
├── sonuc/                 # Trained models (v1–v6)
└── assets/                # README figures
```

---

## Algorithm

**REINFORCE** (Williams, 1992) — Monte Carlo Policy Gradient

- Full return computed at end of each episode (γ = 0.99)
- Normalized returns with baseline for variance reduction
- Entropy bonus for exploration (v1–v5: 0.05, v6: 0.02)

---

## Training History

| Version | Base | Learning Rate | Entropy | Best Score |
|---|---|---|---|---|
| v1 | scratch | 2×10⁻⁴ | 0.05 | +1018.20 |
| v2 | v1 | 2×10⁻⁴ | 0.05 | +1167.07 |
| v3 | v2 | 2×10⁻⁴ | 0.05 | +1292.85 |
| v4 | v3 | 2×10⁻⁴ | 0.05 | +1437.40 |
| v5 | v4 | 2×10⁻⁴ | 0.05 | +1446.22 |
| **v6** | v5 | **1×10⁻⁴** | **0.02** | **+1826.63** ✓ |

~65,000 total episodes trained on Kaggle (CPU).

---

## Requirements

- Python 3.10
- PyTorch ≥ 1.13
- PyBullet ≥ 3.2.5
- Gymnasium ≥ 0.26
- NumPy, Matplotlib

---

## License

[MIT](LICENSE)
