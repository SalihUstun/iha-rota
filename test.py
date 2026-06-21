import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.distributions import Categorical
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.iha_env import IhaEnv
from network.policy import IhaPolicy
from visualization.viz_2d import TopDownViz
from config import MODEL_PATH, NUM_TARGETS, MAX_STEPS


def test():
    env    = IhaEnv(render_mode="human")
    policy = IhaPolicy()

    if not os.path.exists(MODEL_PATH):
        print(f"HATA: '{MODEL_PATH}' bulunamadı!")
        return

    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        policy.load_state_dict(checkpoint["model_state"])
        print(f"Model yüklendi: '{MODEL_PATH}' (Rekor: {checkpoint.get('max_skor', '?'):.2f})")
    else:
        policy.load_state_dict(checkpoint)
        print(f"Model yüklendi: '{MODEL_PATH}'")
    policy.eval()

    viz = TopDownViz()

    for gorev in range(5):
        print(f"\n Görev {gorev + 1} Başlıyor...")
        obs, _ = env.reset()
        done   = False
        toplam_odul = 0.0
        gecmis = []

        step = 0
        while not done:
            with torch.no_grad():
                st    = torch.FloatTensor(obs)
                probs = policy(st)

                konum = tuple(np.round(env.drone_pos[:2], 1))
                gecmis.append(konum)
                if len(gecmis) > 10:
                    gecmis.pop(0)

                if gecmis.count(konum) >= 4:
                    action = Categorical(probs).sample().item()
                else:
                    action = torch.argmax(probs).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            toplam_odul += reward
            step += 1

            viz.update(
                env.obstacle_grid, env.drone_pos,
                env.enemy_positions, env.target_positions,
                env.active_target, gorev + 1, step, toplam_odul)

        if step >= MAX_STEPS:
            print(f"Zaman aşımı. (Puan: {toplam_odul:.2f} | Adım: {step})")
        else:
            print(f"Görev başarılı! Hedef tamamlandı. (Puan: {toplam_odul:.2f} | Adım: {step})")

        import time
        time.sleep(1.5)

    print("\n Tüm test görevleri tamamlandı")
    viz.close()
    env.close()


if __name__ == "__main__":
    test()
