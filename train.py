import os
import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.iha_env import IhaEnv
from network.policy import IhaPolicy
from visualization.viz_2d import TopDownViz
from config import (
    LR, GAMMA, ENTROPY_COEF, TOTAL_EPISODES,
    MAP_RESET_EVERY, SAVE_EVERY, MODEL_PATH, BASE_PATH, VERSION,
)


def main(visualize=True):
    os.makedirs("sonuc", exist_ok=True)
    render_ep   = 50
    env         = IhaEnv(render_mode="headless")
    policy      = IhaPolicy()
    optimizer   = optim.Adam(policy.parameters(), lr=LR)
    viz         = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    print(f"Cihaz: {device} | Versiyon: {VERSION}")

    en_iyi_ortalama = float('-inf')

    def _load(path, label):
        nonlocal en_iyi_ortalama
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            policy.load_state_dict(ckpt["model_state"])
            en_iyi_ortalama = ckpt.get("max_skor", float('-inf'))
            print(f"{label} '{path}' yüklendi. Rekor: {en_iyi_ortalama:.2f}")
        else:
            policy.load_state_dict(ckpt)
            print(f"{label} '{path}' yüklendi.")

    if os.path.exists(MODEL_PATH):
        _load(MODEL_PATH, f"[{VERSION}]")
    elif BASE_PATH and os.path.exists(BASE_PATH):
        _load(BASE_PATH, f"[Baz model]")
        print(f"Yeni versiyon ({VERSION}) baz modelden başlıyor — baz rekor korundu: {en_iyi_ortalama:.2f}")
    else:
        print(f"Kayıtlı model bulunamadı. [{VERSION}] sıfırdan başlıyor.")

    print("IHA Rota Eğitimi Başlıyor")
    son_100_odul    = []

    obs, _ = env.reset_map()

    for ep in range(TOTAL_EPISODES):

        if ep > 0 and ep % MAP_RESET_EVERY == 0:
            obs, _ = env.reset_map()
        else:
            obs, _ = env.reset()

        log_probs, rewards, entropies = [], [], []
        toplam_odul = 0.0
        render_this = visualize and (ep % render_ep == 0)

        if render_this and viz is None:
            viz = TopDownViz()
        elif not render_this and viz is not None:
            viz.close()
            viz = None

        step = 0
        while True:
            st    = torch.FloatTensor(obs).to(device)
            probs = policy(st)
            m     = Categorical(probs)
            action = m.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            entropies.append(m.entropy())
            toplam_odul += reward
            step += 1

            if render_this and viz is not None:
                viz.update(
                    env.obstacle_grid, env.drone_pos,
                    env.enemy_positions, env.target_positions,
                    env.active_target, ep, step, toplam_odul)

            if done:
                break

        son_100_odul.append(toplam_odul)
        if len(son_100_odul) > 100:
            son_100_odul.pop(0)

        R       = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss   = -(torch.stack(log_probs) * returns).sum()
        entropy_bonus = ENTROPY_COEF * torch.stack(entropies).sum()
        loss          = policy_loss - entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        if ep > 0 and ep % SAVE_EVERY == 0:
            guncel_ort = np.mean(son_100_odul)
            print(f"Sorti {ep:>6} | Son 100 Ort: {guncel_ort:>8.2f} | Adım: {step}" )
            if guncel_ort > en_iyi_ortalama:
                en_iyi_ortalama = guncel_ort
                torch.save({"model_state": policy.state_dict(), "max_skor": float(en_iyi_ortalama)}, MODEL_PATH)
                print(f"YENİ REKOR (Ort: {en_iyi_ortalama:.2f}) — model kaydedildi!" )

    if viz is not None:
        viz.close()
    env.close()


if __name__ == "__main__":
    main(visualize=True)
