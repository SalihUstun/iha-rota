import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.distributions import Categorical
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.iha_env import IhaEnv
from network.policy import IhaPolicy
from config import MODEL_PATH, MAX_STEPS

TOPLAM_GOREV  = 100_000
HARITA_DEGIS  = 5   


def uzun_test():
    env    = IhaEnv(render_mode=None)
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

    skorlar     = []
    basarili    = 0
    zaman_asimi = 0

    for gorev in range(TOPLAM_GOREV):
        harita_no = gorev // HARITA_DEGIS + 1

        if gorev % HARITA_DEGIS == 0 and gorev > 0:
            obs, _ = env.reset_map()
        else:
            obs, _ = env.reset()

        done        = False
        toplam_odul = 0.0
        gecmis      = []
        step        = 0

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
            done         = terminated or truncated
            toplam_odul += reward
            step        += 1

        skorlar.append(toplam_odul)

        if step >= MAX_STEPS:
            zaman_asimi += 1
        else:
            basarili += 1

        if (gorev + 1) % 1000 == 0:
            print(f"  {gorev + 1}/{TOPLAM_GOREV} tamamlandı — Başarı: %{basarili / (gorev + 1) * 100:.1f}")

    print("\n" + "=" * 50)
    print(f"  UZUN TEST SONUÇLARI ({TOPLAM_GOREV} Görev)")
    print("=" * 50)
    print(f"  Ortalama Puan : {np.mean(skorlar):.2f}")
    print(f"  En Yüksek     : {max(skorlar):.2f}")
    print(f"  En Düşük      : {min(skorlar):.2f}")
    print(f"  Başarılı      : {basarili}/{TOPLAM_GOREV}  (%{basarili / TOPLAM_GOREV * 100:.1f})")
    print(f"  Zaman Aşımı   : {zaman_asimi}/{TOPLAM_GOREV}  (%{zaman_asimi / TOPLAM_GOREV * 100:.1f})")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    uzun_test()
