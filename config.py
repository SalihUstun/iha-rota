
# ALAN
AREA_SIZE       = 20.0                          # 20x20 metre
CELL_SIZE       = 0.5                           # Grid hücre boyutu 
GRID_SIZE       = int(AREA_SIZE / CELL_SIZE)    # 40x40


# SİMÜLASYON
SIM_FREQ        = 240           # Fizik adım frekansı 
CTRL_FREQ       = 30            # Kontrol frekansı 
MAX_STEPS       = 3000


# DRONE

DRONE_HEIGHT    = 1.5           # uçuş yüksekliği 
DRONE_RADIUS    = 0.15          # Çarpışma yarıçapı 
SPAWN_X         = -8.0          # Başlangıç X 
SPAWN_Y         = -8.0          # Başlangıç Y
SAFE_ZONE_RADIUS = 2.5          # Spawn çevresinde engel/düşman yok

# HAREKET — 9 YÖN 
MAX_SPEED       = 2.0           
ACTION_DIM      = 9


# SINIR DUVARLARI
WALL_THICKNESS  = 0.3
WALL_HEIGHT     = 3.0
WALL_SENSOR_DIST = 1.5


# ENGELLER 
OBSTACLE_CONFIG = [
    {"type": "box",      "size": [0.8, 0.8, 1.2], "color": [0.55, 0.27, 0.07, 1]},
    {"type": "box",      "size": [1.0, 0.6, 1.0], "color": [0.55, 0.27, 0.07, 1]},
    {"type": "box",      "size": [0.6, 1.0, 1.4], "color": [0.55, 0.27, 0.07, 1]},
    {"type": "box",      "size": [1.2, 0.5, 0.9], "color": [0.50, 0.50, 0.50, 1]},
    {"type": "box",      "size": [0.5, 1.1, 1.1], "color": [0.50, 0.50, 0.50, 1]},
    {"type": "box",      "size": [0.9, 0.7, 0.8], "color": [0.60, 0.40, 0.10, 1]},
    {"type": "cylinder", "size": [0.7, 1.8],       "color": [0.20, 0.60, 0.20, 1]},
    {"type": "cylinder", "size": [0.8, 2.0],       "color": [0.20, 0.60, 0.20, 1]},
    {"type": "cylinder", "size": [0.6, 1.5],       "color": [0.15, 0.50, 0.15, 1]},
    {"type": "cylinder", "size": [0.9, 2.2],       "color": [0.10, 0.45, 0.10, 1]},
]


OBSTACLE_MIN_SPACING   = 0.8    # Engeller arası minimum boşluk 
OBSTACLE_WALL_MARGIN   = 1.0    # Duvara minimum mesafe 
OBSTACLE_SPAWN_MARGIN  = 3.0    # Drone spawn bölgesine mesafe 
MAX_PLACEMENT_TRIES    = 500


# LiDAR
LIDAR_RAYS      = 16
LIDAR_RANGE     = 5.0           # metre


# DÜŞMANLAR 
NUM_ENEMIES     = 2
ENEMY_RADIUS    = 0.2
ENEMY_SPEED     = 0.8           # m/s
ENEMY_NEAR_DIST = 1.2           # Bu mesafede ceza


# HEDEFLER 
NUM_TARGETS     = 3
TARGET_RADIUS   = 0.8           # Hedefe "ulaşma" eşiği (m)
TARGET_MIN_FROM_SPAWN = 4.0     # Spawn'dan minimum uzaklık (m)


# GÖZLEM BOYUTU
# drone_pos(3) + drone_vel(3) + hedef_pos(3) + yakin_dusman_pos(3) + lidar(16) = 28
OBS_DIM = 3 + 3 + 3 + 3 + LIDAR_RAYS


# ÖDÜLLER 
REWARD_STEP         = -0.1
REWARD_ENEMY_HIT    = -250.0
REWARD_OBSTACLE_HIT = -5.0
REWARD_NEAR_ENEMY   = -0.5
REWARD_TARGET_MID   = 400.0
REWARD_TARGET_FINAL = 1000.0
REWARD_DIST_SCALE   = 10.0


# VERSİYON — yeni oturum öncesi VERSION'ı artır, BASE_VERSION bir önceki yap
VERSION         = "v6"
BASE_VERSION    = "v5"

# EĞİTİM
LR              = 7e-5
GAMMA           = 0.99
ENTROPY_COEF    = 0.02
TOTAL_EPISODES  = 60000
MAP_RESET_EVERY = 10
SAVE_EVERY      = 100
MODEL_PATH      = f"sonuc/{VERSION}_iha_beyni.pth"
BASE_PATH       = f"sonuc/{BASE_VERSION}_iha_beyni.pth" if BASE_VERSION else None


# GÖRSELLEŞTİRME
VIZ_UPDATE_FREQ = 5
VIZ_WINDOW_SIZE = 600
