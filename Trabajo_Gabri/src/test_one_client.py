from src.features.load_data import load_data
from src.features.build_features import build_features

data = load_data()

# ==============================
# IDs elegidas a mano
# ==============================

GOOD_ID  = 265600   # cliente bueno (ejemplo)
BAD_ID   = 100234   # cliente malo
HEAVY_ID = 265681   # cliente con muchos préstamos previos

print("\n==============================")
print("CLIENTE BUENO")
print("==============================")
print(build_features(GOOD_ID, data).to_string(index=True))

print("\n==============================")
print("CLIENTE MALO")
print("==============================")
print(build_features(BAD_ID, data).to_string(index=True))

print("\n==============================")
print("CLIENTE HEAVY USER")
print("==============================")
print(build_features(HEAVY_ID, data).to_string(index=True))