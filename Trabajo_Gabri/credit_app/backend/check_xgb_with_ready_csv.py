"""
check_xgb_with_ready_csv.py
Verifica si xgb_new_dataset.pkl encaja con home_credit_train_ready.csv
"""

from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "assets" / "data" / "home_credit_train_ready.csv"
MODEL_PATH = BASE_DIR / "assets" / "models" / "xgb_new_dataset.pkl"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def main():
    print("=== CHECK XGB vs READY CSV ===")
    print("Dataset:", DATA_PATH)
    print("Modelo:", MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas | {df.shape[1]:,} columnas")

    if TARGET_COL not in df.columns:
        raise SystemExit(f"❌ El dataset no tiene {TARGET_COL}. No parece training-ready.")
    if ID_COL not in df.columns:
        raise SystemExit(f"❌ El dataset no tiene {ID_COL}.")

    model = load_pickle(MODEL_PATH)
    print("✅ Modelo:", type(model))

    # features del dataset
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    X = df.iloc[[0]][feature_cols].copy()

    # features del modelo
    model_feats = list(getattr(model, "feature_names_in_", []))
    if not model_feats:
        raise SystemExit("❌ El modelo no expone feature_names_in_. (raro en sklearn wrapper)")

    ds_set = set(feature_cols)
    m_set = set(model_feats)

    missing_in_ds = sorted(list(m_set - ds_set))
    extra_in_ds = sorted(list(ds_set - m_set))

    print("\n=== CHECK columnas (modelo vs dataset) ===")
    print("Modelo features :", len(model_feats))
    print("Dataset features:", len(feature_cols))

    if missing_in_ds:
        print("❌ Faltan en dataset (modelo las espera):", len(missing_in_ds))
        print("   Ejemplo:", missing_in_ds[:25])
    if extra_in_ds:
        print("⚠️ Sobran en dataset (modelo no las usa):", len(extra_in_ds))
        print("   Ejemplo:", extra_in_ds[:25])

    if missing_in_ds:
        raise SystemExit("\n🚫 Sigue habiendo mismatch. Este CSV tampoco es el correcto.")

    # Alinear orden
    X = X[model_feats]

    # Predicción
    proba = model.predict_proba(X)
    p_default = float(proba[0][1])

    print("\n✅ OK: Predicción hecha")
    print("P(default=1):", p_default)

if __name__ == "__main__":
    main()
