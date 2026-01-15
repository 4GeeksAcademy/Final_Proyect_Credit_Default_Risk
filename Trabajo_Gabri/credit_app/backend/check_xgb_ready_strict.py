"""
check_xgb_ready_strict.py
Check blindado:
- Verifica que el modelo cargado es XGBClassifier
- Verifica que X tiene features
- Convierte categoricas a CODES (ordinal simple) SOLO para probar predict_proba
  (esto NO es el encoding final de prod; es para validar que el modelo puede correr)
"""

from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "assets" / "data" / "home_credit_train_ready.csv"
MODEL_PATH = BASE_DIR / "assets" / "models" / "xgb_new_dataset.pkl"  # <- DEBE ser este

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def main():
    print("=== STRICT CHECK (XGB + ready CSV) ===")
    print("Dataset:", DATA_PATH)
    print("Model  :", MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset shape: {df.shape}")

    model = load_pickle(MODEL_PATH)
    print("✅ Loaded model type:", type(model))

    # 1) Confirmar que es XGB
    if "xgboost" not in str(type(model)).lower():
        raise SystemExit("❌ Estás cargando un modelo que NO es XGBoost. Revisa MODEL_PATH.")

    # 2) Construir X
    if ID_COL not in df.columns or TARGET_COL not in df.columns:
        raise SystemExit("❌ El CSV no tiene SK_ID_CURR/TARGET, no es 'ready'.")

    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    if len(feature_cols) == 0:
        raise SystemExit("❌ No hay features en el dataset después de quitar ID/TARGET.")

    X = df.iloc[[0]][feature_cols].copy()
    print("✅ Initial X shape:", X.shape)

    # 3) Alinear orden al modelo
    model_feats = list(getattr(model, "feature_names_in_", []))
    if not model_feats:
        raise SystemExit("❌ El modelo no expone feature_names_in_.")

    X = X[model_feats]
    print("✅ Aligned X shape:", X.shape)

    # 4) Convertir categoricas a códigos (ordinal simple) solo para probar
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print("ℹ️ object columns:", len(obj_cols))
    for c in obj_cols:
        X[c] = X[c].astype("category").cat.codes.astype("int32")

    # 5) Predicción
    proba = model.predict_proba(X)
    p_default = float(proba[0][1])
    print("✅ OK predict_proba")
    print("P(default=1):", p_default)

if __name__ == "__main__":
    main()
