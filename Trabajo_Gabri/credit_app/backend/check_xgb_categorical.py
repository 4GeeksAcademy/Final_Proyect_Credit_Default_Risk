"""
check_xgb_categorical.py
Prueba si el XGBClassifier funciona convirtiendo objetos a category
y habilitando enable_categorical.
"""

from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "assets" / "data" / "home_credit_train_ready.csv"
MODEL_PATH = BASE_DIR / "assets" / "models" / "catboost_new_dataset.pkl"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def main():
    df = pd.read_csv(DATA_PATH)
    model = load_pickle(MODEL_PATH)

    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    model_feats = list(getattr(model, "feature_names_in_", []))
    X = df.iloc[[0]][feature_cols].copy()

    # Alinear orden
    X = X[model_feats]

    # Convertir object -> category
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        X[c] = X[c].astype("category")

    # Intentar activar enable_categorical (si existe)
    try:
        model.set_params(enable_categorical=True)
        print("✅ set_params(enable_categorical=True) aplicado.")
    except Exception as e:
        print("⚠️ No se pudo setear enable_categorical:", e)

    # Predicción
    proba = model.predict_proba(X)
    print("✅ OK predict_proba")
    print("P(default=1):", float(proba[0][1]))

if __name__ == "__main__":
    main()
