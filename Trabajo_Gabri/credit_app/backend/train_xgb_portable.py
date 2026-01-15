"""
train_xgb_portable.py
Entrena un XGBoost "rápido" con home_credit_train_ready.csv y guarda modelo portable.

Outputs:
- assets/models/xgb_model.json   (portable)
- assets/models/xgb_model.joblib (para sklearn wrapper)
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "assets" / "data" / "home_credit_train_ready.csv"
OUT_DIR = BASE_DIR / "assets" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

def main():
    df = pd.read_csv(DATA_PATH)

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Mantén ID fuera de features
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    # Encoding mínimo estable: object -> category codes
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category").cat.codes.astype("int32")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    p = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p)
    print(f"✅ AUC validación: {auc:.5f}")

    # Guardar portable
    json_path = OUT_DIR / "xgb_model.json"
    model.get_booster().save_model(str(json_path))
    print("✅ Guardado:", json_path)

    # Guardar wrapper (opcional)
    joblib_path = OUT_DIR / "xgb_model.joblib"
    joblib.dump(model, joblib_path)
    print("✅ Guardado:", joblib_path)

if __name__ == "__main__":
    main()
