"""
check_model_dataset.py
Verifica que el MODELO y el DATASET FINAL son compatibles y predice 1 ejemplo.

Qué hace:
- Carga el parquet final (train_final_advanced_features.parquet)
- Carga el modelo (.pkl)
- Detecta columnas de features: dataset sin TARGET y sin SK_ID_CURR
- Prueba predict_proba() con una fila real
- Si hay mismatch, imprime columnas faltantes / extra
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import pandas as pd


# =========================
# CONFIG (ajusta solo si cambias rutas/nombres)
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "assets" / "data" / "train_final_advanced_features.parquet"

# Cambia este nombre si quieres probar CatBoost:
MODEL_PATH = BASE_DIR / "assets" / "models" / "catboost_new_dataset.pkl"
# MODEL_PATH = BASE_DIR / "assets" / "models" / "catboost_new_dataset.pkl"
# MODEL_PATH = BASE_DIR / "assets" / "models" / "catboost_best_scores.pkl"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> int:
    print("=== CHECK: dataset + modelo ===")
    print(f"Dataset: {DATA_PATH}")
    print(f"Modelo:  {MODEL_PATH}")

    if not DATA_PATH.exists():
        print(f"❌ No existe el dataset: {DATA_PATH}")
        return 1

    if not MODEL_PATH.exists():
        print(f"❌ No existe el modelo: {MODEL_PATH}")
        return 1

    # 1) Cargar dataset final
    df = pd.read_parquet(DATA_PATH)
    print(f"\n✅ Dataset cargado: {df.shape[0]:,} filas | {df.shape[1]:,} columnas")

    if ID_COL not in df.columns:
        print(f"❌ El dataset NO tiene {ID_COL}. Columnas disponibles: {list(df.columns)[:20]} ...")
        return 1

    # 2) Preparar features: quitar TARGET si existe y quitar ID
    df_feat = df.copy()
    if TARGET_COL in df_feat.columns:
        df_feat = df_feat.drop(columns=[TARGET_COL])
        print("ℹ️ TARGET detectado y eliminado para inferencia.")

    feature_cols = [c for c in df_feat.columns if c != ID_COL]
    print(f"✅ Features detectadas: {len(feature_cols):,}")

    # 3) Cargar modelo
    model = load_pickle(MODEL_PATH)
    print(f"\n✅ Modelo cargado: {type(model)}")

    # 4) Elegir una fila real
    # Tomamos la primera fila del dataset (existe seguro)
    row = df_feat.iloc[[0]].copy()
    sk_id_curr = int(row[ID_COL].iloc[0])
    X = row[feature_cols]

    # 5) Check de columnas vs modelo (si el modelo expone feature_names)
    model_feature_names = None
    for attr in ("feature_names_in_", "feature_names", "feature_name_"):
        if hasattr(model, attr):
            try:
                model_feature_names = list(getattr(model, attr))
                break
            except Exception:
                pass

    if model_feature_names is not None:
        ds_set = set(feature_cols)
        m_set = set(model_feature_names)

        missing_in_ds = sorted(list(m_set - ds_set))
        extra_in_ds = sorted(list(ds_set - m_set))

        print("\n=== CHECK columnas (modelo vs dataset) ===")
        print(f"Modelo features:  {len(model_feature_names):,}")
        print(f"Dataset features: {len(feature_cols):,}")

        if missing_in_ds:
            print(f"❌ Faltan en dataset (pero el modelo las espera): {len(missing_in_ds)}")
            print("   Ejemplo:", missing_in_ds[:20])
        if extra_in_ds:
            print(f"⚠️ Sobran en dataset (modelo no las usa): {len(extra_in_ds)}")
            print("   Ejemplo:", extra_in_ds[:20])

        if missing_in_ds:
            print("\n🚫 Con este mismatch NO va a predecir bien. Hay que alinear columnas.")
            return 2

        # Reordenar X al orden del modelo si procede
        X = X[model_feature_names]
        print("✅ Columnas alineadas al orden del modelo.")

    # 6) Predicción
    if not hasattr(model, "predict_proba"):
        print("❌ El modelo NO tiene predict_proba(). No puedo sacar probas.")
        return 3

    try:
        proba = model.predict_proba(X)
        p_default = float(proba[0][1])
    except Exception as e:
        print("❌ Error en predict_proba():", str(e))
        print("\nTip: casi siempre es mismatch de columnas/dtypes.")
        return 4

    print("\n=== RESULTADO ===")
    print(f"SK_ID_CURR usado: {sk_id_curr}")
    print(f"P(default=1): {p_default:.6f}")
    print("✅ OK: modelo y dataset predicen 1 fila sin reventar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
