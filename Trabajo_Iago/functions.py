"""
function.py

Funciones de negocio para scoring + explicación (SHAP) para Credit App.

Objetivo:
- Buscar cliente por SK_ID_CURR en el dataset "ready"
- Sobrescribir inputs del frontend (AMT_CREDIT y NAME_CONTRACT_TYPE)
- Limpiar NaN/Inf y preparar categóricas
- Predecir probabilidad (predict_proba)
- Calcular SHAP (TreeExplainer) para top factores

Diseñado para integrarse con FastAPI y tu HTML:
- El frontend llama POST /api/score
- FastAPI usa: search_user() -> predict() -> explain()
- FastAPI devuelve JSON con proba.default + shap.top_risk_increasing/decreasing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import Pool


# ============================================================
# Paths robustos (NO dependes del working directory)
# Ajusta PROJECT_ROOT si tu repo está en otra forma.
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# Caso típico: .../credit_app/backend/function.py
# y data/models están en raíz del proyecto:
PROJECT_ROOT = BASE_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "home_credit_train_ready.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_new_dataset.pkl"


# ============================================================
# Cache global (se cargan 1 vez)
# ============================================================
_DB: pd.DataFrame | None = None
_MODEL: Any | None = None


def get_db() -> pd.DataFrame:
    """
    Carga el dataset una sola vez en memoria (cache).
    """
    global _DB
    if _DB is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset no encontrado en: {DATA_PATH}")
        _DB = pd.read_csv(DATA_PATH)
    return _DB


def get_model() -> Any:
    """
    Carga el modelo una sola vez en memoria (cache).
    """
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def _safe_value(v: Any) -> Any:
    """
    Convierte valores pandas/numpy a tipos serializables en JSON.
    - NaN -> None
    - numpy scalar -> python scalar
    """
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, (np.integer, np.floating)):
        return v.item()

    return v


# ============================================================
# Core functions (misma lógica que tu versión)
# ============================================================
def search_user(sk_id: int, ammount: float, credit_type: str) -> pd.DataFrame:
    """
    Busca un usuario por SK_ID_CURR en el dataset "ready" y construye
    el vector de features (1 fila) sobrescribiendo los inputs del frontend.

    Nota:
    - Retorna DataFrame vacío si no existe el SK_ID_CURR.
    - No lanza HTTP errors aquí: eso lo maneja FastAPI.
    """
    db = get_db()

    user = db.loc[db["SK_ID_CURR"] == sk_id].copy()
    if user.empty:
        return user  # DataFrame vacío

    # Sobrescribir inputs de la solicitud
    user["AMT_CREDIT"] = float(ammount)
    user["NAME_CONTRACT_TYPE"] = str(credit_type)

    # Limpieza numérica
    user.replace([np.inf, -np.inf], np.nan, inplace=True)
    user.fillna(0, inplace=True)

    # Preparación de categóricas (misma lógica que tu código)
    obj_cols = user.select_dtypes("object").columns
    for col in obj_cols:
        user[col] = user[col].replace(0, "missing")
    for col in obj_cols:
        user[col] = user[col].astype("category")

    # Drop columnas no usadas por el modelo (si existen)
    drop_cols = [c for c in ["TARGET", "SK_ID_CURR"] if c in user.columns]
    user = user.drop(columns=drop_cols)

    return user


def predict(user: pd.DataFrame) -> np.ndarray:
    """
    Predicción compatible con XGBoost y CatBoost.
    Retorna predict_proba shape (1, 2): [P(class0), P(class1)].
    """
    model = get_model()
    model_type = type(model).__name__

    # Intento de forzar CPU si fuera XGBoost (sin romper)
    if "XGB" in model_type or "Booster" in model_type:
        try:
            model.set_params(device="cpu", eval_metric=None)
        except Exception:
            try:
                model.set_params(device="cpu")
            except Exception:
                pass

    # CatBoost puede predecir directamente con DataFrame (categorías tipo category)
    # Tu código declaraba Pool pero no lo usaba en predict(); lo mantenemos igual.
    pred = model.predict_proba(user)
    return np.asarray(pred)


def explain(user_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explicación SHAP compatible con XGBoost y CatBoost.

    Retorna:
      top_bad  (shap alto positivo) -> aumenta riesgo
      top_good (shap negativo)      -> baja riesgo
    """
    try:
        model = get_model()
        model_type = type(model).__name__

        # XGBoost: codificar category -> codes
        if "XGB" in model_type or "Booster" in model_type:
            try:
                model.set_params(device="cpu", eval_metric=None)
            except Exception:
                try:
                    model.set_params(device="cpu")
                except Exception:
                    pass

            user_encoded = user_df.copy()
            cat_cols = user_encoded.select_dtypes("category").columns
            for col in cat_cols:
                user_encoded[col] = user_encoded[col].cat.codes

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_encoded)

        # CatBoost: usar Pool con cat_features
        elif "CatBoost" in model_type:
            cat_features = user_df.select_dtypes("category").columns.tolist()
            pool = Pool(user_df, cat_features=cat_features)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pool)

        # Fallback genérico
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_df)

        # Clasificación binaria: shap puede venir como list [clase0, clase1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # clase 1 (Default)

        shap_row = shap_values[0]  # solo 1 cliente

        result = pd.DataFrame(
            {
                "feature": user_df.columns,
                "value": user_df.iloc[0].values,
                "shap": shap_row,
            }
        )

        result.replace([np.inf, -np.inf], 0, inplace=True)
        result.fillna(0, inplace=True)

        top_bad = result.sort_values("shap", ascending=False).head(10)
        top_good = result.sort_values("shap", ascending=True).head(10)

        return top_bad, top_good

    except Exception as e:
        # En producción, loguearías con logging. Aquí lo dejamos claro.
        print(f"[function.explain] Error: {e}")
        empty = pd.DataFrame(columns=["feature", "value", "shap"])
        return empty, empty


# ============================================================
# Helper extra: formato listo para el HTML
# (FastAPI puede llamar esto y devolverlo tal cual)
# ============================================================
def format_shap_for_frontend(top_bad: pd.DataFrame, top_good: pd.DataFrame) -> Dict[str, Any]:
    """
    Convierte los DataFrames de SHAP a la estructura exacta que tu HTML usa:

    {
      "enabled": True,
      "top_risk_increasing": [{"feature","value","shap"}, ...],
      "top_risk_decreasing": [{"feature","value","shap"}, ...]
    }
    """

    def pack(df: pd.DataFrame) -> List[dict]:
        if df is None or df.empty:
            return []
        out: List[dict] = []
        for _, r in df.iterrows():
            out.append(
                {
                    "feature": str(r["feature"]),
                    "value": _safe_value(r["value"]),
                    "shap": float(r["shap"]),
                }
            )
        return out

    return {
        "enabled": True,
        "top_risk_increasing": pack(top_bad),
        "top_risk_decreasing": pack(top_good),
    }
