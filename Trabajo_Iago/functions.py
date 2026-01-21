"""
Trabajo_Iago/functions.py

Pipeline unificado para:
- Cliente existente (search_user)
- Nuevo cliente (new_user)
- Predicción (predict)
- Explicación SHAP (explain)
- Formateo para frontend (format_shap_for_frontend)

✅ Arreglos clave:
- Ya NO usa rutas relativas frágiles ('../...') por defecto.
- Expone get_model() y get_db() (cacheados).
- Permite que TU api.py sobreescriba DATA_PATH y MODEL_PATH:
    iago_fn.DATA_PATH = <Path a tu csv>
    iago_fn.MODEL_PATH = <Path a tu pkl>
    (y si quieres reset: iago_fn.get_model.cache_clear(); iago_fn.get_db.cache_clear())
- Limpieza robusta: inf/nan, strings numéricos, categories, 'missing'
- Compatible con CatBoost y XGBoost en predict() y explain()

⚠️ Nota realista:
- SHAP con CatBoost vía shap.TreeExplainer puede fallar según versión.
  Si falla, devolvemos data vacía (y tu API lo captura).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# CatBoost/SHAP son opcionales en runtime si solo quieres predict sin explain
try:
    from catboost import Pool  # type: ignore
except Exception:
    Pool = None  # type: ignore

try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore


# =============================================================================
# Config (TU API puede sobreescribir estas variables)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Por defecto: mismos paths que tu compañero solía usar, pero en Path absoluto
DATA_PATH: Path = (BASE_DIR / ".." / "data" / "processed" / "home_credit_train_ready.csv").resolve()
MODEL_PATH: Path = (BASE_DIR / ".." / "models" / "catboost_best_scores.pkl").resolve()

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# -----------------------------------------------------------------------------
# Helpers internos
# -----------------------------------------------------------------------------
def _is_cat_or_obj_dtype(s: pd.Series) -> bool:
    return (pd.api.types.is_categorical_dtype(s.dtype) or s.dtype == "object")


def _safe_num(x: Any) -> Optional[float]:
    """Intenta convertir a número si viene como string numérico."""
    if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            v = float(s)
            if np.isfinite(v):
                return v
        except Exception:
            return None
    return None


def _clean_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _fill_missing_by_type(df: pd.DataFrame, cat_cols: Iterable[str]) -> pd.DataFrame:
    """
    - Columnas categóricas: 'missing'
    - Numéricas: 0
    """
    out = df.copy()
    cat_cols = set(cat_cols)
    for c in out.columns:
        if c in cat_cols:
            out[c] = out[c].astype("object").where(out[c].notna(), "missing")
            out[c] = out[c].replace(0, "missing")
        else:
            out[c] = out[c].where(out[c].notna(), 0)
    return out


def _infer_cat_columns_from_db(db: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Devuelve:
      - original_dtypes: dtype objetivo por columna (sin ID/TARGET)
      - cat_columns: categorías permitidas por columna categórica/obj
    """
    X = db.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in db.columns], errors="ignore")

    original_dtypes = X.dtypes.to_dict()

    cat_columns: Dict[str, List[Any]] = {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_categorical_dtype(s.dtype):
            cats = list(s.cat.categories)
            if "missing" not in cats:
                cats = cats + ["missing"]
            cat_columns[col] = cats
        elif s.dtype == "object":
            cats = list(pd.Series(s.dropna().unique()).astype(str).unique())
            if "missing" not in cats:
                cats = cats + ["missing"]
            cat_columns[col] = cats

    return original_dtypes, cat_columns


def _apply_dtypes_and_categories(
    user: pd.DataFrame,
    original_dtypes: Dict[str, Any],
    cat_columns: Dict[str, List[Any]],
) -> pd.DataFrame:
    """
    Convierte el DF al contrato del entrenamiento:
      - categóricas: pd.Categorical con categorías entrenadas + 'missing'
      - numéricas: cast a dtype original si se puede
    """
    out = user.copy()

    # 1) fill missing antes de casteos
    out = _clean_inf_nan(out)
    out = _fill_missing_by_type(out, cat_columns.keys())

    # 2) aplicar categorías/dtypes
    for col, dtype in original_dtypes.items():
        if col not in out.columns:
            continue
        if col in cat_columns:
            out[col] = pd.Categorical(out[col].astype(str), categories=cat_columns[col])
        else:
            # si viene string numérico, intenta parsear
            if out[col].dtype == "object":
                v = _safe_num(out[col].iloc[0]) if len(out) else None
                if v is not None:
                    out[col] = v
            try:
                out[col] = out[col].astype(dtype)
            except Exception:
                # si no se puede, lo dejamos tal cual (mejor que romper)
                pass

    return out


def _ensure_single_row(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) > 1:
        return df.iloc[[0]].copy()
    return df.copy()


# =============================================================================
# Caches: DB y Modelo
# =============================================================================
@lru_cache(maxsize=1)
def get_db() -> pd.DataFrame:
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATA_PATH}")
    return pd.read_csv(str(DATA_PATH))


@lru_cache(maxsize=1)
def get_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
    return joblib.load(str(MODEL_PATH))


# =============================================================================
# Public API
# =============================================================================
def search_user(sk_id: int, ammount: float, credit_type: str) -> pd.DataFrame:
    """
    Cliente existente:
    - Busca SK_ID_CURR en el dataset
    - Sobrescribe AMT_CREDIT y NAME_CONTRACT_TYPE
    - Limpia + prepara dtypes/categorías como entrenamiento
    - Devuelve X listo (sin TARGET ni SK_ID_CURR)
    """
    db = get_db()

    if ID_COL not in db.columns:
        raise KeyError(f"El dataset no contiene la columna {ID_COL}")

    user = db.loc[db[ID_COL] == sk_id].copy()
    user = _ensure_single_row(user)
    if user.empty:
        return user

    # Sobrescrituras solicitadas desde frontend
    if "AMT_CREDIT" in user.columns:
        user["AMT_CREDIT"] = ammount
    else:
        # si no existe, la creamos para no romper
        user["AMT_CREDIT"] = ammount

    if "NAME_CONTRACT_TYPE" in user.columns:
        user["NAME_CONTRACT_TYPE"] = credit_type
    else:
        user["NAME_CONTRACT_TYPE"] = credit_type

    original_dtypes, cat_columns = _infer_cat_columns_from_db(db)

    # drop ID/TARGET antes de tipar (contrato de input)
    X = user.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in user.columns], errors="ignore")

    X = _apply_dtypes_and_categories(X, original_dtypes, cat_columns)
    return X.reset_index(drop=True)


def new_user(**kwargs) -> pd.DataFrame:
    """
    Nuevo cliente:
    - Construye un único registro con columnas EXACTAS del entrenamiento
    - Rellena lo no dado con 0 / 'missing'
    - Aplica categorías/dtypes según dataset de entrenamiento
    """
    db = get_db()
    original_dtypes, cat_columns = _infer_cat_columns_from_db(db)

    # Base con todas las columnas del entrenamiento (sin ID/TARGET)
    user_data: Dict[str, Any] = {col: np.nan for col in original_dtypes.keys()}

    # Rellenar con kwargs
    for key, value in kwargs.items():
        if key in user_data:
            user_data[key] = value

    user = pd.DataFrame([user_data])

    user = _apply_dtypes_and_categories(user, original_dtypes, cat_columns)

    return user.reset_index(drop=True)


def predict(user: pd.DataFrame) -> np.ndarray:
    """
    Predicción (CatBoost o XGBoost) -> predict_proba
    Retorna shape (1,2): [P(no_default), P(default)]
    """
    model = get_model()
    model_type = type(model).__name__

    X = _ensure_single_row(user)

    # XGBoost: categorías -> codes
    if ("XGB" in model_type) or ("Booster" in model_type):
        X2 = X.copy()
        cat_cols = X2.select_dtypes("category").columns.tolist()
        for col in cat_cols:
            X2[col] = X2[col].cat.codes
        try:
            # algunos wrappers aceptan set_params
            model.set_params(device="cpu", eval_metric=None)
        except Exception:
            try:
                model.set_params(device="cpu")
            except Exception:
                pass
        return np.asarray(model.predict_proba(X2))

    # CatBoost
    if "CatBoost" in model_type:
        # CatBoost acepta directamente DataFrame con categóricas en muchos casos
        return np.asarray(model.predict_proba(X))

    # Fallback genérico
    return np.asarray(model.predict_proba(X))


def explain(user_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (top_bad, top_good) con columnas: feature, value, shap
    - top_bad: mayor shap (sube riesgo)
    - top_good: menor shap (baja riesgo)
    """
    if shap is None:
        # si no hay shap instalado
        empty = pd.DataFrame(columns=["feature", "value", "shap"])
        return empty, empty

    try:
        model = get_model()
        model_type = type(model).__name__
        X = _ensure_single_row(user_df)

        if ("XGB" in model_type) or ("Booster" in model_type):
            # XGBoost: categories -> codes
            X2 = X.copy()
            cat_cols = X2.select_dtypes("category").columns.tolist()
            for col in cat_cols:
                X2[col] = X2[col].cat.codes

            try:
                model.set_params(device="cpu", eval_metric=None)
            except Exception:
                try:
                    model.set_params(device="cpu")
                except Exception:
                    pass

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X2)

        elif "CatBoost" in model_type:
            # CatBoost: ideal usar Pool si está disponible
            if Pool is not None:
                cat_features = X.select_dtypes("category").columns.tolist()
                pool = Pool(X, cat_features=cat_features)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(pool)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

        # binario: list -> clase 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_row = np.asarray(shap_values)[0]

        result = pd.DataFrame(
            {
                "feature": list(X.columns),
                "value": X.iloc[0].values,
                "shap": shap_row,
            }
        )

        result = result.replace([np.inf, -np.inf], 0).fillna(0)

        top_bad = result.sort_values("shap", ascending=False).head(10).reset_index(drop=True)
        top_good = result.sort_values("shap", ascending=True).head(10).reset_index(drop=True)

        return top_bad, top_good

    except Exception:
        empty = pd.DataFrame(columns=["feature", "value", "shap"])
        return empty, empty


def format_shap_for_frontend(
    top_bad: pd.DataFrame,
    top_good: pd.DataFrame,
    *,
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Convierte (top_bad, top_good) al formato que tu frontend ya consume:
    {
      "enabled": true,
      "top_risk_increasing": [{"feature":..., "shap":..., "value":...}, ...],
      "top_risk_decreasing": [{"feature":..., "shap":..., "value":...}, ...]
    }
    """
    def row_to_item(r: pd.Series) -> Dict[str, Any]:
        v = r.get("value", None)
        # JSON-friendly
        try:
            if pd.isna(v):
                v = None
        except Exception:
            pass
        if isinstance(v, (np.integer, np.floating)):
            v = v.item()
        if isinstance(v, (np.bool_)):
            v = bool(v)

        s = r.get("shap", 0.0)
        if isinstance(s, (np.integer, np.floating)):
            s = float(s.item())
        else:
            try:
                s = float(s)
            except Exception:
                s = 0.0

        return {
            "feature": str(r.get("feature", "")),
            "shap": s,
            "value": v,
        }

    inc = []
    dec = []

    if top_bad is not None and not top_bad.empty:
        inc = [row_to_item(top_bad.iloc[i]) for i in range(len(top_bad))]

    if top_good is not None and not top_good.empty:
        dec = [row_to_item(top_good.iloc[i]) for i in range(len(top_good))]

    return {
        "enabled": bool(enabled),
        "top_risk_increasing": inc,
        "top_risk_decreasing": dec,
    }