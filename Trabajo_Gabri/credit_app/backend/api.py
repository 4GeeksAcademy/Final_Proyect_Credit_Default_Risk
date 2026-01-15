"""
api.py
Backend FastAPI para Credit App (modelo real + SHAP siempre activo)

Qué hace:
- Carga dataset "ready" (home_credit_train_ready.csv)
- Carga XGBoost portable (xgb_model.json)
- Aplica el MISMO encoding que en entrenamiento:
    object -> category codes (con categorías fijadas desde el dataset)
- POST /api/score: scoring por SK_ID_CURR + AMT_CREDIT + NAME_CONTRACT_TYPE
- SHAP siempre activo (top features que suben/bajan riesgo)
- Sirve el frontend en GET /app

Notas:
- Este pipeline es consistente con train_xgb_portable.py (mismo encoding).
- Para producción real, lo ideal es guardar el encoder explícito (pero esto funciona).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field

import xgboost as xgb


# =========================
# Paths / Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DATA_DIR = ASSETS_DIR / "data"

DATA_FILE = DATA_DIR / "home_credit_train_ready.csv"
MODEL_JSON = MODELS_DIR / "xgb_model.json"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# Frontend values -> dataset values
CONTRACT_TYPE_MAP = {
    "revolving_loans": "Revolving loans",
    "cash_loans": "Cash loans",
}
CreditType = Literal["revolving_loans", "cash_loans"]


# =========================
# App / Templates
# =========================
app = FastAPI(title="Credit Risk API", version="1.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =========================
# Globals (loaded once)
# =========================
DF_RAW: Optional[pd.DataFrame] = None
FEATURE_COLS: Optional[List[str]] = None

CAT_COLS: Optional[List[str]] = None
CAT_MAP: Optional[Dict[str, List[str]]] = None  # col -> categories list

MODEL: Optional[xgb.Booster] = None
EXPLAINER: Optional[Any] = None


# =========================
# Schemas
# =========================
class ScoreRequest(BaseModel):
    sk_id_curr: int = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    credit_type: CreditType


# =========================
# Helpers
# =========================
def _safe_value(v: Any) -> Any:
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def build_category_maps(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, List[str]]:
    """
    Construye el "diccionario" de categorías por columna desde el dataset.
    Esto fija el mapping a codes de forma estable (igual que en entrenamiento).
    """
    mapping: Dict[str, List[str]] = {}
    for c in cat_cols:
        # Ojo: convertimos a string para no romper por NaNs raros
        vals = df[c].astype("string")
        cats = pd.Series(vals.dropna().unique()).sort_values().tolist()
        mapping[c] = cats
    return mapping


def encode_like_training(X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el MISMO encoding que el script de entrenamiento:
      - Para cada columna categórica: category con categorías fijas -> codes int32
      - NaNs -> -1 (igual que cat.codes)
    """
    assert CAT_COLS is not None and CAT_MAP is not None, "Encoder no inicializado"
    X2 = X.copy()

    for c in CAT_COLS:
        if c not in X2.columns:
            continue
        cats = CAT_MAP[c]
        X2[c] = pd.Categorical(X2[c].astype("string"), categories=cats).codes.astype("int32")

    # Si quedara algún object por error, lo pasamos a category codes genérico
    for c in X2.select_dtypes(include=["object"]).columns:
        X2[c] = X2[c].astype("category").cat.codes.astype("int32")

    return X2


def predict_proba_default(X_num: pd.DataFrame) -> float:
    """
    Predice P(default=1) con Booster XGBoost (modelo JSON portable).
    """
    assert MODEL is not None, "Modelo no cargado"
    dm = xgb.DMatrix(X_num, feature_names=list(X_num.columns))
    pred = MODEL.predict(dm)
    return float(pred[0])  # ya es probabilidad (binary:logistic)


def shap_top(explainer: Any, X_num: pd.DataFrame, top_n: int = 8) -> Dict[str, Any]:
    """
    SHAP para 1 fila (TreeExplainer).
    Devuelve top features que aumentan/disminuyen riesgo.
    """
    sv = explainer.shap_values(X_num)
    # sv puede ser (1, n_features) para binary logistic
    sv_vec = sv[0] if hasattr(sv, "__len__") and len(np.array(sv).shape) == 2 else sv
    sv_series = pd.Series(sv_vec, index=X_num.columns)

    inc = sv_series[sv_series > 0].sort_values(ascending=False).head(top_n)
    dec = sv_series[sv_series < 0].sort_values(ascending=True).head(top_n)

    def pack(s: pd.Series) -> List[dict]:
        return [
            {"feature": k, "shap": float(v), "value": _safe_value(X_num.iloc[0][k])}
            for k, v in s.items()
        ]

    return {
        "enabled": True,
        "top_risk_increasing": pack(inc),
        "top_risk_decreasing": pack(dec),
    }


# =========================
# Startup
# =========================
@app.on_event("startup")
def startup() -> None:
    global DF_RAW, FEATURE_COLS, CAT_COLS, CAT_MAP, MODEL, EXPLAINER

    # --- sanity checks
    if not DATA_FILE.exists():
        raise RuntimeError(f"Dataset no encontrado: {DATA_FILE}")
    if not MODEL_JSON.exists():
        raise RuntimeError(f"Modelo JSON no encontrado: {MODEL_JSON}")

    # --- load data
    DF_RAW = pd.read_csv(DATA_FILE)

    if ID_COL not in DF_RAW.columns or TARGET_COL not in DF_RAW.columns:
        raise RuntimeError("El dataset no tiene SK_ID_CURR/TARGET. No es 'ready'.")

    # Features = todo menos ID y TARGET
    FEATURE_COLS = [c for c in DF_RAW.columns if c not in (ID_COL, TARGET_COL)]

    # Detectar categóricas (object)
    CAT_COLS = DF_RAW[FEATURE_COLS].select_dtypes(include=["object"]).columns.tolist()
    CAT_MAP = build_category_maps(DF_RAW[FEATURE_COLS], CAT_COLS)

    # --- load model booster (portable)
    MODEL = xgb.Booster()
    MODEL.load_model(str(MODEL_JSON))

    # --- build SHAP explainer con background sample
    # Encoding del background (mismo que prod)
    bg = DF_RAW.sample(n=min(400, len(DF_RAW)), random_state=42)[FEATURE_COLS].copy()
    bg_num = encode_like_training(bg)
    EXPLAINER = shap.TreeExplainer(MODEL, bg_num)

    print(f"✅ Loaded dataset rows={len(DF_RAW):,} features={len(FEATURE_COLS):,} cat_cols={len(CAT_COLS):,}")
    print(f"✅ Loaded model: {MODEL_JSON.name}")
    print("✅ SHAP explainer ready")


# =========================
# Routes
# =========================
@app.get("/", tags=["health"])
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "dataset_exists": DATA_FILE.exists(),
        "model_exists": MODEL_JSON.exists(),
        "rows_loaded": 0 if DF_RAW is None else int(len(DF_RAW)),
        "features_loaded": 0 if FEATURE_COLS is None else int(len(FEATURE_COLS)),
        "cat_cols": 0 if CAT_COLS is None else int(len(CAT_COLS)),
    }


@app.get("/app", response_class=HTMLResponse, tags=["frontend"])
def app_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/score", tags=["scoring"])
def score(req: ScoreRequest) -> Dict[str, Any]:
    if DF_RAW is None or FEATURE_COLS is None:
        raise HTTPException(status_code=500, detail="Dataset no cargado")
    if MODEL is None or EXPLAINER is None:
        raise HTTPException(status_code=500, detail="Modelo/SHAP no cargados")

    # 1) Buscar cliente base por ID
    row = DF_RAW.loc[DF_RAW[ID_COL] == req.sk_id_curr]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {req.sk_id_curr} no existe en el dataset.")

    # 2) Construir X (1 fila) y sobrescribir inputs del frontend
    X = row.iloc[[0]][FEATURE_COLS].copy()

    # Sobrescribir AMT_CREDIT si existe
    if "AMT_CREDIT" in X.columns:
        X.loc[:, "AMT_CREDIT"] = float(req.amt_credit)

    # Sobrescribir tipo de crédito
    contract = CONTRACT_TYPE_MAP[req.credit_type]
    if "NAME_CONTRACT_TYPE" in X.columns:
        X.loc[:, "NAME_CONTRACT_TYPE"] = contract

    # 3) Encoding consistente
    X_num = encode_like_training(X)

    # 4) Predicción
    p_default = predict_proba_default(X_num)
    threshold = 0.60
    decision = "REJECT" if p_default >= threshold else "APPROVE"

    # 5) SHAP (siempre)
    try:
        shap_block = shap_top(EXPLAINER, X_num, top_n=8)
    except Exception as e:
        shap_block = {"enabled": True, "error": str(e), "top_risk_increasing": [], "top_risk_decreasing": []}

    return {
        "sk_id_curr": req.sk_id_curr,
        "input": {
            "sk_id_curr": req.sk_id_curr,
            "amt_credit": req.amt_credit,
            "credit_type": req.credit_type,
            "name_contract_type": contract,
        },
        "proba": {"no_default": 1 - p_default, "default": p_default},
        "decision": decision,
        "threshold": threshold,
        "shap": shap_block,
    }
