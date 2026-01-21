"""
api.py
Backend FastAPI para Credit App (CatBoost best_score vía Trabajo_Iago/functions.py)

Qué hace:
- Importa y reutiliza Trabajo_Iago/functions.py (pipeline del compañero)
- Fuerza a usar:
    assets/data/home_credit_train_ready.csv
    assets/models/catboost_best_scores.pkl
- Corrige errores típicos CatBoost:
    1) Orden/selección de columnas (alineación exacta al modelo)
    2) Categóricas mal marcadas (cat_features según el modelo)
- POST /api/score: scoring por SK_ID_CURR + AMT_CREDIT + NAME_CONTRACT_TYPE
- SHAP: usa iago_fn.explain() + format_shap_for_frontend() tal cual
- Sirve frontend en GET /app
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field

from feature_labels import COLUMN_MAPPING_UI

# =========================
# Helpers - Define early
# =========================
def _find_repo_root(start: Path) -> Path:
    """
    Busca hacia arriba un directorio que contenga 'Trabajo_Iago'.
    Así no dependes de parents[x] exacto.
    """
    cur = start
    for _ in range(8):
        if (cur / "Trabajo_Iago").exists():
            return cur
        cur = cur.parent
    # Fallback: asume 3 niveles arriba (backend -> credit_app -> Trabajo_Gabri -> repo root)
    return start.parents[3]


# =========================
# Paths / Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
REPO_ROOT = _find_repo_root(BASE_DIR)
MODELS_DIR = REPO_ROOT / "models"

MODEL_PKL = MODELS_DIR / "catboost_best_scores.pkl"
DATA_DIR = REPO_ROOT / "assets" / "data"

DATA_FILE = DATA_DIR / "home_credit_train_ready.csv"
MODEL_PKL = MODELS_DIR / "catboost_best_scores.pkl"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

# Frontend values -> dataset values
CONTRACT_TYPE_MAP = {
    "revolving_loans": "Revolving loans",
    "cash_loans": "Cash loans",
}
CreditType = Literal["revolving_loans", "cash_loans"]


# =========================
# Import Trabajo_Iago/functions.py (sin tocar su archivo)
# =========================


REPO_ROOT = _find_repo_root(BASE_DIR)
IAGO_DIR = REPO_ROOT / "Trabajo_Iago"

if not IAGO_DIR.exists():
    raise RuntimeError(f"No se encontró Trabajo_Iago en: {IAGO_DIR}")

if str(IAGO_DIR) not in sys.path:
    sys.path.insert(0, str(IAGO_DIR))

from Trabajo_Iago import functions as iago_fn


# Forzar que SU pipeline use TU dataset/modelo best_score (sin modificar functions.py)
iago_fn.DATA_PATH = DATA_FILE
iago_fn.MODEL_PATH = MODEL_PKL

# Reset de cachés por si ya se cargó otro dataset/modelo antes
if hasattr(iago_fn, "_DB"):
    iago_fn._DB = None
if hasattr(iago_fn, "_MODEL"):
    iago_fn._MODEL = None


# =========================
# App / Templates
# =========================
app = FastAPI(title="Credit Risk API", version="1.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/api/feature-labels", response_class=JSONResponse, tags=["frontend"])
def feature_labels() -> Dict[str, str]:
    """Devuelve el diccionario técnico -> nombre legible para UI."""
    return COLUMN_MAPPING_UI


# =========================
# Helpers
# =========================
def _safe_value(v: Any) -> Any:
    """
    Convierte valores numpy/pandas a tipos JSON-friendly.
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


def _get_cb_feature_names(model: Any) -> List[str]:
    """
    Devuelve el orden EXACTO de features del modelo CatBoost.
    Sin esto, CatBoost puede interpretar columnas por posición y romper.
    """
    names = None

    if hasattr(model, "feature_names_"):
        try:
            names = list(model.feature_names_)
        except Exception:
            names = None

    if not names and hasattr(model, "get_feature_names"):
        try:
            names = list(model.get_feature_names())
        except Exception:
            names = None

    if not names:
        raise RuntimeError("El modelo CatBoost no expone feature names; no se puede alinear el input.")

    names = [str(x) for x in names if str(x).strip() != ""]
    if not names:
        raise RuntimeError("feature_names del modelo está vacío.")
    return names


def _align_and_cast_for_catboost(user_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    - Alinea columnas al orden EXACTO del modelo
      (evita: 'Revolving loans' -> float en feature_idx=0).
    - Fuerza categóricas según el modelo
      (evita: 'Categorical in model but marked different in the dataset').

    Nota: NO modifica functions.py, sólo prepara el DF antes de llamar a predict/explain.
    """
    feats = _get_cb_feature_names(model)

    missing = [c for c in feats if c not in user_df.columns]
    if missing:
        raise RuntimeError(
            f"Faltan columnas para el modelo: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    X = user_df.loc[:, feats].copy()

    cat_idx: List[int] = []
    if hasattr(model, "get_cat_feature_indices"):
        try:
            cat_idx = list(model.get_cat_feature_indices())
        except Exception:
            cat_idx = []

    for i in cat_idx:
        if 0 <= i < X.shape[1]:
            col = X.columns[i]
            # CatBoost + dtype correcto (category) para que su explain() también funcione
            X[col] = X[col].astype("string").astype("category")

    return X


# =========================
# Schemas
# =========================
class ScoreRequest(BaseModel):
    sk_id_curr: int = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    credit_type: CreditType


# =========================
# Routes
# =========================
@app.get("/", tags=["health"])
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "dataset_exists": DATA_FILE.exists(),
        "model_exists": MODEL_PKL.exists(),
        "paths": {
            "dataset": str(DATA_FILE),
            "model": str(MODEL_PKL),
            "iago_dir": str(IAGO_DIR),
            "repo_root": str(REPO_ROOT),
        },
    }


@app.get("/app", response_class=HTMLResponse, tags=["frontend"])
def app_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/score", tags=["scoring"])
def score(req: ScoreRequest) -> Dict[str, Any]:
    # Validaciones rápidas de archivos (evita errores raros en runtime)
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail=f"Dataset no encontrado: {DATA_FILE}")
    if not MODEL_PKL.exists():
        raise HTTPException(status_code=500, detail=f"Modelo no encontrado: {MODEL_PKL}")

    # 1) Cargar el modelo del compañero (ya parcheado a best_score)
    try:
        model = iago_fn.get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo (Trabajo_Iago/functions.py): {str(e)}")

    # 2) Mapear credit_type del frontend
    contract = CONTRACT_TYPE_MAP[req.credit_type]

    # 3) Construir el user con SU función (pipeline de entrenamiento)
    try:
        user = iago_fn.search_user(
            sk_id=req.sk_id_curr,
            ammount=req.amt_credit,
            credit_type=contract,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en search_user(): {str(e)}")

    if user is None or user.empty:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {req.sk_id_curr} no existe en el dataset.")

    # 4) FIX CRÍTICO: alinear columnas y marcar categóricas según el modelo
    try:
        user = _align_and_cast_for_catboost(user, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Input preparation falló: {str(e)}")

    # 5) Predicción usando SU predict()
    try:
        pred = iago_fn.predict(user)  # shape (1,2)
        p_default = float(np.asarray(pred)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict falló: {str(e)}")

    # 6) Decisión
    threshold = 0.60
    decision = "REJECT" if p_default >= threshold else "APPROVE"

    # 7) SHAP usando SU explain() + formatter
    try:
        top_bad, top_good = iago_fn.explain(user)
        shap_block = iago_fn.format_shap_for_frontend(top_bad, top_good)
    except Exception as e:
        shap_block = {
            "enabled": True,
            "error": str(e),
            "top_risk_increasing": [],
            "top_risk_decreasing": [],
        }

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