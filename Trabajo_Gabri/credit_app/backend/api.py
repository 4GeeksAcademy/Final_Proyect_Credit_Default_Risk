"""
api.py
Backend FastAPI para Credit App (CatBoost best_score vía Trabajo_Iago/functions.py)

Qué hace:
- Importa y reutiliza Trabajo_Iago/functions.py (pipeline del compañero)
- Fuerza a usar:
    backend/assets/data/home_credit_train_ready.csv
    repo_root/models/catboost_best_scores.pkl
- Corrige errores típicos CatBoost:
    1) Orden/selección de columnas (alineación exacta al modelo)
    2) Categóricas mal marcadas (cat_features según el modelo)
- POST /api/score: scoring por SK_ID_CURR + AMT_CREDIT + NAME_CONTRACT_TYPE (cliente existente)
- POST /api/new-client: registro de nuevo cliente (devuelve SK virtual)
- GET /api/new-client/schema: schema de features para nuevo cliente (form dinámico)
- POST /api/new-client/score: scoring para nuevo cliente usando iago_fn.new_user(**features)
  - Soporta sk_id_curr virtual para reutilizar features guardadas en el registro
- SHAP: usa iago_fn.explain() y iago_fn.format_shap_for_frontend() si existen
- Sirve frontend en GET /app
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from feature_labels import COLUMN_MAPPING_UI


# =============================================================================
# Helpers
# =============================================================================
def _find_repo_root(start: Path) -> Path:
    """
    Busca hacia arriba un directorio que contenga 'Trabajo_Iago'.
    """
    cur = start
    for _ in range(10):
        if (cur / "Trabajo_Iago").exists():
            return cur
        cur = cur.parent
    try:
        return start.parents[3]
    except Exception:
        return start


def _safe_json_value(v: Any) -> Any:
    """Convierte tipos numpy/pandas a JSON-friendly."""
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _get_cb_feature_names(model: Any) -> List[str]:
    """Orden EXACTO de features del modelo CatBoost."""
    names: Optional[List[str]] = None

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

    names = [str(x) for x in names if str(x).strip()]
    if not names:
        raise RuntimeError("feature_names del modelo está vacío.")
    return names


def _align_and_cast_for_catboost(user_df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Alinea columnas al orden exacto del modelo y fuerza categóricas según el modelo.
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
            X[col] = X[col].astype("string").astype("category")

    return X


def _load_model_fallback(model_path: Path) -> Any:
    """
    Fallback si el functions.py del compa NO tiene get_model().
    Carga el pkl directamente.
    """
    import joblib

    if not model_path.exists():
        raise RuntimeError(f"Modelo no encontrado: {model_path}")
    return joblib.load(model_path)


def _format_shap_fallback(top_bad: pd.DataFrame, top_good: pd.DataFrame) -> Dict[str, Any]:
    """
    Fallback simple si no existe format_shap_for_frontend() en functions.py
    """
    def _to_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if df is None or df.empty:
            return out
        for _, r in df.iterrows():
            out.append(
                {
                    "feature": str(r.get("feature")),
                    "value": _safe_json_value(r.get("value")),
                    "shap": float(r.get("shap")) if r.get("shap") is not None else 0.0,
                }
            )
        return out

    return {
        "enabled": True,
        "top_risk_increasing": _to_list(top_bad),
        "top_risk_decreasing": _to_list(top_good),
    }


def _norm_email(s: str) -> str:
    return (s or "").strip().lower()


def _norm_phone(s: str) -> str:
    return re.sub(r"\D+", "", (s or "").strip())


def _guess_default_for_feature(name: str, dtype: str) -> Any:
    """
    Default razonable si el frontend no envía esa feature.
    - Categóricas: 'missing'
    - Numéricas: 0
    """
    if dtype == "cat":
        return "missing"
    return 0


# =============================================================================
# Paths / Config
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

REPO_ROOT = _find_repo_root(BASE_DIR)
IAGO_DIR = REPO_ROOT / "Trabajo_Iago"

# Dataset en TU backend (según tu estructura)
DATA_DIR = BASE_DIR / "assets" / "data"
DATA_FILE = DATA_DIR / "home_credit_train_ready.csv"

# Modelo en repo root / models
MODELS_DIR = REPO_ROOT / "models"
MODEL_PKL = MODELS_DIR / "catboost_best_scores.pkl"

# Frontend values -> dataset values
CONTRACT_TYPE_MAP = {
    "revolving_loans": "Revolving loans",
    "cash_loans": "Cash loans",
}
CreditType = Literal["revolving_loans", "cash_loans"]


# =============================================================================
# Import Trabajo_Iago/functions.py
# =============================================================================
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from Trabajo_Iago import functions as iago_fn  # type: ignore
except Exception as e:
    raise RuntimeError(
        f"No se pudo importar Trabajo_Iago.functions. REPO_ROOT={REPO_ROOT}. Error: {e}"
    )

# Forzar que SU pipeline use TUS paths (si su archivo los respeta)
try:
    iago_fn.DATA_PATH = DATA_FILE
except Exception:
    pass

try:
    iago_fn.MODEL_PATH = MODEL_PKL
except Exception:
    pass

# ✅ Reset correcto para lru_cache (tu compañero lo usa en get_db/get_model)
try:
    if hasattr(iago_fn, "get_db"):
        iago_fn.get_db.cache_clear()
except Exception:
    pass

try:
    if hasattr(iago_fn, "get_model"):
        iago_fn.get_model.cache_clear()
except Exception:
    pass


# =============================================================================
# In-memory register (DEMO)
# =============================================================================
# Registro “suficiente” para demo: persistencia temporal en RAM.
# - sk_id_curr virtual: 900000000, 900000001, ...
# - dedupe básico por email/dni (para no generar mil SK por error)
REGISTERED_CLIENTS: Dict[int, Dict[str, Any]] = {}
REGISTER_INDEX_EMAIL: Dict[str, int] = {}
REGISTER_INDEX_DNI: Dict[str, int] = {}


def _next_dataset_sk() -> int:
    """
    Devuelve el siguiente SK_ID_CURR basado en el dataset real:
    next = max(SK_ID_CURR) + 1
    """
    try:
        db = iago_fn.get_db() if hasattr(iago_fn, "get_db") else pd.read_csv(str(DATA_FILE))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo leer dataset para generar SK: {e}")

    if "SK_ID_CURR" not in db.columns:
        raise HTTPException(status_code=500, detail="Dataset no contiene SK_ID_CURR.")

    mx = pd.to_numeric(db["SK_ID_CURR"], errors="coerce").max()
    if not np.isfinite(mx):
        raise HTTPException(status_code=500, detail="No se pudo calcular max(SK_ID_CURR).")

    return int(mx) + 1


def _create_or_get_virtual_sk(full_name: str, dni: str, email: str, phone: str, features: Dict[str, Any]) -> int:
    """
    Crea un SK "realista" = max(SK_ID_CURR) + 1 (y +2, +3...).
    Dedupe por email o dni para no crear duplicados en demo.
    Persistencia en RAM (REGISTERED_CLIENTS) solo para reutilizar features.
    """
    dni_k = (dni or "").strip().upper()
    email_k = _norm_email(email)

    # Dedupe: si ya existe, devuelve el mismo SK
    if email_k and email_k in REGISTER_INDEX_EMAIL:
        return REGISTER_INDEX_EMAIL[email_k]
    if dni_k and dni_k in REGISTER_INDEX_DNI:
        return REGISTER_INDEX_DNI[dni_k]

    # Generar SK siguiente basado en dataset y en los ya registrados en RAM
    base_next = _next_dataset_sk()
    if REGISTERED_CLIENTS:
        base_next = max(base_next, max(REGISTERED_CLIENTS.keys()) + 1)

    sk = int(base_next)

    REGISTERED_CLIENTS[sk] = {
        "full_name": (full_name or "").strip(),
        "dni": dni_k,
        "email": email_k,
        "phone": _norm_phone(phone),
        "features": dict(features or {}),
    }
    if email_k:
        REGISTER_INDEX_EMAIL[email_k] = sk
    if dni_k:
        REGISTER_INDEX_DNI[dni_k] = sk

    return sk



# =============================================================================
# App / Templates
# =============================================================================
app = FastAPI(title="Credit Risk API", version="1.2")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# =============================================================================
# Frontend helpers
# =============================================================================
@app.get("/api/feature-labels", response_class=JSONResponse, tags=["frontend"])
def feature_labels() -> Dict[str, str]:
    """Devuelve el diccionario técnico -> nombre legible para UI."""
    return COLUMN_MAPPING_UI


# =============================================================================
# Schemas
# =============================================================================
class ScoreRequest(BaseModel):
    sk_id_curr: int = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    credit_type: CreditType


class NewClientRegisterRequest(BaseModel):
    """
    Registro del nuevo cliente (solo demo / flujo).
    Permite también guardar "features" opcionales para luego reutilizarlas en el scoring del nuevo.
    """
    full_name: str = Field(..., min_length=2)
    dni: str = Field(..., min_length=5)
    email: str = Field(..., min_length=5)
    phone: str = Field(..., min_length=5)
    features: Dict[str, Any] = Field(default_factory=dict)


class NewClientRegisterResponse(BaseModel):
    ok: bool
    sk_id_curr: int


class NewClientScoreRequest(BaseModel):
    """
    Nuevo cliente:
    - sk_id_curr: opcional (si viene, reutiliza features guardadas en /api/new-client)
    - amt_credit + credit_type siempre
    - features: dict con columnas EXACTAS del dataset para sobreescribir/añadir
    """
    sk_id_curr: Optional[int] = Field(default=None, gt=0)
    amt_credit: float = Field(..., gt=0)
    credit_type: CreditType
    features: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Routes
# =============================================================================
@app.get("/", tags=["health"])
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "exists": {
            "repo_root": REPO_ROOT.exists(),
            "iago_dir": IAGO_DIR.exists(),
            "dataset": DATA_FILE.exists(),
            "model": MODEL_PKL.exists(),
            "templates_dir": TEMPLATES_DIR.exists(),
        },
        "paths": {
            "repo_root": str(REPO_ROOT),
            "iago_dir": str(IAGO_DIR),
            "dataset": str(DATA_FILE),
            "model": str(MODEL_PKL),
            "templates_dir": str(TEMPLATES_DIR),
        },
        "functions_has": {
            "get_db": hasattr(iago_fn, "get_db"),
            "get_model": hasattr(iago_fn, "get_model"),
            "search_user": hasattr(iago_fn, "search_user"),
            "new_user": hasattr(iago_fn, "new_user"),
            "predict": hasattr(iago_fn, "predict"),
            "explain": hasattr(iago_fn, "explain"),
            "format_shap_for_frontend": hasattr(iago_fn, "format_shap_for_frontend"),
        },
        "registered_clients": {
            "count": len(REGISTERED_CLIENTS),
            "sk_example": next(iter(REGISTERED_CLIENTS.keys()), None),
        },
    }


@app.get("/app", response_class=HTMLResponse, tags=["frontend"])
def app_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------------------------------------------------------
# Nuevo cliente: registro
# -----------------------------------------------------------------------------
@app.post("/api/new-client", response_class=JSONResponse, tags=["frontend"])
def new_client_register(req: NewClientRegisterRequest) -> Dict[str, Any]:
    """
    Registra un nuevo cliente y devuelve un SK virtual.
    - Deduplica por email o dni para evitar duplicados en demo.
    - Guarda features opcionales para usar luego en /api/new-client/score
    """
    sk = _create_or_get_virtual_sk(
        full_name=req.full_name,
        dni=req.dni,
        email=req.email,
        phone=req.phone,
        features=req.features,
    )
    return {"ok": True, "sk_id_curr": sk}


# -----------------------------------------------------------------------------
# Nuevo cliente: schema (para form dinámico)
# -----------------------------------------------------------------------------
@app.get("/api/new-client/schema", response_class=JSONResponse, tags=["frontend"])
def new_client_schema() -> Dict[str, Any]:
    """
    Devuelve un schema basado en el dataset:
    - name: nombre exacto de la columna
    - dtype: "num" | "cat"
    - label: nombre legible (mapping si existe)
    - allowed: categorías (solo si <= 50; si no, None)
    """
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail=f"Dataset no encontrado: {DATA_FILE}")

    try:
        db = iago_fn.get_db() if hasattr(iago_fn, "get_db") else pd.read_csv(str(DATA_FILE))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo leer dataset: {e}")

    X = db.drop(columns=[c for c in ["SK_ID_CURR", "TARGET"] if c in db.columns], errors="ignore")

    features: List[Dict[str, Any]] = []
    for col in X.columns:
        s = X[col]
        is_cat = pd.api.types.is_categorical_dtype(s.dtype) or s.dtype == "object"
        dtype = "cat" if is_cat else "num"

        item: Dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "label": COLUMN_MAPPING_UI.get(col, col),
        }

        if is_cat:
            cats = sorted(pd.Series(s.dropna().astype(str).unique()).tolist())
            if "missing" not in cats:
                cats.append("missing")
            item["allowed"] = cats if len(cats) <= 50 else None

        features.append(item)

    return {
        "features": features,
        "note": "Envía solo las features que quieras; lo demás se completará como 0/'missing' en new_user().",
    }


# -----------------------------------------------------------------------------
# Cliente existente: scoring
# -----------------------------------------------------------------------------
@app.post("/api/score", tags=["scoring"])
def score(req: ScoreRequest) -> Dict[str, Any]:
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail=f"Dataset no encontrado: {DATA_FILE}")
    if not MODEL_PKL.exists():
        raise HTTPException(status_code=500, detail=f"Modelo no encontrado: {MODEL_PKL}")

    # 1) Modelo
    try:
        model = iago_fn.get_model() if hasattr(iago_fn, "get_model") else _load_model_fallback(MODEL_PKL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")

    # 2) Mapear credit_type del frontend
    contract = CONTRACT_TYPE_MAP[req.credit_type]

    # 3) Construir usuario con pipeline del compañero (cliente existente)
    if not hasattr(iago_fn, "search_user"):
        raise HTTPException(status_code=500, detail="Trabajo_Iago/functions.py no tiene search_user().")

    try:
        user = iago_fn.search_user(
            sk_id=req.sk_id_curr,
            ammount=req.amt_credit,
            credit_type=contract,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en search_user(): {e}")

    if user is None or getattr(user, "empty", False):
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {req.sk_id_curr} no existe en el dataset.")

    # 4) Fix CatBoost: alinear + categóricas
    try:
        user = _align_and_cast_for_catboost(user, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Input preparation falló: {e}")

    # 5) Predicción
    if not hasattr(iago_fn, "predict"):
        raise HTTPException(status_code=500, detail="Trabajo_Iago/functions.py no tiene predict().")

    try:
        pred = iago_fn.predict(user)  # shape (1,2)
        p_default = float(np.asarray(pred)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict falló: {e}")

    # 6) Decisión
    threshold = 0.60
    decision = "REJECT" if p_default >= threshold else "APPROVE"

    # 7) SHAP
    shap_block: Dict[str, Any]
    if hasattr(iago_fn, "explain"):
        try:
            top_bad, top_good = iago_fn.explain(user)
            shap_block = (
                iago_fn.format_shap_for_frontend(top_bad, top_good)
                if hasattr(iago_fn, "format_shap_for_frontend")
                else _format_shap_fallback(top_bad, top_good)
            )
        except Exception as e:
            shap_block = {
                "enabled": True,
                "error": str(e),
                "top_risk_increasing": [],
                "top_risk_decreasing": [],
            }
    else:
        shap_block = {
            "enabled": False,
            "error": "functions.py no tiene explain().",
            "top_risk_increasing": [],
            "top_risk_decreasing": [],
        }

    return {
        "mode": "existing_client",
        "sk_id_curr": req.sk_id_curr,
        "input": {
            "sk_id_curr": req.sk_id_curr,
            "amt_credit": req.amt_credit,
            "credit_type": req.credit_type,
            "name_contract_type": contract,
        },
        "proba": {"no_default": 1.0 - p_default, "default": p_default},
        "decision": decision,
        "threshold": threshold,
        "shap": shap_block,
    }


# -----------------------------------------------------------------------------
# Nuevo cliente: scoring real (con soporte de registro previo)
# -----------------------------------------------------------------------------
@app.post("/api/new-client/score", tags=["scoring"])
def new_client_score(req: NewClientScoreRequest) -> Dict[str, Any]:
    if not DATA_FILE.exists():
        raise HTTPException(status_code=500, detail=f"Dataset no encontrado: {DATA_FILE}")
    if not MODEL_PKL.exists():
        raise HTTPException(status_code=500, detail=f"Modelo no encontrado: {MODEL_PKL}")

    # 1) Modelo
    try:
        model = iago_fn.get_model() if hasattr(iago_fn, "get_model") else _load_model_fallback(MODEL_PKL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")

    # 2) Map credit_type -> dataset value
    contract = CONTRACT_TYPE_MAP[req.credit_type]

    # 3) Construir nuevo usuario
    if not hasattr(iago_fn, "new_user"):
        raise HTTPException(status_code=500, detail="Trabajo_Iago/functions.py no tiene new_user().")

    # 3.1) Mezclar features: primero las guardadas (si hay registro), luego las enviadas ahora
    stored_features: Dict[str, Any] = {}
    if req.sk_id_curr is not None:
        stored = REGISTERED_CLIENTS.get(req.sk_id_curr)
        if stored:
            stored_features = dict(stored.get("features") or {})

    incoming_features = dict(req.features or {})
    feats = {**stored_features, **incoming_features}

    # 3.2) Forzar campos clave del crédito (siempre manda el backend)
    feats["AMT_CREDIT"] = req.amt_credit
    feats["NAME_CONTRACT_TYPE"] = contract

    # 3.3) Si quieres “rellenar” automáticamente lo que falte (útil si el form dinámico no manda todo),
    #     usamos el schema del dataset para defaults num/cat.
    try:
        db = iago_fn.get_db() if hasattr(iago_fn, "get_db") else pd.read_csv(str(DATA_FILE))
        X = db.drop(columns=[c for c in ["SK_ID_CURR", "TARGET"] if c in db.columns], errors="ignore")
        for col in X.columns:
            if col in ["AMT_CREDIT", "NAME_CONTRACT_TYPE"]:
                continue
            if col in feats:
                continue
            s = X[col]
            is_cat = pd.api.types.is_categorical_dtype(s.dtype) or s.dtype == "object"
            dtype = "cat" if is_cat else "num"
            feats[col] = _guess_default_for_feature(col, dtype)
    except Exception:
        # Si falla, no rompas scoring: new_user() del compa debería manejar defaults.
        pass

    try:
        user = iago_fn.new_user(**feats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en new_user(): {e}")

    if user is None or getattr(user, "empty", False):
        raise HTTPException(status_code=500, detail="new_user() devolvió vacío.")

    # 4) Fix CatBoost: alinear + categóricas
    try:
        user = _align_and_cast_for_catboost(user, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Input preparation falló: {e}")

    # 5) Predicción
    if not hasattr(iago_fn, "predict"):
        raise HTTPException(status_code=500, detail="Trabajo_Iago/functions.py no tiene predict().")

    try:
        pred = iago_fn.predict(user)  # shape (1,2)
        p_default = float(np.asarray(pred)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict falló: {e}")

    # 6) Decisión
    threshold = 0.60
    decision = "REJECT" if p_default >= threshold else "APPROVE"

    # 7) SHAP
    shap_block: Dict[str, Any]
    if hasattr(iago_fn, "explain"):
        try:
            top_bad, top_good = iago_fn.explain(user)
            shap_block = (
                iago_fn.format_shap_for_frontend(top_bad, top_good)
                if hasattr(iago_fn, "format_shap_for_frontend")
                else _format_shap_fallback(top_bad, top_good)
            )
        except Exception as e:
            shap_block = {
                "enabled": True,
                "error": str(e),
                "top_risk_increasing": [],
                "top_risk_decreasing": [],
            }
    else:
        shap_block = {
            "enabled": False,
            "error": "functions.py no tiene explain().",
            "top_risk_increasing": [],
            "top_risk_decreasing": [],
        }

    return {
        "mode": "new_client",
        "sk_id_curr": req.sk_id_curr,
        "input": {
            "sk_id_curr": req.sk_id_curr,
            "amt_credit": req.amt_credit,
            "credit_type": req.credit_type,
            "name_contract_type": contract,
            "features_sent": list((req.features or {}).keys()),
            "features_from_register": list(stored_features.keys()) if stored_features else [],
        },
        "proba": {"no_default": 1.0 - p_default, "default": p_default},
        "decision": decision,
        "threshold": threshold,
        "shap": shap_block,
    }