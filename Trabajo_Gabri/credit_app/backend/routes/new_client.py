"""
src/routes/new_client.py

Endpoints para flujo "Nuevo cliente" (DEMO, sin base de datos):

- POST /api/new-client
  Genera un SK_ID_CURR nuevo = max(SK_ID_CURR) + 1 y lo devuelve.
  (No persiste nada, pensado para presentación con 2-3 ejemplos.)

- POST /api/new-client/score
  Hace scoring del "nuevo cliente" SIN existir en el CSV:
    1) Construye una fila plantilla a partir del dataset ready (mediana/moda).
    2) Sobrescribe: SK_ID_CURR, AMT_CREDIT, NAME_CONTRACT_TYPE.
    3) Alinea columnas al modelo (si es posible).
    4) Devuelve probas en formato compatible con tu frontend.

Requisitos del app:
- request.app.state.df_ready -> pandas.DataFrame del dataset ready (con SK_ID_CURR).
- request.app.state.model    -> modelo con predict_proba(X) (sklearn/catboost/xgb wrapper, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

router = APIRouter()


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class NewClientRegisterIn(BaseModel):
    """Datos básicos de registro (demo)."""
    full_name: str = Field(..., min_length=2, max_length=120)
    dni: str = Field(..., min_length=5, max_length=20)
    email: EmailStr
    phone: str = Field(..., min_length=6, max_length=30)


class NewClientRegisterOut(BaseModel):
    """Respuesta del registro: SK_ID_CURR asignado."""
    sk_id_curr: int


class NewClientScoreIn(BaseModel):
    """Payload esperado por tu HTML para scoring del nuevo cliente."""
    sk_id_curr: int = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    credit_type: Literal["revolving_loans", "cash_loans"]
    features: Dict[str, Any] = Field(default_factory=dict)  # reservado (no lo usamos aquí)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _require_state(request: Request) -> tuple[pd.DataFrame, Any]:
    """
    Extrae df_ready y model desde app.state.

    Lanza 500 si no existen para que el error sea evidente en la demo.
    """
    df_ready = getattr(request.app.state, "df_ready", None)
    model = getattr(request.app.state, "model", None)

    if df_ready is None or not isinstance(df_ready, pd.DataFrame) or df_ready.empty:
        raise HTTPException(
            status_code=500,
            detail="Backend no inicializado: falta app.state.df_ready (dataset ready).",
        )
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Backend no inicializado: falta app.state.model (modelo).",
        )
    return df_ready, model


def _next_sk_id(df_ready: pd.DataFrame) -> int:
    """Devuelve max(SK_ID_CURR) + 1 (int)."""
    if "SK_ID_CURR" not in df_ready.columns:
        raise HTTPException(status_code=500, detail="df_ready no contiene columna SK_ID_CURR.")
    mx = pd.to_numeric(df_ready["SK_ID_CURR"], errors="coerce").max()
    if not np.isfinite(mx):
        raise HTTPException(status_code=500, detail="No se pudo calcular max(SK_ID_CURR).")
    return int(mx) + 1


def _contract_type_value(credit_type: str) -> str:
    """
    Mapea el credit_type del frontend a NAME_CONTRACT_TYPE típico del dataset.
    Ajusta aquí si tu dataset usa otros textos.
    """
    mapping = {
        "cash_loans": "Cash loans",
        "revolving_loans": "Revolving loans",
    }
    return mapping.get(credit_type, credit_type)


def _build_template_row(df_ready: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una fila plantilla:
    - numéricas: mediana
    - categóricas/object: moda
    Mantiene el set de columnas del dataset ready.
    """
    # Numéricas -> mediana
    numeric = df_ready.select_dtypes(include=[np.number])
    base_num = numeric.median(numeric_only=True).to_frame().T if not numeric.empty else pd.DataFrame([{}])

    # Categóricas -> moda
    base = base_num.copy()
    for col in df_ready.columns:
        if col in base.columns:
            continue
        # Para object/category/bool u otros: moda; si todo NaN, rellena con "missing"
        s = df_ready[col]
        try:
            mode = s.mode(dropna=True)
            base[col] = mode.iloc[0] if len(mode) else "missing"
        except Exception:
            base[col] = "missing"

    # Ordena columnas como df_ready
    base = base[df_ready.columns.tolist()].copy()

    # Limpieza defensiva
    base.replace([np.inf, -np.inf], np.nan, inplace=True)
    base.fillna(0, inplace=True)
    return base


def _align_to_model(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta alinear columnas al modelo si el modelo expone feature_names_.
    Si no, devuelve X tal cual (porque tu pipeline puede manejarlo).
    """
    names = getattr(model, "feature_names_", None)
    if not names:
        return X

    names = list(names)
    missing = [c for c in names if c not in X.columns]
    if missing:
        # Rellenamos faltantes con 0 (mejor que reventar en demo)
        for c in missing:
            X[c] = 0
    # Reordena exactamente
    return X[names].copy()


def _predict_default_proba(model: Any, X: pd.DataFrame) -> float:
    """
    Devuelve p(default) como float.
    Soporta:
    - sklearn: predict_proba -> [:, 1]
    - modelos que devuelven lista/np array
    """
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="El modelo no expone predict_proba(X).")

    proba = model.predict_proba(X)
    proba = np.asarray(proba)

    if proba.ndim == 1:
        # raro, pero por si devuelve prob de clase positiva directa
        p = float(proba[0])
    else:
        # convención: columna 1 = clase positiva (default)
        p = float(proba[0, 1])
    # clamp defensivo
    p = max(0.0, min(1.0, p))
    return p


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@router.post("/api/new-client", response_model=NewClientRegisterOut)
def register_new_client(payload: NewClientRegisterIn, request: Request) -> NewClientRegisterOut:
    """
    Registro DEMO: devuelve SK nuevo = max + 1.
    No guarda nada.
    """
    df_ready, _ = _require_state(request)
    sk = _next_sk_id(df_ready)
    return NewClientRegisterOut(sk_id_curr=sk)


@router.post("/api/new-client/score")
def score_new_client(payload: NewClientScoreIn, request: Request) -> Dict[str, Any]:
    """
    Scoring DEMO para nuevo cliente:
    - crea fila plantilla del dataset
    - sobrescribe campos del frontend
    - predice prob default/no_default
    """
    df_ready, model = _require_state(request)

    X = _build_template_row(df_ready)

    # Sobrescribir inputs clave
    if "SK_ID_CURR" in X.columns:
        X.loc[:, "SK_ID_CURR"] = int(payload.sk_id_curr)

    if "AMT_CREDIT" in X.columns:
        X.loc[:, "AMT_CREDIT"] = float(payload.amt_credit)

    # Este es el campo típico del dataset; si en tu ready se llama distinto, ajusta aquí.
    if "NAME_CONTRACT_TYPE" in X.columns:
        X.loc[:, "NAME_CONTRACT_TYPE"] = _contract_type_value(payload.credit_type)

    # Alineación a modelo si procede
    X = _align_to_model(model, X)

    # Predicción
    p_default = _predict_default_proba(model, X)
    p_no_default = 1.0 - p_default

    return {
        "sk_id_curr": int(payload.sk_id_curr),
        "input": {
            "sk_id_curr": int(payload.sk_id_curr),
            "amt_credit": float(payload.amt_credit),
            "credit_type": payload.credit_type,
            "name_contract_type": _contract_type_value(payload.credit_type),
        },
        "proba": {"no_default": float(p_no_default), "default": float(p_default)},
        "shap": {"enabled": False, "top_risk_increasing": [], "top_risk_decreasing": []},
    }
