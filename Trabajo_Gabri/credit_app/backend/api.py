"""
api.py
API para conectar el frontend (FastAPI + HTML servido por el backend).
- GET  /                -> healthcheck (+ diagnóstico templates)
- GET  /app             -> sirve el frontend (HTML)
- POST /api/score        -> scoring (mock) + SHAP siempre activo
- POST /api/new-client   -> registro
"""

from pathlib import Path
import random
from typing import Any, Dict, Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# =========================
# App + templates (ruta absoluta)
# =========================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="Credit App API", version="0.2")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

CreditType = Literal["revolving_loans", "cash_loans"]


# =========================
# Schemas
# =========================
class ScoreRequest(BaseModel):
    """Request de scoring. SHAP está siempre activo (no se envía desde frontend)."""
    sk_id_curr: int = Field(..., gt=0)
    amt_credit: float = Field(..., gt=0)
    credit_type: CreditType


class NewClientRequest(BaseModel):
    full_name: str = Field(..., min_length=2)
    dni: str = Field(..., min_length=3)
    email: str = Field(..., min_length=5)
    phone: str = Field(..., min_length=5)


# =========================
# Routes
# =========================
@app.get("/", tags=["health"])
def root():
    """Healthcheck + diagnóstico rápido de plantillas."""
    return {
        "status": "ok",
        "message": "Credit Risk API is running",
        "templates_dir": str(TEMPLATES_DIR),
        "index_exists": (TEMPLATES_DIR / "index.html").exists(),
    }


@app.get("/app", response_class=HTMLResponse, tags=["frontend"])
def serve_app(request: Request):
    """Sirve el frontend HTML desde /templates/index.html."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/score", tags=["scoring"])
def score(req: ScoreRequest) -> Dict[str, Any]:
    """Scoring MOCK + SHAP siempre activo (por ahora dummy)."""
    base = min(req.amt_credit / 50000.0, 1.0)
    noise = random.uniform(-0.05, 0.05)
    p_default = max(0.01, min(0.99, 0.10 + 0.35 * base + noise))

    threshold = 0.60
    decision = "REJECT" if p_default >= threshold else "APPROVE"

    # SHAP dummy (hasta conectar el modelo real)
    shap_block = {
        "enabled": True,
        "top_risk_increasing": [],
        "top_risk_decreasing": [],
        "note": "SHAP siempre activo. Ahora mismo es DEMO (sin modelo real).",
    }

    return {
        "sk_id_curr": req.sk_id_curr,
        "input": req.model_dump(),
        "proba": {"no_default": 1 - p_default, "default": p_default},
        "decision": decision,
        "threshold": threshold,
        "shap": shap_block,
    }


@app.post("/api/new-client", tags=["clients"])
def new_client(req: NewClientRequest) -> Dict[str, Any]:
    """Registro DEMO: devuelve un client_id aleatorio."""
    client_id = random.randint(9000, 9999)
    return {"client_id": client_id, "status": "CREATED", "input": req.model_dump()}
