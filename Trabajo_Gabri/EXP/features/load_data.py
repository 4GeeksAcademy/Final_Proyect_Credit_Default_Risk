import pandas as pd
from pathlib import Path
from .installments import build_installments_agg
from .pos_cash import build_pos_cash_agg
from .previous_application import build_prev_app_agg
from .bureau import build_bureau_agg

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]

def load_data():
    app = pd.read_csv(BASE_DIR / "data/raw/application_tr.csv")
    prev = pd.read_csv(BASE_DIR / "data/raw/previous_application.csv")
    pos = pd.read_csv(BASE_DIR / "data/raw/POS_CASH_balance.csv")
    inst = pd.read_csv(BASE_DIR / "data/raw/installments_payments.csv")
    bureau = pd.read_csv(BASE_DIR / "data/raw/bureau.csv")

    # Pre-agregaciones a nivel cliente
    pos_agg = build_pos_cash_agg(pos, prev)
    inst_agg = build_installments_agg(inst, prev)
    prev_agg = build_prev_app_agg(prev)
    bureau_agg = build_bureau_agg(bureau)

    return {
        "app": app,
        "prev_agg": prev_agg,
        "bureau_agg": bureau_agg,
        "pos_agg": pos_agg,
        "inst_agg": inst_agg
    }
