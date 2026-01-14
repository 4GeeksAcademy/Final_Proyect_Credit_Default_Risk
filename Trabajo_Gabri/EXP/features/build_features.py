import pandas as pd
from .application import application_features
from .previous import previous_features


def build_features(sk_id, data):
    # Base cliente
    app_f = application_features(data["app"], sk_id)

    # Historial de préstamos previos
    prev_f = previous_features(data["prev"], sk_id)

    # POS ya agregado (una fila por cliente)
    pos_f = data["pos_agg"].loc[
        data["pos_agg"]["SK_ID_CURR"] == sk_id
    ].drop(columns=["SK_ID_CURR"], errors="ignore")

    # Installments ya agregado (una fila por cliente)
    inst_f = data["inst_agg"].loc[
        data["inst_agg"]["SK_ID_CURR"] == sk_id
    ].drop(columns=["SK_ID_CURR"], errors="ignore")

    return pd.concat([app_f, prev_f, pos_f, inst_f], axis=1)
