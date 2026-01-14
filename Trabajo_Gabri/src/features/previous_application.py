import pandas as pd
import numpy as np


def build_prev_app_agg(prev: pd.DataFrame) -> pd.DataFrame:
    prev.columns = prev.columns.str.strip()

    # Flags útiles
    prev["APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    # Diferencia importe solicitado vs concedido
    prev["AMT_DIFF"] = (
        prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    )

    agg = (
        prev
        .groupby("SK_ID_CURR")
        .agg(
            PREV_COUNT=("SK_ID_PREV", "count"),

            PREV_APPROVED_SUM=("APPROVED", "sum"),
            PREV_REFUSED_SUM=("REFUSED", "sum"),

            PREV_APPROVAL_RATE=("APPROVED", "mean"),

            PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
            PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
            PREV_AMT_DIFF_MEAN=("AMT_DIFF", "mean"),

            PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
            PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),

            PREV_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
            PREV_GOODS_PRICE_MEAN=("AMT_GOODS_PRICE", "mean"),
        )
        .reset_index()
    )

    # Ratios robustos
    agg["PREV_REFUSED_RATIO"] = (
        agg["PREV_REFUSED_SUM"] / agg["PREV_COUNT"]
    )

    return agg
