import pandas as pd
import numpy as np


def build_bureau_agg(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau.columns = bureau.columns.str.strip()

    # Flags útiles
    bureau["HAS_DPD"] = (bureau["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    bureau["ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)

    agg = (
        bureau
        .groupby("SK_ID_CURR")
        .agg(
            BUREAU_COUNT=("SK_ID_BUREAU", "count"),

            BUREAU_ACTIVE_SUM=("ACTIVE", "sum"),
            BUREAU_CLOSED_SUM=("CLOSED", "sum"),

            BUREAU_DPD_SUM=("HAS_DPD", "sum"),
            BUREAU_DPD_RATE=("HAS_DPD", "mean"),

            BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
            BUREAU_CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),

            BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
            BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),

            BUREAU_AMT_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
            BUREAU_AMT_CREDIT_SUM_DEBT_MAX=("AMT_CREDIT_SUM_DEBT", "max"),

            BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
            BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        )
        .reset_index()
    )

    # Ratios robustos
    agg["BUREAU_ACTIVE_RATIO"] = (
        agg["BUREAU_ACTIVE_SUM"] / agg["BUREAU_COUNT"]
    )

    return agg