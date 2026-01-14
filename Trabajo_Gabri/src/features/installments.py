import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def _installments_features_from_df(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "inst_late_ratio_all": 0,
            "inst_dpd_mean_all": 0,
            "inst_dpd_max_all": 0,
            "inst_payment_ratio_all": 1,
            "inst_late_ratio_last_6m": 0,
            "inst_dpd_trend": 0,
            "months_since_last_late_inst": np.nan,
        }

    dpd = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)

    late_ratio_all = (dpd > 0).mean()
    dpd_mean_all = dpd.mean()
    dpd_max_all = dpd.max()

    total_inst = df["AMT_INSTALMENT"].sum()
    payment_ratio_all = (
        df["AMT_PAYMENT"].sum() / total_inst if total_inst > 0 else 1
    )

    last_6m = df[df["DAYS_INSTALMENT"] >= -180]
    dpd_last_6m = (
        (last_6m["DAYS_ENTRY_PAYMENT"] - last_6m["DAYS_INSTALMENT"])
        .clip(lower=0)
        if not last_6m.empty
        else pd.Series(dtype=float)
    )

    late_ratio_last_6m = (dpd_last_6m > 0).mean() if not dpd_last_6m.empty else 0
    inst_dpd_trend = (
        dpd_last_6m.mean() - dpd_mean_all if not dpd_last_6m.empty else 0
    )

    late_rows = df.loc[dpd > 0, "DAYS_INSTALMENT"]
    months_since_last_late = (
        abs(late_rows.max()) // 30 if not late_rows.empty else np.nan
    )

    return {
        "inst_late_ratio_all": late_ratio_all,
        "inst_dpd_mean_all": dpd_mean_all,
        "inst_dpd_max_all": dpd_max_all,
        "inst_payment_ratio_all": payment_ratio_all,
        "inst_late_ratio_last_6m": late_ratio_last_6m,
        "inst_dpd_trend": inst_dpd_trend,
        "months_since_last_late_inst": months_since_last_late,
    }


def build_installments_agg(inst: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    inst.columns = inst.columns.str.strip()
    prev.columns = prev.columns.str.strip()

    # 🔥 INDEXAR UNA SOLA VEZ (CLAVE PARA VELOCIDAD)
    inst_by_prev = inst.groupby("SK_ID_PREV")

    rows = []

    for sk_id, prev_grp in tqdm(
        prev.groupby("SK_ID_CURR"),
        total=prev["SK_ID_CURR"].nunique(),
        desc="⚡ Building installments features"
    ):
        prev_ids = prev_grp["SK_ID_PREV"].unique()

        dfs = [
            inst_by_prev.get_group(pid)
            for pid in prev_ids
            if pid in inst_by_prev.groups
        ]

        df_client = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        feats = _installments_features_from_df(df_client)
        feats["SK_ID_CURR"] = sk_id
        rows.append(feats)

    return pd.DataFrame(rows)