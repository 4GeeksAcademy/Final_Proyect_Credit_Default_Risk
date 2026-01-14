import pandas as pd
import numpy as np
from tqdm import tqdm


def _pos_cash_features_from_df(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "pos_dpd_mean_all": 0,
            "pos_late_ratio_all": 0,
            "pos_dpd_mean_last_6m": 0,
            "pos_late_ratio_last_6m": 0,
            "pos_dpd_mean_older": 0,
            "pos_late_ratio_older": 0,
            "pos_dpd_trend": 0,
            "months_since_last_late_pos": np.nan,
        }

    df = df.sort_values("MONTHS_BALANCE")

    def safe_mean(x): 
        return x.mean() if not x.empty else 0

    def safe_ratio(x): 
        return (x > 0).mean() if not x.empty else 0

    dpd_all = df["SK_DPD"]

    last_6m = df[df["MONTHS_BALANCE"] >= -6]
    older = df[df["MONTHS_BALANCE"] < -6]

    late_months = df.loc[df["SK_DPD"] > 0, "MONTHS_BALANCE"]
    months_since_last_late = (
        abs(late_months.max()) if not late_months.empty else np.nan
    )

    return {
        "pos_dpd_mean_all": safe_mean(dpd_all),
        "pos_late_ratio_all": safe_ratio(dpd_all),
        "pos_dpd_mean_last_6m": safe_mean(last_6m["SK_DPD"]),
        "pos_late_ratio_last_6m": safe_ratio(last_6m["SK_DPD"]),
        "pos_dpd_mean_older": safe_mean(older["SK_DPD"]),
        "pos_late_ratio_older": safe_ratio(older["SK_DPD"]),
        "pos_dpd_trend": safe_mean(last_6m["SK_DPD"]) - safe_mean(older["SK_DPD"]),
        "months_since_last_late_pos": months_since_last_late,
    }


def build_pos_cash_agg(pos: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    pos.columns = pos.columns.str.strip()
    prev.columns = prev.columns.str.strip()

    # 🔥 INDEXAR UNA SOLA VEZ
    pos_by_prev = pos.groupby("SK_ID_PREV")

    rows = []

    for sk_id, prev_grp in tqdm(
        prev.groupby("SK_ID_CURR"),
        total=prev["SK_ID_CURR"].nunique(),
        desc="⚡ Building POS_CASH features"
    ):
        prev_ids = prev_grp["SK_ID_PREV"].unique()

        dfs = [
            pos_by_prev.get_group(pid)
            for pid in prev_ids
            if pid in pos_by_prev.groups
        ]

        df_client = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        feats = _pos_cash_features_from_df(df_client)
        feats["SK_ID_CURR"] = sk_id
        rows.append(feats)

    return pd.DataFrame(rows)
