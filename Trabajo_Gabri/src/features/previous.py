import pandas as pd

def previous_features(prev, sk_id):
    df = prev[prev["SK_ID_CURR"] == sk_id]

    if df.empty:
        return pd.DataFrame({
            "n_prev_loans": [0],
            "prev_approved_ratio": [0],
            "prev_refused_ratio": [0],
        })

    return pd.DataFrame({
        "n_prev_loans": [len(df)],
        "prev_approved_ratio": [
            (df["NAME_CONTRACT_STATUS"] == "Approved").mean()
        ],
        "prev_refused_ratio": [
            (df["NAME_CONTRACT_STATUS"] == "Refused").mean()
        ],
    })