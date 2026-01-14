import pandas as pd

def application_features(app, sk_id):
    row = app[app["SK_ID_CURR"] == sk_id]

    if row.empty:
        raise ValueError("Cliente no encontrado")

    return pd.DataFrame({
        "income": row["AMT_INCOME_TOTAL"].values,
        "credit": row["AMT_CREDIT"].values,
        "annuity": row["AMT_ANNUITY"].values,
        "credit_income_ratio": (
            row["AMT_CREDIT"].values / row["AMT_INCOME_TOTAL"].values
        ),
        "days_birth": row["DAYS_BIRTH"].values,
        "days_employed": row["DAYS_EMPLOYED"].values,
    })