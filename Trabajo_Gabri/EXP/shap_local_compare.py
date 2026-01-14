import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.model_selection import train_test_split

# =====================
# CONFIG
# =====================
THRESHOLD = 0.59
OUTPUT_DIR = "reports/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Load data
# =====================
df = pd.read_parquet("data/processed/features.parquet")

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category")

X = df.drop(columns=["TARGET", "SK_ID_CURR"])
y = df["TARGET"]

cat_cols = X.select_dtypes(include="category").columns.tolist()

# =====================
# Train / Validation split
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================
# Train model (best iteration fixed)
# =====================
model = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    n_estimators=493,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train, categorical_feature=cat_cols)

# =====================
# Predict probabilities
# =====================
y_val_proba = model.predict_proba(X_val)[:, 1]

# =====================
# Select one ACCEPTED and one REJECTED client
# =====================
accepted_idx = np.where(y_val_proba < THRESHOLD)[0][0]
rejected_idx = np.where(y_val_proba >= THRESHOLD)[0][0]

clients = {
    "ACEPTADO": accepted_idx,
    "RECHAZADO": rejected_idx
}

# =====================
# SHAP
# =====================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# =====================
# Plot comparison
# =====================
for label, idx in clients.items():
    proba = y_val_proba[idx]

    print("===================================")
    print(f"Cliente {label}")
    print(f"Índice: {idx}")
    print(f"Probabilidad de default: {proba:.4f}")
    print(f"Threshold aplicado: {THRESHOLD}")
    print("===================================")

    shap_exp = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_val.iloc[idx],
        feature_names=X_val.columns
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap_exp,
        max_display=10,
        show=False
    )

    plt.title(
        f"{label} | PD={proba:.2f}",
        fontsize=11
    )

    plt.savefig(
        f"{OUTPUT_DIR}/shap_waterfall_{label}.png",
        bbox_inches="tight"
    )
    plt.close()

    print(f"✅ SHAP {label} guardado en {OUTPUT_DIR}")
