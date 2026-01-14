import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split

# =====================
# Paths
# =====================
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
# Train model
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

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
# SHAP
# =====================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Guardar valores SHAP y datos
np.save(f"{OUTPUT_DIR}/shap_values.npy", shap_values)
X_val.to_parquet(f"{OUTPUT_DIR}/X_val.parquet")

# =====================
# Global plots
# =====================
shap.summary_plot(
    shap_values,
    X_val,
    max_display=20,
    show=False
)
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", bbox_inches="tight")
plt.close()

shap.summary_plot(
    shap_values,
    X_val,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.savefig(f"{OUTPUT_DIR}/shap_summary_bar.png", bbox_inches="tight")
plt.close()

print("✅ SHAP GLOBAL guardado en reports/shap/")