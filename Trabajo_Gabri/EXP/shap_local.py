import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

# =====================
# CONFIG
# =====================
THRESHOLD = 0.59  # threshold oficial definido en el análisis previo
OUTPUT_DIR = "reports/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Load data
# =====================
df = pd.read_parquet("data/processed/features.parquet")

# Convert object columns to category (LightGBM)
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
# Train model (fixed n_estimators = best iteration)
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
    n_estimators=493,  # best iteration del entrenamiento
    random_state=42,
    verbose=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=cat_cols
)

# =====================
# Predict probabilities
# =====================
y_val_proba = model.predict_proba(X_val)[:, 1]

# =====================
# SHAP
# =====================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# =====================
# Explain ONE client with decision
# =====================
idx = 0  # cambia este índice si quieres otro cliente

client_proba = y_val_proba[idx]
decision = "RECHAZADO" if client_proba >= THRESHOLD else "ACEPTADO"

print("===================================")
print(f"Cliente índice: {idx}")
print(f"Probabilidad de default: {client_proba:.4f}")
print(f"Threshold aplicado: {THRESHOLD}")
print(f"DECISIÓN DEL MODELO: {decision}")
print("===================================")

# =====================
# SHAP WATERFALL (mejor legibilidad)
# =====================
shap_values_single = shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_val.iloc[idx],
    feature_names=X_val.columns
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(
    shap_values_single,
    max_display=10,   # solo top 10 features
    show=False
)

plt.title(
    f"Cliente {idx} | PD={client_proba:.2f} | {decision}",
    fontsize=11
)

plt.savefig(
    f"{OUTPUT_DIR}/shap_waterfall_client_{idx}_{decision}.png",
    bbox_inches="tight"
)
plt.close()

print(f"✅ SHAP LOCAL guardado en {OUTPUT_DIR}/shap_local_client_{idx}_{decision}.png")
