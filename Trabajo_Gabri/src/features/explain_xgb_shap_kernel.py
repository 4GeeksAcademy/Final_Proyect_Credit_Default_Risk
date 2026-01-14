import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

print("EJECUTANDO SHAP (KERNEL):", __file__)

# === 1. Cargar dataset ===
df = pd.read_parquet("data/processed/features.parquet")

y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

# === 2. Codificar categóricas PARA SHAP ===
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category").cat.codes

# === 3. Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# === 4. Entrenar modelo SOLO PARA SHAP ===
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
print("Modelo SHAP entrenado")

# === 5. KernelExplainer (estable) ===
# Usar muestras pequeñas (clave)
X_background = shap.sample(X_train, 100, random_state=42)
X_shap = shap.sample(X_val, 500, random_state=42)

def model_predict(X):
    return model.predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(model_predict, X_background)
shap_values = explainer.shap_values(X_shap, nsamples=200)

# === 6. Guardar figura ===
os.makedirs("reports/figures", exist_ok=True)

shap.summary_plot(
    shap_values,
    X_shap,
    max_display=20,
    show=False
)

plt.tight_layout()
plt.savefig("reports/figures/shap_summary_xgb.png", dpi=200)
plt.close()

print("SHAP generado en reports/figures/shap_summary_xgb.png")
