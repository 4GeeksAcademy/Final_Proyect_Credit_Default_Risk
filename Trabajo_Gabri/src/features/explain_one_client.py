import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

THRESHOLD = 0.2
CLIENT_INDEX = None  # si quieres forzar uno concreto luego

print("EJECUTANDO SHAP LOCAL")

# === 1. Cargar dataset ===
df = pd.read_parquet("data/processed/features.parquet")

y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

# Codificar categóricas (solo para SHAP)
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category").cat.codes

# === 2. Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# === 3. Entrenar modelo (equivalente) ===
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

# === 4. Elegir cliente con riesgo ≥ threshold ===
probas = model.predict_proba(X_val)[:, 1]

if CLIENT_INDEX is None:
    idx = np.where(probas >= THRESHOLD)[0][0]
else:
    idx = CLIENT_INDEX

x_client = X_val.iloc[[idx]]
p_client = probas[idx]

print(f"Cliente seleccionado | Probabilidad default = {p_client:.3f}")

# === 5. SHAP local (Kernel) ===
X_background = shap.sample(X_train, 100, random_state=42)

def model_predict(X):
    return model.predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(model_predict, X_background)
shap_values = explainer.shap_values(x_client, nsamples=200)

# === 6. Guardar waterfall plot ===
os.makedirs("reports/figures", exist_ok=True)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=x_client.iloc[0],
        feature_names=x_client.columns
    ),
    show=False
)

plt.tight_layout()
plt.savefig("reports/figures/shap_local_client.png", dpi=200)
plt.close()

print("SHAP local guardado en reports/figures/shap_local_client.png")
