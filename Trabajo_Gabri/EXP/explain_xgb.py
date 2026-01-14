import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

print("EJECUTANDO ESTE ARCHIVO:", __file__)

# === 1. Cargar dataset FULL ===
df = pd.read_parquet("data/processed/features.parquet")

y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

# object -> category
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# === 2. Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# === 3. Modelo ===
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=-1,
    random_state=42
)

# === 4. ENTRENAR (CLAVE) ===
model.fit(X_train, y_train)

# 🔒 Verificación dura
try:
    model.get_booster()
    print("Modelo entrenado correctamente")
except NotFittedError:
    raise RuntimeError("El modelo NO está entrenado. Revisa el fit().")

# === 5. SHAP ===
booster = model.get_booster()
explainer = shap.TreeExplainer(booster)


X_shap = X_val.sample(n=5000, random_state=42)
shap_values = explainer.shap_values(X_shap)

# === 6. Guardar SHAP ===
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

print("SHAP guardado en reports/figures/shap_summary_xgb.png")
