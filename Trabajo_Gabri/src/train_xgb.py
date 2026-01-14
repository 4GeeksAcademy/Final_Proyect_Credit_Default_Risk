import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import precision_score, recall_score

# === 1. Cargar dataset FULL ===
df = pd.read_parquet("data/processed/features.parquet")
print("Dataset cargado:", df.shape)

# === 2. Separar X / y ===
y = df["TARGET"]
X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

# === 3. Convertir object -> category (CLAVE)
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

print("Categorical columns:", X.select_dtypes(include="category").shape[1])

# === 4. Split estratificado
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# === 5. Modelo XGBoost con categóricas activadas
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,   # 🔥 CLAVE
    n_jobs=-1,
    random_state=42
)

# === 6. Entrenar
model.fit(X_train, y_train)

# === 7. Evaluar
y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)

print(f"AUC FULL dataset (XGB): {auc:.4f}")

# === Evaluación ===
y_val_proba = model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba >= 0.2).astype(int)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

print("\n=== Threshold tuning ===")
print("thr | precision | recall | positives_pred")

for thr in thresholds:
    y_pred_thr = (y_val_proba >= thr).astype(int)

    precision = precision_score(y_val, y_pred_thr, zero_division=0)
    recall = recall_score(y_val, y_pred_thr)
    positives = y_pred_thr.sum()

    print(
        f"{thr:>3} | "
        f"{precision:>9.3f} | "
        f"{recall:>6.3f} | "
        f"{positives:>14}"
    )

auc = roc_auc_score(y_val, y_val_proba)
print(f"AUC Validation: {auc:.4f}")

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_val, y_val_proba)

os.makedirs("reports/figures", exist_ok=True)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Credit Risk")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/roc_curve_xgb.png", dpi=200)
plt.close()

# === Confusion Matrix ===
cm = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Threshold 0.2")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix_xgb_thr_02.png", dpi=200)
plt.close()

# === Classification Report ===
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

# =========================
# === CALIBRACIÓN =========
# =========================

print("\n=== Probability Calibration (Isotonic) ===")

# Modelo calibrado usando el modelo ya entrenado
calibrated_model = CalibratedClassifierCV(
    model,
    method="isotonic",
    cv="prefit"
)

calibrated_model.fit(X_val, y_val)

# Probabilidades
y_val_proba_uncal = y_val_proba
y_val_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]

# Brier score
brier_uncal = brier_score_loss(y_val, y_val_proba_uncal)
brier_cal = brier_score_loss(y_val, y_val_proba_cal)

print(f"Brier score (uncalibrated): {brier_uncal:.4f}")
print(f"Brier score (calibrated)  : {brier_cal:.4f}")

# Curvas de calibración
prob_true_uncal, prob_pred_uncal = calibration_curve(
    y_val, y_val_proba_uncal, n_bins=10
)
prob_true_cal, prob_pred_cal = calibration_curve(
    y_val, y_val_proba_cal, n_bins=10
)

# === Plot ===
plt.figure(figsize=(6, 5))
plt.plot(prob_pred_uncal, prob_true_uncal, "o-", label="Uncalibrated")
plt.plot(prob_pred_cal, prob_true_cal, "o-", label="Calibrated (Isotonic)")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Predicted probability")
plt.ylabel("Observed default rate")
plt.title("Calibration Curve - XGBoost")
plt.legend()
plt.tight_layout()

plt.savefig("reports/figures/calibration_curve_xgb.png", dpi=200)
plt.close()

print("Calibration curve guardada en reports/figures/calibration_curve_xgb.png")
