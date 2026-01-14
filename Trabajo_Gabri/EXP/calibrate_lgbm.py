import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, confusion_matrix

# =====================
# Load dataset
# =====================
df = pd.read_parquet("data/processed/features.parquet")

# Convert categoricals
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].astype("category")

X = df.drop(columns=["TARGET", "SK_ID_CURR"])
y = df["TARGET"]

cat_cols = X.select_dtypes(include="category").columns.tolist()

# =====================
# Split
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================
# LightGBM sklearn API
# =====================
lgbm = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    n_estimators=493,      # 🔧 CAMBIADO (best iteration con bureau)
    random_state=42,
    verbose=-1
)

lgbm.fit(
    X_train,
    y_train,
    categorical_feature=cat_cols
)

# =====================
# Calibración (Platt / Sigmoid)
# =====================
calibrator = CalibratedClassifierCV(
    estimator=lgbm,
    method="sigmoid",
    cv="prefit"
)

calibrator.fit(X_val, y_val)

# =====================
# Evaluación
# =====================
y_val_proba = calibrator.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_proba)

print(f"AUC calibrado: {auc:.4f}")

# =====================
# Threshold table
# =====================
results = []

for thr in np.arange(0.05, 0.51, 0.01):
    y_pred = (y_val_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    results.append({
        "threshold": round(thr, 2),
        "fp_rate": fp / (fp + tn),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp),
        "positives_pred": tp + fp
    })

df_thr = pd.DataFrame(results)

print("\n=== Thresholds con FP <= 15% (CALIBRADO) ===")
print(
    df_thr[df_thr["fp_rate"] <= 0.15]
    .sort_values("recall", ascending=False)
    .head(10)
)
