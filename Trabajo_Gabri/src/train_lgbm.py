import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import confusion_matrix
from src.evaluation.threshold_analysis import threshold_analysis

# =====================
# Load data
# =====================
df = pd.read_parquet("data/processed/features.parquet")

# Convert object columns to category (CRUCIAL for LightGBM)
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype("category")

# =====================
# Split features / target
# =====================
X = df.drop(columns=["TARGET", "SK_ID_CURR"])
y = df["TARGET"]

cat_cols = X.select_dtypes(include="category").columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================
# LightGBM datasets
# =====================
train_set = lgb.Dataset(
    X_train,
    y_train,
    categorical_feature=cat_cols
)

val_set = lgb.Dataset(
    X_val,
    y_val,
    categorical_feature=cat_cols
)

# =====================
# Model params
# =====================
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
    "verbose": -1
}

# =====================
# Train
# =====================
model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[val_set],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(100)
    ]
)

# =====================
# Evaluate
# =====================
y_val_pred = model.predict(X_val)
auc = roc_auc_score(y_val, y_val_pred)

print(f"AUC LightGBM: {auc:.4f}")

# Probabilidades
y_val_proba = y_val_pred

results = []

for thr in np.arange(0.05, 0.51, 0.01):
    y_pred = (y_val_proba >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    fp_rate = fp / (fp + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    results.append({
        "threshold": round(thr, 2),
        "fp_rate": fp_rate,
        "recall": recall,
        "precision": precision,
        "positives_pred": tp + fp
    })

results_df = pd.DataFrame(results)

# =====================
# Threshold analysis
# =====================
y_val_proba = y_val_pred

results = []

for thr in np.arange(0.05, 0.51, 0.01):
    y_pred = (y_val_proba >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    results.append({
        "threshold": round(thr, 2),
        "fp_rate": round(fp_rate, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "positives_pred": int(tp + fp)
    })

results_df = pd.DataFrame(results)

# =====================
# 1️⃣ PRINT FULL TABLE (IMPORTANT)
# =====================
print("\n=== Tabla COMPLETA de thresholds ===")
print(results_df)

# =====================
# 2️⃣ FILTER BY FP RATE
# =====================
filtered = results_df[results_df["fp_rate"] <= 0.15]

print("\n=== Thresholds con FP <= 15% ===")
print(filtered.sort_values("recall", ascending=False).head(10))

# =====================
# Threshold analysis (USANDO MODULO)
# =====================

df_thresh = threshold_analysis(
    y_true=y_val,
    y_proba=y_val_proba
)

print("\n=== Tabla COMPLETA de thresholds ===")
print(df_thresh.head(20))

print("\n=== Thresholds con FP rate <= 15% ===")
print(
    df_thresh[df_thresh["fp_rate"] <= 0.15]
    .sort_values("recall", ascending=False)
    .head(10)
)