import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def threshold_analysis(
    y_true,
    y_proba,
    thresholds=np.arange(0.05, 0.95, 0.01)
):
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)

        rows.append({
            "threshold": round(t, 3),
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "precision": precision,
            "recall": recall
        })

    return pd.DataFrame(rows)
