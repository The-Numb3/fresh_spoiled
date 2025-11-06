# meatscan/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    mets = dict(
        accuracy = accuracy_score(y_true, y_pred),
        precision= precision_score(y_true, y_pred, zero_division=0),
        recall   = recall_score(y_true, y_pred, zero_division=0),
        f1       = f1_score(y_true, y_pred, zero_division=0),
        roc_auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else np.nan,
        pr_auc   = average_precision_score(y_true, y_prob) if len(np.unique(y_true))==2 else np.nan,
        confusion_matrix = confusion_matrix(y_true, y_pred).tolist()
    )
    return mets
