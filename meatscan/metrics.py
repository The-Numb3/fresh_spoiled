import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix
)

def _binary_metrics(y_true, p_pos, threshold=0.5):
    y_pred = (p_pos >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # 점수 계산이 불가능할 수 있는 경우를 대비해 try/except
    try:
        roc = roc_auc_score(y_true, p_pos)
    except Exception:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, p_pos)
    except Exception:
        pr = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc, "pr_auc": pr, "confusion_matrix": cm
    }

def _multiclass_metrics(y_true, P):  # P: (N, C) softmax 확률
    y_pred = P.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    # macro/micro/weighted 모두 리포팅
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    # 멀티클래스 ROC-AUC/PR-AUC (ovr, macro-average) 시도
    C = P.shape[1]
    roc = pr = float("nan")
    try:
        # one-vs-rest용 이진표현
        Y = np.eye(C, dtype=int)[y_true]
        roc = roc_auc_score(Y, P, average="macro", multi_class="ovr")
    except Exception:
        pass
    try:
        Y = np.eye(C, dtype=int)[y_true]
        pr = average_precision_score(Y, P, average="macro")
    except Exception:
        pass

    return {
        "accuracy": acc,
        "precision_micro": prec_micro, "recall_micro": rec_micro, "f1_micro": f1_micro,
        "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro,
        "precision_weighted": prec_w, "recall_weighted": rec_w, "f1_weighted": f1_w,
        "roc_auc_macro_ovr": roc, "pr_auc_macro_ovr": pr,
        "confusion_matrix": cm
    }

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    - y_pred_proba.ndim == 1  → binary (양성 확률)
    - y_pred_proba.ndim == 2  → multiclass (각 클래스 확률)
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if y_pred_proba.ndim == 1:
        return _binary_metrics(y_true, y_pred_proba, threshold=threshold)
    elif y_pred_proba.ndim == 2:
        return _multiclass_metrics(y_true, y_pred_proba)
    else:
        raise ValueError("y_pred_proba must be 1D (binary) or 2D (multiclass).")
