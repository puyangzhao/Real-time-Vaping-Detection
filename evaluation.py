from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def safe_auc(y_true, score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, score)

def youden_threshold(y_true, score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5, np.nan
    fpr, tpr, thresholds = roc_curve(y_true, score)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx]), float(roc_auc_score(y_true, score))

def classification_metrics(y_true, score, threshold):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    auc_val = safe_auc(y_true, score)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AUC": auc_val,
        "Accuracy": accuracy_score(y_true, pred),
        "Sensitivity": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "Specificity": tn/(tn+fp) if (tn+fp)>0 else np.nan,
        "FPR": fp/(fp+tn) if (fp+tn)>0 else np.nan,
        "PPV": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "NPV": tn/(tn+fn) if (tn+fn)>0 else np.nan,
        "F1": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }
