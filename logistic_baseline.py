from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .evaluation import youden_threshold, classification_metrics

def add_cyclic_time_features(df, time_col="segment_end_time"):
    out = df.copy()
    ts = pd.to_datetime(out[time_col])
    hour = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    out["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    out["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    return out

def run_logistic_baseline(train_df, valid_df, test_df, feature_cols=("intercept","slope","hour_sin","hour_cos"), random_state=12345):
    train_df, valid_df, test_df = map(add_cyclic_time_features, [train_df, valid_df, test_df])
    if train_df["is_vaping"].nunique() < 2:
        raise RuntimeError("Training data must contain both classes.")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=random_state))
    ])
    model.fit(train_df[list(feature_cols)], train_df["is_vaping"].astype(int))
    valid_score = model.predict_proba(valid_df[list(feature_cols)])[:, 1]
    tau, valid_auc = youden_threshold(valid_df["is_vaping"], valid_score)
    test_score = model.predict_proba(test_df[list(feature_cols)])[:, 1]
    metrics = classification_metrics(test_df["is_vaping"], test_score, tau)
    metrics["Valid_AUC"] = valid_auc
    metrics["Threshold"] = tau
    return metrics
