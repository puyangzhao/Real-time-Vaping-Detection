from __future__ import annotations
import numpy as np

def add_linear_trend_features(segments, values_col="heart_rate_values"):
    out = segments.copy()
    intercepts, slopes = [], []
    for values in out[values_col]:
        y = np.asarray(values, dtype=float)
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        mask = np.isfinite(y)
        if mask.sum() < 2:
            intercepts.append(np.nan); slopes.append(np.nan); continue
        X = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        intercepts.append(float(beta[0])); slopes.append(float(beta[1]))
    out["intercept"] = intercepts
    out["slope"] = slopes
    return out.dropna(subset=["intercept", "slope"]).reset_index(drop=True)
