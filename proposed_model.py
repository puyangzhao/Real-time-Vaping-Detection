from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .evaluation import safe_auc, youden_threshold, classification_metrics

class TimeAwareVapingDetector:
    def __init__(self, feature_cols=("intercept", "slope"),
                 b_grid=(48,24,12,8,6,4), eps_grid=(0,1e-2,1e-3,1e-4,1e-5,1e-6),
                 random_state=12345):
        self.feature_cols = list(feature_cols)
        self.b_grid = list(b_grid)
        self.eps_grid = list(eps_grid)
        self.random_state = random_state
        self.b_star_ = None
        self.eps_star_ = None
        self.tau_star_ = None
        self.use_time_ = None
        self.kde_obj_ = None
        self.time_obj_ = None

    @staticmethod
    def _time_bin(timestamps, b):
        ts = pd.to_datetime(timestamps)
        hour = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
        bins = np.floor(hour / (24.0/b)).astype(int)
        return np.clip(bins, 0, b-1)

    def _fit_kde(self, df):
        vap = df[df["is_vaping"] == 1]
        non = df[df["is_vaping"] == 0]
        if len(vap) < 2 or len(non) < 2:
            return None
        try:
            return {
                "kde_v": gaussian_kde(vap[self.feature_cols].values.T),
                "kde_0": gaussian_kde(non[self.feature_cols].values.T),
                "prior_v": len(vap) / len(df)
            }
        except (LinAlgError, ValueError):
            return None

    def _kde_score(self, df, kde):
        if kde is None:
            return np.full(len(df), np.nan)
        X = df[self.feature_cols].values.T
        f_v, f_0, pi = kde["kde_v"](X), kde["kde_0"](X), kde["prior_v"]
        den = f_v*pi + f_0*(1-pi)
        return np.clip(np.divide(f_v*pi, den, out=np.zeros_like(den), where=den>0), 0, 1)

    def _fit_time_models(self, df, b):
        temp = df.copy()
        temp["time_bin"] = self._time_bin(temp["segment_end_time"], b)
        if temp["time_bin"].nunique() < 2:
            return None
        scaler_all = StandardScaler()
        X_all = scaler_all.fit_transform(temp[self.feature_cols])
        y_all = temp["time_bin"].astype(int)
        try:
            model_all = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(X_all, y_all)
        except Exception:
            return None

        vap = temp[temp["is_vaping"] == 1]
        if len(vap) < 2 or vap["time_bin"].nunique() < 2:
            model_vap, scaler_vap = None, None
        else:
            scaler_vap = StandardScaler()
            X_vap = scaler_vap.fit_transform(vap[self.feature_cols])
            y_vap = vap["time_bin"].astype(int)
            try:
                model_vap = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(X_vap, y_vap)
            except Exception:
                model_vap, scaler_vap = None, None
        return {"b": b, "model_all": model_all, "scaler_all": scaler_all,
                "model_vap": model_vap, "scaler_vap": scaler_vap}

    def _prob_class(self, model, scaler, X_df, cls):
        if model is None or scaler is None or len(X_df) == 0:
            return np.zeros(len(X_df))
        prob = model.predict_proba(scaler.transform(X_df[self.feature_cols]))
        classes = list(model.classes_)
        return prob[:, classes.index(cls)] if cls in classes else np.zeros(len(X_df))

    def _time_score(self, df, kde, time_obj, eps):
        base = self._kde_score(df, kde)
        if time_obj is None or np.isinf(eps):
            return base
        temp = df.copy()
        temp["time_bin"] = self._time_bin(temp["segment_end_time"], time_obj["b"])
        p_all = np.zeros(len(temp)); p_vap = np.zeros(len(temp))
        for r in sorted(temp["time_bin"].unique()):
            idx = np.where(temp["time_bin"].values == r)[0]
            X = temp[self.feature_cols].iloc[idx]
            p_all[idx] = self._prob_class(time_obj["model_all"], time_obj["scaler_all"], X, r)
            p_vap[idx] = self._prob_class(time_obj["model_vap"], time_obj["scaler_vap"], X, r)
        score = base.copy()
        use = p_all > eps
        score[use] = (p_vap[use] / p_all[use]) * base[use]
        return np.clip(score, 0, 1)

    def fit_with_lodo_cv(self, dev_df):
        dates = sorted(dev_df["date"].unique())
        fold_base, rows = [], []
        for val_day in dates:
            train = dev_df[dev_df["date"] != val_day]
            val = dev_df[dev_df["date"] == val_day]
            if train["is_vaping"].nunique() < 2 or val["is_vaping"].nunique() < 2:
                continue
            kde = self._fit_kde(train)
            if kde is None: continue
            bscore = self._kde_score(val, kde)
            bauc = safe_auc(val["is_vaping"], bscore)
            if np.isnan(bauc): continue
            fold_base.append(bauc)
            for b in self.b_grid:
                time_obj = self._fit_time_models(train, b)
                for eps in self.eps_grid:
                    score = self._time_score(val, kde, time_obj, eps)
                    eauc = safe_auc(val["is_vaping"], score)
                    if not np.isnan(eauc):
                        rows.append({"b": b, "eps": eps, "base_auc": bauc,
                                     "enhanced_auc": eauc, "effective_auc": max(eauc, bauc)})
        if not fold_base or not rows:
            raise RuntimeError("Unable to perform LODO-CV tuning.")
        cv = pd.DataFrame(rows)
        base_auc = float(np.mean(fold_base))
        summ = cv.groupby(["b","eps"], as_index=False).agg(mean_effective_auc=("effective_auc","mean"))
        summ["delta_auc"] = summ["mean_effective_auc"] - base_auc
        best = summ.sort_values("delta_auc", ascending=False).iloc[0]
        self.b_star_ = int(best["b"])
        self.eps_star_ = float(best["eps"]) if best["delta_auc"] > 0 else np.inf
        self.use_time_ = bool(best["delta_auc"] > 0)

        thresholds = []
        for val_day in dates:
            train = dev_df[dev_df["date"] != val_day]
            val = dev_df[dev_df["date"] == val_day]
            if train["is_vaping"].nunique() < 2 or val["is_vaping"].nunique() < 2: continue
            kde = self._fit_kde(train)
            if kde is None: continue
            time_obj = self._fit_time_models(train, self.b_star_) if self.use_time_ else None
            score = self._time_score(val, kde, time_obj, self.eps_star_)
            tau, _ = youden_threshold(val["is_vaping"], score)
            thresholds.append(tau)
        self.tau_star_ = float(np.mean(thresholds)) if thresholds else 0.5
        self.kde_obj_ = self._fit_kde(dev_df)
        self.time_obj_ = self._fit_time_models(dev_df, self.b_star_) if self.use_time_ else None
        return {"base_auc": base_auc, "enhanced_auc": float(best["mean_effective_auc"]),
                "delta_auc": float(best["delta_auc"]), "b_star": self.b_star_,
                "eps_star": self.eps_star_, "use_time": self.use_time_, "tau_star": self.tau_star_}

    def predict_score(self, df):
        return self._time_score(df, self.kde_obj_, self.time_obj_, self.eps_star_)

    def predict_baseline_score(self, df):
        return self._kde_score(df, self.kde_obj_)

    def evaluate(self, test_df):
        return classification_metrics(test_df["is_vaping"], self.predict_score(test_df), self.tau_star_)
