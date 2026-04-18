"""Baseline predictors of log forward realized volatility (Stage 1).

All three baselines return a numpy array of predictions aligned with
`test_df` rows. Expected columns (present in both `train_df` and
`test_df`, produced by `stage1_training.compute_target_and_derived`):

    target          — log(sigma_rv), the regression target
    sigma_gk_lag1   — sigma_gk at bar t-1 (strictly positive after filter)
    sigma_gk_short  — 12-bar rolling mean of sigma_gk (strictly positive)
    sigma_gk_long   — 60-bar rolling mean of sigma_gk (strictly positive)
    vpin            — rolling-20 mean of |imbalance_t|

B1 (naive persistence):
    log_sigma_rv_hat = log(sigma_gk_lag1) + 0.5 * log(12)
    No training; `train_df` is unused.

B2 (VPIN + lagged sigma OLS):
    log_sigma_rv ~ a0 + a1 * vpin + a2 * log(sigma_gk_lag1)

B3 (HAR-RV OLS, Corsi 2009 analog at 5min resolution):
    log_sigma_rv ~ b0 + b1 * log(sigma_gk_lag1)
                     + b2 * log(sigma_gk_short)
                     + b3 * log(sigma_gk_long)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

HORIZON = 12
_FLOOR = 1e-10


def _log_pos(arr) -> np.ndarray:
    """Safe log with a small floor to avoid -inf if any non-positive values slip through filters."""
    return np.log(np.clip(np.asarray(arr, dtype=float), _FLOOR, None))


def fit_predict_b1(train_df, test_df) -> np.ndarray:
    """Naive persistence. No training performed."""
    return _log_pos(test_df['sigma_gk_lag1'].values) + 0.5 * np.log(HORIZON)


def fit_predict_b2(train_df, test_df) -> np.ndarray:
    """VPIN + log(sigma_gk_lag1) OLS with intercept."""
    X_tr = np.column_stack([
        train_df['vpin'].values,
        _log_pos(train_df['sigma_gk_lag1'].values),
    ])
    y_tr = train_df['target'].values
    X_te = np.column_stack([
        test_df['vpin'].values,
        _log_pos(test_df['sigma_gk_lag1'].values),
    ])
    model = LinearRegression().fit(X_tr, y_tr)
    return model.predict(X_te)


def fit_predict_b3(train_df, test_df) -> np.ndarray:
    """HAR-RV OLS on {log(sigma_gk_lag1), log(sigma_gk_short), log(sigma_gk_long)} with intercept."""
    cols = ('sigma_gk_lag1', 'sigma_gk_short', 'sigma_gk_long')
    X_tr = np.column_stack([_log_pos(train_df[c].values) for c in cols])
    y_tr = train_df['target'].values
    X_te = np.column_stack([_log_pos(test_df[c].values) for c in cols])
    model = LinearRegression().fit(X_tr, y_tr)
    return model.predict(X_te)


