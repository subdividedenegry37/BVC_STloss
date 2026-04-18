"""Per-fold metrics helpers for the Stage 1 full run.

Functions here are pure: they take arrays, return dicts/lists. No I/O.

Metrics defined (all on log-sigma-rv scale):
  * rho          — Spearman rank correlation of (target, prediction)
  * rmse         — sqrt(mean((target - prediction)^2))
  * mae          — mean(|target - prediction|)
  * hit_rate     — P(sign(pred - pivot) == sign(target - pivot))
                   where pivot is the training-set median of the target
                   (passed in). This operationalizes the spec's
                   "above/below rolling mean" directional metric as a
                   leakage-free direction sign, using train-only info.
  * n            — row count in the slice

Authorized sanity bounds (per full-run authorization, 2026-04-18):
  FINAL_SANITY = {
      'rho_har_lower': 0.50, 'rho_har_upper': 0.80,
      'rho_lgb_lower': 0.50, 'rho_lgb_upper': 0.85,
      'diff_lower': -0.02,   'diff_upper':  0.12,
  }
Rationale is documented in STAGE1_RESULTS.md.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr

FINAL_SANITY = {
    'rho_har_lower': 0.50, 'rho_har_upper': 0.80,
    'rho_lgb_lower': 0.50, 'rho_lgb_upper': 0.85,
    'diff_lower': -0.02, 'diff_upper': 0.12,
}

MODEL_KEYS = ('lgb', 'b1_naive', 'b2_vpin', 'b3_har')


def slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, pivot: float) -> dict:
    """Compute {rho, rmse, mae, hit_rate, n} on one (y_true, y_pred) slice.
    `pivot` is the train-set median of the target; used for hit_rate only."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    if n < 2:
        return {'n': n, 'rho': float('nan'), 'rmse': float('nan'),
                'mae': float('nan'), 'hit_rate': float('nan')}
    resid = y_true - y_pred
    rho, _ = spearmanr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean(resid * resid)))
    mae = float(np.mean(np.abs(resid)))
    hit = float(np.mean(np.sign(y_pred - pivot) == np.sign(y_true - pivot)))
    return {'n': n, 'rho': float(rho), 'rmse': rmse, 'mae': mae, 'hit_rate': hit}


def collect_fold_metrics(
    *, fold_id: int, y_true: np.ndarray, preds: Mapping[str, np.ndarray],
    contracts: np.ndarray, contract_order: Iterable[str], pivot: float,
) -> list[dict]:
    """Produce long-format rows:
        [{'fold_id', 'model', 'slice', 'n', 'rho', 'rmse', 'mae', 'hit_rate'}, ...]
    One row per (model, slice) where slice is 'pooled' or a contract symbol.
    """
    rows: list[dict] = []
    for model_name in MODEL_KEYS:
        if model_name not in preds:
            continue
        yhat = np.asarray(preds[model_name], dtype=float)
        m = slice_metrics(y_true, yhat, pivot)
        m.update({'fold_id': int(fold_id), 'model': model_name, 'slice': 'pooled'})
        rows.append(m)
        for c in contract_order:
            mask = (contracts == c)
            if mask.sum() < 10:
                continue
            m = slice_metrics(y_true[mask], yhat[mask], pivot)
            m.update({'fold_id': int(fold_id), 'model': model_name, 'slice': c})
            rows.append(m)
    return rows


def success_criteria(per_fold_delta: np.ndarray, per_contract_delta: Mapping[str, float],
                     pooled_rho_lgb: float, pooled_rho_b3: float) -> dict:
    """Spec §7 criteria:
      (1) pooled Δρ >= 0.03
      (2) lgb beats b3 in >= 70% of folds (sign of per-fold delta)
      (3) per-contract: lgb beats b3 on at least 2 of 3 contracts individually
    """
    d1 = float(pooled_rho_lgb - pooled_rho_b3)
    crit1 = d1 >= 0.03
    pos = int(np.sum(np.asarray(per_fold_delta) > 0))
    crit2 = pos / max(1, len(per_fold_delta)) >= 0.70
    n_contracts_beating = sum(1 for v in per_contract_delta.values() if v > 0)
    crit3 = n_contracts_beating >= 2
    return {
        'c1_pooled_delta': {'pass': bool(crit1), 'value': d1, 'threshold': 0.03},
        'c2_sign_consistency': {
            'pass': bool(crit2),
            'value': pos / max(1, len(per_fold_delta)),
            'threshold': 0.70,
            'folds_positive': pos,
            'folds_total': int(len(per_fold_delta)),
        },
        'c3_per_contract_generality': {
            'pass': bool(crit3),
            'n_beating': int(n_contracts_beating),
            'threshold': 2,
            'per_contract_delta': {k: float(v) for k, v in per_contract_delta.items()},
        },
        'all_pass': bool(crit1 and crit2 and crit3),
    }
