"""Single-fold validation of the Stage 1 scaffold.

Runs fold 0 only (first walk-forward block). Writes
`scaffold_validation_report.md` containing target distribution, single-fold
Spearman rho (pooled + per-contract) for LightGBM and the three baselines,
top-5 features by gain, wall-clock time, and sanity-check pass/fail flags.

This module performs scaffold validation only.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import stage1_training as T  # noqa: E402
import baselines as B  # noqa: E402

SANITY = {
    'rho_har_lower': 0.25, 'rho_har_upper': 0.65,
    'rho_lgb_lower': 0.25, 'rho_lgb_upper': 0.75,
    'diff_lower': -0.05, 'diff_upper': 0.15,
}


def _rho(y, yhat):
    r, _ = spearmanr(y, yhat)
    return float(r)


def _target_stats(tgt: pd.Series) -> dict:
    return {
        'count': int(len(tgt)),
        'mean': float(tgt.mean()),
        'std': float(tgt.std()),
        'min': float(tgt.min()),
        'p5': float(tgt.quantile(0.05)),
        'p25': float(tgt.quantile(0.25)),
        'p50': float(tgt.quantile(0.50)),
        'p75': float(tgt.quantile(0.75)),
        'p95': float(tgt.quantile(0.95)),
        'max': float(tgt.max()),
    }


def _sanity_flags(pooled_rho: dict) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    if pooled_rho['b3'] < SANITY['rho_har_lower']:
        flags.append(('BLOCK', f"HAR-RV rho {pooled_rho['b3']:.3f} < {SANITY['rho_har_lower']} (likely B3 bug)"))
    elif pooled_rho['b3'] > SANITY['rho_har_upper']:
        flags.append(('WARN', f"HAR-RV rho {pooled_rho['b3']:.3f} > {SANITY['rho_har_upper']} (unexpectedly high)"))
    if pooled_rho['lgb'] > SANITY['rho_lgb_upper']:
        flags.append(('BLOCK', f"LightGBM rho {pooled_rho['lgb']:.3f} > {SANITY['rho_lgb_upper']} (possible target leakage)"))
    elif pooled_rho['lgb'] < SANITY['rho_lgb_lower']:
        flags.append(('WARN', f"LightGBM rho {pooled_rho['lgb']:.3f} < {SANITY['rho_lgb_lower']} (underperforming)"))
    diff = pooled_rho['lgb'] - pooled_rho['b3']
    if diff > SANITY['diff_upper']:
        flags.append(('WARN', f"(lgb - har) {diff:+.3f} > {SANITY['diff_upper']} (single-fold optimism, non-blocking)"))
    if diff < SANITY['diff_lower']:
        flags.append(('WARN', f"(lgb - har) {diff:+.3f} < {SANITY['diff_lower']} (lgb underperforms HAR on single fold, non-blocking)"))
    return flags


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('bvc.validate')

    t_start = time.time()
    log.info('Loading pooled data and computing target/derived features...')
    pooled = T.load_pooled_with_features()
    log.info('Pooled rows (pre-filter): %d', len(pooled))
    filtered = T.filter_training_pairs(pooled)
    log.info('Filtered training pairs: %d', len(filtered))

    tgt_stats = _target_stats(filtered['target'])

    folds = T.generate_folds(filtered.index.min(), filtered.index.max())
    (_HERE / 'fold_dates.json').write_text(json.dumps(folds, indent=2))
    log.info('Wrote %d folds to fold_dates.json', len(folds))
    fold0 = folds[0]

    train_df, test_df = T.slice_fold(filtered, fold0)
    log.info('Fold 0 train rows: %d, test rows: %d', len(train_df), len(test_df))
    train_by_c = train_df['contract'].value_counts().to_dict()
    test_by_c = test_df['contract'].value_counts().to_dict()

    log.info('Fitting LightGBM on fold 0...')
    t_lgb = time.time()
    model = T.fit_lightgbm(train_df)
    lgb_pred = T.predict_lightgbm(model, test_df)
    lgb_elapsed = time.time() - t_lgb
    log.info('LightGBM fit+predict: %.1f s, best_iter=%d / %d',
             lgb_elapsed, model.best_iteration, model.current_iteration())

    log.info('Fitting baselines (B1/B2/B3) on fold 0...')
    b1_pred = B.fit_predict_b1(train_df, test_df)
    b2_pred = B.fit_predict_b2(train_df, test_df)
    b3_pred = B.fit_predict_b3(train_df, test_df)

    y_true = test_df['target'].values
    pooled_rho = {
        'lgb': _rho(y_true, lgb_pred),
        'b1': _rho(y_true, b1_pred),
        'b2': _rho(y_true, b2_pred),
        'b3': _rho(y_true, b3_pred),
    }
    log.info('Pooled rho — lgb=%.4f b3=%.4f b2=%.4f b1=%.4f',
             pooled_rho['lgb'], pooled_rho['b3'], pooled_rho['b2'], pooled_rho['b1'])

    per_contract: dict[str, dict] = {}
    for c in T.CONTRACTS:
        m = (test_df['contract'].values == c)
        n_c = int(m.sum())
        if n_c < 10:
            per_contract[c] = {'n': n_c, 'lgb': None, 'b1': None, 'b2': None, 'b3': None}
            continue
        per_contract[c] = {
            'n': n_c,
            'lgb': _rho(y_true[m], np.asarray(lgb_pred)[m]),
            'b1': _rho(y_true[m], np.asarray(b1_pred)[m]),
            'b2': _rho(y_true[m], np.asarray(b2_pred)[m]),
            'b3': _rho(y_true[m], np.asarray(b3_pred)[m]),
        }

    flags = _sanity_flags(pooled_rho)
    gains = model.feature_importance(importance_type='gain').tolist()
    splits = model.feature_importance(importance_type='split').tolist()
    names = model.feature_name()
    top5_gain = sorted(zip(names, gains), key=lambda x: -x[1])[:5]

    total_elapsed = time.time() - t_start
    _write_report(
        fold0=fold0, tgt_stats=tgt_stats, train_by_c=train_by_c, test_by_c=test_by_c,
        pooled_rho=pooled_rho, per_contract=per_contract, flags=flags,
        top5_gain=top5_gain, gains=gains, splits=splits, names=names,
        lgb_time=lgb_elapsed, total_time=total_elapsed,
        best_iter=int(model.best_iteration),
        total_iter=int(model.current_iteration()),
        n_folds=len(folds), filtered_n=len(filtered),
    )
    log.info('Wrote scaffold_validation_report.md (wall-clock total: %.1f s)', total_elapsed)


# --- report writer ---------------------------------------------------------
def _write_report(**kw) -> None:
    from _report_template import render
    text = render(**kw)
    (_HERE / 'scaffold_validation_report.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()


