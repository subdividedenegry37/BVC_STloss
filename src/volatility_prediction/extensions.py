"""Stage 1 extensions: quantile regression (A) and per-contract SHAP (B).

Reuses the Stage 1 data loader, fold schedule, and feature list. Does NOT
modify predictions.parquet, fold_metrics.parquet, or feature_importance.parquet.

Extension A — tau=0.9 quantile regression on the same 31 folds:
    writes predictions_q90.parquet with columns
      {ts, instrument_id, contract, fold_id, target_log_sigma_rv, pred_lgb_q90}.

Extension B — per-contract SHAP on fold_id=15 (middle fold):
    retrains the mean-regression model on that fold, calls LightGBM's native
    TreeSHAP via `model.predict(..., pred_contrib=True)` on the test set, and
    writes shap_fold15.parquet with columns
      {ts, instrument_id, contract, fold_id, <one column per feature>,
       shap_expected_value}.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import stage1_training as T  # noqa: E402


# -- Quantile params: same shape as LGB_PARAMS but objective/metric swapped.
LGB_PARAMS_Q90 = {
    'objective': 'quantile',
    'alpha': 0.9,
    'metric': 'quantile',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}


def fit_lightgbm_quantile(train_df: pd.DataFrame, alpha: float = 0.9,
                          val_fraction: float = 0.1,
                          num_boost_round: int = 5000,
                          early_stopping_rounds: int = 100):
    """Same internal-val split as T.fit_lightgbm, but quantile objective."""
    n = len(train_df)
    n_val = max(1, int(n * val_fraction))
    tr = train_df.iloc[:n - n_val]
    va = train_df.iloc[n - n_val:]
    X_tr, y_tr = T._prep_features(tr), tr['target'].values
    X_va, y_va = T._prep_features(va), va['target'].values
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=[T.CAT_FEATURE], free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=[T.CAT_FEATURE],
                      reference=dtr, free_raw_data=False)
    params = dict(LGB_PARAMS_Q90)
    params['alpha'] = alpha
    model = lgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dva],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def run_extension_a(out_dir: Path | None = None) -> dict:
    """31-fold quantile-regression pass (alpha=0.9). Same folds as Stage 1 full run."""
    out_dir = Path(out_dir) if out_dir else RUN_DIR
    log = logging.getLogger('bvc.ext_a')

    t0 = time.time()
    log.info('[Ext A] Loading pooled data + computing target/derived features...')
    pooled = T.load_pooled_with_features()
    filtered = T.filter_training_pairs(pooled)
    folds = T.generate_folds(filtered.index.min(), filtered.index.max())
    log.info('[Ext A] Filtered rows=%d  folds=%d', len(filtered), len(folds))

    pred_frames: list[pd.DataFrame] = []
    for fold in folds:
        fid = fold['fold_id']
        tr, te = T.slice_fold(filtered, fold)
        if len(tr) < 1000 or len(te) < 100:
            log.warning('[Ext A Fold %d] SKIPPED train=%d test=%d', fid, len(tr), len(te))
            continue
        t_fold = time.time()
        model = fit_lightgbm_quantile(tr, alpha=0.9)
        yhat = model.predict(T._prep_features(te), num_iteration=model.best_iteration)
        pred_frames.append(pd.DataFrame({
            'ts': te.index.values,
            'instrument_id': te['instrument_id'].values,
            'contract': te['contract'].astype(str).values,
            'fold_id': fid,
            'target_log_sigma_rv': te['target'].values.astype(float),
            'pred_lgb_q90': yhat,
        }))
        log.info('[Ext A Fold %2d/%d] train=%d test=%d  best_iter=%d  %.1fs',
                 fid, len(folds) - 1, len(tr), len(te), model.best_iteration,
                 time.time() - t_fold)

    if not pred_frames:
        raise RuntimeError('Ext A produced no predictions.')
    out = pd.concat(pred_frames, ignore_index=True)
    dst = out_dir / 'predictions_q90.parquet'
    out.to_parquet(dst, index=False)
    total = time.time() - t0
    log.info('[Ext A] Wrote %s (%d rows, %.1fs)', dst, len(out), total)
    return {'folds_run': len(pred_frames), 'n_predictions': int(len(out)),
            'wall_clock_s': total, 'out_path': str(dst)}


def run_extension_b(target_fold_id: int = 15, out_dir: Path | None = None) -> dict:
    """Retrain mean model on fold `target_fold_id`, compute TreeSHAP on its test set."""
    out_dir = Path(out_dir) if out_dir else RUN_DIR
    log = logging.getLogger('bvc.ext_b')

    t0 = time.time()
    log.info('[Ext B] Loading pooled data...')
    pooled = T.load_pooled_with_features()
    filtered = T.filter_training_pairs(pooled)
    folds = T.generate_folds(filtered.index.min(), filtered.index.max())
    fold = next(f for f in folds if f['fold_id'] == target_fold_id)
    tr, te = T.slice_fold(filtered, fold)
    log.info('[Ext B] Fold %d: train=%d test=%d  train_end=%s test_window=[%s, %s)',
             target_fold_id, len(tr), len(te), fold['train_end'],
             fold['test_start'], fold['test_end'])

    log.info('[Ext B] Fitting mean-regression model on fold %d...', target_fold_id)
    model = T.fit_lightgbm(tr)
    log.info('[Ext B] best_iter=%d', model.best_iteration)

    # TreeSHAP via LightGBM native pred_contrib.
    # Shape: (n_test, n_features + 1). Last column is the model's expected value.
    X_te = T._prep_features(te)
    log.info('[Ext B] Computing SHAP on %d test rows (pred_contrib=True)...', len(X_te))
    t_shap = time.time()
    contrib = model.predict(X_te, num_iteration=model.best_iteration, pred_contrib=True)
    log.info('[Ext B] SHAP done in %.1fs  shape=%s', time.time() - t_shap, contrib.shape)

    feature_names = list(model.feature_name())
    assert contrib.shape[1] == len(feature_names) + 1, \
        f'Unexpected SHAP shape {contrib.shape} vs {len(feature_names)}+1 features'

    # Prefix SHAP feature columns so the `contract` feature's SHAP column
    # does not collide with the `contract` identifier column.
    shap_cols = [f'shap__{nm}' for nm in feature_names]
    shap_df = pd.DataFrame(contrib[:, :-1], columns=shap_cols)
    shap_df.insert(0, 'ts', te.index.values)
    shap_df.insert(1, 'instrument_id', te['instrument_id'].values)
    shap_df.insert(2, 'contract', te['contract'].astype(str).values)
    shap_df.insert(3, 'fold_id', target_fold_id)
    shap_df['shap_expected_value'] = contrib[:, -1]
    dst = out_dir / f'shap_fold{target_fold_id}.parquet'
    shap_df.to_parquet(dst, index=False)
    total = time.time() - t0
    log.info('[Ext B] Wrote %s (%d rows, %.1fs)', dst, len(shap_df), total)
    return {'fold_id': target_fold_id, 'n_test_rows': int(len(shap_df)),
            'feature_names': feature_names, 'wall_clock_s': total,
            'out_path': str(dst)}


def main():
    ap = argparse.ArgumentParser(description='Stage 1 extensions.')
    ap.add_argument('--ext-a', action='store_true', help='Run Extension A (quantile regression).')
    ap.add_argument('--ext-b', action='store_true', help='Run Extension B (fold-15 SHAP).')
    ap.add_argument('--all', action='store_true', help='Run both extensions (A then B).')
    ap.add_argument('--fold-id', type=int, default=15, help='Fold id for Extension B (default 15).')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('bvc.extensions')

    summaries: dict = {}
    if args.all or args.ext_a:
        summaries['ext_a'] = run_extension_a()
    if args.all or args.ext_b:
        summaries['ext_b'] = run_extension_b(target_fold_id=args.fold_id)
    if not summaries:
        log.warning('Nothing to do. Use --ext-a, --ext-b, or --all.')
        return
    log.info('Summaries: %s', json.dumps(summaries, indent=2, default=str))


if __name__ == '__main__':
    main()
