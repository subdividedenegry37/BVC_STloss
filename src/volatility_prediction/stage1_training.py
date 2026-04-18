"""Stage 1 scaffold — forward realized volatility.

See the Stage 1 specification for full spec. This scaffold:
  * Loads the three 2026-04-18 physics-feature pickles and pools them
  * Computes target log(sigma_rv_{t+1..t+12}) per instrument_id
  * Derives HAR-analog features and VPIN, per instrument_id (no roll crossing)
  * Generates expanding walk-forward folds (12-month warmup, 2-month test, 1-day purge)
  * Provides fit/predict helpers for LightGBM (10% internal-val early stopping)

The `--run-full` flag is GATED OFF by default. Executing the 30-fold pass
requires explicit human authorization after review of
`scaffold_validation_report.md`.

Naming deviations vs spec text:
  * spec says `DER`; the pickle column is `der` (lowercase).
  * spec says `ts_event`; the pickle index is named `ts`.
  * spec §1.4 says `instrument_id` categorical. The pickles carry a
    per-expiration instrument_id (~79 unique values across contract rolls),
    which is almost pure noise at the model level. The spec's rationale
    ("ES vs NQ vs RTY differences") is at the contract-family level, so we
    use a derived `contract` column (ES/NQ/RTY) as the LightGBM categorical.
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
PICKLE_DIR = RUN_DIR.parent / '2026-04-18_physics_feature_expansion'
CONTRACTS = ('ES', 'NQ', 'RTY')

CORE_BVC = ['z', 'imbalance_t', 'der', 'sign_concordance',
            'clv_mean', 'clv_var', 'subbar_imbalance']
CORE_PHYSICS = ['sigma_yz', 'body_to_range', 'gel_fraction', 'wick_asymmetry',
                'amihud', 'ko_W', 'polar_order_P']
DERIVED_HAR = ['sigma_gk_lag1', 'sigma_gk_short', 'sigma_gk_long']
CAT_FEATURE = 'contract'
FEATURE_COLS = CORE_BVC + CORE_PHYSICS + DERIVED_HAR + [CAT_FEATURE]

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

TARGET_HORIZON = 12
HAR_SHORT_WINDOW = 12
HAR_LONG_WINDOW = 60
VPIN_WINDOW = 20


def _compute_one_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Add target, HAR-analog rolling features, and VPIN to a single-contract frame.
    Groupby is by `instrument_id` (per expiration) so no computation crosses a contract roll."""
    df = df.sort_values(['instrument_id']).sort_index(kind='mergesort')
    grp_close = df.groupby('instrument_id', sort=False, observed=True)['close']
    df['log_ret_1'] = np.log(df['close']) - grp_close.transform(lambda s: np.log(s).shift(1))

    grp_lr = df.groupby('instrument_id', sort=False, observed=True)['log_ret_1']
    df['forward_ss'] = grp_lr.transform(
        lambda s: (s * s).rolling(TARGET_HORIZON, min_periods=TARGET_HORIZON).sum().shift(-TARGET_HORIZON)
    )
    df['target'] = 0.5 * np.log(df['forward_ss'].replace(0, np.nan))

    grp_gk = df.groupby('instrument_id', sort=False, observed=True)['sigma_gk']
    df['sigma_gk_lag1'] = grp_gk.shift(1)
    df['sigma_gk_short'] = grp_gk.transform(
        lambda s: s.rolling(HAR_SHORT_WINDOW, min_periods=HAR_SHORT_WINDOW).mean()
    )
    df['sigma_gk_long'] = grp_gk.transform(
        lambda s: s.rolling(HAR_LONG_WINDOW, min_periods=HAR_LONG_WINDOW).mean()
    )
    df['vpin'] = df.groupby('instrument_id', sort=False, observed=True)['imbalance_t'].transform(
        lambda s: s.abs().rolling(VPIN_WINDOW, min_periods=VPIN_WINDOW).mean()
    )
    return df


def load_pooled_with_features() -> pd.DataFrame:
    """Load three per-contract pickles, compute target/derived features per contract, pool."""
    frames = []
    for c in CONTRACTS:
        df = pd.read_pickle(PICKLE_DIR / f'phase2_features_expanded_{c}.pkl')
        df['contract'] = c
        df = _compute_one_contract(df)
        frames.append(df)
    pooled = pd.concat(frames, axis=0, copy=False)
    # Time-ordered for fold slicing and internal-validation split
    pooled = pooled.reset_index().sort_values(['ts', 'instrument_id'], kind='mergesort').set_index('ts')
    return pooled


def filter_training_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars with is_train_valid_bar, valid target, and no NaN in any model input."""
    required = CORE_BVC + CORE_PHYSICS + DERIVED_HAR + ['vpin', 'target']
    mask = df['is_train_valid_bar'] & df['target'].notna()
    for c in required:
        mask &= df[c].notna()
    for c in DERIVED_HAR:
        mask &= (df[c] > 0)
    return df.loc[mask].copy()


def generate_folds(min_ts, max_ts, warmup_months: int = 12,
                   block_months: int = 2, purge_days: int = 1) -> list[dict]:
    """Expanding walk-forward: train = [anchor, test_start - purge], test = [test_start, test_start + block).
    Anchor is floor(min_ts) at day granularity. First test_start = anchor + warmup + purge.
    Fold is emitted only if its test_end <= max_ts."""
    anchor = pd.Timestamp(min_ts).normalize()
    first_test = (anchor + pd.DateOffset(months=warmup_months)
                  + pd.Timedelta(days=purge_days)).normalize()
    max_ts = pd.Timestamp(max_ts)
    folds = []
    fid = 0
    ts = first_test
    while True:
        te = ts + pd.DateOffset(months=block_months)
        if te > max_ts:
            break
        train_end = ts - pd.Timedelta(seconds=1)
        folds.append({
            'fold_id': fid,
            'train_start': anchor.isoformat(),
            'train_end': train_end.isoformat(),
            'purge_start': train_end.isoformat(),
            'test_start': ts.isoformat(),
            'test_end': te.isoformat(),
        })
        fid += 1
        ts = te
    return folds


def slice_fold(df: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = df.index
    train_mask = (idx >= pd.Timestamp(fold['train_start'])) & (idx <= pd.Timestamp(fold['train_end']))
    test_mask = (idx >= pd.Timestamp(fold['test_start'])) & (idx < pd.Timestamp(fold['test_end']))
    return df.loc[train_mask], df.loc[test_mask]


def _prep_features(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[FEATURE_COLS].copy()
    X[CAT_FEATURE] = X[CAT_FEATURE].astype('category')
    return X


def fit_lightgbm(train_df: pd.DataFrame, val_fraction: float = 0.1,
                 num_boost_round: int = 5000, early_stopping_rounds: int = 100):
    """Fit LightGBM with a time-ordered internal validation split (last `val_fraction` of train)."""
    n = len(train_df)
    n_val = max(1, int(n * val_fraction))
    tr = train_df.iloc[:n - n_val]
    va = train_df.iloc[n - n_val:]
    X_tr, y_tr = _prep_features(tr), tr['target'].values
    X_va, y_va = _prep_features(va), va['target'].values
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=[CAT_FEATURE], free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=[CAT_FEATURE],
                      reference=dtr, free_raw_data=False)
    model = lgb.train(
        LGB_PARAMS,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dva],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model


def predict_lightgbm(model, test_df: pd.DataFrame) -> np.ndarray:
    return model.predict(_prep_features(test_df), num_iteration=model.best_iteration)


def write_fold_dates(dest: Path | None = None) -> Path:
    """Compute the full fold schedule and write fold_dates.json."""
    pooled = load_pooled_with_features()
    filtered = filter_training_pairs(pooled)
    folds = generate_folds(filtered.index.min(), filtered.index.max())
    dest = dest or (RUN_DIR / 'fold_dates.json')
    dest.write_text(json.dumps(folds, indent=2))
    return dest


def _log(msg, *args, level='info'):
    logging.getLogger('bvc.stage1').log(
        {'info': logging.INFO, 'warn': logging.WARNING, 'error': logging.ERROR}[level], msg, *args)


def run_full_training(out_dir: Path | None = None) -> dict:
    """Execute the 31-fold walk-forward run. Writes three parquet artifacts:

      * predictions.parquet         — all test-fold predictions, long format
      * feature_importance.parquet  — per-fold gain + split count
      * fold_metrics.parquet        — per-fold (rho, rmse, mae, hit_rate), pooled + per-contract
    Returns a small summary dict for the CLI caller.
    """
    sys.path.insert(0, str(RUN_DIR))
    import baselines as B  # local import to avoid scaffold-time dep
    import metrics as M

    out_dir = Path(out_dir) if out_dir else RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    _log('Loading pooled data and computing target/derived features...')
    pooled = load_pooled_with_features()
    filtered = filter_training_pairs(pooled)
    _log('Filtered training pairs: %d', len(filtered))

    folds = generate_folds(filtered.index.min(), filtered.index.max())
    (out_dir / 'fold_dates.json').write_text(json.dumps(folds, indent=2))
    _log('Generated %d folds; starting walk-forward loop', len(folds))

    pred_frames: list[pd.DataFrame] = []
    importance_rows: list[dict] = []
    metric_rows: list[dict] = []

    for fold in folds:
        fid = fold['fold_id']
        tr, te = slice_fold(filtered, fold)
        n_tr, n_te = len(tr), len(te)
        if n_tr < 1000 or n_te < 100:
            _log('[Fold %d] SKIPPED — train=%d test=%d too small', fid, n_tr, n_te, level='warn')
            continue

        t_fold = time.time()
        model = fit_lightgbm(tr)
        lgb_pred = predict_lightgbm(model, te)
        b1 = B.fit_predict_b1(tr, te)
        b2 = B.fit_predict_b2(tr, te)
        b3 = B.fit_predict_b3(tr, te)
        y_true = te['target'].values.astype(float)

        pred_frames.append(pd.DataFrame({
            'ts': te.index.values,
            'instrument_id': te['instrument_id'].values,
            'contract': te['contract'].astype(str).values,
            'fold_id': fid,
            'target_log_sigma_rv': y_true,
            'pred_lgb': lgb_pred,
            'pred_b1_naive': b1,
            'pred_b2_vpin': b2,
            'pred_b3_har': b3,
        }))

        gains = model.feature_importance(importance_type='gain').tolist()
        splits = model.feature_importance(importance_type='split').tolist()
        for nm, g, s in zip(model.feature_name(), gains, splits):
            importance_rows.append({'fold_id': fid, 'feature': nm, 'gain': float(g),
                                    'split_count': int(s), 'best_iteration': int(model.best_iteration)})

        pivot = float(np.median(tr['target'].values))
        contracts_arr = te['contract'].astype(str).values
        preds = {'lgb': lgb_pred, 'b1_naive': b1, 'b2_vpin': b2, 'b3_har': b3}
        metric_rows.extend(M.collect_fold_metrics(
            fold_id=fid, y_true=y_true, preds=preds,
            contracts=contracts_arr, contract_order=CONTRACTS, pivot=pivot,
        ))

        elapsed = time.time() - t_fold
        d_lgb = next(r['rho'] for r in metric_rows if r['fold_id'] == fid and r['model'] == 'lgb' and r['slice'] == 'pooled')
        d_b3 = next(r['rho'] for r in metric_rows if r['fold_id'] == fid and r['model'] == 'b3_har' and r['slice'] == 'pooled')
        _log('[Fold %2d/%d] train=%d test=%d  ρ_lgb=%+.4f ρ_b3=%+.4f Δ=%+.4f  best_iter=%d  %.1fs',
             fid, len(folds) - 1, n_tr, n_te, d_lgb, d_b3, d_lgb - d_b3,
             model.best_iteration, elapsed)

    if not pred_frames:
        raise RuntimeError('No folds produced predictions; check fold filter thresholds.')

    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df.to_parquet(out_dir / 'predictions.parquet', index=False)
    pd.DataFrame(importance_rows).to_parquet(out_dir / 'feature_importance.parquet', index=False)
    pd.DataFrame(metric_rows).to_parquet(out_dir / 'fold_metrics.parquet', index=False)

    total = time.time() - t_start
    _log('Full run complete: %d folds, %d test predictions, wall-clock %.1fs',
         len(pred_frames), len(pred_df), total)
    return {
        'folds_run': len(pred_frames),
        'n_predictions': int(len(pred_df)),
        'wall_clock_s': total,
        'out_dir': str(out_dir),
    }


def main():
    ap = argparse.ArgumentParser(description='Stage 1 training.')
    ap.add_argument('--emit-folds', action='store_true',
                    help='Write fold_dates.json and exit. Safe (no training).')
    ap.add_argument('--run-full', action='store_true',
                    help='Execute the full 31-fold training run. Authorized 2026-04-18.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('bvc.stage1')

    if args.emit_folds:
        out = write_fold_dates()
        log.info('Wrote fold_dates.json to %s', out)
        return

    if not args.run_full:
        log.warning('Scaffold mode. Run `python validate_scaffold.py` for a single-fold '
                    'validation. Use --run-full for the authorized full pass.')
        return

    summary = run_full_training()
    log.info('Summary: %s', summary)


if __name__ == '__main__':
    main()

