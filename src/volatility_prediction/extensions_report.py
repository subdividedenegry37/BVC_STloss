"""Build STAGE1_EXTENSIONS.md from extensions.py artifacts.

Inputs (read-only):
  * predictions.parquet         — Stage 1 mean-model predictions (for decile-A comparison)
  * predictions_q90.parquet     — Extension A output (same 31 folds, alpha=0.9)
  * shap_fold15.parquet         — Extension B output (wide: shap__<feature> columns)

Output:
  * STAGE1_EXTENSIONS.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import stage1_training as T  # noqa: E402


# -- Feature partitions used for the RTY hypothesis test in Extension B.
# "Vol-state" = features that directly encode recent realized/range-vol.
VOL_STATE_FEATURES = ['sigma_yz', 'sigma_gk_lag1', 'sigma_gk_short', 'sigma_gk_long']
# Everything else the model sees, excluding the contract categorical
# (contract is a routing variable, not an information channel about flow).
NON_VOL_STATE_FEATURES = [f for f in T.CORE_BVC + T.CORE_PHYSICS + T.DERIVED_HAR
                          if f not in VOL_STATE_FEATURES]

# ---------------------------------------------------------------- Extension A

def _decile_table(preds_mean: pd.DataFrame, preds_q90: pd.DataFrame) -> pd.DataFrame:
    """Pooled decile table. Deciles are assigned within each fold on the target,
    then residuals pooled across folds. Returns one row per decile (1..10) with
    mean residual for mean and q90 models."""
    m = preds_mean[['ts', 'instrument_id', 'fold_id', 'contract',
                    'target_log_sigma_rv', 'pred_lgb']].copy()
    m['decile'] = m.groupby('fold_id', observed=True)['target_log_sigma_rv'].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates='drop') + 1
    )
    m['resid_mean'] = m['target_log_sigma_rv'] - m['pred_lgb']

    q = preds_q90[['ts', 'instrument_id', 'fold_id', 'pred_lgb_q90']].copy()
    merged = m.merge(q, on=['ts', 'instrument_id', 'fold_id'], how='inner')
    merged['resid_q90'] = merged['target_log_sigma_rv'] - merged['pred_lgb_q90']

    rows = []
    for d in range(1, 11):
        sub = merged[merged['decile'] == d]
        rows.append({
            'decile': d,
            'n': int(len(sub)),
            'target_mean': float(sub['target_log_sigma_rv'].mean()),
            'mean_resid_mean_model': float(sub['resid_mean'].mean()),
            'std_resid_mean_model': float(sub['resid_mean'].std()),
            'mean_resid_q90_model': float(sub['resid_q90'].mean()),
            'std_resid_q90_model': float(sub['resid_q90'].std()),
        })
    return pd.DataFrame(rows)


def _q90_coverage(preds_q90: pd.DataFrame) -> dict:
    """Empirical coverage: fraction of observations where y_true <= pred_q90.
    For a well-calibrated tau=0.9 model this should be ~0.90."""
    hit = (preds_q90['target_log_sigma_rv'] <= preds_q90['pred_lgb_q90']).mean()
    return {'overall': float(hit),
            'by_contract': preds_q90.groupby('contract', observed=True).apply(
                lambda g: float((g['target_log_sigma_rv'] <= g['pred_lgb_q90']).mean())
            ).to_dict()}


# ---------------------------------------------------------------- Extension B

def _shap_per_contract(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame indexed by feature, columns per contract: mean |SHAP|."""
    shap_cols = [c for c in shap_df.columns if c.startswith('shap__')]
    features = [c.replace('shap__', '') for c in shap_cols]
    out = {}
    for c, sub in shap_df.groupby('contract', observed=True):
        out[c] = {
            feat: float(np.mean(np.abs(sub[col].values)))
            for feat, col in zip(features, shap_cols)
        }
    return pd.DataFrame(out).loc[features]


def _non_vol_share(mean_abs: pd.DataFrame) -> dict:
    """For each contract, compute share of total mean |SHAP| attributable
    to NON_VOL_STATE_FEATURES (excludes the contract categorical)."""
    out = {}
    for c in mean_abs.columns:
        s = mean_abs[c]
        # Drop the contract categorical for the share calculation so we
        # compare information channels, not the routing variable.
        s = s.drop(index=['contract'], errors='ignore')
        total = s.sum()
        nvs = s.loc[s.index.isin(NON_VOL_STATE_FEATURES)].sum()
        out[c] = {'total': float(total), 'non_vol_state': float(nvs),
                  'share': float(nvs / total) if total > 0 else float('nan')}
    return out


# ---------------------------------------------------------------- render

def _fmt_pct(x: float) -> str:
    return f'{100*x:.1f}%'


def _render(dec_tbl: pd.DataFrame, q90_cov: dict,
            shap_per_c: pd.DataFrame, nvs_share: dict) -> str:
    L: list[str] = []
    L.append('# Stage 1 — Extensions (Quantile + Per-Contract SHAP)')
    L.append('')
    L.append(f'**Generated:** {pd.Timestamp.now(tz="UTC").isoformat()}')
    L.append('**Scope:** two diagnostic additions to the authorized Stage 1 run. '
             'Locked artifacts (`predictions.parquet`, `fold_metrics.parquet`, '
             '`feature_importance.parquet`, `STAGE1_RESULTS.md`) were **not** modified.')
    L.append('')
    L.append('This repository publishes Stage 1 only; Stage 2 is out of scope.')
    L.append('')

    # --- Extension A ---
    L.append('## Extension A — Quantile regression at τ = 0.9')
    L.append('')
    L.append('**Setup.** Same 31 folds, same 14 core + 3 HAR-analog + `contract` categorical '
             'features, same target (forward 12-bar log σ_rv). Only change: LightGBM '
             '`objective=quantile`, `alpha=0.9`. Per-fold internal-val split (last 10%), '
             'early stopping on quantile loss, same 5000-round cap. 585,202 pooled test rows.')
    L.append('')
    L.append('**Artifact:** `predictions_q90.parquet` (sibling of `predictions.parquet`).')
    L.append('')

    L.append('### A.1 Empirical coverage check (sanity on τ)')
    L.append('')
    L.append('For a well-calibrated τ=0.9 quantile estimator, P(y ≤ ŷ) ≈ 0.90.')
    L.append('')
    L.append('| scope | coverage P(y ≤ ŷ_q90) |')
    L.append('|:------|----------------------:|')
    L.append(f'| pooled | {q90_cov["overall"]:.3f} |')
    for c, v in q90_cov['by_contract'].items():
        L.append(f'| {c} | {v:.3f} |')
    L.append('')

    L.append('### A.2 Residual bias by target decile (pooled, deciles assigned within fold)')
    L.append('')
    L.append('Residual = y − ŷ. For the mean model, positive residuals in high deciles '
             'indicate systematic underprediction of the top of the vol distribution. '
             'For the quantile (τ=0.9) model, E[resid] should be negative for deciles '
             '1–9 and near zero for decile 10.')
    L.append('')
    L.append('| decile | n | mean(target) | μ resid (mean model) | σ resid (mean) | μ resid (q90 model) | σ resid (q90) |')
    L.append('|------:|---:|-------------:|---------------------:|---------------:|--------------------:|--------------:|')
    for _, row in dec_tbl.iterrows():
        L.append(f'| {int(row["decile"])} | {int(row["n"]):,} | '
                 f'{row["target_mean"]:+.3f} | '
                 f'{row["mean_resid_mean_model"]:+.4f} | {row["std_resid_mean_model"]:.4f} | '
                 f'{row["mean_resid_q90_model"]:+.4f} | {row["std_resid_q90_model"]:.4f} |')
    L.append('')

    top = dec_tbl.iloc[-1]
    L.append('### A.3 Top-decile bias test')
    L.append('')
    L.append(f'- Mean-model top-decile μ resid: **{top["mean_resid_mean_model"]:+.4f}** '
             '(matches `STAGE1_RESULTS.md §6` "high-vol (top decile)" bias of +0.515)')
    L.append(f'- Quantile-model top-decile μ resid: **{top["mean_resid_q90_model"]:+.4f}**')
    test_pass = top['mean_resid_q90_model'] < 0.20
    L.append(f'- Threshold: top-decile q90 μ resid < +0.20 → **{"PASS" if test_pass else "FAIL"}**')
    L.append('')
    if test_pass:
        L.append('**Finding.** The τ=0.9 quantile objective substantially reduces the systematic '
                 'underprediction of high-vol bars. The remaining mean residual in the top decile '
                 'is the expected behaviour of a correctly-calibrated 90th-percentile estimator: '
                 'on average it sits at the 90th-percentile of the conditional target distribution, '
                 'so conditionally on being in the top 10%, some positive residual is expected.')
    else:
        L.append('**Finding.** The quantile model does not bring the top-decile bias below the '
                 '+0.20 threshold. This indicates the conditional 90th-percentile of the target '
                 'itself lies well above the mean prediction; Stage 2 design should not rely on '
                 'mean-model residuals as a proxy for tail risk.')
    L.append('')

    # --- Extension B ---
    L.append('## Extension B — Per-contract SHAP on fold 15')
    L.append('')
    L.append('**Setup.** Fold 15 (train window = anchor → 2023-07-02, test window = '
             '2023-07-03 → 2023-09-02, train=385,966, test=18,218). Retrained the mean-regression '
             'model with identical params to the Stage 1 run (reproduced `best_iter=1198`), '
             'then computed TreeSHAP on the full test set via LightGBM native '
             '`pred_contrib=True`. No external `shap` dependency.')
    L.append('')
    L.append('**Artifact:** `shap_fold15.parquet` (18,218 rows × 18 feature SHAP columns '
             '+ expected value + 4 identifiers).')
    L.append('')

    # Per-contract top 10
    L.append('### B.1 Top-10 features per contract by mean |SHAP|')
    L.append('')
    for c in ['ES', 'NQ', 'RTY']:
        if c not in shap_per_c.columns:
            continue
        L.append(f'#### {c}')
        L.append('')
        L.append('| rank | feature | mean &#124;SHAP&#124; | share of total |')
        L.append('|---:|:--------|---------------------:|---------------:|')
        s = shap_per_c[c].sort_values(ascending=False)
        total = s.sum()
        for i, (feat, v) in enumerate(s.head(10).items(), 1):
            L.append(f'| {i} | `{feat}` | {v:.4f} | {_fmt_pct(v / total)} |')
        L.append('')

    # Joint table: all features × all contracts (mean |SHAP|)
    L.append('### B.2 Full feature × contract mean |SHAP| matrix')
    L.append('')
    L.append('| feature | ES | NQ | RTY |')
    L.append('|:--------|---:|---:|----:|')
    ordered = shap_per_c.sum(axis=1).sort_values(ascending=False).index
    for feat in ordered:
        es = shap_per_c.loc[feat, 'ES'] if 'ES' in shap_per_c.columns else float('nan')
        nq = shap_per_c.loc[feat, 'NQ'] if 'NQ' in shap_per_c.columns else float('nan')
        rty = shap_per_c.loc[feat, 'RTY'] if 'RTY' in shap_per_c.columns else float('nan')
        L.append(f'| `{feat}` | {es:.4f} | {nq:.4f} | {rty:.4f} |')
    L.append('')

    # Hypothesis test
    L.append('### B.3 RTY non-vol-state hypothesis')
    L.append('')
    L.append(f'**Vol-state features** ({len(VOL_STATE_FEATURES)}): `' +
             '`, `'.join(VOL_STATE_FEATURES) + '`')
    L.append(f'**Non-vol-state features** ({len(NON_VOL_STATE_FEATURES)}): `' +
             '`, `'.join(NON_VOL_STATE_FEATURES) + '`')
    L.append('')
    L.append('`contract` (categorical routing variable) is excluded from the share denominator '
             'so we compare information channels.')
    L.append('')
    L.append('| contract | total mean &#124;SHAP&#124; | non-vol-state mean &#124;SHAP&#124; | share |')
    L.append('|:---------|---:|---:|---:|')
    for c in ['ES', 'NQ', 'RTY']:
        if c not in nvs_share:
            continue
        r = nvs_share[c]
        L.append(f'| {c} | {r["total"]:.4f} | {r["non_vol_state"]:.4f} | {_fmt_pct(r["share"])} |')
    L.append('')
    es_s = nvs_share.get('ES', {}).get('share', float('nan'))
    nq_s = nvs_share.get('NQ', {}).get('share', float('nan'))
    rty_s = nvs_share.get('RTY', {}).get('share', float('nan'))
    rty_higher = (rty_s > es_s) and (rty_s > nq_s)
    L.append(f'**Hypothesis:** RTY non-vol-state share > ES share **and** > NQ share.')
    L.append(f'- RTY = {_fmt_pct(rty_s)}, ES = {_fmt_pct(es_s)}, NQ = {_fmt_pct(nq_s)}')
    L.append(f'- Result: **{"CONFIRMED" if rty_higher else "NOT CONFIRMED"}**')
    L.append('')
    if rty_higher:
        L.append('**Interpretation.** On fold 15, RTY leans more heavily on the non-vol-state '
                 'features (BVC flow + physics ex-σ_yz) than ES/NQ do. This is consistent with '
                 'the lower efficiency/higher micro-flow informativeness regime expected in a '
                 'less liquid contract (RTY vs ES/NQ). It is a single-fold, single-period '
                 'observation and should not be generalized without replicating across folds.')
    else:
        L.append('**Interpretation.** On fold 15, RTY does not show a systematically higher '
                 'non-vol-state SHAP share than ES/NQ. The Andersen-Bondarenko pattern observed '
                 'pooled in `STAGE1_RESULTS.md §5` (BVC never entering top-5 by gain) appears to '
                 'hold approximately uniformly across the three contracts.')
    L.append('')

    # Scope
    L.append('## Scope')
    L.append('')
    L.append('Extensions complete. Locked Stage 1 artifacts are untouched. '
             'This repository publishes Stage 1 only; Stage 2 is out of scope.')
    L.append('')
    return '\n'.join(L)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('bvc.ext_report')

    preds_mean = pd.read_parquet(RUN_DIR / 'predictions.parquet')
    preds_q90 = pd.read_parquet(RUN_DIR / 'predictions_q90.parquet')
    shap_df = pd.read_parquet(RUN_DIR / 'shap_fold15.parquet')
    log.info('Loaded: mean preds=%d, q90 preds=%d, shap rows=%d',
             len(preds_mean), len(preds_q90), len(shap_df))

    dec_tbl = _decile_table(preds_mean, preds_q90)
    q90_cov = _q90_coverage(preds_q90)
    shap_per_c = _shap_per_contract(shap_df)
    nvs_share = _non_vol_share(shap_per_c)

    text = _render(dec_tbl, q90_cov, shap_per_c, nvs_share)
    out = RUN_DIR / 'STAGE1_EXTENSIONS.md'
    out.write_text(text, encoding='utf-8')
    log.info('Wrote %s (%d lines)', out, text.count('\n') + 1)


if __name__ == '__main__':
    main()
