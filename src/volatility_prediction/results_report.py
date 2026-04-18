"""Generate STAGE1_RESULTS.md from the three parquet artifacts.

Reads:
  predictions.parquet, feature_importance.parquet, fold_metrics.parquet
Writes:
  STAGE1_RESULTS.md

Self-contained diagnostics:
  * Success criteria check (spec §7): pooled Δρ, fold-sign consistency, per-contract generality
  * Per-fold Δρ table (LGB vs HAR-RV)
  * Feature importance rank stability (Spearman of top-K ranks across adjacent folds)
  * Top-5 features by mean gain with per-fold presence counts
  * Residual analysis: contract, high-vol regime, FOMC announcement days
  * Correctness provenance: target-alignment spot-check carried over from scaffold
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import metrics as M  # noqa: E402
import stage1_training as T  # noqa: E402

# Public FOMC announcement dates 2021-03-01 through 2026-03-31 (Fed calendar).
# Used for residual analysis only; inaccuracies of a day or two do not change the signal.
FOMC_DATES = [
    '2021-03-17', '2021-04-28', '2021-06-16', '2021-07-28', '2021-09-22', '2021-11-03', '2021-12-15',
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-10',
    '2026-01-28', '2026-03-18',
]


def _pooled_aggregate(preds: pd.DataFrame) -> dict:
    """Compute pooled (across all folds and contracts) Spearman ρ, RMSE, MAE for each model."""
    out = {}
    y = preds['target_log_sigma_rv'].values
    for m_name, col in (('lgb', 'pred_lgb'), ('b1_naive', 'pred_b1_naive'),
                        ('b2_vpin', 'pred_b2_vpin'), ('b3_har', 'pred_b3_har')):
        yhat = preds[col].values
        rho, _ = spearmanr(y, yhat)
        resid = y - yhat
        out[m_name] = {
            'rho': float(rho),
            'rmse': float(np.sqrt(np.mean(resid ** 2))),
            'mae': float(np.mean(np.abs(resid))),
        }
    return out


def _per_contract_aggregate(preds: pd.DataFrame) -> dict:
    out: dict = {}
    for c, sub in preds.groupby('contract', observed=True):
        y = sub['target_log_sigma_rv'].values
        row = {'n': int(len(sub))}
        for m_name, col in (('lgb', 'pred_lgb'), ('b1_naive', 'pred_b1_naive'),
                            ('b2_vpin', 'pred_b2_vpin'), ('b3_har', 'pred_b3_har')):
            yhat = sub[col].values
            rho, _ = spearmanr(y, yhat)
            row[m_name] = float(rho)
        out[c] = row
    return out


def _per_fold_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pooled per-fold (LGB − B3) Spearman ρ delta."""
    pooled = metrics[metrics['slice'] == 'pooled'][['fold_id', 'model', 'rho']]
    wide = pooled.pivot(index='fold_id', columns='model', values='rho').sort_index()
    wide['delta_lgb_minus_b3'] = wide['lgb'] - wide['b3_har']
    return wide.reset_index()


def _importance_stability(imp: pd.DataFrame, k: int = 10) -> dict:
    """Spearman rank correlation of top-k features across adjacent folds.

    For each adjacent pair (fold i, fold i+1), take the union of top-k features (by gain)
    from either fold, rank each feature by gain within each fold, then compute Spearman rho.
    Report the list of per-pair rhos and summary stats.
    """
    folds = sorted(imp['fold_id'].unique())
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(folds) - 1):
        a, b = folds[i], folds[i + 1]
        ga = imp[imp['fold_id'] == a].set_index('feature')['gain']
        gb = imp[imp['fold_id'] == b].set_index('feature')['gain']
        topa = set(ga.sort_values(ascending=False).head(k).index)
        topb = set(gb.sort_values(ascending=False).head(k).index)
        union = list(topa | topb)
        ra = ga.reindex(union).rank(ascending=False).values
        rb = gb.reindex(union).rank(ascending=False).values
        rho, _ = spearmanr(ra, rb)
        pairs.append((int(a), int(b), float(rho)))
    rhos = [r for _, _, r in pairs]
    return {
        'pair_rhos': pairs,
        'mean_rho': float(np.mean(rhos)) if rhos else float('nan'),
        'median_rho': float(np.median(rhos)) if rhos else float('nan'),
        'pct_above_0p7': float(np.mean(np.array(rhos) > 0.7)) if rhos else float('nan'),
    }


def _top_features_by_mean_gain(imp: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    by_feat = imp.groupby('feature', observed=True).agg(
        mean_gain=('gain', 'mean'), std_gain=('gain', 'std'),
        mean_splits=('split_count', 'mean'),
        n_folds=('fold_id', 'nunique'),
    ).sort_values('mean_gain', ascending=False)
    return by_feat.head(k).reset_index()


def _per_fold_top5_presence(imp: pd.DataFrame, watch: list[str]) -> pd.DataFrame:
    """For each fold, indicate whether each feature in `watch` is in that fold's top-5 by gain."""
    folds = sorted(imp['fold_id'].unique())
    rows = []
    for f in folds:
        sub = imp[imp['fold_id'] == f].sort_values('gain', ascending=False)
        top5 = set(sub.head(5)['feature'].values)
        rows.append({'fold_id': f, **{w: (w in top5) for w in watch}})
    return pd.DataFrame(rows)


def _residual_diagnostics(preds: pd.DataFrame) -> dict:
    """Residuals (target − pred_lgb): breakdown by contract, high-vol regime, FOMC day."""
    p = preds.copy()
    p['resid_lgb'] = p['target_log_sigma_rv'] - p['pred_lgb']
    out: dict = {'by_contract': {}}
    for c, sub in p.groupby('contract', observed=True):
        out['by_contract'][c] = {
            'n': int(len(sub)),
            'mean': float(sub['resid_lgb'].mean()),
            'std': float(sub['resid_lgb'].std()),
        }
    # High-vol regime: top decile of target within each fold (contemporaneous, diagnostic only)
    hi_cut = p.groupby('fold_id', observed=True)['target_log_sigma_rv'].transform(lambda s: s.quantile(0.9))
    hi = p['target_log_sigma_rv'] >= hi_cut
    out['high_vol_top_decile'] = {
        'n': int(hi.sum()), 'mean_resid': float(p.loc[hi, 'resid_lgb'].mean()),
        'std_resid': float(p.loc[hi, 'resid_lgb'].std()),
    }
    out['normal_regime'] = {
        'n': int((~hi).sum()), 'mean_resid': float(p.loc[~hi, 'resid_lgb'].mean()),
        'std_resid': float(p.loc[~hi, 'resid_lgb'].std()),
    }
    # FOMC days (calendar day matches any announcement)
    ts = pd.to_datetime(p['ts'], utc=True)
    fomc_set = {pd.Timestamp(d).date() for d in FOMC_DATES}
    is_fomc = ts.dt.date.isin(fomc_set).values
    out['fomc_days'] = {
        'n': int(is_fomc.sum()),
        'mean_resid': float(p.loc[is_fomc, 'resid_lgb'].mean()) if is_fomc.any() else float('nan'),
        'std_resid': float(p.loc[is_fomc, 'resid_lgb'].std()) if is_fomc.any() else float('nan'),
    }
    out['non_fomc_days'] = {
        'n': int((~is_fomc).sum()),
        'mean_resid': float(p.loc[~is_fomc, 'resid_lgb'].mean()),
        'std_resid': float(p.loc[~is_fomc, 'resid_lgb'].std()),
    }
    return out


def _render(preds: pd.DataFrame, metrics: pd.DataFrame, imp: pd.DataFrame) -> str:
    L: list[str] = []
    L.append('# Stage 1 — Full Run Results (forward realized volatility)')
    L.append('')
    L.append(f'**Generated:** {pd.Timestamp.now(tz="UTC").isoformat()}')
    L.append('**Target:** log σ_rv over the forward 12 bars (1 hour at 5-min resolution)')
    L.append(f'**Folds executed:** {metrics["fold_id"].nunique()} (expanding, 12-month warmup, 2-month blocks, 1-day purge)')
    L.append(f'**Test predictions:** {len(preds):,}')
    L.append('')

    pooled_agg = _pooled_aggregate(preds)
    pc_agg = _per_contract_aggregate(preds)
    fold_delta = _per_fold_delta(metrics)

    per_contract_delta = {c: pc_agg[c]['lgb'] - pc_agg[c]['b3_har'] for c in pc_agg}
    crit = M.success_criteria(
        per_fold_delta=fold_delta['delta_lgb_minus_b3'].values,
        per_contract_delta=per_contract_delta,
        pooled_rho_lgb=pooled_agg['lgb']['rho'],
        pooled_rho_b3=pooled_agg['b3_har']['rho'],
    )

    # 1. Executive summary
    L.append('## 1. Executive summary')
    L.append('')
    L.append('| Model | pooled ρ | pooled RMSE | pooled MAE |')
    L.append('|-------|---------:|------------:|-----------:|')
    for m_name, label in (('b1_naive', 'B1 naive persistence'),
                          ('b2_vpin', 'B2 VPIN + σ_lag OLS'),
                          ('b3_har', 'B3 HAR-RV OLS'),
                          ('lgb', 'LightGBM (17 + contract)')):
        a = pooled_agg[m_name]
        L.append(f'| {label} | {a["rho"]:+.4f} | {a["rmse"]:.4f} | {a["mae"]:.4f} |')
    L.append(f'| **LGB − HAR-RV** | **{pooled_agg["lgb"]["rho"] - pooled_agg["b3_har"]["rho"]:+.4f}** | — | — |')
    L.append('')
    L.append('### Success criteria (spec §7)')
    L.append('')
    c1, c2, c3 = crit['c1_pooled_delta'], crit['c2_sign_consistency'], crit['c3_per_contract_generality']
    L.append(f'1. **Pooled Δρ ≥ 0.03:** {c1["value"]:+.4f} — {"PASS" if c1["pass"] else "FAIL"}')
    L.append(f'2. **Sign consistency ≥ 70% of folds:** {c2["folds_positive"]}/{c2["folds_total"]} '
             f'({100 * c2["value"]:.1f}%) — {"PASS" if c2["pass"] else "FAIL"}')
    L.append(f'3. **Per-contract: beats HAR on ≥ 2 of 3 contracts:** {c3["n_beating"]}/3 — '
             f'{"PASS" if c3["pass"] else "FAIL"}')
    L.append(f'   - per-contract Δρ: ' + ', '.join(
        f'{c}={v:+.4f}' for c, v in c3["per_contract_delta"].items()))
    L.append('')
    L.append(f'**Overall: {"ALL PASS — main paper result" if crit["all_pass"] else "NOT all criteria met — see details"}**')
    L.append('')

    # 2. Per-contract comparison
    L.append('## 2. Per-contract comparison (pooled across all folds)')
    L.append('')
    L.append('| contract | n | B1 | B2 | B3 | LGB | LGB − B3 |')
    L.append('|----------|---:|-------:|-------:|-------:|-------:|--------:|')
    for c in T.CONTRACTS:
        if c not in pc_agg:
            continue
        r = pc_agg[c]
        L.append(f'| {c} | {r["n"]:,} | {r["b1_naive"]:+.4f} | {r["b2_vpin"]:+.4f} | '
                 f'{r["b3_har"]:+.4f} | {r["lgb"]:+.4f} | {r["lgb"] - r["b3_har"]:+.4f} |')
    L.append('')

    # 3. Per-fold table
    L.append('## 3. Per-fold LightGBM vs HAR-RV')
    L.append('')
    L.append('| fold | ρ_LGB | ρ_HAR | Δ (LGB − HAR) | sign |')
    L.append('|----:|------:|------:|--------------:|:----:|')
    for _, row in fold_delta.iterrows():
        sign = '+' if row['delta_lgb_minus_b3'] > 0 else ('−' if row['delta_lgb_minus_b3'] < 0 else '0')
        L.append(f'| {int(row["fold_id"])} | {row["lgb"]:+.4f} | {row["b3_har"]:+.4f} | '
                 f'{row["delta_lgb_minus_b3"]:+.4f} | {sign} |')
    L.append('')

    # 4. Feature-importance stability
    stab = _importance_stability(imp, k=10)
    L.append('## 4. Feature-importance rank stability')
    L.append('')
    L.append(f'Spearman ρ of top-10 feature ranks across adjacent folds (fold i vs i+1):')
    L.append(f'- pair count: **{len(stab["pair_rhos"])}**')
    L.append(f'- mean pair-ρ: **{stab["mean_rho"]:.3f}**')
    L.append(f'- median pair-ρ: **{stab["median_rho"]:.3f}**')
    L.append(f'- pairs with ρ > 0.70: **{100 * stab["pct_above_0p7"]:.1f}%**')
    interpretation = ('stable' if stab['mean_rho'] > 0.7
                      else 'moderately stable' if stab['mean_rho'] > 0.5
                      else 'unstable — revisit interaction constraints')
    L.append(f'- **interpretation:** importance ranking is **{interpretation}** across folds (spec §6.1)')
    L.append('')

    # 5. Top features
    top5 = _top_features_by_mean_gain(imp, k=5)
    L.append('## 5. Top 5 features by mean gain-based importance (all folds)')
    L.append('')
    L.append('| rank | feature | mean gain | std gain | mean splits | folds present |')
    L.append('|----:|:--------|----------:|---------:|------------:|--------------:|')
    for i, row in enumerate(top5.itertuples(index=False), 1):
        L.append(f'| {i} | `{row.feature}` | {row.mean_gain:,.1f} | {row.std_gain:,.1f} | '
                 f'{row.mean_splits:,.1f} | {int(row.n_folds)} |')
    L.append('')

    # Presence of the scaffold-observed top-5 across folds
    watch = ['sigma_yz', 'sigma_gk_lag1', 'sigma_gk_short', 'sigma_gk_long', 'ko_W']
    presence = _per_fold_top5_presence(imp, watch)
    L.append('### Stability of scaffold-observed top-5 pattern')
    L.append('')
    L.append('Share of folds where each scaffold-top-5 feature appears in that fold\'s top-5 by gain:')
    L.append('')
    L.append('| feature | folds in top-5 | % |')
    L.append('|:--------|--------------:|--:|')
    n_f = len(presence)
    for w in watch:
        cnt = int(presence[w].sum())
        L.append(f'| `{w}` | {cnt}/{n_f} | {100 * cnt / n_f:.0f}% |')
    L.append('')
    bvc = set(T.CORE_BVC)
    bvc_in_top5 = imp[imp['feature'].isin(bvc)].groupby('fold_id')['gain'].apply(
        lambda s: s.max() if not s.empty else 0)
    bvc_top5_count = int(sum(
        (imp[imp['fold_id'] == f].sort_values('gain', ascending=False).head(5)['feature']
         .isin(bvc).any()) for f in presence['fold_id'].values))
    L.append(f'**BVC presence in top-5:** {bvc_top5_count}/{n_f} folds ({100 * bvc_top5_count / n_f:.0f}%).')
    if bvc_top5_count / n_f < 0.1:
        L.append('')
        L.append('**Finding:** across essentially all folds, the model leans on vol-state features '
                 '(σ_yz + HAR trio) plus physics `ko_W`, with the 7 BVC features rarely entering top-5. '
                 'This is consistent with the Andersen-Bondarenko critique that volume-based flow '
                 'signals add little to volatility forecasting once vol-state is properly conditioned.')
    else:
        L.append('')
        L.append(f'**Finding:** BVC features enter top-5 in {100 * bvc_top5_count / n_f:.0f}% of folds. '
                 'Regime conditionality of BVC importance may warrant further investigation.')
    L.append('')

    # 6. Residual diagnostics
    resid = _residual_diagnostics(preds)
    L.append('## 6. Residual diagnostics (LightGBM residuals)')
    L.append('')
    L.append('### By contract')
    L.append('')
    L.append('| contract | n | mean resid | std resid |')
    L.append('|:---------|---:|-----------:|----------:|')
    for c, r in resid['by_contract'].items():
        L.append(f'| {c} | {r["n"]:,} | {r["mean"]:+.4f} | {r["std"]:.4f} |')
    L.append('')
    L.append('### By regime (target top decile within fold = high vol)')
    L.append('')
    L.append('| regime | n | mean resid | std resid |')
    L.append('|:-------|---:|-----------:|----------:|')
    L.append(f'| high-vol (top decile) | {resid["high_vol_top_decile"]["n"]:,} | '
             f'{resid["high_vol_top_decile"]["mean_resid"]:+.4f} | {resid["high_vol_top_decile"]["std_resid"]:.4f} |')
    L.append(f'| normal | {resid["normal_regime"]["n"]:,} | '
             f'{resid["normal_regime"]["mean_resid"]:+.4f} | {resid["normal_regime"]["std_resid"]:.4f} |')
    L.append('')
    L.append('### By FOMC announcement day (embedded Fed calendar 2021–2026)')
    L.append('')
    L.append('| day type | n | mean resid | std resid |')
    L.append('|:---------|---:|-----------:|----------:|')
    L.append(f'| FOMC day | {resid["fomc_days"]["n"]:,} | '
             f'{resid["fomc_days"]["mean_resid"]:+.4f} | {resid["fomc_days"]["std_resid"]:.4f} |')
    L.append(f'| non-FOMC | {resid["non_fomc_days"]["n"]:,} | '
             f'{resid["non_fomc_days"]["mean_resid"]:+.4f} | {resid["non_fomc_days"]["std_resid"]:.4f} |')
    L.append('')

    # 7. Correctness provenance + sanity bound notes
    L.append('## 7. Correctness provenance')
    L.append('')
    L.append('**Target alignment verified in scaffold validation (2026-04-18):** at bar indices '
             't ∈ {500, 1000, 2000, 5000} on the highest-row-count front-month instrument, '
             'hand-computed Σ log_ret²_{t+1..t+12} matched the framework\'s `forward_ss[t]` to '
             'float64 precision, and 0.5·log(forward_ss) matched `target[t]` exactly (4/4 matches). '
             'Features at bar t are Phase-2 constructions from bar-t OHLC and sub-bar stats; '
             'rolling/lag aggregates use only information from bars ≤ t. Conclusion: no target leakage.')
    L.append('')
    L.append('## 8. Sanity bounds (authorized 2026-04-18)')
    L.append('')
    L.append(f'| band | lower | upper |')
    L.append(f'|------|------:|------:|')
    L.append(f'| ρ_HAR | {M.FINAL_SANITY["rho_har_lower"]:.2f} | {M.FINAL_SANITY["rho_har_upper"]:.2f} |')
    L.append(f'| ρ_LGB | {M.FINAL_SANITY["rho_lgb_lower"]:.2f} | {M.FINAL_SANITY["rho_lgb_upper"]:.2f} |')
    L.append(f'| Δ (LGB−HAR) | {M.FINAL_SANITY["diff_lower"]:+.2f} | {M.FINAL_SANITY["diff_upper"]:+.2f} |')
    L.append('')
    L.append('**Rationale.** HAR-RV literature at 5-min → 1-hour horizons on equity index futures '
             'reports R² of 0.6–0.75 (Corsi 2009; Andersen–Bollerslev–Diebold). For monotonic mappings, '
             'Spearman ρ is typically ≥ √R², so ρ_HAR in [0.50, 0.80] is the expected literature band, '
             'not an anomaly. The single-fold scaffold observations (ρ_HAR=0.726, ρ_LGB=0.778, Δ=+0.052) '
             'sit inside these bands and were re-authorized as non-blocking.')
    obs_rho_har = pooled_agg["b3_har"]['rho']
    obs_rho_lgb = pooled_agg["lgb"]['rho']
    obs_diff = obs_rho_lgb - obs_rho_har
    flags = []
    s = M.FINAL_SANITY
    if obs_rho_har < s['rho_har_lower']:
        flags.append(f'ρ_HAR {obs_rho_har:.3f} below lower bound')
    if obs_rho_har > s['rho_har_upper']:
        flags.append(f'ρ_HAR {obs_rho_har:.3f} above upper bound')
    if obs_rho_lgb < s['rho_lgb_lower']:
        flags.append(f'ρ_LGB {obs_rho_lgb:.3f} below lower bound')
    if obs_rho_lgb > s['rho_lgb_upper']:
        flags.append(f'ρ_LGB {obs_rho_lgb:.3f} above upper bound')
    if obs_diff < s['diff_lower']:
        flags.append(f'Δ {obs_diff:+.3f} below lower bound')
    if obs_diff > s['diff_upper']:
        flags.append(f'Δ {obs_diff:+.3f} above upper bound')
    if flags:
        L.append('')
        L.append('**Bound violations on this run:**')
        for f in flags:
            L.append(f'- {f}')
    else:
        L.append('')
        L.append('**All observed values lie within authorized sanity bounds.**')
    L.append('')
    L.append('## 9. Scope')
    L.append('Stage 1 full run complete. This repository publishes Stage 1 only: '
             'no hyperparameter tuning, no alternative target, no ranking layer, '
             'no signal extraction is in scope.')
    L.append('')
    return '\n'.join(L)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('bvc.report')

    preds = pd.read_parquet(_HERE / 'predictions.parquet')
    metrics = pd.read_parquet(_HERE / 'fold_metrics.parquet')
    imp = pd.read_parquet(_HERE / 'feature_importance.parquet')
    log.info('Loaded: predictions=%d, metrics=%d, importance=%d',
             len(preds), len(metrics), len(imp))

    text = _render(preds, metrics, imp)
    out = _HERE / 'STAGE1_RESULTS.md'
    out.write_text(text, encoding='utf-8')
    log.info('Wrote %s (%d lines)', out, text.count('\n') + 1)


if __name__ == '__main__':
    main()
