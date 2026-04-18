"""Phase 4 final validation — ES post-only, NQ full, RTY full.

Produces comparison table: old baseline vs proposed new params, with
full validation (PIT, shoulder, split-half, skew-by-|z|, per-year).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gc
from student_t_bvc import (
    aggregate_to_bars, compute_sigma_causal_session_isolated,
    fit_student_t, pit_uniformity, shoulder_cdf_deviation,
    split_half_stability, skewness_by_magnitude,
    PHASE1_CAUSAL_PARAMS_5MIN,
)

# ES_POST_CUT: ES regime-start used for post-regime validation fits and for
# gating `is_valid_bar` in the input pickles.
#
# Retained at 2020-03-01 on economic grounds (COVID volatility peak, Fed
# intervention, structural market events cluster in Feb-Mar 2020).
#
# The monthly-resolution diagnostic in
#   runs/2026-04-18_es_monthly_stabilization/ES_MONTHLY_STABILIZATION.md
# confirmed the regime break is discrete rather than gradual: post-break ν is
# insensitive to boundary choice in Jan-Mar 2020 (Δν < 0.02 across eight
# candidate cutoffs; earliest cutoff passing all stabilization criteria is
# 2020-01-01). That finding strengthens the regime-split methodology but does
# not move the boundary. Training-window gating of NQ and RTY at 2020-01-01
# is applied via the `is_train_valid_bar` column in
# runs/2026-04-18_physics_feature_expansion/phase2_features_expanded_*.pkl
# (see runs/2026-04-18_es_monthly_stabilization/PHASE1_LOCK_FINAL_v2_TRAINING_WINDOW.md).
ES_POST_CUT = pd.Timestamp('2020-03-01', tz='UTC')


def load_or_compute_z(path, cache, warmup=80, gap='15min', span=20, bar_minutes=5):
    if Path(cache).exists():
        print(f"  cached: {cache}", flush=True)
        return pd.read_parquet(cache)
    raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close',
                                          'volume', 'symbol', 'instrument_id'])
    bars = aggregate_to_bars(raw, bar_minutes)
    del raw; gc.collect()
    bars['log_ret'] = np.log(bars['close'] / bars['open'])
    bars['_row_id'] = np.arange(len(bars))
    bars['_ts'] = bars.index
    bars_sorted = bars.sort_values(['instrument_id', '_ts']).copy()
    sigma, _ = compute_sigma_causal_session_isolated(
        bars_sorted, span=span, warmup_bars=warmup, gap_threshold=gap)
    bars_sorted['_sigma'] = sigma
    bars_sorted2 = bars_sorted.sort_values('_row_id')
    bars['sigma'] = bars_sorted2['_sigma'].values
    bars['z'] = bars['log_ret'] / bars['sigma']
    z_df = pd.DataFrame({'ts': bars.index, 'z': bars['z'].values}).dropna()
    z_df = z_df[np.isfinite(z_df['z']) & (np.abs(z_df['z']) < 50)].reset_index(drop=True)
    z_df.to_parquet(cache)
    del bars, bars_sorted, bars_sorted2; gc.collect()
    return z_df


def full_validate(z_vals, label):
    nu, loc, scale = fit_student_t(z_vals)
    pit = pit_uniformity(z_vals, nu, loc, scale)
    sh = shoulder_cdf_deviation(z_vals, nu, loc, scale)
    sh_max = max(sh['max_dev_left_shoulder'], sh['max_dev_right_shoulder'])
    sp = split_half_stability(z_vals)
    skew = skewness_by_magnitude(z_vals)
    print(f"\n  [{label}]  n={len(z_vals):,}")
    print(f"    ν={nu:.4f} loc={loc:+.4f} scale={scale:.4f}")
    print(f"    PIT KS={pit['ks_stat']:.4f} shape={pit['shape']} asym={pit['asymmetry']:+.3f}")
    print(f"    shoulder L={sh['max_dev_left_shoulder']:.4f} R={sh['max_dev_right_shoulder']:.4f} "
          f"(pass<0.01: {sh_max < 0.01})")
    print(f"    split-half Δν={sp['delta']['nu']:.3f} Δscale={sp['delta']['scale']:.3f} "
          f"H1-skew={sp['h1']['skew']:+.3f} H2-skew={sp['h2']['skew']:+.3f} stable={sp['stable']}")
    print(f"    skewness-by-|z|:")
    print(skew.to_string(index=False).replace('\n', '\n      '))
    return {
        'label': label, 'n': len(z_vals),
        'nu': nu, 'loc': loc, 'scale': scale,
        'ks': pit['ks_stat'], 'pit_shape': pit['shape'], 'pit_asym': pit['asymmetry'],
        'shoulder_max': sh_max,
        'shoulder_left': sh['max_dev_left_shoulder'],
        'shoulder_right': sh['max_dev_right_shoulder'],
        'split_half': sp, 'skew_by_mag': skew,
    }


def per_year(z_df, label):
    df = z_df.copy()
    df['year'] = pd.DatetimeIndex(df['ts']).year
    rows = []
    for y, grp in df.groupby('year'):
        z = grp['z'].values
        if len(z) < 5000:
            continue
        nu, loc, scale = fit_student_t(z)
        pit = pit_uniformity(z, nu, loc, scale)
        sh = shoulder_cdf_deviation(z, nu, loc, scale)
        sh_max = max(sh['max_dev_left_shoulder'], sh['max_dev_right_shoulder'])
        rows.append({
            'contract': label, 'year': int(y), 'n': len(z),
            'nu': nu, 'loc': loc, 'scale': scale,
            'ks': pit['ks_stat'], 'shoulder_max': sh_max,
            'quality': 'GOOD' if (pit['ks_stat'] < 0.05 and sh_max < 0.01) else 'FAIL'
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--nq', required=True)
    parser.add_argument('--rty', required=True)
    parser.add_argument('--outdir', default='./outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    proposed = {}
    per_year_tables = []

    # --- ES post-only ---
    print("=" * 70 + f"\nES post-only (start={ES_POST_CUT.date()})\n" + "=" * 70, flush=True)
    z_es = load_or_compute_z(args.es, outdir / 'es_z_cleaned.parquet')
    ts = pd.DatetimeIndex(z_es['ts'])
    z_es_post = z_es[np.asarray(ts >= ES_POST_CUT)].reset_index(drop=True)
    r_es = full_validate(z_es_post['z'].values, 'ES-post')
    proposed['ES'] = r_es
    per_year_tables.append(per_year(z_es_post, 'ES'))
    del z_es, z_es_post; gc.collect()

    # --- NQ full ---
    print("\n" + "=" * 70 + "\nNQ full sample\n" + "=" * 70, flush=True)
    z_nq = load_or_compute_z(args.nq, outdir / 'nq_z_cleaned.parquet')
    r_nq = full_validate(z_nq['z'].values, 'NQ-full')
    proposed['NQ'] = r_nq
    per_year_tables.append(per_year(z_nq, 'NQ'))
    del z_nq; gc.collect()

    # --- RTY full ---
    print("\n" + "=" * 70 + "\nRTY full sample\n" + "=" * 70, flush=True)
    z_rty = load_or_compute_z(args.rty, outdir / 'rty_z_cleaned.parquet')
    r_rty = full_validate(z_rty['z'].values, 'RTY-full')
    proposed['RTY'] = r_rty
    per_year_tables.append(per_year(z_rty, 'RTY'))
    del z_rty; gc.collect()

    # --- Comparison table ---
    print("\n" + "=" * 70 + "\nPROPOSED vs OLD baseline parameters\n" + "=" * 70)
    rows = []
    for c, r in proposed.items():
        old = PHASE1_CAUSAL_PARAMS_5MIN[c]
        rows.append({
            'contract': c, 'sample': r['label'], 'n': r['n'],
            'old_nu': old['nu'], 'new_nu': r['nu'],
            'old_loc': old['loc'], 'new_loc': r['loc'],
            'old_scale': old['scale'], 'new_scale': r['scale'],
            'new_ks': r['ks'], 'new_shoulder_max': r['shoulder_max'],
        })
    df_cmp = pd.DataFrame(rows)
    print(df_cmp.to_string(index=False))
    df_cmp.to_csv(outdir / 'proposed_params_comparison.csv', index=False)

    pd.concat(per_year_tables).to_csv(outdir / 'per_year_validation.csv', index=False)

    with open(outdir / 'phase4_final_validation.pkl', 'wb') as f:
        pickle.dump({'proposed': proposed, 'per_year': per_year_tables}, f)
    print(f"\nSaved: {outdir}/proposed_params_comparison.csv")
