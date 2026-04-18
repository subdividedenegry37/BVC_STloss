"""
phase1_causal_diagnostic.py — baseline predictive calibration (causal σ)

Compares concurrent σ (Phase 1 baseline) against strictly causal σ (uses only
t-1 and earlier). This is the calibration Stage 1 ML requires.

KEY FINDING from 1-year data:
  ν drops from ~6.8 (concurrent) to ~4.4 (causal) across all contracts.
  Scale drops by ~0.04. Parameters cross "meaningful shift" thresholds.

  Interpretation: concurrent σ was hiding the true tail index by absorbing
  surprise events into its own denominator. Causal σ correctly reports its
  ignorance of the current bar, and z explodes during FOMC/tariff shocks.

  ν ≈ 4.4 is the HONEST tail index. Stage 1 uses these parameters.

Validation diagnostics run:
  - MLE fit comparison (concurrent vs causal vs causal+span19)
  - PIT uniformity on causal fit
  - Shoulder CDF fit on causal fit
  - Split-half SKEWNESS stability (not ν) — decides symmetric t vs skew-t
  - Skewness by |z| bucket — shows asymmetry is concentrated in tails

USAGE:
    python phase1_causal_diagnostic.py --es ... --nq ... --rty ... --outdir ./outputs

EXPECTED RESULTS (1-year test data):
    ES:  ν=4.607, loc=0.0132, scale=0.7716
    NQ:  ν=4.383, loc=0.0174, scale=0.7519
    RTY: ν=4.460, loc=0.0094, scale=0.8279

DECISION RULE on 15-year data:
  - If causal parameters match 1-year values within ±1.0 on ν: EXPECTED, proceed.
  - If ν drifts sharply across the 15 years (e.g., 2008 crisis, COVID): flag for
    regime-conditional calibration in downstream analyses.
  - If split-half skewness shows STRUCTURAL asymmetry (same sign, similar magnitude,
    Δ < 0.3 across halves in at least 2 of 3 contracts): implement Hansen skew-t.
  - If split-half skewness is unstable (sign flips, >3× variation): stay with
    symmetric t. Full-sample skewness is a sample artifact, not structural.
"""

import argparse
import pickle
from pathlib import Path
import psutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.interpolate import interp1d

from student_t_bvc import (
    aggregate_to_bars,
    compute_gk_variance,
    compute_sigma_concurrent,
    compute_sigma_causal,
    compute_sigma_causal_session_isolated,
    fit_student_t,
    pit_uniformity,
    shoulder_cdf_deviation,
    split_half_stability,
    skewness_by_magnitude,
    PHASE1_CAUSAL_PARAMS_5MIN,
)


def compute_variant_c(df, span=19):
    """Variant C: causal with span adjusted to compensate information loss."""
    out = []
    for sym, grp in df.groupby('symbol'):
        gv = compute_gk_variance(grp)
        gv_lagged = gv.shift(1)
        sigma = np.sqrt(gv_lagged.ewm(span=span, min_periods=10).mean())
        out.append(sigma)
    return pd.concat(out).sort_index()


def run_causal_diagnostic(parquet_paths, outdir, bar_minutes=5, span=20, warmup=None, gap=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in parquet_paths.items():
        print(f"\n{'='*70}")
        print(f"CAUSAL σ DIAGNOSTIC: {name}")
        print(f"{'='*70}")

        mem_before = psutil.Process().memory_info().rss / 1e9
        print(f"Memory before loading: {mem_before:.2f} GB")

        raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])

        mem_after_load = psutil.Process().memory_info().rss / 1e9
        print(f"Memory after loading: {mem_after_load:.2f} GB")

        bars = aggregate_to_bars(raw, bar_minutes)
        print(f"  {len(bars):,} {bar_minutes}-min bars")

        del raw
        import gc; gc.collect()
        mem_after_agg = psutil.Process().memory_info().rss / 1e9
        print(f"Memory after agg and GC: {mem_after_agg:.2f} GB")

        bars['log_ret'] = np.log(bars['close'] / bars['open'])

        # Three σ variants
        sigma_A = compute_sigma_concurrent(bars, span=span)    # Phase 1 baseline
        if warmup is not None and gap is not None:
            # Sort for session isolation without destroying index
            bars['_row_id'] = np.arange(len(bars))
            bars['_ts'] = bars.index
            bars_sorted = bars.sort_values(['instrument_id', '_ts']).copy()
            sigma_B_series, _ = compute_sigma_causal_session_isolated(bars_sorted, span=span, warmup_bars=warmup, gap_threshold=gap)
            bars_sorted['sigma_B'] = sigma_B_series
            bars_sorted = bars_sorted.sort_values('_row_id')
            sigma_B = bars_sorted['sigma_B'].values
            bars = bars.drop(columns=['_row_id', '_ts'])
        else:
            sigma_B = compute_sigma_causal(bars, span=span)         # strict causal

        sigma_C = compute_variant_c(bars, span=span-1)          # causal + span comp.

        bars['sigma_A'] = sigma_A
        bars['sigma_B'] = sigma_B
        bars['sigma_C'] = sigma_C
        bars['z_A'] = bars['log_ret'] / bars['sigma_A']
        bars['z_B'] = bars['log_ret'] / bars['sigma_B']
        bars['z_C'] = bars['log_ret'] / bars['sigma_C']

        paired = bars[['z_A', 'z_B', 'z_C', 'sigma_A', 'sigma_B', 'sigma_C']].dropna()

        # Fit all three variants
        nu_a, loc_a, scale_a = fit_student_t(paired['z_A'].values)
        nu_b, loc_b, scale_b = fit_student_t(paired['z_B'].values)
        nu_c, loc_c, scale_c = fit_student_t(paired['z_C'].values)

        print(f"\n  {'Variant':<20} {'ν':>8} {'loc':>10} {'scale':>10}")
        print(f"  {'A Concurrent':<20} {nu_a:>8.3f} {loc_a:>10.4f} {scale_a:>10.4f}")
        print(f"  {'B Causal':<20} {nu_b:>8.3f} {loc_b:>10.4f} {scale_b:>10.4f}")
        print(f"  {'C Causal+span19':<20} {nu_c:>8.3f} {loc_c:>10.4f} {scale_c:>10.4f}")

        # Parameter shifts
        dnu = nu_a - nu_b
        dscale = scale_a - scale_b
        dloc = loc_a - loc_b
        meaningful = (abs(dnu) > 0.5) or (abs(dscale) > 0.03) or (abs(dloc) > 0.005)
        print(f"\n  Shift (Concurrent - Causal): Δν={dnu:+.3f}, Δscale={dscale:+.4f}, Δloc={dloc:+.4f}")
        print(f"  Meaningful (thresholds Δν>0.5, Δscale>0.03): {meaningful}")

        # Correlations
        sig_corr = paired['sigma_A'].corr(paired['sigma_B'])
        z_corr = paired['z_A'].corr(paired['z_B'])
        print(f"  σ_A vs σ_B correlation: {sig_corr:.4f}")
        print(f"  z_A vs z_B correlation: {z_corr:.4f}")

        # z-range expansion (the surprise-event story)
        print(f"\n  z-range concurrent: [{paired['z_A'].min():.2f}, {paired['z_A'].max():.2f}]")
        print(f"  z-range causal:     [{paired['z_B'].min():.2f}, {paired['z_B'].max():.2f}]")
        n_extreme_a = (np.abs(paired['z_A']) > 5).sum()
        n_extreme_b = (np.abs(paired['z_B']) > 5).sum()
        print(f"  |z|>5 concurrent: {n_extreme_a} bars  ({n_extreme_a/len(paired)*100:.3f}%)")
        print(f"  |z|>5 causal:     {n_extreme_b} bars  ({n_extreme_b/len(paired)*100:.3f}%)")

        # Full validation on CAUSAL fit (Phase 1')
        z_b_clean = paired['z_B'].values
        pit_res = pit_uniformity(z_b_clean, nu_b, loc_b, scale_b)
        shoulder_res = shoulder_cdf_deviation(z_b_clean, nu_b, loc_b, scale_b)
        stab = split_half_stability(z_b_clean)
        skew_mag = skewness_by_magnitude(z_b_clean)

        print(f"\n  CAUSAL-FIT VALIDATION:")
        print(f"    PIT KS:            {pit_res['ks_stat']:.4f}  ({pit_res['shape']})")
        print(f"    PIT asymmetry:     {pit_res['asymmetry']:+.3f}")
        print(f"    Shoulder PASS:     {shoulder_res['passes_0.01_threshold']}")

        # Split-half SKEWNESS — the skew-t decision
        skew_h1 = stab['h1']['skew']
        skew_h2 = stab['h2']['skew']
        print(f"\n  Split-half SKEWNESS (skew-t decision):")
        print(f"    H1: skew={skew_h1:+.3f}")
        print(f"    H2: skew={skew_h2:+.3f}")
        print(f"    Δ skew: {abs(skew_h1 - skew_h2):.3f}")

        sign_flip = np.sign(skew_h1) != np.sign(skew_h2)
        big_diff = abs(skew_h1 - skew_h2) > 0.3
        if sign_flip:
            verdict = 'SAMPLE ARTIFACT (sign flip) → symmetric t is correct'
        elif big_diff:
            verdict = 'UNSTABLE (Δ > 0.3) → symmetric t is correct'
        elif abs(skew_h1) > 0.3 and abs(skew_h2) > 0.3:
            verdict = 'STRUCTURAL → Hansen skew-t recommended'
        else:
            verdict = 'MODERATE — symmetric t acceptable'
        print(f"    VERDICT: {verdict}")

        print(f"\n  Skewness by |z| bucket:")
        for _, row in skew_mag.iterrows():
            print(f"    {row['range']:<12} n={row['n']:>8,}  "
                  f"pct={row['pct_of_total']:>5.2f}%  skew={row['skewness']:+.3f}")

        # Compare with 1-year locked causal parameters
        if name in PHASE1_CAUSAL_PARAMS_5MIN:
            locked = PHASE1_CAUSAL_PARAMS_5MIN[name]
            print(f"\n  vs 1-year locked causal ({name}):")
            print(f"    Δν     = {nu_b - locked['nu']:+.3f}")
            print(f"    Δloc   = {loc_b - locked['loc']:+.4f}")
            print(f"    Δscale = {scale_b - locked['scale']:+.4f}")

        results[name] = {
            'n_bars': len(paired),
            'concurrent': {'nu': nu_a, 'loc': loc_a, 'scale': scale_a},
            'causal':     {'nu': nu_b, 'loc': loc_b, 'scale': scale_b},
            'causal_sp19':{'nu': nu_c, 'loc': loc_c, 'scale': scale_c},
            'sigma_corr': sig_corr,
            'z_corr':     z_corr,
            'z_range_concurrent': [float(paired['z_A'].min()), float(paired['z_A'].max())],
            'z_range_causal':     [float(paired['z_B'].min()), float(paired['z_B'].max())],
            'pit_causal':      pit_res,
            'shoulder_causal': shoulder_res,
            'stability':       stab,
            'skew_mag':        skew_mag.to_dict('records'),
            'skew_verdict':    verdict,
            'z_B_sample':      z_b_clean[:100000],  # keep a sample for plotting
        }

    # Figure
    _plot_causal_diagnostics(results, outdir / 'causal_pit_diagnostics.png')

    # Save
    pkl_path = outdir / 'causal_diagnostics.pkl'
    slim = {k: {kk: vv for kk, vv in v.items() if kk != 'z_B_sample'}
            for k, v in results.items()}
    with open(pkl_path, 'wb') as f:
        pickle.dump(slim, f)

    # Summary
    with open(outdir / 'phase1_causal_summary.txt', 'w') as f:
        f.write("PHASE 1' CAUSAL CALIBRATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("LOCKED CAUSAL PARAMETERS ((causal, predictive)):\n\n")
        for name, r in results.items():
            c = r['causal']
            f.write(f"  {name}:  ν={c['nu']:.3f}  loc={c['loc']:.4f}  scale={c['scale']:.4f}\n")
        f.write("\n\nVALIDATION:\n")
        for name, r in results.items():
            f.write(f"  {name}:\n")
            f.write(f"    PIT KS = {r['pit_causal']['ks_stat']:.4f}\n")
            f.write(f"    Shoulder PASS = {r['shoulder_causal']['passes_0.01_threshold']}\n")
            f.write(f"    Skew verdict:   {r['skew_verdict']}\n")

    print(f"\n\nResults saved to {outdir}")


def _plot_causal_diagnostics(results, outpath):
    """3 rows (ES, NQ, RTY) × 3 cols (PIT, CDF residual, skew-by-|z|)."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("baseline Validation — causal GK σ, Student-t 3-parameter fit",
                 fontsize=14, fontweight='bold', y=0.99)
    gs = GridSpec(len(results), 3, figure=fig, hspace=0.4, wspace=0.3)

    for row, (name, r) in enumerate(results.items()):
        z = r['z_B_sample']
        z = z[np.isfinite(z) & (np.abs(z) < 50)]
        c = r['causal']
        nu, loc, scale = c['nu'], c['loc'], c['scale']

        # PIT
        ax = fig.add_subplot(gs[row, 0])
        pit = stats.t.cdf(z, df=nu, loc=loc, scale=scale)
        ax.hist(pit, bins=50, alpha=0.7, color='purple', density=True, edgecolor='black', lw=0.3)
        ax.axhline(1.0, color='black', ls='--', lw=1, label='Uniform target')
        ax.set_title(f'{name} — PIT (causal)\nKS = {r["pit_causal"]["ks_stat"]:.4f}', fontsize=11)
        ax.set_xlabel('PIT'); ax.set_ylabel('Density')
        ax.set_ylim(0.5, 1.6)
        ax.legend(fontsize=8)

        # Shoulder CDF residual
        ax = fig.add_subplot(gs[row, 1])
        z_sorted = np.sort(z); n = len(z_sorted)
        ecdf_vals = np.arange(1, n+1) / (n+1)
        ecdf_func = interp1d(z_sorted, ecdf_vals, bounds_error=False, fill_value=(0, 1))
        z_eval = np.linspace(-4, 4, 1000)
        t_cdf = stats.t.cdf(z_eval, df=nu, loc=loc, scale=scale)
        n_cdf = stats.norm.cdf(z_eval, loc=np.mean(z), scale=np.std(z))
        emp = ecdf_func(z_eval)
        ax.plot(z_eval, (emp - t_cdf) * 100, color='purple', lw=1.5, label='Emp − Student-t')
        ax.plot(z_eval, (emp - n_cdf) * 100, color='red', lw=1.5, label='Emp − Gaussian')
        ax.axhline(0, color='black', ls='--', lw=0.5)
        ax.axhspan(-1, 1, alpha=0.1, color='green')
        ax.set_title(f'{name} — CDF Residual (%)', fontsize=11)
        ax.set_xlabel('z'); ax.set_ylabel('Residual (%)')
        ax.set_xlim(-4, 4); ax.set_ylim(-6, 6)
        ax.legend(fontsize=7)

        # Skewness by |z|
        ax = fig.add_subplot(gs[row, 2])
        sm = pd.DataFrame(r['skew_mag']).dropna(subset=['skewness'])
        x = range(len(sm))
        colors = ['green' if abs(s) < 0.15 else ('orange' if abs(s) < 0.3 else 'red')
                  for s in sm['skewness']]
        ax.bar(x, sm['skewness'], alpha=0.7, color=colors, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(sm['range'], rotation=30, fontsize=8)
        ax.axhline(0, color='black', lw=0.5)
        ax.axhline(0.15, color='orange', ls='--', lw=1, alpha=0.7, label='|0.15| threshold')
        ax.axhline(-0.15, color='orange', ls='--', lw=1, alpha=0.7)
        ax.set_title(f'{name} — Skewness by |z| bucket (causal)', fontsize=11)
        ax.set_ylabel('Skewness within bucket')
        ax.legend(fontsize=8)

    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--nq', required=True)
    parser.add_argument('--rty', required=True)
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--bar-minutes', type=int, default=5)
    parser.add_argument('--span', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--gap', type=str, default=None)
    args = parser.parse_args()

    paths = {'ES': args.es, 'NQ': args.nq, 'RTY': args.rty}
    run_causal_diagnostic(paths, args.outdir, bar_minutes=args.bar_minutes, span=args.span, warmup=args.warmup, gap=args.gap)
