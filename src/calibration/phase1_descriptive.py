"""
phase1_calibration.py — Descriptive Student-t BVC calibration (concurrent σ)

Fits Student-t parameters per contract using Garman-Klass σ that includes
the current bar's OHLCV. This is the DESCRIPTIVE calibration used for
retrospective analysis and cross-validation with the physics-finance project.

For PREDICTIVE use (Stage 1 ML), run phase1_causal_diagnostic.py instead —
that uses strictly causal σ and produces different (lower ν) parameters.

USAGE:
    python phase1_calibration.py --es /path/to/es.parquet --nq /path/to/nq.parquet --rty /path/to/rty.parquet --outdir ./outputs
    
    Or edit the paths at the bottom and run directly.

EXPECTED RESULTS (1-year test data, for comparison):
    ES:  ν=6.79, loc=0.0117, scale=0.8076
    NQ:  ν=6.43, loc=0.0156, scale=0.7887
    RTY: ν=6.73, loc=0.0081, scale=0.8705

Validation thresholds:
    PIT KS stat < 0.05 — distribution well-specified
    Shoulder |ΔCDF| < 0.01 — tail fit adequate
    Split-half Δν < 1.0, Δscale < 0.03 — parameters stable
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
    compute_sigma_concurrent,
    compute_sigma_ewma,
    compute_z,
    fit_student_t,
    pit_uniformity,
    shoulder_cdf_deviation,
    split_half_stability,
    skewness_by_magnitude,
    PHASE1_PARAMS_5MIN,
)


def run_phase1(parquet_paths, outdir, bar_minutes=5, span=20):
    """
    Run full Phase 1 calibration:
      1. Load and aggregate each contract
      2. Fit Student-t on both EWMA and GK sigma (to demonstrate GK breakthrough)
      3. Run PIT, shoulder CDF, split-half stability, skewness-by-magnitude
      4. Save figures and pickled results
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in parquet_paths.items():
        print(f"\n{'='*70}")
        print(f"PHASE 1 CALIBRATION: {name}")
        print(f"{'='*70}")

        mem_before = psutil.Process().memory_info().rss / 1e9
        print(f"Memory before loading: {mem_before:.2f} GB")

        print(f"Loading {path}...")
        raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
        print(f"  {len(raw):,} 1-min bars")
        mem_after_load = psutil.Process().memory_info().rss / 1e9
        print(f"Memory after loading: {mem_after_load:.2f} GB")

        bars = aggregate_to_bars(raw, bar_minutes)
        print(f"  {len(bars):,} {bar_minutes}-min bars after aggregation")

        del raw
        import gc; gc.collect()
        mem_after_agg = psutil.Process().memory_info().rss / 1e9
        print(f"Memory after agg and GC: {mem_after_agg:.2f} GB")

        # Show the GK breakthrough: EWMA vs GK sigma
        sigma_ewma = compute_sigma_ewma(bars, span=span)
        sigma_gk = compute_sigma_concurrent(bars, span=span)

        z_ewma = compute_z(bars, sigma_ewma).dropna()
        z_gk = compute_z(bars, sigma_gk).dropna()

        nu_e, loc_e, scale_e = fit_student_t(z_ewma.values)
        nu_g, loc_g, scale_g = fit_student_t(z_gk.values)

        print(f"\n  EWMA(C-O) fit: ν={nu_e:.2f}, loc={loc_e:.4f}, scale={scale_e:.4f}")
        print(f"  GK fit:        ν={nu_g:.2f}, loc={loc_g:.4f}, scale={scale_g:.4f}")
        print(f"  → GK reveals {(nu_e - nu_g):.1f} units lower ν (hidden tails)")

        # Diagnostics on the GK fit (this is the baseline spec)
        pit_res = pit_uniformity(z_gk.values, nu_g, loc_g, scale_g)
        print(f"\n  PIT KS stat:       {pit_res['ks_stat']:.4f}  ({pit_res['shape']})")
        print(f"  PIT asymmetry:     {pit_res['asymmetry']:+.3f}")

        shoulder_res = shoulder_cdf_deviation(z_gk.values, nu_g, loc_g, scale_g)
        print(f"  Shoulder |ΔCDF| R: {shoulder_res['max_dev_right_shoulder']:.4f}")
        print(f"  Shoulder |ΔCDF| L: {shoulder_res['max_dev_left_shoulder']:.4f}")
        print(f"  Shoulder PASS:     {shoulder_res['passes_0.01_threshold']}")

        stab = split_half_stability(z_gk.values)
        print(f"\n  Split-half stability:")
        print(f"    H1: ν={stab['h1']['nu']:.2f}, scale={stab['h1']['scale']:.4f}")
        print(f"    H2: ν={stab['h2']['nu']:.2f}, scale={stab['h2']['scale']:.4f}")
        print(f"    Δν={stab['delta']['nu']:.3f}, Δscale={stab['delta']['scale']:.4f}")
        print(f"    STABLE: {stab['stable']}")

        skew_mag = skewness_by_magnitude(z_gk.values)
        print(f"\n  Skewness by |z| bucket:")
        for _, row in skew_mag.iterrows():
            print(f"    {row['range']:<12} n={row['n']:>8,}  "
                  f"pct={row['pct_of_total']:>5.2f}%  skew={row['skewness']:+.3f}")

        # Compare with locked 1-year parameters
        if name in PHASE1_PARAMS_5MIN:
            locked = PHASE1_PARAMS_5MIN[name]
            print(f"\n  vs 1-year locked ({name}):")
            print(f"    Δν     = {nu_g - locked['nu']:+.3f}")
            print(f"    Δloc   = {loc_g - locked['loc']:+.4f}")
            print(f"    Δscale = {scale_g - locked['scale']:+.4f}")

        results[name] = {
            'n_bars': len(bars),
            'ewma_fit':  {'nu': nu_e, 'loc': loc_e, 'scale': scale_e},
            'gk_fit':    {'nu': nu_g, 'loc': loc_g, 'scale': scale_g},
            'pit':       pit_res,
            'shoulder':  shoulder_res,
            'stability': stab,
            'skew_mag':  skew_mag.to_dict('records'),
            'z_gk':      z_gk,
            'sigma_gk':  sigma_gk,
        }

    # Diagnostic figure
    _plot_phase1_diagnostics(results, outdir / 'phase1_diagnostics.png')

    # Save pickled results
    pkl_path = outdir / 'phase1_results.pkl'
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ('z_gk', 'sigma_gk')}
            for k, v in results.items()}
    with open(pkl_path, 'wb') as f:
        pickle.dump(slim, f)

    # Summary text
    summary_path = outdir / 'phase1_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PHASE 1 CALIBRATION SUMMARY (concurrent σ, descriptive)\n")
        f.write("=" * 70 + "\n\n")
        for name, r in results.items():
            g = r['gk_fit']
            f.write(f"{name}:  ν={g['nu']:.3f}  loc={g['loc']:.4f}  scale={g['scale']:.4f}\n")
            f.write(f"       PIT KS={r['pit']['ks_stat']:.4f}  "
                    f"Shoulder max|ΔF|={max(r['shoulder']['max_dev_left_shoulder'], r['shoulder']['max_dev_right_shoulder']):.4f}  "
                    f"Split-half STABLE={r['stability']['stable']}\n\n")

    print(f"\n\nResults saved to {outdir}")
    print(f"  phase1_diagnostics.png — validation figure")
    print(f"  phase1_results.pkl     — full results")
    print(f"  phase1_summary.txt     — human-readable summary")

    return results


def _plot_phase1_diagnostics(results, outpath):
    """3 rows (ES, NQ, RTY) × 3 cols (PIT, CDF residual, skew-by-|z|)."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Phase 1 Validation — concurrent GK σ, Student-t 3-parameter fit',
                 fontsize=14, fontweight='bold', y=0.99)
    gs = GridSpec(len(results), 3, figure=fig, hspace=0.4, wspace=0.3)

    for row, (name, r) in enumerate(results.items()):
        z = r['z_gk'].values
        z = z[np.isfinite(z) & (np.abs(z) < 50)]
        g = r['gk_fit']
        nu, loc, scale = g['nu'], g['loc'], g['scale']

        # PIT
        ax = fig.add_subplot(gs[row, 0])
        pit = stats.t.cdf(z, df=nu, loc=loc, scale=scale)
        ax.hist(pit, bins=50, alpha=0.7, color='green', density=True, edgecolor='black', lw=0.3)
        ax.axhline(1.0, color='black', ls='--', lw=1, label='Uniform target')
        ax.set_title(f'{name} — PIT (GK)\nKS = {r["pit"]["ks_stat"]:.4f}', fontsize=11)
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
        ax.plot(z_eval, (emp - t_cdf) * 100, color='green', lw=1.5, label='Emp − Student-t')
        ax.plot(z_eval, (emp - n_cdf) * 100, color='red', lw=1.5, label='Emp − Gaussian')
        ax.axhline(0, color='black', ls='--', lw=0.5)
        ax.axhspan(-1, 1, alpha=0.1, color='green')
        ax.axvspan(1, 3, alpha=0.08, color='blue')
        ax.axvspan(-3, -1, alpha=0.08, color='orange')
        ax.set_title(f'{name} — CDF Residual (%)', fontsize=11)
        ax.set_xlabel('z'); ax.set_ylabel('Residual (%)')
        ax.set_xlim(-4, 4); ax.set_ylim(-6, 6)
        ax.legend(fontsize=7)

        # Skewness by |z|
        ax = fig.add_subplot(gs[row, 2])
        sm = pd.DataFrame(r['skew_mag'])
        sm = sm.dropna(subset=['skewness'])
        x = range(len(sm))
        colors = ['green' if abs(s) < 0.15 else ('orange' if abs(s) < 0.3 else 'red')
                  for s in sm['skewness']]
        ax.bar(x, sm['skewness'], alpha=0.7, color=colors, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(sm['range'], rotation=30, fontsize=8)
        ax.axhline(0, color='black', lw=0.5)
        ax.axhline(0.15, color='orange', ls='--', lw=1, alpha=0.7, label='|0.15| threshold')
        ax.axhline(-0.15, color='orange', ls='--', lw=1, alpha=0.7)
        ax.set_title(f'{name} — Skewness by |z| bucket', fontsize=11)
        ax.set_ylabel('Skewness within bucket')
        ax.legend(fontsize=8)

    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True, help='Path to ES parquet')
    parser.add_argument('--nq', required=True, help='Path to NQ parquet')
    parser.add_argument('--rty', required=True, help='Path to RTY parquet')
    parser.add_argument('--outdir', default='./outputs', help='Output directory')
    parser.add_argument('--bar-minutes', type=int, default=5)
    parser.add_argument('--span', type=int, default=20, help='EWMA span for sigma smoothing')
    args = parser.parse_args()

    paths = {'ES': args.es, 'NQ': args.nq, 'RTY': args.rty}
    run_phase1(paths, args.outdir, bar_minutes=args.bar_minutes, span=args.span)
