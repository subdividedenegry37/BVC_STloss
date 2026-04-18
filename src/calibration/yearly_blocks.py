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

from student_t_bvc import (
    aggregate_to_bars,
    compute_sigma_causal,
    compute_sigma_causal_session_isolated,
    compute_z,
    fit_student_t,
    pit_uniformity,
)

def run_yearly_blocks(parquet_paths, outdir, bar_minutes=5, span=20, warmup=None, gap=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, path in parquet_paths.items():
        print(f"\n{'='*70}")
        print(f"YEARLY BLOCKS (CAUSAL σ): {name}")
        print(f"{'='*70}")

        mem_before = psutil.Process().memory_info().rss / 1e9

        raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
        bars = aggregate_to_bars(raw, bar_minutes)

        del raw
        import gc; gc.collect()

        bars['log_ret'] = np.log(bars['close'] / bars['open'])

        if warmup is not None and gap is not None:
            # Sort for session isolation without destroying index
            bars['_row_id'] = np.arange(len(bars))
            bars['_ts'] = bars.index
            bars_sorted = bars.sort_values(['instrument_id', '_ts']).copy()
            sigma_series, _ = compute_sigma_causal_session_isolated(bars_sorted, span=span, warmup_bars=warmup, gap_threshold=gap)
            bars_sorted['sigma_causal'] = sigma_series
            bars_sorted = bars_sorted.sort_values('_row_id')
            sigma_causal = bars_sorted['sigma_causal'].values
            bars = bars.drop(columns=['_row_id', '_ts'])
        else:
            sigma_causal = compute_sigma_causal(bars, span=span)

        bars['sigma'] = sigma_causal
        bars['z'] = bars['log_ret'] / bars['sigma']
        
        paired = bars[['z', 'sigma']].dropna()
        
        # Full sample fit
        z_full = paired['z'].values
        z_clean_full = z_full[np.isfinite(z_full) & (np.abs(z_full) < 50)]
        nu_full, loc_full, scale_full = fit_student_t(z_clean_full)
        
        print(f"  Full sample fit ({len(z_clean_full)} bars): ν={nu_full:.3f}, loc={loc_full:.4f}, scale={scale_full:.4f}")
        
        # Split into yearly blocks
        paired['year'] = paired.index.year
        
        yearly_stats = []
        
        for year, group in paired.groupby('year'):
            z_y = group['z'].values
            z_clean = z_y[np.isfinite(z_y) & (np.abs(z_y) < 50)]
            
            if len(z_clean) < 10000:
                print(f"  {year}: SKIP (only {len(z_clean)} bars, < 10000)")
                continue
                
            nu, loc, scale = fit_student_t(z_clean)
            skewness = stats.skew(z_clean)
            pit_res = pit_uniformity(z_clean, nu, loc, scale)
            
            flag = ""
            if abs(nu - nu_full) > 1.0:
                flag = f"*** REGIME CHANGE (Δν = {nu - nu_full:+.2f}) ***"
                
            print(f"  {year} ({len(z_clean):>6} bars): ν={nu:>6.2f} loc={loc:>7.4f} scale={scale:>6.4f} skew={skewness:>6.3f} KS={pit_res['ks_stat']:.4f} {flag}")
            
            yearly_stats.append({
                'year': year,
                'n_bars': len(z_clean),
                'nu': nu,
                'loc': loc,
                'scale': scale,
                'skewness': skewness,
                'ks_stat': pit_res['ks_stat'],
                'd_nu': nu - nu_full
            })
            
        all_results[name] = {
            'full': {'nu': nu_full, 'loc': loc_full, 'scale': scale_full},
            'yearly': pd.DataFrame(yearly_stats)
        }
        
    # Plotting
    _plot_yearly_drift(all_results, outdir / 'yearly_blocks_drift.png')
    
    # Save results
    with open(outdir / 'yearly_blocks_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
        
    with open(outdir / 'yearly_blocks_summary.txt', 'w') as f:
        f.write("PHASE 1 YEARLY BLOCKS SUMMARY (causal σ)\n")
        f.write("=" * 70 + "\n\n")
        for name, r in all_results.items():
            f.write(f"\n{name} (Full Sample: ν={r['full']['nu']:.3f})\n")
            f.write("-" * 50 + "\n")
            df_y = r['yearly']
            if df_y.empty: continue
            f.write(f"{'Year':<6} {'N bars':>8} {'ν':>8} {'loc':>8} {'scale':>8} {'skew':>8} {'KS':>8} {'Flag':>15}\n")
            for _, row in df_y.iterrows():
                flag = "***" if abs(row['d_nu']) > 1.0 else ""
                f.write(f"{int(row['year']):<6} {int(row['n_bars']):>8} {row['nu']:>8.3f} {row['loc']:>8.4f} {row['scale']:>8.4f} {row['skewness']:>8.3f} {row['ks_stat']:>8.4f} {flag:>15}\n")
                
    print(f"\nResults saved to {outdir}")

def _plot_yearly_drift(all_results, outpath):
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Yearly Drift in Student-t Parameters (Causal σ)', fontsize=16, fontweight='bold', y=0.95)
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)
    
    ax_nu = fig.add_subplot(gs[0, 0])
    ax_loc = fig.add_subplot(gs[1, 0])
    ax_scale = fig.add_subplot(gs[2, 0])
    
    colors = {'ES': 'blue', 'NQ': 'green', 'RTY': 'red'}
    
    for name, r in all_results.items():
        df = r['yearly']
        if df.empty: continue
        
        color = colors.get(name, 'black')
        
        ax_nu.plot(df['year'], df['nu'], marker='o', color=color, label=f"{name} (Full ν={r['full']['nu']:.2f})")
        ax_nu.axhline(r['full']['nu'], color=color, linestyle='--', alpha=0.5)
        
        ax_loc.plot(df['year'], df['loc'], marker='o', color=color, label=name)
        ax_loc.axhline(r['full']['loc'], color=color, linestyle='--', alpha=0.5)
        
        ax_scale.plot(df['year'], df['scale'], marker='o', color=color, label=name)
        ax_scale.axhline(r['full']['scale'], color=color, linestyle='--', alpha=0.5)
        
    ax_nu.set_ylabel('Degrees of Freedom (ν)')
    ax_nu.set_title('Tail Index (ν) Drift')
    ax_nu.legend()
    ax_nu.grid(True, alpha=0.3)
    
    ax_loc.set_ylabel('Location (loc)')
    ax_loc.set_title('Location Drift')
    ax_loc.grid(True, alpha=0.3)
    
    ax_scale.set_ylabel('Scale (scale)')
    ax_scale.set_title('Scale Drift')
    ax_scale.grid(True, alpha=0.3)
    
    # Set x-ticks to years
    all_years = []
    for r in all_results.values():
        all_years.extend(r['yearly']['year'].tolist())
    if all_years:
        min_yr, max_yr = int(min(all_years)), int(max(all_years))
        for ax in [ax_nu, ax_loc, ax_scale]:
            ax.set_xticks(range(min_yr, max_yr + 1))
            
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
    run_yearly_blocks(paths, args.outdir, bar_minutes=args.bar_minutes, span=args.span, warmup=args.warmup, gap=args.gap)
