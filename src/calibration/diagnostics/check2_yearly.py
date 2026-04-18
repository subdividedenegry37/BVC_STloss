"""Check 2: Per-year PIT quality on cleaned σ (w=80, gap=15min)."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from student_t_bvc import aggregate_to_bars, compute_sigma_causal_session_isolated, fit_student_t, pit_uniformity

def shoulder_residual(z_clean, nu, loc, scale):
    z_sorted = np.sort(z_clean)
    emp_cdf = np.arange(1, len(z_sorted) + 1) / len(z_sorted)
    theo_cdf = stats.t.cdf((z_sorted - loc) / scale, df=nu)
    abs_diff = np.abs(emp_cdf - theo_cdf)
    idx_pos = (z_sorted >= 1) & (z_sorted <= 3)
    idx_neg = (z_sorted >= -3) & (z_sorted <= -1)
    mp = np.max(abs_diff[idx_pos]) if np.any(idx_pos) else 0.0
    mn = np.max(abs_diff[idx_neg]) if np.any(idx_neg) else 0.0
    return max(mp, mn)

def run(parquet_paths, outdir, warmup=80, gap='15min', bar_minutes=5, span=20):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, path in parquet_paths.items():
        print(f"\n{'='*70}\nPER-YEAR PIT: {name} (w={warmup}, gap={gap})\n{'='*70}", flush=True)
        raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
        bars = aggregate_to_bars(raw, bar_minutes)
        del raw
        import gc; gc.collect()

        bars['log_ret'] = np.log(bars['close'] / bars['open'])
        bars['_row_id'] = np.arange(len(bars))
        bars['_ts'] = bars.index
        bars_sorted = bars.sort_values(['instrument_id', '_ts']).copy()
        sigma, _ = compute_sigma_causal_session_isolated(bars_sorted, span=span, warmup_bars=warmup, gap_threshold=gap)
        bars_sorted['_sigma'] = sigma
        bars_sorted2 = bars_sorted.sort_values('_row_id')
        bars['sigma'] = bars_sorted2['_sigma'].values
        bars['z'] = bars['log_ret'] / bars['sigma']

        # Yearly blocks
        year_idx = bars.index.year if hasattr(bars.index, 'year') else pd.to_datetime(bars['_ts']).dt.year
        paired = pd.DataFrame({'z': bars['z'].values, 'year': year_idx.values})
        paired = paired.dropna()
        paired = paired[np.isfinite(paired['z']) & (np.abs(paired['z']) < 50)]

        year_stats = []
        for year, grp in paired.groupby('year'):
            z_clean = grp['z'].values
            if len(z_clean) < 5000:
                continue
            nu, loc, scale = fit_student_t(z_clean)
            pit_res = pit_uniformity(z_clean, nu, loc, scale)
            shoulder = shoulder_residual(z_clean, nu, loc, scale)
            quality = 'GOOD' if (pit_res['ks_stat'] < 0.05 and shoulder < 0.01) else 'FAIL'
            year_stats.append({
                'year': int(year), 'n': len(z_clean),
                'nu': nu, 'loc': loc, 'scale': scale,
                'ks': pit_res['ks_stat'], 'shoulder': shoulder, 'quality': quality
            })
            print(f"  {year} N={len(z_clean):>7,} ν={nu:6.3f} loc={loc:+.4f} scale={scale:.4f} KS={pit_res['ks_stat']:.4f} shoulder={shoulder:.4f} {quality}")

        df_y = pd.DataFrame(year_stats)
        all_results[name] = df_y
        df_y.to_csv(outdir / f'check2_yearly_{name}.csv', index=False)
        del bars, bars_sorted, bars_sorted2, paired
        gc.collect()

    # Save
    with open(outdir / 'check2_yearly_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # KS bar charts
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (name, df_y) in zip(axes, all_results.items()):
        if df_y.empty: continue
        colors = ['green' if q == 'GOOD' else 'red' for q in df_y['quality']]
        ax.bar(df_y['year'], df_y['ks'], color=colors, edgecolor='black')
        ax.axhline(0.05, color='black', linestyle='--', label='KS = 0.05 threshold')
        ax.set_title(f"{name} — Per-Year PIT KS (cleaned σ)")
        ax.set_xlabel('Year')
        ax.set_ylabel('KS statistic')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'check2_ks_by_year.png', dpi=150)
    plt.close()

    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--nq', required=True)
    parser.add_argument('--rty', required=True)
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--warmup', type=int, default=80)
    parser.add_argument('--gap', default='15min')
    args = parser.parse_args()

    paths = {'ES': args.es, 'NQ': args.nq, 'RTY': args.rty}
    run(paths, args.outdir, warmup=args.warmup, gap=args.gap)
