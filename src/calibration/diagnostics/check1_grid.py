"""Check 1: Full 3x3 ν matrix for each contract, with bars, loc, scale, KS."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from itertools import product
from student_t_bvc import aggregate_to_bars, compute_sigma_causal_session_isolated, fit_student_t, pit_uniformity

def run_grid(parquet_paths, outdir, bar_minutes=5, span=20):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    warmup_options = [40, 80, 200]
    gap_options = ['5min', '15min', '60min']
    grid = list(product(warmup_options, gap_options))

    all_results = {}

    for name, path in parquet_paths.items():
        print(f"\n{'='*70}\nGRID: {name}\n{'='*70}", flush=True)

        raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
        bars = aggregate_to_bars(raw, bar_minutes)
        del raw
        import gc; gc.collect()

        bars['log_ret'] = np.log(bars['close'] / bars['open'])
        bars['_row_id'] = np.arange(len(bars))
        bars['_ts'] = bars.index
        bars_sorted = bars.sort_values(['instrument_id', '_ts']).copy()

        contract_results = []
        for w, g in grid:
            print(f"  [w={w}, gap={g}]", end='', flush=True)
            sigma, _ = compute_sigma_causal_session_isolated(bars_sorted, span=span, warmup_bars=w, gap_threshold=g)
            bars_sorted['_sigma'] = sigma
            # Resort to original order
            bars_sorted2 = bars_sorted.sort_values('_row_id')
            sigma_orig = bars_sorted2['_sigma'].values
            z = bars['log_ret'].values / sigma_orig

            z_clean = z[np.isfinite(z) & (np.abs(z) < 50)]
            n_bars = len(z_clean)
            nu, loc, scale = fit_student_t(z_clean)
            pit_res = pit_uniformity(z_clean, nu, loc, scale)

            contract_results.append({
                'warmup': w, 'gap': g,
                'nu': nu, 'loc': loc, 'scale': scale,
                'n_bars': n_bars, 'ks': pit_res['ks_stat']
            })
            print(f" ν={nu:.3f} N={n_bars:,} KS={pit_res['ks_stat']:.4f}")

        del bars, bars_sorted, bars_sorted2
        gc.collect()

        df_res = pd.DataFrame(contract_results)
        all_results[name] = df_res

    # Save pickle
    with open(outdir / 'check1_grid_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Print nicely formatted tables
    print("\n\n" + "="*70)
    print("FINAL GRID TABLES")
    print("="*70)
    for name, df_res in all_results.items():
        print(f"\n{name} — ν by (gap, warmup):")
        piv_nu = df_res.pivot(index='gap', columns='warmup', values='nu').loc[['5min', '15min', '60min']]
        print(piv_nu.round(3).to_string())
        print(f"\n{name} — N bars by (gap, warmup):")
        piv_n = df_res.pivot(index='gap', columns='warmup', values='n_bars').loc[['5min', '15min', '60min']]
        print(piv_n.to_string())
        print(f"\n{name} — KS by (gap, warmup):")
        piv_ks = df_res.pivot(index='gap', columns='warmup', values='ks').loc[['5min', '15min', '60min']]
        print(piv_ks.round(4).to_string())

    # Heatmap figure (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, df_res) in zip(axes, all_results.items()):
        piv = df_res.pivot(index='gap', columns='warmup', values='nu').loc[['5min', '15min', '60min']]
        im = ax.imshow(piv.values, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel('warmup_bars')
        ax.set_ylabel('gap_threshold')
        ax.set_title(f"{name} ν")
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                ax.text(j, i, f"{piv.values[i,j]:.2f}", ha='center', va='center', fontsize=10, color='black')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outdir / 'check1_nu_heatmap.png', dpi=150)
    plt.close()

    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--nq', required=True)
    parser.add_argument('--rty', required=True)
    parser.add_argument('--outdir', default='./outputs')
    args = parser.parse_args()

    paths = {'ES': args.es, 'NQ': args.nq, 'RTY': args.rty}
    run_grid(paths, args.outdir)
