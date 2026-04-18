import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from student_t_bvc import aggregate_to_bars, compute_sigma_causal_session_isolated, fit_student_t, pit_uniformity

def t_cdf(z, nu, loc, scale):
    return stats.t.cdf((z - loc) / scale, df=nu)

def run_grid(parquet_paths, outdir, bar_minutes=5, span=20):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    warmup_options = [40, 80, 200]
    gap_options = ['5min', '15min', '60min']
    grid = list(product(warmup_options, gap_options))
    
    all_results = {}
    
    for name, path in parquet_paths.items():
        print(f"\n{'='*70}")
        print(f"PHASE 2: SENSITIVITY GRID for {name}")
        print(f"{'='*70}")
        
        raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
        bars = aggregate_to_bars(raw, bar_minutes)
        del raw
        import gc; gc.collect()
        
        bars['log_ret'] = np.log(bars['close'] / bars['open'])
        
        # Sort properly for session isolation
        bars = bars.reset_index().rename(columns={'index': 'ts_event', 'ts_event': 'ts_event'})
        if 'ts_event' not in bars.columns and 'level_0' in bars.columns:
            bars = bars.rename(columns={'level_0': 'ts_event'})
        bars = bars.sort_values(['instrument_id', 'ts_event']).copy()
        
        contract_results = []
        
        for w, g in grid:
            print(f"  [warmup={w}, gap={g}]", end='', flush=True)
            
            sigma, _ = compute_sigma_causal_session_isolated(bars, span=span, warmup_bars=w, gap_threshold=g)
            z = bars['log_ret'] / sigma
            
            z_valid = z.dropna().values
            z_clean = z_valid[np.isfinite(z_valid) & (np.abs(z_valid) < 50)]
            n_bars = len(z_clean)
            
            nu, loc, scale = fit_student_t(z_clean)
            
            pit_res = pit_uniformity(z_clean, nu, loc, scale)
            ks_stat = pit_res['ks_stat']
            
            # Empirical CDF
            z_sorted = np.sort(z_clean)
            emp_cdf = np.arange(1, len(z_sorted) + 1) / len(z_sorted)
            theo_cdf = t_cdf(z_sorted, nu, loc, scale)
            abs_diff = np.abs(emp_cdf - theo_cdf)
            
            # Shoulders
            idx_pos = (z_sorted >= 1) & (z_sorted <= 3)
            max_res_pos = np.max(abs_diff[idx_pos]) if np.any(idx_pos) else 0.0
            
            idx_neg = (z_sorted >= -3) & (z_sorted <= -1)
            max_res_neg = np.max(abs_diff[idx_neg]) if np.any(idx_neg) else 0.0
            
            max_res_shoulder = max(max_res_pos, max_res_neg)
            
            cdf_z2 = t_cdf(2.0, nu, loc, scale)
            
            contract_results.append({
                'warmup': w, 'gap': g,
                'nu': nu, 'loc': loc, 'scale': scale,
                'n_bars': n_bars, 'ks': ks_stat,
                'shoulder_res': max_res_shoulder,
                'cdf_z2': cdf_z2
            })
            
            print(f" -> ν={nu:.3f}, N={n_bars:,}")
            
        all_results[name] = pd.DataFrame(contract_results)
        
        # Plot Heatmap
        df_grid = all_results[name].pivot(index='gap', columns='warmup', values='nu')
        # match gap sorting: 5min, 15min, 60min
        df_grid = df_grid.loc[['5min', '15min', '60min']]
        
        plt.figure(figsize=(6, 5))
        median_nu = all_results[name]['nu'].median()
        sns.heatmap(df_grid, annot=True, fmt=".3f", cmap="coolwarm", center=median_nu)
        plt.title(f"{name} Grid: Degrees of Freedom (ν)")
        plt.tight_layout()
        plt.savefig(outdir / f'heatmap_{name}.png', dpi=150)
        plt.close()
        
    # Plateau Detection
    print("\n" + "="*70)
    print("PHASE 3: PLATEAU DETECTION")
    print("="*70)
    
    plateau_passed = {}
    for name, df_res in all_results.items():
        max_cdf_diff = 0.0
        cdf_vals = df_res['cdf_z2'].values
        for i in range(len(cdf_vals)):
            for j in range(i+1, len(cdf_vals)):
                diff = abs(cdf_vals[i] - cdf_vals[j])
                if diff > max_cdf_diff:
                    max_cdf_diff = diff
                    
        passed = max_cdf_diff < 0.005
        plateau_passed[name] = passed
        print(f"  {name}: max ΔF_t(z=2) = {max_cdf_diff:.5f} -> {'PASS' if passed else 'FAIL'}")
        
    if all(plateau_passed.values()):
        print("\n  *** CLEAN PLATEAU *** (All contracts passed criterion)")
        print("  Recommending: warmup=40, gap_threshold='60min' (maximizes data)")
    else:
        print("\n  *** CONDITIONAL PLATEAU *** (One or more contracts failed criterion)")
        print("  Check heatmaps to identify the driver of sensitivity.")
        
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
