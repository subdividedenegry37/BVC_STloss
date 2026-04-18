"""Check 3: Reproduce the RTY ν=5.85 result using the OLD diagnostic logic.

Old diagnostic approach:
  1) Compute σ with original compute_sigma_causal (EWMA rolls across sessions)
  2) Post-filter: drop first 200 bars after each boundary (contract change OR >60min gap)

Then compare to new grid at w=200, gap=60min which uses session-isolated EWMA.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from student_t_bvc import (
    aggregate_to_bars, compute_sigma_causal, compute_sigma_causal_session_isolated,
    fit_student_t, pit_uniformity
)

def run_old_diagnostic(path, label, bar_minutes=5, span=20, drop_bars=200, gap_minutes=60):
    """Old diagnostic: cross-session EWMA + post-filter."""
    print(f"\n  [OLD DIAGNOSTIC] {label}: drop_bars={drop_bars}, gap={gap_minutes}min")
    raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
    bars = aggregate_to_bars(raw, bar_minutes)
    del raw

    bars['log_ret'] = np.log(bars['close'] / bars['open'])
    bars['sigma'] = compute_sigma_causal(bars, span=span)
    bars['z'] = bars['log_ret'] / bars['sigma']

    bars = bars.reset_index()
    if 'ts_event' not in bars.columns and 'level_0' in bars.columns:
        bars = bars.rename(columns={'level_0': 'ts_event'})

    bars = bars.sort_values(['instrument_id', 'ts_event']).copy()
    bars['ts_diff'] = bars.groupby('instrument_id')['ts_event'].diff()
    bars['is_boundary'] = bars['ts_diff'].isna() | (bars['ts_diff'] > pd.Timedelta(minutes=gap_minutes))
    drop_mask = bars.groupby('instrument_id')['is_boundary'].transform(
        lambda x: x.rolling(drop_bars, min_periods=1).max() > 0
    )

    filtered = bars[~drop_mask].copy()
    z_vals = filtered['z'].dropna().values
    z_clean = z_vals[np.isfinite(z_vals) & (np.abs(z_vals) < 50)]
    nu, loc, scale = fit_student_t(z_clean)
    pit_res = pit_uniformity(z_clean, nu, loc, scale)
    print(f"    N={len(z_clean):,}, ν={nu:.3f}, loc={loc:.4f}, scale={scale:.4f}, KS={pit_res['ks_stat']:.4f}")
    return {'method': 'OLD_postfilter', 'nu': nu, 'loc': loc, 'scale': scale, 'n': len(z_clean), 'ks': pit_res['ks_stat']}

def run_new_grid_cell(path, label, warmup, gap, bar_minutes=5, span=20):
    """New session-isolated σ at (warmup, gap)."""
    print(f"\n  [NEW GRID] {label}: warmup={warmup}, gap={gap}")
    raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
    bars = aggregate_to_bars(raw, bar_minutes)
    del raw

    bars['log_ret'] = np.log(bars['close'] / bars['open'])
    bars['_row_id'] = np.arange(len(bars))
    bars['_ts'] = bars.index
    bars_sorted = bars.sort_values(['instrument_id', '_ts']).copy()

    sigma, _ = compute_sigma_causal_session_isolated(bars_sorted, span=span, warmup_bars=warmup, gap_threshold=gap)
    bars_sorted['_sigma'] = sigma
    bars_sorted2 = bars_sorted.sort_values('_row_id')
    sigma_orig = bars_sorted2['_sigma'].values

    z = bars['log_ret'].values / sigma_orig
    z_clean = z[np.isfinite(z) & (np.abs(z) < 50)]
    nu, loc, scale = fit_student_t(z_clean)
    pit_res = pit_uniformity(z_clean, nu, loc, scale)
    print(f"    N={len(z_clean):,}, ν={nu:.3f}, loc={loc:.4f}, scale={scale:.4f}, KS={pit_res['ks_stat']:.4f}")
    return {'method': f'NEW_w{warmup}_g{gap}', 'nu': nu, 'loc': loc, 'scale': scale, 'n': len(z_clean), 'ks': pit_res['ks_stat']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rty-1yr', required=True)
    parser.add_argument('--rty-15yr', required=True)
    parser.add_argument('--outdir', default='./outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CHECK 3: RTY ν RECONCILIATION")
    print("="*70)

    results = []

    # Old diagnostic reproduction
    r = run_old_diagnostic(args.rty_1yr, "RTY 1yr", drop_bars=200, gap_minutes=60)
    r['dataset'] = '1yr'; results.append(r)

    r = run_old_diagnostic(args.rty_15yr, "RTY 15yr", drop_bars=200, gap_minutes=60)
    r['dataset'] = '15yr'; results.append(r)

    # New session-isolated equivalent
    r = run_new_grid_cell(args.rty_1yr, "RTY 1yr", warmup=200, gap='60min')
    r['dataset'] = '1yr'; results.append(r)

    r = run_new_grid_cell(args.rty_15yr, "RTY 15yr", warmup=200, gap='60min')
    r['dataset'] = '15yr'; results.append(r)

    # Chosen grid parameters for comparison
    r = run_new_grid_cell(args.rty_1yr, "RTY 1yr", warmup=80, gap='15min')
    r['dataset'] = '1yr'; results.append(r)

    r = run_new_grid_cell(args.rty_15yr, "RTY 15yr", warmup=80, gap='15min')
    r['dataset'] = '15yr'; results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(outdir / 'check3_rty_reconciliation.csv', index=False)
    print("\n\nFINAL RECONCILIATION TABLE:")
    print(df.to_string(index=False))
