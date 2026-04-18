import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from student_t_bvc import aggregate_to_bars, compute_sigma_causal, fit_student_t, pit_uniformity
import psutil

def run_rty_2025_2026(rty_path, bar_minutes=5, span=20):
    print(f"\n{'='*70}")
    print(f"RTY DIAGNOSTIC: Mar 2025 - Mar 2026")
    print(f"{'='*70}")

    raw = pd.read_parquet(rty_path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
    bars = aggregate_to_bars(raw, bar_minutes)

    del raw
    import gc; gc.collect()

    bars['log_ret'] = np.log(bars['close'] / bars['open'])
    bars['sigma'] = compute_sigma_causal(bars, span=span)
    bars['z'] = bars['log_ret'] / bars['sigma']

    # We need to sort by instrument_id and time to correctly identify sequential gaps
    bars = bars.reset_index().rename(columns={'index': 'ts_event', 'ts_event': 'ts_event'})
    if 'ts_event' not in bars.columns and 'level_0' in bars.columns:
        bars = bars.rename(columns={'level_0': 'ts_event'})

    bars = bars.sort_values(['instrument_id', 'ts_event']).copy()

    # Calculate boundaries
    bars['ts_diff'] = bars.groupby('instrument_id')['ts_event'].diff()
    bars['is_boundary'] = bars['ts_diff'].isna() | (bars['ts_diff'] > pd.Timedelta(minutes=60))
    drop_mask = bars.groupby('instrument_id')['is_boundary'].transform(lambda x: x.rolling(200, min_periods=1).max() > 0)

    # Unfiltered (Base)
    paired_base = bars[['z', 'sigma', 'instrument_id']].dropna()
    z_base = paired_base['z'].values
    z_clean_base = z_base[np.isfinite(z_base) & (np.abs(z_base) < 50)]
    nu_base, loc_base, scale_base = fit_student_t(z_clean_base)
    print(f"  Unfiltered ({len(z_clean_base):,} bars): ν={nu_base:.3f}, loc={loc_base:.4f}, scale={scale_base:.4f}")

    # Filtered
    filtered_window = bars[~drop_mask[bars.index]].copy()
    paired_filtered = filtered_window[['z', 'sigma', 'instrument_id']].dropna()
    z_filtered = paired_filtered['z'].values
    z_clean_filtered = z_filtered[np.isfinite(z_filtered) & (np.abs(z_filtered) < 50)]

    nu_f, loc_f, scale_f = fit_student_t(z_clean_filtered)
    print(f"  Filtered ({len(z_clean_filtered):,} bars): ν={nu_f:.3f}, loc={loc_f:.4f}, scale={scale_f:.4f}")

    if nu_f < 5.0:
        print(f"\n  *** VERDICT: Filtered ν is {nu_f:.3f} (≈ 4.5). The 12-month calibration is capturing a genuinely heavier-tailed current regime. ***")
    else:
        print(f"\n  *** VERDICT: Filtered ν is {nu_f:.3f} (≈ 5.7). The 15-year filtered estimate is correct and the 12-month calibration has a methodological issue. ***")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rty', required=True)
    parser.add_argument('--bar-minutes', type=int, default=5)
    parser.add_argument('--span', type=int, default=20)
    args = parser.parse_args()

    run_rty_2025_2026(args.rty, bar_minutes=args.bar_minutes, span=args.span)
