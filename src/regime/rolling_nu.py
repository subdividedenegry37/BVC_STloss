"""Rolling 3-year ν analysis (quarterly step) on cleaned session-isolated σ.

Fits Student-t on 3-year rolling windows of cleaned (w=80, g=15min) z values for
each contract. Also computes a pre/post-2020-03-01 full-regime fit for ES to
overlay as reference horizontal lines.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import gc
from student_t_bvc import (
    aggregate_to_bars, compute_sigma_causal_session_isolated,
    fit_student_t, pit_uniformity,
)

WINDOW_YEARS = 3
STEP_MONTHS = 3
ES_REGIME_CUT = pd.Timestamp('2020-03-01', tz='UTC')


def load_z_series(path, warmup=80, gap='15min', span=20, bar_minutes=5):
    """Return DataFrame indexed by ts_event with columns [z, year]. NaN/|z|>50 dropped."""
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

    z = bars[['z']].copy()
    z['ts'] = bars.index
    z = z.dropna()
    z = z[np.isfinite(z['z']) & (np.abs(z['z']) < 50)]
    del bars, bars_sorted, bars_sorted2; gc.collect()
    return z


def rolling_nu(z_df, window_years=WINDOW_YEARS, step_months=STEP_MONTHS):
    """Fit Student-t on rolling windows. Return DataFrame with center, nu, loc, scale, n, ks."""
    ts_all = pd.DatetimeIndex(z_df['ts'])
    start = ts_all.min().normalize()
    end = ts_all.max().normalize()

    window = pd.DateOffset(years=window_years)
    step = pd.DateOffset(months=step_months)

    rows = []
    w_start = start
    while w_start + window <= end + pd.DateOffset(days=1):
        w_end = w_start + window
        mask = (ts_all >= w_start) & (ts_all < w_end)
        mask_arr = np.asarray(mask)
        z_win = z_df['z'].values[mask_arr]
        if len(z_win) < 10_000:
            w_start = w_start + step
            continue
        nu, loc, scale = fit_student_t(z_win)
        pit = pit_uniformity(z_win, nu, loc, scale)
        center = w_start + (w_end - w_start) / 2
        rows.append({
            'window_start': w_start, 'window_end': w_end, 'center': center,
            'nu': nu, 'loc': loc, 'scale': scale, 'n': len(z_win),
            'ks': pit['ks_stat']
        })
        w_start = w_start + step
    return pd.DataFrame(rows)


def fit_regime(z_df, start_ts=None, end_ts=None):
    ts_all = pd.DatetimeIndex(z_df['ts'])
    mask = np.ones(len(z_df), dtype=bool)
    if start_ts is not None:
        mask &= np.asarray(ts_all >= start_ts)
    if end_ts is not None:
        mask &= np.asarray(ts_all < end_ts)
    z_win = z_df['z'].values[mask]
    if len(z_win) < 1000:
        return None
    nu, loc, scale = fit_student_t(z_win)
    pit = pit_uniformity(z_win, nu, loc, scale)
    return {'nu': nu, 'loc': loc, 'scale': scale, 'n': len(z_win), 'ks': pit['ks_stat']}


def plot_rolling(results, regimes, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = {'ES': 'tab:blue', 'NQ': 'tab:green', 'RTY': 'tab:red'}
    for ax, name in zip(axes, ['ES', 'NQ', 'RTY']):
        df_r = results[name]
        if df_r.empty:
            continue
        ax.plot(df_r['center'], df_r['nu'], marker='o', color=colors[name],
                label=f'{name} rolling ν (3-yr window, quarterly step)')
        if name == 'ES' and regimes.get('ES_pre') and regimes.get('ES_post'):
            ax.axhline(regimes['ES_pre']['nu'], color='k', linestyle='--', alpha=0.6,
                       label=f"ES-pre (<{ES_REGIME_CUT.date()}) ν={regimes['ES_pre']['nu']:.3f}")
            ax.axhline(regimes['ES_post']['nu'], color='k', linestyle=':', alpha=0.6,
                       label=f"ES-post (>={ES_REGIME_CUT.date()}) ν={regimes['ES_post']['nu']:.3f}")
            ax.axvline(ES_REGIME_CUT, color='red', linestyle='-', alpha=0.4)
        ax.set_ylabel('ν')
        ax.set_title(f'{name} — rolling ν')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    axes[-1].set_xlabel('Window center date')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--nq', required=True)
    parser.add_argument('--rty', required=True)
    parser.add_argument('--outdir', default='./outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    regimes = {}
    for name, path in [('ES', args.es), ('NQ', args.nq), ('RTY', args.rty)]:
        print(f"\n{'='*70}\nROLLING ν: {name}\n{'='*70}", flush=True)
        z_df = load_z_series(path)
        print(f"  z rows: {len(z_df):,}  [{z_df['ts'].min()} .. {z_df['ts'].max()}]")
        df_roll = rolling_nu(z_df)
        results[name] = df_roll
        df_roll.to_csv(outdir / f'rolling_nu_{name}.csv', index=False)
        print(df_roll[['center', 'nu', 'scale', 'n', 'ks']].to_string(index=False))
        if name == 'ES':
            regimes['ES_pre'] = fit_regime(z_df, end_ts=ES_REGIME_CUT)
            regimes['ES_post'] = fit_regime(z_df, start_ts=ES_REGIME_CUT)
            print(f"\n  ES-pre  (< {ES_REGIME_CUT.date()}): {regimes['ES_pre']}")
            print(f"  ES-post (>={ES_REGIME_CUT.date()}): {regimes['ES_post']}")
        del z_df; gc.collect()

    with open(outdir / 'rolling_nu_results.pkl', 'wb') as f:
        pickle.dump({'rolling': results, 'es_regimes': regimes}, f)
    plot_rolling(results, regimes, outdir / 'rolling_nu.png')
    print(f"\nSaved: {outdir}/rolling_nu.png")
