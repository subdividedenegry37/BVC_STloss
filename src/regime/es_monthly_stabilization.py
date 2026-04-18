"""ES monthly-resolution stabilization diagnostic.

Given the cached ES z-series produced by the locked session-isolated causal GK σ
pipeline (warmup=80, gap=15min, span=20, 5-min bars), sweep 16 monthly cutoffs
from 2019-09-01 through 2020-12-01 to locate the exact month at which post-COVID
ES enters its current stationary regime.

Outputs (runs/2026-04-18_es_monthly_stabilization/):
  - candidate_cutoffs.csv          global fit statistics per cutoff
  - per_month_winning_cutoff.csv   per-calendar-month PIT KS at winning global fit
  - rolling6m_nu.csv               rolling 6-month ν stepped monthly
  - rolling6m_nu.png               plot of rolling ν across 2019-2021
  - stabilization_summary.json     winning cutoff and pass/fail details
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from student_t_bvc import fit_student_t, pit_uniformity

LOCKED_NU = 4.602
NU_TOL = 0.15
GLOBAL_KS_MAX = 0.05
PER_MONTH_KS_MAX = 0.08
POST_END = pd.Timestamp('2026-05-01', tz='UTC')  # exclusive; covers through 2026-04


def month_range(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[pd.Timestamp]:
    end_inclusive = end_exclusive - pd.Timedelta(days=1)
    return list(pd.date_range(start=start, end=end_inclusive, freq='MS', tz='UTC'))


def fit_row(z: np.ndarray) -> dict:
    if len(z) < 1000:
        return {'n': len(z), 'nu': np.nan, 'loc': np.nan, 'scale': np.nan,
                'ks': np.nan, 'skew': np.nan}
    nu, loc, scale = fit_student_t(z)
    ks = pit_uniformity(z, nu, loc, scale)['ks_stat']
    return {'n': int(len(z)), 'nu': float(nu), 'loc': float(loc),
            'scale': float(scale), 'ks': float(ks), 'skew': float(stats.skew(z))}


def per_month_ks(z_df: pd.DataFrame, nu: float, loc: float, scale: float,
                 start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    df = z_df.copy()
    df = df[(df['ts'] >= start) & (df['ts'] < end_exclusive)]
    df['month'] = df['ts'].dt.to_period('M')
    rows = []
    for m, grp in df.groupby('month'):
        z = grp['z'].values
        if len(z) < 500:
            rows.append({'month': str(m), 'n': int(len(z)), 'ks': np.nan})
            continue
        pit = stats.t.cdf(z, df=nu, loc=loc, scale=scale)
        ks = float(stats.kstest(pit, 'uniform').statistic)
        rows.append({'month': str(m), 'n': int(len(z)), 'ks': ks})
    return pd.DataFrame(rows).sort_values('month').reset_index(drop=True)


def rolling_nu(z_df: pd.DataFrame, anchor_start: pd.Timestamp,
               anchor_end_exclusive: pd.Timestamp, window_months: int = 6) -> pd.DataFrame:
    anchors = month_range(anchor_start, anchor_end_exclusive)
    rows = []
    for a in anchors:
        lo = (a - pd.DateOffset(months=window_months - 1)).tz_convert('UTC')
        hi = (a + pd.DateOffset(months=1)).tz_convert('UTC')
        mask = (z_df['ts'] >= lo) & (z_df['ts'] < hi)
        z = z_df.loc[mask, 'z'].values
        r = fit_row(z)
        rows.append({'anchor_month': a.strftime('%Y-%m'),
                     'window_start': lo.strftime('%Y-%m-%d'),
                     'window_end_exclusive': hi.strftime('%Y-%m-%d'),
                     **r})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z-cache', default='runs/2026-04-17_regime_validation/es_z_cleaned.parquet')
    ap.add_argument('--outdir', default='runs/2026-04-18_es_monthly_stabilization')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.z_cache}", flush=True)
    z_df = pd.read_parquet(args.z_cache)
    z_df['ts'] = pd.to_datetime(z_df['ts'], utc=True)
    z_df = z_df[(z_df['ts'] < POST_END)].reset_index(drop=True)
    print(f"       N={len(z_df):,}  range={z_df['ts'].min()} -> {z_df['ts'].max()}", flush=True)

    # --- 16 monthly cutoffs ---
    cutoffs = month_range(pd.Timestamp('2019-09-01', tz='UTC'),
                          pd.Timestamp('2021-01-01', tz='UTC'))  # exclusive -> 2020-12-01 last
    assert len(cutoffs) == 16, f"expected 16, got {len(cutoffs)}"
    print(f"[sweep] {len(cutoffs)} cutoffs", flush=True)

    cand_rows = []
    for c in cutoffs:
        z = z_df.loc[(z_df['ts'] >= c) & (z_df['ts'] < POST_END), 'z'].values
        r = fit_row(z)
        r['cutoff'] = c.strftime('%Y-%m-%d')
        r['nu_delta_vs_lock'] = abs(r['nu'] - LOCKED_NU) if np.isfinite(r['nu']) else np.nan
        cand_rows.append(r)
        print(f"  {r['cutoff']}  n={r['n']:>7,}  ν={r['nu']:.4f}  loc={r['loc']:+.4f}  "
              f"scale={r['scale']:.4f}  KS={r['ks']:.4f}  skew={r['skew']:+.3f}", flush=True)

    cand_df = pd.DataFrame(cand_rows)[
        ['cutoff', 'n', 'nu', 'loc', 'scale', 'ks', 'skew', 'nu_delta_vs_lock']
    ]
    cand_df.to_csv(outdir / 'candidate_cutoffs.csv', index=False)

    # --- Identify stabilization cutoff ---
    winner = None
    winner_per_month = None
    for _, row in cand_df.iterrows():
        if not np.isfinite(row['nu']):
            continue
        if row['nu_delta_vs_lock'] > NU_TOL:
            continue
        if row['ks'] > GLOBAL_KS_MAX:
            continue
        c = pd.Timestamp(row['cutoff'], tz='UTC')
        pm = per_month_ks(z_df, row['nu'], row['loc'], row['scale'], c, POST_END)
        finite_ks = pm['ks'].dropna()
        if (finite_ks <= PER_MONTH_KS_MAX).all():
            winner = row
            winner_per_month = pm
            break

    summary = {
        'locked_nu': LOCKED_NU,
        'nu_tolerance': NU_TOL,
        'global_ks_max': GLOBAL_KS_MAX,
        'per_month_ks_max': PER_MONTH_KS_MAX,
        'post_end_exclusive': str(POST_END),
    }
    if winner is not None:
        summary['winning_cutoff'] = winner['cutoff']
        summary['winning_fit'] = {k: float(winner[k]) for k in
                                   ['n', 'nu', 'loc', 'scale', 'ks', 'skew', 'nu_delta_vs_lock']}
        winner_per_month.to_csv(outdir / 'per_month_winning_cutoff.csv', index=False)
        print(f"\n[winner] {winner['cutoff']}  ν={winner['nu']:.4f}  KS={winner['ks']:.4f}",
              flush=True)
    else:
        summary['winning_cutoff'] = None
        print("\n[winner] NONE — no cutoff satisfies all three criteria", flush=True)

    # --- Rolling 6-month ν across 2019-2021 ---
    roll_df = rolling_nu(
        z_df,
        anchor_start=pd.Timestamp('2019-01-01', tz='UTC'),
        anchor_end_exclusive=pd.Timestamp('2022-01-01', tz='UTC'),
        window_months=6,
    )
    roll_df.to_csv(outdir / 'rolling6m_nu.csv', index=False)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = pd.to_datetime(roll_df['anchor_month'] + '-15')
    ax.step(x, roll_df['nu'], where='mid', lw=1.6, color='#2c3e50', label='6-mo rolling ν')
    ax.axhline(LOCKED_NU, color='#c0392b', ls='--', lw=1.0,
               label=f'baseline post-ν = {LOCKED_NU:.3f}')
    ax.axhspan(LOCKED_NU - NU_TOL, LOCKED_NU + NU_TOL, color='#c0392b', alpha=0.10,
               label=f'±{NU_TOL} tolerance')
    if winner is not None:
        wc = pd.Timestamp(winner['cutoff'])
        ax.axvline(wc, color='#27ae60', ls=':', lw=1.3,
                   label=f'stabilization cutoff {winner["cutoff"]}')
    ax.set_xlabel('Anchor month (window end)')
    ax.set_ylabel('Student-t ν (6-mo rolling, stepped monthly)')
    ax.set_title('ES rolling 6-month ν — 2019-2021')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / 'rolling6m_nu.png', dpi=150)
    plt.close(fig)

    with open(outdir / 'stabilization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[write] {outdir}", flush=True)


if __name__ == '__main__':
    main()
