"""Extended ES break-date study.

Phase 3 showed ES-pre fails KS<0.05 for every candidate cutoff in {2019-10..2020-09}
because the 2010-2019 window is internally non-stationary (rolling ν drifts 5.67 -> 4.85).

This script explores:
(A) Restrict ES-pre START date (cut 2010-2013 out): does a narrower stationary window pass?
(B) Two-breakpoint structure: (2010-06-01 .. t0) | excluded transition | (t1 .. present).
(C) Single-regime post-only fit with various start dates.

All use cutoff=2020-06-01 as the post-start (winner of Phase 3 combined KS).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from student_t_bvc import fit_student_t, pit_uniformity, shoulder_cdf_deviation


POST_CUT = pd.Timestamp('2020-06-01', tz='UTC')


def fit_segment(z_df, start_ts, end_ts):
    ts = pd.DatetimeIndex(z_df['ts'])
    mask = np.ones(len(z_df), dtype=bool)
    if start_ts is not None:
        mask &= np.asarray(ts >= start_ts)
    if end_ts is not None:
        mask &= np.asarray(ts < end_ts)
    z = z_df['z'].values[mask]
    if len(z) < 1000:
        return None
    nu, loc, scale = fit_student_t(z)
    pit = pit_uniformity(z, nu, loc, scale)
    sh = shoulder_cdf_deviation(z, nu, loc, scale)
    return {
        'start': start_ts, 'end': end_ts, 'n': len(z),
        'nu': nu, 'loc': loc, 'scale': scale,
        'ks': pit['ks_stat'],
        'shoulder_max': max(sh['max_dev_left_shoulder'], sh['max_dev_right_shoulder']),
        'passes': (pit['ks_stat'] < 0.05),
    }


def scan_pre_starts(z_df, starts, end_ts):
    rows = []
    for s in starts:
        r = fit_segment(z_df, s, end_ts)
        if r is None:
            continue
        rows.append({
            'pre_start': s.date() if s is not None else None,
            'pre_end': end_ts.date() if end_ts is not None else None,
            'n': r['n'], 'nu': r['nu'], 'scale': r['scale'],
            'ks': r['ks'], 'shoulder_max': r['shoulder_max'], 'passes': r['passes']
        })
    return pd.DataFrame(rows)


def dual_breakpoint(z_df, t0_starts, t1_starts_after_post=None):
    """For each (t_pre_end, t_post_start=POST_CUT) pair with excluded transition,
    fit ES-pre on [data_start, t_pre_end) and ES-post on [POST_CUT, end)."""
    rows = []
    for t_pre_end in t0_starts:
        pre = fit_segment(z_df, None, t_pre_end)
        post = fit_segment(z_df, POST_CUT, None)
        if pre is None or post is None:
            continue
        combined_ks = (pre['ks'] * pre['n'] + post['ks'] * post['n']) / (pre['n'] + post['n'])
        excluded_n = len(z_df) - pre['n'] - post['n']
        rows.append({
            'pre_end': t_pre_end.date(), 'post_start': POST_CUT.date(),
            'excluded_n': excluded_n, 'excluded_pct': 100 * excluded_n / len(z_df),
            'nu_pre': pre['nu'], 'scale_pre': pre['scale'], 'n_pre': pre['n'],
            'ks_pre': pre['ks'], 'shoulder_pre': pre['shoulder_max'],
            'nu_post': post['nu'], 'scale_post': post['scale'], 'n_post': post['n'],
            'ks_post': post['ks'], 'shoulder_post': post['shoulder_max'],
            'ks_combined': combined_ks,
            'pre_passes': pre['ks'] < 0.05, 'post_passes': post['ks'] < 0.05,
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, help='Path to cached es_z_cleaned.parquet')
    parser.add_argument('--outdir', default='./outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z_df = pd.read_parquet(args.cache)
    print(f"Loaded z: {len(z_df):,}")

    # (A) Scan ES-pre start dates with end = 2020-06-01
    print("\n" + "="*70 + "\n(A) ES-pre start-date scan (end=2020-06-01)\n" + "="*70)
    starts = [pd.Timestamp(d, tz='UTC') for d in
              ['2010-06-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01',
               '2016-01-01','2017-01-01','2018-01-01']]
    df_a = scan_pre_starts(z_df, starts, POST_CUT)
    print(df_a.to_string(index=False))
    df_a.to_csv(outdir / 'es_pre_start_scan.csv', index=False)

    # (B) Dual breakpoint: vary pre-end between 2017 and 2020
    print("\n" + "="*70 + "\n(B) Dual-breakpoint structure (pre-end, post=2020-06-01)\n" + "="*70)
    pre_ends = [pd.Timestamp(d, tz='UTC') for d in
                ['2017-01-01','2017-06-01','2018-01-01','2018-06-01','2019-01-01',
                 '2019-06-01','2019-10-01','2020-01-01','2020-03-01','2020-06-01']]
    df_b = dual_breakpoint(z_df, pre_ends)
    print(df_b[['pre_end','n_pre','nu_pre','ks_pre','pre_passes','n_post','nu_post','ks_post','post_passes','excluded_pct','ks_combined']].to_string(index=False))
    df_b.to_csv(outdir / 'es_dual_breakpoint.csv', index=False)

    # (C) Scan ES-post start
    print("\n" + "="*70 + "\n(C) Single-regime post-only scan (start, end=full)\n" + "="*70)
    post_starts = [pd.Timestamp(d, tz='UTC') for d in
                   ['2019-10-01','2020-01-01','2020-03-01','2020-06-01','2020-09-01','2021-01-01']]
    df_c = scan_pre_starts(z_df, post_starts, None)
    print(df_c.to_string(index=False))
    df_c.to_csv(outdir / 'es_post_start_scan.csv', index=False)

    out = {'pre_start_scan': df_a, 'dual_breakpoint': df_b, 'post_start_scan': df_c}
    with open(outdir / 'es_extended_breakdate.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved: {outdir}/es_extended_breakdate.pkl")
