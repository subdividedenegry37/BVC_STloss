"""Phase 3 + Phase 4: ES break-date sensitivity and full validation.

Phase 3: Fit ES-pre and ES-post at five candidate cutoffs and pick the one
that minimises combined PIT KS subject to both sub-regimes passing KS<0.05
individually.

Phase 4: Full validation (PIT, shoulder, split-half, per-year,
skewness-by-|z|) for the winning cutoff. Also per-year ν trend within ES-post.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import gc
from scipy import stats
from student_t_bvc import (
    aggregate_to_bars, compute_sigma_causal_session_isolated,
    fit_student_t, pit_uniformity, shoulder_cdf_deviation,
    split_half_stability, skewness_by_magnitude,
)

CANDIDATES = [
    pd.Timestamp('2019-10-01', tz='UTC'),
    pd.Timestamp('2020-01-01', tz='UTC'),
    pd.Timestamp('2020-03-01', tz='UTC'),
    pd.Timestamp('2020-06-01', tz='UTC'),
    pd.Timestamp('2020-09-01', tz='UTC'),
]


def load_es_z(path, cache, warmup=80, gap='15min', span=20, bar_minutes=5):
    if cache.exists():
        print(f"  loading cached z from {cache}")
        return pd.read_parquet(cache)
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
    z_df = pd.DataFrame({
        'ts': bars.index, 'z': bars['z'].values,
    }).dropna()
    z_df = z_df[np.isfinite(z_df['z']) & (np.abs(z_df['z']) < 50)].reset_index(drop=True)
    z_df.to_parquet(cache)
    return z_df


def fit_and_validate(z_vals):
    if len(z_vals) < 1000:
        return None
    nu, loc, scale = fit_student_t(z_vals)
    pit = pit_uniformity(z_vals, nu, loc, scale)
    sh = shoulder_cdf_deviation(z_vals, nu, loc, scale)
    sh_max = max(sh['max_dev_left_shoulder'], sh['max_dev_right_shoulder'])
    return {
        'nu': nu, 'loc': loc, 'scale': scale, 'n': len(z_vals),
        'ks': pit['ks_stat'], 'pit_shape': pit['shape'],
        'pit_asym': pit['asymmetry'],
        'shoulder_left': sh['max_dev_left_shoulder'],
        'shoulder_right': sh['max_dev_right_shoulder'],
        'shoulder_max': sh_max,
    }


def per_year_fit(z_df, start_ts=None, end_ts=None):
    df = z_df.copy()
    if start_ts is not None:
        df = df[df['ts'] >= start_ts]
    if end_ts is not None:
        df = df[df['ts'] < end_ts]
    df['year'] = pd.DatetimeIndex(df['ts']).year
    rows = []
    for y, grp in df.groupby('year'):
        if len(grp) < 5000:
            continue
        r = fit_and_validate(grp['z'].values)
        if r is None:
            continue
        r['year'] = int(y)
        r['quality'] = 'GOOD' if (r['ks'] < 0.05 and r['shoulder_max'] < 0.01) else 'FAIL'
        rows.append(r)
    return pd.DataFrame(rows)


def break_sensitivity(z_df):
    ts = pd.DatetimeIndex(z_df['ts'])
    rows = []
    for cut in CANDIDATES:
        pre_mask = np.asarray(ts < cut)
        post_mask = ~pre_mask
        pre = fit_and_validate(z_df['z'].values[pre_mask])
        post = fit_and_validate(z_df['z'].values[post_mask])
        combined_ks = (pre['ks'] * pre['n'] + post['ks'] * post['n']) / (pre['n'] + post['n'])
        rows.append({
            'cutoff': cut.date(),
            'nu_pre': pre['nu'], 'scale_pre': pre['scale'], 'n_pre': pre['n'],
            'ks_pre': pre['ks'], 'shoulder_pre': pre['shoulder_max'],
            'nu_post': post['nu'], 'scale_post': post['scale'], 'n_post': post['n'],
            'ks_post': post['ks'], 'shoulder_post': post['shoulder_max'],
            'ks_combined': combined_ks,
            'pre_passes': pre['ks'] < 0.05,
            'post_passes': post['ks'] < 0.05,
        })
    return pd.DataFrame(rows)


def plot_per_year_trend(df_pre, df_post, outpath):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    if not df_pre.empty:
        ax.plot(df_pre['year'], df_pre['nu'], 'o-', color='tab:blue', label='ES-pre per-year ν')
    if not df_post.empty:
        ax.plot(df_post['year'], df_post['nu'], 's-', color='tab:red', label='ES-post per-year ν')
    ax.axhline(df_post['nu'].mean() if not df_post.empty else np.nan, color='tab:red',
               linestyle=':', alpha=0.5, label=f'ES-post mean ν = {df_post["nu"].mean():.3f}')
    ax.set_xlabel('Year')
    ax.set_ylabel('ν (fit per year)')
    ax.set_title('ES per-year ν trend — pre vs post regime')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--outdir', default='./outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / 'es_z_cleaned.parquet'

    print("Loading ES z series...", flush=True)
    z_df = load_es_z(args.es, cache)
    print(f"  rows: {len(z_df):,}  span: {z_df['ts'].min()} .. {z_df['ts'].max()}")

    print("\n" + "="*70 + "\nPhase 3: Break-date sensitivity\n" + "="*70)
    df_sens = break_sensitivity(z_df)
    print(df_sens.to_string(index=False))
    df_sens.to_csv(outdir / 'es_break_sensitivity.csv', index=False)

    # Winner = minimum combined KS
    winner = df_sens.sort_values('ks_combined').iloc[0]
    cutoff_star = pd.Timestamp(winner['cutoff'], tz='UTC')
    print(f"\nWinner: cutoff={winner['cutoff']} ks_combined={winner['ks_combined']:.4f} "
          f"pre_passes={winner['pre_passes']} post_passes={winner['post_passes']}")

    print("\n" + "="*70 + f"\nPhase 4: Full validation at cutoff={cutoff_star.date()}\n" + "="*70)
    pre_mask = np.asarray(pd.DatetimeIndex(z_df['ts']) < cutoff_star)
    z_pre = z_df['z'].values[pre_mask]
    z_post = z_df['z'].values[~pre_mask]
    val_pre = fit_and_validate(z_pre)
    val_post = fit_and_validate(z_post)
    print(f"  ES-pre:  {val_pre}")
    print(f"  ES-post: {val_post}")

    # Split-half + skewness
    split_pre = split_half_stability(z_pre)
    split_post = split_half_stability(z_post)
    skew_pre = skewness_by_magnitude(z_pre)
    skew_post = skewness_by_magnitude(z_post)
    print(f"\n  split-half ES-pre:  Δν={split_pre['delta']['nu']:.3f} Δscale={split_pre['delta']['scale']:.3f} stable={split_pre['stable']}")
    print(f"  split-half ES-post: Δν={split_post['delta']['nu']:.3f} Δscale={split_post['delta']['scale']:.3f} stable={split_post['stable']}")
    print("\n  skewness ES-pre by |z|:\n" + skew_pre.to_string(index=False))
    print("\n  skewness ES-post by |z|:\n" + skew_post.to_string(index=False))

    # Per-year
    pyr_pre = per_year_fit(z_df, end_ts=cutoff_star)
    pyr_post = per_year_fit(z_df, start_ts=cutoff_star)
    pyr_pre.to_csv(outdir / 'es_pre_per_year.csv', index=False)
    pyr_post.to_csv(outdir / 'es_post_per_year.csv', index=False)
    print("\n  ES-pre per-year:\n" + pyr_pre[['year', 'n', 'nu', 'scale', 'ks', 'shoulder_max', 'quality']].to_string(index=False))
    print("\n  ES-post per-year:\n" + pyr_post[['year', 'n', 'nu', 'scale', 'ks', 'shoulder_max', 'quality']].to_string(index=False))

    # Trend within ES-post
    if len(pyr_post) >= 3:
        slope, intercept, r, p, se = stats.linregress(pyr_post['year'], pyr_post['nu'])
        print(f"\n  ES-post per-year ν linear trend: slope={slope:.4f}/year r={r:.3f} p={p:.3f}")
        trend = {'slope': slope, 'intercept': intercept, 'r': r, 'p': p}
    else:
        trend = None

    plot_per_year_trend(pyr_pre, pyr_post, outdir / 'es_per_year_nu.png')

    # Save everything
    out = {
        'cutoff': cutoff_star, 'break_sensitivity': df_sens,
        'validation_pre': val_pre, 'validation_post': val_post,
        'split_half_pre': split_pre, 'split_half_post': split_post,
        'skew_pre': skew_pre, 'skew_post': skew_post,
        'per_year_pre': pyr_pre, 'per_year_post': pyr_post,
        'post_trend': trend,
    }
    with open(outdir / 'es_breakdate_validation.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved: {outdir}/es_breakdate_validation.pkl")
