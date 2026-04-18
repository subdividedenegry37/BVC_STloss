"""Check 4: Sunday-evening warm-up convergence of σ.

For each long-gap session (>48h gap = Sunday-evening open), track σ bar-by-bar
for 200 bars. Measure relative deviation from terminal value at bar 200.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from student_t_bvc import aggregate_to_bars, compute_gk_variance

def convergence_for_contract(path, name, outdir, bar_minutes=5, span=20,
                              track_bars=200, min_session_bars=500, ref_bar=500):
    """Measure relative deviation of session-isolated σ_n vs σ at far-ahead reference bar.

    For each long-gap session with >= min_session_bars bars:
      σ(n) = EWMA of lagged gk_var within this session, at bar n
      ref  = σ(ref_bar)  # well-converged reference
      rel  = |σ(n) - ref| / ref

    Plot median and P90 of rel over n ∈ [1, track_bars].
    """
    print(f"\n{'='*70}\nCONVERGENCE: {name}\n{'='*70}", flush=True)
    raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
    bars = aggregate_to_bars(raw, bar_minutes)
    del raw
    import gc; gc.collect()

    bars = bars.reset_index()
    if 'ts_event' not in bars.columns:
        ts_col = bars.columns[0]
        bars = bars.rename(columns={ts_col: 'ts_event'})

    bars = bars.sort_values(['instrument_id', 'ts_event']).copy()
    bars['gk_var'] = compute_gk_variance(bars)
    bars['ts_diff'] = bars.groupby('instrument_id')['ts_event'].diff()

    contract_change = bars['instrument_id'].ne(bars['instrument_id'].shift())
    long_gap = bars['ts_diff'] > pd.Timedelta(hours=48)
    bars['session_start'] = contract_change | long_gap
    bars['session_id'] = bars['session_start'].cumsum()

    long_gap_sessions = bars[bars['session_start'] & long_gap]['session_id'].unique()
    print(f"  Found {len(long_gap_sessions)} long-gap (>48h) sessions")

    alpha = 2.0 / (span + 1)
    curves = []
    for sid in long_gap_sessions:
        grp = bars[bars['session_id'] == sid]
        if len(grp) < min_session_bars:
            continue
        gk_var = grp['gk_var'].values[:min_session_bars]
        lagged = np.concatenate([[np.nan], gk_var[:-1]])

        s = np.full(min_session_bars, np.nan)
        ewma = np.nan
        for i in range(min_session_bars):
            x = lagged[i]
            if np.isnan(x):
                continue
            if np.isnan(ewma):
                ewma = x
            else:
                ewma = alpha * x + (1 - alpha) * ewma
            s[i] = np.sqrt(ewma) if ewma > 0 else np.nan
        curves.append(s)

    if not curves:
        print("  No sessions found.")
        return None

    curves_arr = np.array(curves)
    ref_sigma = curves_arr[:, ref_bar - 1]
    valid = np.isfinite(ref_sigma) & (ref_sigma > 0)
    curves_arr = curves_arr[valid]
    ref_sigma = ref_sigma[valid]

    rel = np.abs(curves_arr[:, :track_bars] - ref_sigma[:, None]) / ref_sigma[:, None]
    median_curve = np.nanmedian(rel, axis=0)
    p90_curve = np.nanpercentile(rel, 90, axis=0)

    median_stable = np.where(median_curve < 0.05)[0]
    median_bar_stable = median_stable[0] + 1 if len(median_stable) > 0 else None
    p90_stable = np.where(p90_curve < 0.05)[0]
    p90_bar_stable = p90_stable[0] + 1 if len(p90_stable) > 0 else None

    median_stable_10 = np.where(median_curve < 0.10)[0]
    median_bar_stable_10 = median_stable_10[0] + 1 if len(median_stable_10) > 0 else None

    print(f"  N valid sessions: {int(valid.sum())} / {len(curves)}")
    print(f"  Median within ±5%: bar {median_bar_stable}, P90 within ±5%: bar {p90_bar_stable}")
    print(f"  Median within ±10%: bar {median_bar_stable_10}")

    return {
        'name': name,
        'n_sessions': int(valid.sum()),
        'median_curve': median_curve,
        'p90_curve': p90_curve,
        'median_bar_stable': median_bar_stable,
        'p90_bar_stable': p90_bar_stable,
        'median_bar_stable_10': median_bar_stable_10
    }

def plot_all(results, outpath):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = {'ES': 'blue', 'NQ': 'green', 'RTY': 'red'}
    for name, r in results.items():
        if r is None: continue
        bars_x = np.arange(1, len(r['median_curve']) + 1)
        ax.plot(bars_x, r['median_curve'], color=colors[name], label=f"{name} median (N={r['n_sessions']})")
        ax.plot(bars_x, r['p90_curve'], color=colors[name], linestyle='--', alpha=0.5, label=f"{name} P90")
    ax.axhline(0.05, color='black', linestyle=':', label='±5% threshold')
    ax.set_xlabel('Bar number after session open')
    ax.set_ylabel('Relative deviation of σ from terminal value')
    ax.set_title('Sunday-evening σ convergence')
    ax.legend()
    ax.grid(alpha=0.3)
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

    paths = {'ES': args.es, 'NQ': args.nq, 'RTY': args.rty}
    results = {}
    for name, path in paths.items():
        results[name] = convergence_for_contract(path, name, outdir)

    with open(outdir / 'check4_convergence_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    plot_all(results, outdir / 'check4_convergence.png')
    print("\nDone.")
