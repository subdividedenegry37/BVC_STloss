"""
phase2_features.py — Sub-bar feature engineering

Computes 5-min bars with seven sub-bar features:
  1. z                  — standardized log return (BVC input)
  2. imbalance_t        — Student-t BVC signed imbalance (monotonic transform of z)
  3. der                — Directional Efficiency Ratio
  4. sign_concordance   — fraction of sub-bars agreeing with parent direction  [ORTHOGONAL TO BVC]
  5. clv_mean           — mean Close Location Value across sub-bars
  6. clv_var            — variance of CLV across sub-bars
  7. subbar_imbalance   — volume-weighted sub-bar BVC aggregation

Plus vol_skew and real_kurt which are computed but should be DROPPED
(too noisy at n=5 sub-bars — documented in the state doc).

KEY FINDING from 1-year data:
  sign_concordance has r ≈ 0.00 with BVC imbalance — carries ORTHOGONAL
  information. No published BVC study has this feature. It captures flow
  PERSISTENCE while BVC captures flow MAGNITUDE.

USAGE:
    python phase2_features.py --es ... --nq ... --rty ... --outdir ./outputs --params concurrent

    --params: 'concurrent' (Phase 1) or 'causal' (Phase 1', for Stage 1)
"""

import argparse
import pickle
from pathlib import Path
import psutil

import numpy as np
import pandas as pd

from student_t_bvc import (
    compute_subbar_features,
    PHASE1_PARAMS_5MIN,
    PHASE1_CAUSAL_PARAMS_5MIN,
)


def run_phase2(parquet_paths, outdir, params='concurrent', warmup=None, gap=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if params == 'concurrent':
        params_dict = PHASE1_PARAMS_5MIN
        label = 'Phase 1 (concurrent σ, descriptive)'
    elif params == 'causal':
        params_dict = PHASE1_CAUSAL_PARAMS_5MIN
        label = "baseline (causal σ, predictive — for Stage 1)"
    else:
        raise ValueError(f"params must be 'concurrent' or 'causal', got {params!r}")

    # Inject warmup and gap into params_dict
    if warmup is not None and gap is not None:
        # copy to avoid mutating the global dict for subsequent calls
        params_dict = params_dict.copy()
        for k in params_dict:
            if isinstance(params_dict[k], dict):
                params_dict[k] = params_dict[k].copy()
                params_dict[k]['warmup'] = warmup
                params_dict[k]['gap'] = gap

    print(f"Using parameters: {label}")

    results = {}
    for name, path in parquet_paths.items():
        print(f"\n{'='*70}")
        print(f"PHASE 2 FEATURES: {name}")
        print(f"{'='*70}")

        mem_before = psutil.Process().memory_info().rss / 1e9
        print(f"Memory before loading: {mem_before:.2f} GB")

        raw = pd.read_parquet(path, columns=['ts_event', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'instrument_id'])
        print(f"  Loaded {len(raw):,} 1-min bars")

        mem_after_load = psutil.Process().memory_info().rss / 1e9
        print(f"Memory after loading: {mem_after_load:.2f} GB")

        print(f"  Computing features...")

        bars = compute_subbar_features(raw, name, params_dict)

        del raw
        import gc; gc.collect()
        mem_after_feat = psutil.Process().memory_info().rss / 1e9
        print(f"Memory after features and GC: {mem_after_feat:.2f} GB")

        # Save individually
        pkl_path = outdir / f'phase2_features_{name}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(bars, f)
        print(f"  Saved features to {pkl_path}")

        # Keep in results for correlation matrix printing later
        results[name] = bars

        print(f"  Produced {len(bars):,} 5-min bars with features")

        features = ['z', 'imbalance_t', 'der', 'sign_concordance',
                    'clv_mean', 'clv_var', 'vol_skew', 'real_kurt',
                    'subbar_imbalance']
        print(f"\n  Feature summary:")
        print(f"    {'feature':<20} {'mean':>10} {'std':>10} {'P05':>10} {'P95':>10}  NaN%")
        for f in features:
            s = bars[f].dropna()
            nan_pct = bars[f].isna().mean() * 100
            print(f"    {f:<20} {s.mean():>10.4f} {s.std():>10.4f} "
                  f"{s.quantile(0.05):>10.4f} {s.quantile(0.95):>10.4f} "
                  f"{nan_pct:>5.1f}%")

        # Correlation matrix
        core = ['z', 'imbalance_t', 'der', 'sign_concordance',
                'clv_mean', 'clv_var', 'subbar_imbalance']
        valid = bars[core].dropna()
        corr = valid.corr()

        print(f"\n  Key correlations:")
        print(f"    imbalance_t ↔ subbar_imbalance:  r = {corr.loc['imbalance_t','subbar_imbalance']:+.3f}")
        print(f"    imbalance_t ↔ clv_mean:          r = {corr.loc['imbalance_t','clv_mean']:+.3f}")
        print(f"    imbalance_t ↔ sign_concordance:  r = {corr.loc['imbalance_t','sign_concordance']:+.3f}  "
              f"{'← should be ~0 (orthogonal)' if abs(corr.loc['imbalance_t','sign_concordance']) < 0.1 else '← UNEXPECTED'}")
        print(f"    der ↔ clv_var:                   r = {corr.loc['der','clv_var']:+.3f}")

    # Save correlation matrix to text
    with open(outdir / 'phase2_correlations.txt', 'w') as f:
        f.write(f"PHASE 2 FEATURE CORRELATIONS — {label}\n")
        f.write("=" * 70 + "\n\n")
        for name, bars in results.items():
            f.write(f"\n{name}:\n")
            core = ['z', 'imbalance_t', 'der', 'sign_concordance',
                    'clv_mean', 'clv_var', 'subbar_imbalance']
            valid = bars[core].dropna()
            f.write(valid.corr().to_string())
            f.write("\n")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--nq', required=True)
    parser.add_argument('--rty', required=True)
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--params', default='concurrent', choices=['concurrent', 'causal'])
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--gap', type=str, default=None)
    args = parser.parse_args()

    paths = {'ES': args.es, 'NQ': args.nq, 'RTY': args.rty}
    run_phase2(paths, args.outdir, params=args.params, warmup=args.warmup, gap=args.gap)
