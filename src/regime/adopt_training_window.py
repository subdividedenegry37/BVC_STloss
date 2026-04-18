"""Add the unified training-window gate to the expanded feature pickles.

Adopts the 2020-01-01 unified cutoff identified by the monthly-
stabilization analysis (see
results/regime_break/monthly_stabilization/ES_MONTHLY_STABILIZATION.md).

Strictly additive: loads each expanded per-contract pickle, computes

    is_train_valid_bar = is_valid_bar AND (ts_event >= 2020-01-01 UTC)

and writes it back. No feature values are altered; no existing column is
modified; `is_valid_bar` stays intact.

Usage:
    python adopt_training_window.py --all
    python adopt_training_window.py --contract ES --dry-run
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

EXPANDED_DIR = Path("runs/2026-04-18_physics_feature_expansion")
OUT_DIR = Path("runs/2026-04-18_es_monthly_stabilization")
TRAIN_CUTOFF = pd.Timestamp('2020-01-01', tz='UTC')

# Expected ranges from the adoption spec. Used only for reporting.
EXPECTED = {
    'ES':  (380_000 * 0.80, 380_000 * 1.20),
    'NQ':  (450_000 * 0.80, 500_000 * 1.20),
    'RTY': (330_000 * 0.80, 380_000 * 1.20),
}
EXPECTED_MIDPOINT = {'ES': 380_000, 'NQ': 475_000, 'RTY': 355_000}


def process_contract(contract: str, dry_run: bool = False) -> dict:
    path = EXPANDED_DIR / f"phase2_features_expanded_{contract}.pkl"
    with open(path, 'rb') as f:
        df = pickle.load(f)

    if 'is_valid_bar' not in df.columns:
        raise ValueError(f"[{contract}] missing required column is_valid_bar")
    if df.index.tz is None:
        raise ValueError(f"[{contract}] index must be tz-aware")

    import numpy as np
    valid = df['is_valid_bar'].astype(bool).values
    in_window = np.asarray(df.index >= TRAIN_CUTOFF)
    is_train_valid_bar = valid & in_window

    n_total = int(len(df))
    n_valid = int(valid.sum())
    n_in_window = int(in_window.sum())
    n_train = int(is_train_valid_bar.sum())

    # Baseline expectations:
    lo, hi = EXPECTED[contract]
    mid = EXPECTED_MIDPOINT[contract]
    dev_pct = 100.0 * (n_train - mid) / mid
    within_20pct = abs(dev_pct) <= 20.0

    info = {
        'contract': contract,
        'path': str(path),
        'n_total': n_total,
        'n_valid': n_valid,
        'n_in_window_post_cutoff': n_in_window,
        'n_train_valid_bar': n_train,
        'train_cutoff': str(TRAIN_CUTOFF),
        'expected_range': [int(lo), int(hi)],
        'expected_midpoint': mid,
        'deviation_pct_vs_midpoint': round(dev_pct, 2),
        'within_20pct_tolerance': bool(within_20pct),
    }

    if dry_run:
        info['written'] = False
        return info

    # Strictly additive write — overwrite the pickle in place.
    df = df.copy()
    df['is_train_valid_bar'] = is_train_valid_bar
    tmp = path.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(df, f)
    tmp.replace(path)
    info['written'] = True
    info['n_columns_after'] = int(len(df.columns))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', choices=['ES', 'NQ', 'RTY'])
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.all:
        contracts = ['ES', 'NQ', 'RTY']
    elif args.contract is not None:
        contracts = [args.contract]
    else:
        ap.error("specify --all or --contract {ES,NQ,RTY}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    any_out_of_tolerance = False
    for c in contracts:
        info = process_contract(c, dry_run=args.dry_run)
        results.append(info)
        tol_marker = 'OK' if info['within_20pct_tolerance'] else 'OUT-OF-TOLERANCE'
        if not info['within_20pct_tolerance']:
            any_out_of_tolerance = True
        print(
            f"[{c}] n_total={info['n_total']:,}  n_valid={info['n_valid']:,}  "
            f"n_train_valid_bar={info['n_train_valid_bar']:,}  "
            f"dev={info['deviation_pct_vs_midpoint']:+.1f}%  "
            f"[{tol_marker}]  written={info.get('written', False)}",
            flush=True,
        )

    report_path = OUT_DIR / 'training_window_counts.json'
    payload = {
        'train_cutoff': str(TRAIN_CUTOFF),
        'results': results,
        'any_out_of_tolerance': any_out_of_tolerance,
    }
    with open(report_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\n[write] {report_path}", flush=True)


if __name__ == '__main__':
    main()
