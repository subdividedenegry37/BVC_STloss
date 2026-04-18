"""Phase 6: Rebuild Phase 2 features with cleaned session-isolated σ and
the baseline' parameters from 2026-04-17 .

Config:
  - compute_sigma_causal_session_isolated(warmup_bars=80, gap_threshold='15min', span=20)
  - Student-t params per contract (see LOCKED dict)
  - ES regime gating: bars strictly before 2020-03-01 have is_valid_bar=False and
    imbalance_t, v_buy_t, subbar_imbalance = NaN (no BVC computed pre-break).

Outputs per contract:
  phase2_features_cleaned_{ES,NQ,RTY}.pkl
Plus:
  phase6_correlations.txt   — full correlation matrix per contract
  phase6_retention.csv      — data retention statistics
  phase6_orthogonality.csv  — sign_concordance vs imbalance_t correlations
"""
import argparse
import pickle
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from student_t_bvc import compute_subbar_features

LOCKED = {
    'ES':  {'nu': 4.6023, 'loc': 0.0152, 'scale': 0.8463},
    'NQ':  {'nu': 4.4098, 'loc': 0.0156, 'scale': 0.8510},
    'RTY': {'nu': 4.4809, 'loc': 0.0108, 'scale': 0.8854},
}
ES_REGIME_START = pd.Timestamp('2020-03-01', tz='UTC')
WARMUP = 80
GAP = '15min'

CORE_FEATURES = ['z', 'imbalance_t', 'der', 'sign_concordance',
                 'clv_mean', 'clv_var', 'subbar_imbalance']


def build_for_contract(parquet_path, name, outdir):
    print(f"\n{'='*70}\nPhase 6: {name}\n{'='*70}", flush=True)
    raw = pd.read_parquet(parquet_path, columns=['ts_event', 'open', 'high', 'low',
                                                  'close', 'volume', 'symbol',
                                                  'instrument_id'])
    print(f"  loaded {len(raw):,} 1-min bars")

    # Build params dict with top-level warmup/gap (as compute_subbar_features checks
    # params_dict directly, not the per-contract sub-dict).
    params_dict = {k: dict(v) for k, v in LOCKED.items()}
    params_dict['warmup'] = WARMUP
    params_dict['gap'] = GAP

    bars = compute_subbar_features(raw, name, params_dict)
    del raw; gc.collect()
    print(f"  produced {len(bars):,} 5-min bars")

    # is_valid_bar = calibration validity (trustworthy σ AND z)
    bars['is_valid_bar'] = bars['warmup_valid'].astype(bool) & bars['sigma_gk'].notna()

    # ES regime gating: pre-break bars are not in calibration support
    pre_break_n = 0
    if name == 'ES':
        ts_idx = bars.index
        pre_mask = ts_idx < ES_REGIME_START
        pre_break_n = int(pre_mask.sum())
        bars.loc[pre_mask, 'is_valid_bar'] = False
        for col in ['imbalance_t', 'v_buy_t', 'subbar_imbalance']:
            if col in bars.columns:
                bars.loc[pre_mask, col] = np.nan
        print(f"  ES regime gating: nulled {pre_break_n:,} bars before {ES_REGIME_START.date()}")

    # Save per-contract
    out_pkl = Path(outdir) / f'phase2_features_cleaned_{name}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(bars, f)
    print(f"  saved {out_pkl}")

    total_n = len(bars)
    valid_n = int(bars['is_valid_bar'].sum())
    print(f"  total_bars={total_n:,}  valid_bars={valid_n:,}  "
          f"retention={100*valid_n/total_n:.1f}%  pre_break_excluded={pre_break_n:,}")

    return {
        'name': name,
        'bars': bars,
        'total': total_n,
        'valid': valid_n,
        'pre_break_excluded': pre_break_n,
    }


def orthogonality_check(bars, name):
    """Correlation of sign_concordance vs imbalance_t on valid bars only."""
    sub = bars.loc[bars['is_valid_bar'], ['sign_concordance', 'imbalance_t']].dropna()
    r = sub['sign_concordance'].corr(sub['imbalance_t'])
    status = 'preserved' if abs(r) < 0.05 else ('marginal' if abs(r) < 0.10 else 'BROKEN')
    return {'contract': name, 'n': len(sub), 'r': r, 'abs_r': abs(r), 'status': status}


def corr_matrix(bars):
    valid = bars.loc[bars['is_valid_bar'], CORE_FEATURES].dropna()
    return valid.corr(), len(valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--es', required=True)
    parser.add_argument('--nq', required=True)
    parser.add_argument('--rty', required=True)
    parser.add_argument('--outdir', default='./outputs')
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in [('ES', args.es), ('NQ', args.nq), ('RTY', args.rty)]:
        results[name] = build_for_contract(path, name, outdir)

    # Orthogonality check
    print("\n" + "="*70 + "\nSign-concordance orthogonality\n" + "="*70)
    ortho_rows = []
    for name, r in results.items():
        o = orthogonality_check(r['bars'], name)
        ortho_rows.append(o)
        print(f"  {name}:  r(sign_concordance, imbalance_t) = {o['r']:+.5f}  "
              f"(N={o['n']:,})  → {o['status']}")
    pd.DataFrame(ortho_rows).to_csv(outdir / 'phase6_orthogonality.csv', index=False)

    # Correlation matrices
    print("\n" + "="*70 + "\nCorrelation matrices (valid bars)\n" + "="*70)
    with open(outdir / 'phase6_correlations.txt', 'w') as f:
        f.write("PHASE 6 FEATURE CORRELATIONS — baseline' params (2026-04-17)\n")
        f.write("warmup=80, gap=15min, ES gated pre-2020-03-01\n")
        f.write("=" * 70 + "\n\n")
        for name, r in results.items():
            cm, nvalid = corr_matrix(r['bars'])
            hdr = f"\n{name}  (N_valid={nvalid:,})"
            print(hdr)
            print(cm.round(4).to_string())
            f.write(hdr + "\n")
            f.write(cm.to_string())
            f.write("\n")

    # Retention
    print("\n" + "="*70 + "\nData retention\n" + "="*70)
    retention_rows = []
    for name, r in results.items():
        frac = r['valid'] / r['total']
        retention_rows.append({
            'contract': name, 'total_bars': r['total'],
            'valid_bars': r['valid'],
            'pre_break_excluded': r['pre_break_excluded'],
            'retention_fraction': frac,
        })
        print(f"  {name}:  total={r['total']:,}  valid={r['valid']:,}  "
              f"pre-2020 excluded={r['pre_break_excluded']:,}  retention={100*frac:.1f}%")
    pd.DataFrame(retention_rows).to_csv(outdir / 'phase6_retention.csv', index=False)

    print(f"\nSaved outputs in {outdir}")
