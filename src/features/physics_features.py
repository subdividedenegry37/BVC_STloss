"""physics_features.py — Physics feature expansion

Adds seven candidate features derived from OHLCV and the baseline BVC
outputs:

    sigma_yz         Yang-Zhang rolling 20-bar volatility
    body_to_range    |close - open| / (high - low)
    gel_fraction     |close - midrange| / (high - low)
    wick_asymmetry   ln(H/max(O,C)) - ln(min(O,C)/L)
    amihud           |log_ret| / volume
    ko_W             sigma_gk * close * volume
    polar_order_P    |rolling_sum_50(sign(log_ret))| / 50

Excluded after validation:
    v_star_C         |imbalance_t| / EMA(session_mean_|imbalance_t|, span=10)
                     — dropped due to structural redundancy with the
                     DER / body_to_range cluster (r=0.767 with DER,
                     r=0.855 with body_to_range on all three contracts).
                     compute_v_star_C() is retained below for provenance.

Usage:
    python physics_features.py --all
    python physics_features.py --contract ES
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

PHASE2_INPUT_DIR = Path("runs/2026-04-17_regime_validation")
OUTPUT_DIR = Path("runs/2026-04-18_physics_feature_expansion")

NEW_FEATURES = ['sigma_yz', 'body_to_range', 'gel_fraction',
                'wick_asymmetry', 'amihud', 'ko_W', 'polar_order_P']


# ---------------------------------------------------------------------------
# Feature computations
# ---------------------------------------------------------------------------

def compute_v_star_C(df, span=10):
    """Feature 8 (EXCLUDED from pipeline; retained for provenance).

    |imbalance_t| / EMA(session-mean |imbalance_t|, span=10).
    Dropped 2026-04-18 after initial validation review — see module docstring.
    """
    abs_imb = df['imbalance_t'].abs()
    dates = df.index.normalize()
    sess_df = pd.DataFrame({
        'iid': df['instrument_id'].values,
        'date': dates,
        'abs_imb': abs_imb.values,
    })
    per_session = (sess_df.groupby(['iid', 'date'], sort=False)['abs_imb']
                          .mean().reset_index())
    per_session = per_session.sort_values(['iid', 'date']).reset_index(drop=True)
    per_session['ema'] = (per_session.groupby('iid', sort=False)['abs_imb']
                          .transform(lambda s: s.ewm(span=span, adjust=False).mean()))
    lookup = per_session.set_index(['iid', 'date'])['ema']
    key = pd.MultiIndex.from_arrays([df['instrument_id'].values, dates])
    ema_baseline = pd.Series(lookup.reindex(key).values, index=df.index)
    ema_baseline = ema_baseline.replace(0, np.nan)
    return abs_imb / ema_baseline


def compute_yang_zhang(df, window=20):
    """Feature 9: Yang-Zhang rolling volatility over n=20 bars per instrument."""
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    prev_close = df.groupby('instrument_id', sort=False)['close'].shift(1)
    overnight = np.log(df['open'] / prev_close)
    o2c = np.log(df['close'] / df['open'])
    rs = (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open'])
          + np.log(df['low'] / df['close']) * np.log(df['low'] / df['open']))

    iid = df['instrument_id']
    var_on = overnight.groupby(iid, sort=False).transform(
        lambda s: s.rolling(n, min_periods=n).var(ddof=1))
    var_o2c = o2c.groupby(iid, sort=False).transform(
        lambda s: s.rolling(n, min_periods=n).var(ddof=1))
    mean_rs = rs.groupby(iid, sort=False).transform(
        lambda s: s.rolling(n, min_periods=n).mean())

    var_yz = var_on + k * var_o2c + (1 - k) * mean_rs
    return np.sqrt(var_yz.clip(lower=0))


def compute_body_to_range(df):
    """Feature 10: |close - open| / (high - low); 0 for zero-range bars."""
    rng = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    out = np.where(rng > 0, body / rng.where(rng > 0), 0.0)
    out = pd.Series(out, index=df.index)
    missing = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
    out[missing] = np.nan
    return out


def compute_gel_fraction(df):
    """Feature 11: |close - midrange| / (high - low); 0 for zero-range bars."""
    rng = df['high'] - df['low']
    mid = 0.5 * (df['high'] + df['low'])
    dev = (df['close'] - mid).abs()
    out = np.where(rng > 0, dev / rng.where(rng > 0), 0.0)
    out = pd.Series(out, index=df.index)
    missing = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
    out[missing] = np.nan
    return out


def compute_wick_asymmetry(df):
    """Feature 12: ln(H/max(O,C)) - ln(min(O,C)/L)."""
    body_high = np.maximum(df['open'], df['close'])
    body_low = np.minimum(df['open'], df['close'])
    with np.errstate(divide='ignore', invalid='ignore'):
        upper = np.log(df['high'] / body_high)
        lower = np.log(body_low / df['low'])
    out = upper - lower
    return out.replace([np.inf, -np.inf], np.nan)


def compute_amihud(df):
    """Feature 13: |log_ret| / volume per instrument_id."""
    prev_close = df.groupby('instrument_id', sort=False)['close'].shift(1)
    log_ret = np.log(df['close'] / prev_close)
    abs_ret = log_ret.abs()
    vol = df['volume'].astype(float)
    safe_vol = vol.where(vol > 0, np.nan)
    out = abs_ret / safe_vol
    return out


def compute_ko_W(df):
    """Feature 14: sigma_gk * close * volume."""
    return df['sigma_gk'] * df['close'] * df['volume'].astype(float)


def compute_polar_order(df, window=50):
    """Feature 15: |rolling_sum_50(sign(log_ret))| / 50 per instrument_id."""
    prev_close = df.groupby('instrument_id', sort=False)['close'].shift(1)
    log_ret = np.log(df['close'] / prev_close).fillna(0.0)
    sign_ret = pd.Series(np.sign(log_ret.values).astype(float), index=df.index)
    roll_sum = sign_ret.groupby(df['instrument_id'], sort=False).transform(
        lambda s: s.rolling(window, min_periods=window).sum())
    return roll_sum.abs() / window


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

REQUIRED_COLS = ['instrument_id', 'open', 'high', 'low', 'close', 'volume',
                 'imbalance_t', 'sigma_gk', 'is_valid_bar']


def process_contract(contract, input_dir=PHASE2_INPUT_DIR, output_dir=OUTPUT_DIR):
    """Load input pickle, compute 8 features, save expanded pickle."""
    input_path = input_dir / f"phase2_features_cleaned_{contract}.pkl"
    output_path = output_dir / f"phase2_features_expanded_{contract}.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n{'='*70}\nPhysics features: {contract}\n{'='*70}", flush=True)
    with open(input_path, 'rb') as f:
        df = pickle.load(f)
    print(f"  loaded {len(df):,} bars, {len(df.columns)} columns", flush=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{contract}] missing required columns: {missing}")

    # Input is already globally monotonic in ts (verified); within each
    # instrument_id the rows are therefore also time-ordered, so groupby+shift
    # and groupby+rolling operate correctly in place.
    if not df.index.is_monotonic_increasing:
        df = df.sort_index(kind='mergesort')

    # v_star_C intentionally excluded — see module docstring.
    df['sigma_yz']       = compute_yang_zhang(df)
    print("  sigma_yz done", flush=True)
    df['body_to_range']  = compute_body_to_range(df)
    df['gel_fraction']   = compute_gel_fraction(df)
    df['wick_asymmetry'] = compute_wick_asymmetry(df)
    df['amihud']         = compute_amihud(df)
    df['ko_W']           = compute_ko_W(df)
    print("  per-bar features done", flush=True)
    df['polar_order_P']  = compute_polar_order(df)
    print("  polar_order_P done", flush=True)

    # Uniform is_valid_bar filter on the seven new features.
    valid = df['is_valid_bar'].astype(bool)
    for col in NEW_FEATURES:
        df[col] = df[col].where(valid, np.nan)

    # Drop any stale v_star_C column if a previous run wrote it.
    if 'v_star_C' in df.columns:
        df = df.drop(columns=['v_star_C'])

    # (sorted in place above if needed)

    with open(output_path, 'wb') as f:
        pickle.dump(df, f)
    dt = time.time() - t0
    print(f"  saved {output_path}  rows={len(df):,}  cols={len(df.columns)}  "
          f"wall={dt:.1f}s", flush=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=["ES", "NQ", "RTY"], default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        contracts = ["ES", "NQ", "RTY"]
    elif args.contract is not None:
        contracts = [args.contract]
    else:
        parser.error("specify --all or --contract {ES,NQ,RTY}")

