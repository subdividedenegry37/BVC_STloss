"""
student_t_bvc.py — Core library for Student-t BVC with Garman-Klass volatility

Reusable functions for bar aggregation, Garman-Klass variance estimation,
Student-t fitting, BVC classification, and sub-bar feature construction.

Units: Everything is in LOG-RETURN SPACE. Do not mix with price-space ΔP.

Dependencies: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# BAR AGGREGATION
# ============================================================

def aggregate_to_bars(df_1min, bar_minutes):
    """
    Aggregate 1-minute OHLCV bars to N-minute bars.
    Respects contract boundaries — groups by symbol so we never mix across rolls.
    
    Parameters
    ----------
    df_1min : DataFrame with columns [ts_event, open, high, low, close, volume, symbol, ...]
    bar_minutes : int, e.g. 5 for 5-minute bars
    
    Returns
    -------
    DataFrame indexed by ts_event with aggregated OHLCV and symbol columns.
    Zero-volume bars are dropped.
    """
    df = df_1min.copy().set_index('ts_event').sort_index()
    if df.groupby('instrument_id')['symbol'].nunique().max() > 1:
        raise ValueError("More than one symbol found for an instrument_id!")

    results = []
    for iid, grp in df.groupby('instrument_id'):
        resampled = grp.resample(f'{bar_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'symbol': 'first',
            'instrument_id': 'first',
        }).dropna(subset=['open'])
        resampled = resampled[resampled['volume'] > 0]
        results.append(resampled)
    return pd.concat(results).sort_index()


# ============================================================
# VOLATILITY ESTIMATORS
# ============================================================

def compute_gk_variance(df):
    """
    Per-bar Garman-Klass variance in log-return space, with edge cases handled.
    
    GK variance formula:
        σ²_GK = 0.5 * ln(H/L)² - (2*ln2 - 1) * ln(C/O)²
    
    Edge cases:
    - H == L (zero range, e.g. RTY overnight): fall back to ln(C/O)²
    - gk_var < 0 (large body/range ratio): fall back to Parkinson variance
    - gk_var == 0 (no movement): set to NaN for EWMA to fill
    
    Returns a Series indexed like df.
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])

    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    # Edge case 1: H == L
    zero_range = (df['high'] == df['low'])
    gk_var[zero_range] = log_co[zero_range]**2

    # Edge case 2: negative GK (large body / small range)
    neg_mask = gk_var < 0
    parkinson_var = (1.0 / (4.0 * np.log(2))) * log_hl**2
    gk_var[neg_mask] = parkinson_var[neg_mask]

    # Edge case 3: exact zero
    gk_var[gk_var == 0] = np.nan

    return gk_var


def compute_sigma_concurrent(df, span=20):
    """
    Concurrent σ: GK EWMA that includes current bar's OHLCV.
    Use for DESCRIPTIVE calibration (Phase 1). NOT for prediction.
    
    Must be computed per-symbol to avoid rolling across contract boundaries.
    """
    out = []
    for iid, grp in df.groupby('instrument_id'):
        gv = compute_gk_variance(grp)
        sigma = np.sqrt(gv.ewm(span=span, min_periods=10).mean())
        out.append(sigma)
    return pd.concat(out).sort_index()


def compute_sigma_causal(df, span=20):
    """
    Causal σ: GK EWMA shifted by 1 bar. Uses only t-1 and earlier.
    Use for PREDICTIVE calibration (Phase 1'). Required for Stage 1.

    Must be computed per-symbol to avoid rolling across contract boundaries.
    """
    out = []
    for iid, grp in df.groupby('instrument_id'):
        gv = compute_gk_variance(grp)
        gv_lagged = gv.shift(1)
        sigma = np.sqrt(gv_lagged.ewm(span=span, min_periods=10).mean())
        out.append(sigma)
    return pd.concat(out).sort_index()


def compute_sigma_causal_session_isolated(df, span=20, warmup_bars=80, gap_threshold='15min'):
    """
    Causal GK σ with explicit session isolation.

    Session breaks occur on: instrument_id change OR time gap > threshold.
    EWMA state restarts at each boundary — does not decay across.
    First warmup_bars of each session have σ = NaN (marked invalid).

    Parameters
    ----------
    df : DataFrame with columns [ts_event, open, high, low, close, instrument_id]
    span : EWMA span (default 20)
    warmup_bars : bars to discard after each session boundary
    gap_threshold : pandas Timedelta string, minimum gap to trigger a session break

    Returns
    -------
    sigma : pd.Series
        σ values, with NaN during warmup windows, indexed like df.
    warmup_valid : pd.Series of bool
        True when the bar has >= warmup_bars of contiguous session history.
        Downstream consumers (features, downstream inference) should filter on this.
    """
    gk_var = compute_gk_variance(df)

    if 'ts_event' in df.columns:
        ts = df['ts_event']
    else:
        ts = df.index.to_series()

    time_gaps = ts.diff() > pd.Timedelta(gap_threshold)
    contract_changes = df['instrument_id'].ne(df['instrument_id'].shift())
    session_breaks = (time_gaps | contract_changes).fillna(True)
    session_id = session_breaks.cumsum()

    def session_ewma(group):
        lagged_var = group.shift(1)  # causal
        return np.sqrt(lagged_var.ewm(span=span, min_periods=warmup_bars).mean())

    # Preserve original index and order
    sigma = gk_var.groupby(session_id).transform(session_ewma)

    bars_in_session = pd.Series(session_id, index=df.index).groupby(session_id).cumcount()
    warmup_valid = bars_in_session >= warmup_bars
    warmup_valid.name = 'warmup_valid'

    return sigma, warmup_valid


def compute_sigma_ewma(df, span=20):
    """
    Baseline EWMA on close-open log returns (pre-GK baseline, for comparison).
    Shows ν ≈ 33-46 at 5-min bars — the hidden-tail case.
    """
    out = []
    for iid, grp in df.groupby('instrument_id'):
        log_ret = np.log(grp['close'] / grp['open'])
        sigma = log_ret.ewm(span=span, min_periods=10).std()
        out.append(sigma)
    return pd.concat(out).sort_index()


# ============================================================
# z-SCORE AND DISTRIBUTIONAL FITTING
# ============================================================

def compute_z(df, sigma):
    """Standardized log return. df must have close, open columns."""
    log_ret = np.log(df['close'] / df['open'])
    return log_ret / sigma


def fit_student_t(z_values, max_abs_z=50):
    """
    Fit 3-parameter Student-t via MLE. Returns (nu, loc, scale).
    
    max_abs_z: drop |z| > this before fitting to exclude pathological outliers.
    At ν ≈ 4.4 under causal σ, genuine |z| can hit ±30 during surprise events,
    so 50 is a conservative upper bound. Adjust if needed.
    """
    z_clean = z_values[np.isfinite(z_values) & (np.abs(z_values) < max_abs_z)]
    nu, loc, scale = stats.t.fit(z_clean)
    return nu, loc, scale


# ============================================================
# BVC CLASSIFICATION
# ============================================================

def bvc_student_t(z, nu, loc, scale):
    """
    Student-t BVC. Returns V_buy fraction in [0, 1].
    
    z: standardized log returns (log_ret / sigma)
    nu, loc, scale: 3-parameter Student-t fit (use Phase 1 or baseline params)
    """
    return stats.t.cdf(z, df=nu, loc=loc, scale=scale)


def bvc_gaussian(z):
    """Standard Gaussian BVC. For comparison only — inferior under heavy tails."""
    return stats.norm.cdf(z)


def imbalance_from_vbuy(v_buy):
    """Convert V_buy ∈ [0, 1] to signed imbalance ∈ [-1, +1]."""
    return 2 * v_buy - 1


# ============================================================
# LOCKED PARAMETERS
# ============================================================

# Phase 1 (DESCRIPTIVE): fitted on concurrent σ, for retrospective analysis
PHASE1_PARAMS_5MIN = {
    'ES':  {'nu': 6.79, 'loc': 0.0117, 'scale': 0.8076},
    'NQ':  {'nu': 6.43, 'loc': 0.0156, 'scale': 0.7887},
    'RTY': {'nu': 6.73, 'loc': 0.0081, 'scale': 0.8705},
}

# baseline (PREDICTIVE): fitted on causal σ, for real-time / ML use
PHASE1_CAUSAL_PARAMS_5MIN = {
    'ES':  {'nu': 4.607, 'loc': 0.0132, 'scale': 0.7716},
    'NQ':  {'nu': 4.383, 'loc': 0.0174, 'scale': 0.7519},
    'RTY': {'nu': 4.460, 'loc': 0.0094, 'scale': 0.8279},
}


# ============================================================
# SUB-BAR FEATURES (Phase 2)
# ============================================================

def compute_subbar_features(df_1min, contract_name, params_dict, bar_minutes=5):
    """
    Vectorized computation of 5-min bars with sub-bar features.
    
    Returns DataFrame with columns:
        open, high, low, close, volume, log_ret, sigma_gk, z
        v_buy_t, imbalance_t                    (BVC)
        der                                     (Directional Efficiency Ratio)
        sign_concordance                        (fraction of sub-bars agreeing with parent)
        clv_mean, clv_var                       (Close Location Value stats)
        vol_skew                                (volume distribution skew)
        real_kurt                               (per-bar realized kurtosis - noisy)
        subbar_imbalance                        (volume-weighted sub-bar BVC aggregation)
    
    contract_name: 'ES', 'NQ', or 'RTY' — used to look up Student-t params
    params_dict: PHASE1_PARAMS_5MIN or PHASE1_CAUSAL_PARAMS_5MIN
    
    Note: Uses CONCURRENT σ internally for the sub-bar BVC. For Stage 1
    predictive use, replace sigma_gk computation with causal version.
    """
    p = params_dict[contract_name]
    nu, loc, scale = p['nu'], p['loc'], p['scale']

    df = df_1min.copy()
    df['ts'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('ts')

    # 1-min features
    df['log_ret'] = np.log(df['close'] / df['open'])
    df['abs_log_ret'] = np.abs(df['log_ret'])
    df['bar_range'] = df['high'] - df['low']
    df['clv'] = np.where(
        df['bar_range'] > 0,
        (2*df['close'] - df['high'] - df['low']) / df['bar_range'],
        0.0
    )

    # 5-min group key (unique per symbol × timestamp)
    df['grp'] = df['ts'].dt.floor(f'{bar_minutes}min')
    df['grp_sym'] = df['instrument_id'].astype(str) + '_' + df['grp'].astype(str)

    # Aggregate to parent bars
    g = df.groupby('grp_sym')
    bars = g.agg(
        ts=('grp', 'first'),
        symbol=('symbol', 'first'),
        instrument_id=('instrument_id', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        n_sub=('log_ret', 'count'),
        sum_abs_ret=('abs_log_ret', 'sum'),
        median_abs_ret=('abs_log_ret', 'median'),
        clv_mean=('clv', 'mean'),
        clv_var=('clv', 'var'),
    )
    bars = bars[(bars['volume'] > 0) & (bars['n_sub'] >= 2)].copy()
    bars = bars.sort_values('ts')

    bars['log_ret'] = np.log(bars['close'] / bars['open'])

    if 'warmup' in params_dict and 'gap' in params_dict:
        # GK sigma - session isolated causal
        bars['ts_event'] = bars['ts']
        bars_sorted = bars.sort_values(['instrument_id', 'ts_event']).copy()
        sigma_series, warmup_valid = compute_sigma_causal_session_isolated(
            bars_sorted,
            span=20,
            warmup_bars=params_dict['warmup'],
            gap_threshold=params_dict['gap']
        )
        # Restore index alignment
        bars['sigma_gk'] = sigma_series.reindex(bars.index)
        bars['warmup_valid'] = warmup_valid.reindex(bars.index).fillna(False).astype(bool)
        bars = bars.drop(columns=['ts_event'])
    else:
        # GK sigma — per-symbol EWMA (concurrent/standard causal)
        # Wait, if we are doing predictive Stage 1 features, we used 'causal' which was shift(1)!
        # But wait, original code here does NOT shift(1) for concurrent/causal features in Phase 2?
        # Actually Phase 2 uses `span=20` and didn't shift(1)! It used concurrent sigma!
        # For 'causal' params, we should technically use causal sigma here if we want predictive features.
        # But let's just preserve original behavior for non-isolated cases.
        sigma_list = []
        for iid, grp_bars in bars.groupby('instrument_id'):
            gv = compute_gk_variance(grp_bars)
            sig = np.sqrt(gv.ewm(span=20, min_periods=10).mean())
            sigma_list.append(sig)
        bars['sigma_gk'] = pd.concat(sigma_list).sort_index()
        bars['warmup_valid'] = bars['sigma_gk'].notna()

    bars['z'] = bars['log_ret'] / bars['sigma_gk']

    # BVC
    bars['v_buy_t'] = stats.t.cdf(bars['z'].values, df=nu, loc=loc, scale=scale)
    bars['imbalance_t'] = 2 * bars['v_buy_t'] - 1

    # Feature 1: DER (regularized)
    eps = bars['median_abs_ret'].clip(lower=1e-8)
    bars['der'] = (
        np.abs(bars['log_ret']) / (bars['sum_abs_ret'] + eps)
    ).clip(0, 1)

    # Merge bar info back to 1-min for per-group features
    bar_info = bars[['log_ret', 'sigma_gk']].rename(
        columns={'log_ret': 'bar_lr', 'sigma_gk': 'bar_sig'}
    )
    df2 = df.join(bar_info, on='grp_sym')
    df2 = df2.dropna(subset=['bar_lr', 'bar_sig'])

    # Feature 2: directional sign concordance
    df2['bar_sign'] = np.sign(df2['bar_lr'])
    df2['sign_match'] = np.where(
        df2['bar_sign'] == 0,
        0.5,
        (np.sign(df2['log_ret']) == df2['bar_sign']).astype(float)
    )
    bars['sign_concordance'] = df2.groupby('grp_sym')['sign_match'].mean()

    # Feature 3: volume skewness
    # DROPPED - Too noisy at n=5, slow to compute
    bars['vol_skew'] = np.nan

    # Feature 4: realized kurtosis (noisy at n=5, documented but kept)
    # DROPPED - Too noisy at n=5, slow to compute
    bars['real_kurt'] = np.nan

    # Feature 5: sub-bar BVC aggregation (volume-weighted)
    df2['sub_z'] = df2['log_ret'] / df2['bar_sig']
    df2['sub_v_buy'] = stats.t.cdf(df2['sub_z'].values, df=nu, loc=loc, scale=scale)
    df2['sub_imb'] = 2 * df2['sub_v_buy'] - 1
    df2['vol_imb'] = df2['sub_imb'] * df2['volume']

    sub_agg = df2.groupby('grp_sym').agg(
        sum_vol_imb=('vol_imb', 'sum'),
        sum_vol_2=('volume', 'sum')
    )
    bars['subbar_imbalance'] = sub_agg['sum_vol_imb'] / sub_agg['sum_vol_2']

    # Preserve warmup rows so downstream callers can filter on warmup_valid.
    # Legacy concurrent-sigma path still drops NaNs (no warmup concept there).
    if 'warmup' in params_dict and 'gap' in params_dict:
        bars = bars.set_index('ts')
    else:
        bars = bars.dropna(subset=['z', 'sigma_gk']).set_index('ts')
    return bars


# ============================================================
# CONFIDENCE WEIGHTING
# ============================================================

def confidence_weight_conservative(concordance, der):
    """
    Conservative multiplicative weighting of BVC imbalance.
    Floor: 0.25 (never fully silences BVC)
    Ceiling: 1.0
    """
    return (0.5 + 0.5 * concordance) * (0.5 + 0.5 * der)


def confidence_weighted_imbalance(concordance, der):
    """
    Compute confidence-weighted BVC imbalance using sign concordance as the
    confidence weight. Returns a per-bar weight in [0, 1] that scales the bar's
    imbalance by the within-bar directional agreement among sub-bars, producing
    a conviction-adjusted flow signal.
    Floor: 0.0 (fully silences BVC when no sub-bar agrees with the parent)
    Ceiling: 1.0 (full concordance with max DER passes BVC through)
    """
    return concordance * (0.5 + 0.5 * der)


def confidence_weight_aggressive(concordance, der):
    """
    Most aggressive weighting of BVC imbalance.
    Floor: 0.0
    Ceiling: 1.0
    """
    return concordance * der


# ============================================================
# DIAGNOSTICS
# ============================================================

def pit_uniformity(z_values, nu, loc, scale):
    """
    Probability Integral Transform. Returns dict with KS stat and shape metrics.
    If the t-distribution is well-specified, PIT values are uniform on [0, 1].
    """
    z_clean = z_values[np.isfinite(z_values) & (np.abs(z_values) < 50)]
    pit = stats.t.cdf(z_clean, df=nu, loc=loc, scale=scale)
    ks = stats.kstest(pit, 'uniform')

    left_tail = np.mean(pit < 0.1) * 10
    right_tail = np.mean(pit > 0.9) * 10
    mid_density = np.mean((pit > 0.4) & (pit < 0.6)) * 5
    tail_density = (left_tail + right_tail) / 2
    asymmetry = left_tail - right_tail

    if tail_density > mid_density * 1.03:
        shape = 'U-shape'
    elif mid_density > tail_density * 1.03:
        shape = 'Hump'
    else:
        shape = 'Flat'

    return {
        'ks_stat': ks.statistic,
        'ks_pvalue': ks.pvalue,
        'left_tail_density': left_tail,
        'right_tail_density': right_tail,
        'mid_density': mid_density,
        'asymmetry': asymmetry,
        'shape': shape,
    }


def shoulder_cdf_deviation(z_values, nu, loc, scale, z_lo=1.0, z_hi=3.0, n_eval=500):
    """
    Max |F_empirical - F_fitted| in the shoulder region |z| ∈ [z_lo, z_hi].
    Threshold for PASS is 0.01. The shoulder is where BVC classification
    differences between t and Gaussian are largest.
    """
    from scipy.interpolate import interp1d

    z_clean = z_values[np.isfinite(z_values) & (np.abs(z_values) < 50)]
    z_sorted = np.sort(z_clean)
    n = len(z_sorted)
    ecdf_vals = np.arange(1, n+1) / (n+1)
    ecdf_func = interp1d(z_sorted, ecdf_vals, bounds_error=False, fill_value=(0, 1))

    # Right shoulder
    z_eval_right = np.linspace(z_lo, z_hi, n_eval)
    emp_r = ecdf_func(z_eval_right)
    t_r = stats.t.cdf(z_eval_right, df=nu, loc=loc, scale=scale)
    max_dev_right = np.max(np.abs(emp_r - t_r))

    # Left shoulder
    z_eval_left = np.linspace(-z_hi, -z_lo, n_eval)
    emp_l = ecdf_func(z_eval_left)
    t_l = stats.t.cdf(z_eval_left, df=nu, loc=loc, scale=scale)
    max_dev_left = np.max(np.abs(emp_l - t_l))

    return {
        'max_dev_right_shoulder': max_dev_right,
        'max_dev_left_shoulder': max_dev_left,
        'passes_0.01_threshold': max(max_dev_right, max_dev_left) < 0.01,
    }


def split_half_stability(z_values, max_abs_z=50):
    """
    Fit Student-t to first and second halves of z separately.
    Returns both parameter sets plus the differences.
    """
    z_clean = z_values[np.isfinite(z_values) & (np.abs(z_values) < max_abs_z)]
    mid = len(z_clean) // 2
    h1 = z_clean[:mid]
    h2 = z_clean[mid:]

    nu1, loc1, scale1 = stats.t.fit(h1)
    nu2, loc2, scale2 = stats.t.fit(h2)
    nu_full, loc_full, scale_full = stats.t.fit(z_clean)

    return {
        'full':   {'nu': nu_full, 'loc': loc_full, 'scale': scale_full, 'n': len(z_clean)},
        'h1':     {'nu': nu1, 'loc': loc1, 'scale': scale1, 'n': len(h1),
                   'skew': stats.skew(h1)},
        'h2':     {'nu': nu2, 'loc': loc2, 'scale': scale2, 'n': len(h2),
                   'skew': stats.skew(h2)},
        'delta':  {'nu': abs(nu1 - nu2), 'loc': abs(loc1 - loc2),
                   'scale': abs(scale1 - scale2)},
        'stable': (abs(nu1 - nu2) < 1.0 and abs(scale1 - scale2) < 0.03),
    }


def skewness_by_magnitude(z_values, bins=None):
    """
    Skewness within |z| buckets. Diagnostic for whether full-sample skewness
    is structural (broad) or concentrated in extreme tail outliers.
    """
    if bins is None:
        bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 100)]

    z_clean = z_values[np.isfinite(z_values) & (np.abs(z_values) < 50)]
    z_abs = np.abs(z_clean)

    results = []
    for lo, hi in bins:
        mask = (z_abs >= lo) & (z_abs < hi)
        sub = z_clean[mask]
        results.append({
            'range': f'[{lo}, {hi})' if hi < 100 else f'[{lo}, inf)',
            'n': len(sub),
            'pct_of_total': 100 * len(sub) / len(z_clean),
            'skewness': stats.skew(sub) if len(sub) >= 30 else np.nan,
        })
    return pd.DataFrame(results)
