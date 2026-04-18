# Physics Feature Expansion — Specification

**Purpose:** Specification of seven physics-motivated candidate features added to the BVC Phase 2 feature pipeline, producing a 14-feature set (7 BVC + 7 physics) for Stage 1 forward realized volatility prediction.

---

## 0. Context and Scope

### 0.1 What This Document Is

This document specifies the exact formulas, implementation path, and validation checks for seven physics-motivated features added to the Phase 2 feature pipeline. Every feature below is computable from OHLCV data (open, high, low, close, volume) plus the existing BVC outputs (`imbalance_t`, `sigma_gk`, `is_valid_bar`).

No new data sources required. No changes to the baseline calibration. No modification of the existing seven BVC features.

### 0.2 What These Features Are For

Candidate predictors for the Stage 1 LightGBM regression on 30-minute forward realized volatility. The combined 14-feature set (7 BVC + 7 physics) is evaluated by feature importance after the first training round; low-ranking features can be pruned before final reporting.

### 0.3 Physics Motivation — Brief Context

The physics feature family draws on established frameworks in statistical mechanics, fluid dynamics, and active matter physics. Each feature is motivated by a specific structural analogy between a microstructure observable and a well-studied physical quantity: Yang-Zhang volatility as a "temperature" variable; body-to-range as a loss tangent; gel-fraction as a network-participation index; wick asymmetry as transverse anisotropy in an intrabar path; Amihud illiquidity as an inverse-permeability (Darcy) coefficient; Kyle-Obizhaeva W as an equation-of-state variable; and the polar order parameter as directional coherence of returns over a relaxation-timescale window.

The physics motivations are descriptive — they explain **why** each feature might carry information. The formulas below are pure OHLCV computations that stand on their own without any physics background; a reader who wants only the numerics can skip the interpretation blocks.

### 0.4 Constraints

- Do not modify `phase2_features_cleaned_{ES,NQ,RTY}.pkl` (the baseline feature pickles).
- Do not modify any function in `student_t_bvc.py`.
- Every formula is specified below; implementations should not deviate.

### 0.5 Data Prerequisites

The baseline feature pickles (`phase2_features_cleaned_{ES,NQ,RTY}.pkl`) must contain these columns:

- `ts_event` (timestamp)
- `instrument_id` (contract identifier for roll handling)
- `open`, `high`, `low`, `close`, `volume` (5-minute bar OHLCV — required for physics features)
- `imbalance_t` (locked BVC feature)
- `sigma_gk` (locked GK σ — required for Yang-Zhang and ko_W)
- `is_valid_bar` (locked validity mask)
- The six other locked BVC features (z, DER, sign_concordance, clv_mean, clv_var, subbar_imbalance) — not strictly required for computing new features, but needed for final validation correlations

If the OHLCV columns are not in the pickles, join them from the source 1-minute OHLCV parquet files using the same aggregation logic the Phase 2 pipeline already uses (5-minute bars, RTH + ETH, session-aware). Verify row alignment after the join.

---

## 1. The Seven Features — Exact Formulas

Each specification includes: (a) the exact algebraic form, (b) expected numerical range, (c) the physics interpretation (for context only, not required for computation), (d) edge case handling, and (e) a validation expectation.

### Feature 8: `sigma_yz` — Yang-Zhang Rolling Volatility

**Formula:**

```
# Per-bar components
overnight_ret_t = ln(open_t / close_{t-1})              # bar-to-bar gap
open_to_close_t = ln(close_t / open_t)                  # intrabar body return
rs_per_bar_t = ln(H_t/C_t)·ln(H_t/O_t) + ln(L_t/C_t)·ln(L_t/O_t)   # Rogers-Satchell

# Yang-Zhang weighting constant
n = 20
k = 0.34 / (1.34 + (n+1)/(n-1))

# Rolling variances (window = 20 bars, min_periods = 20)
var_overnight_t = rolling variance of overnight_ret_t over n bars
var_open_to_close_t = rolling variance of open_to_close_t over n bars
mean_rs_t = rolling mean of rs_per_bar_t over n bars

# Combined Yang-Zhang variance
sigma_yz_squared_t = var_overnight_t + k * var_open_to_close_t + (1 - k) * mean_rs_t

# Final volatility (clip negative to 0 before sqrt for numerical safety)
sigma_yz_t = sqrt(max(sigma_yz_squared_t, 0))
```

**Implementation notes:**
- Rolling operations computed per `instrument_id` (do not let rolling windows cross contract boundaries at rolls).
- The first bar of each contract has no overnight_ret (requires previous close). First n bars of each contract have NaN rolling output. Acceptable — these bars are also `is_valid_bar == False` due to warmup.
- `rs_per_bar_t`: when close = open (doji bar), two terms are zero. When high = low (zero-range bar), all four terms are zero. Both are fine — the formula produces 0 in those cases.
- `overnight_ret_t` across a session boundary (e.g., Friday close to Sunday open) is still included as "overnight" — the Yang-Zhang estimator is designed for this.

**Physics interpretation (context only):**
The "temperature" variable in the physics analogy — a low-bias volatility estimator that combines overnight, open-to-close, and Rogers-Satchell components. Used as a conditioning feature alongside σ_GK, not as the z denominator (σ_GK stays as the z denominator per baseline).

**Edge case handling:**
- If `sigma_yz_squared_t` is negative (can occur with extreme negative autocorrelation in the RS term), clip to 0 before sqrt.
- If any of overnight_ret, open_to_close, or rs_per_bar is NaN (missing previous close, zero prices, etc.), the rolling variance/mean will propagate NaN. Apply `is_valid_bar` filter at the end.

**Expected range:**
- Typical values 0.0001 to 0.002 for 5-minute bars on liquid futures.
- Median per contract should be within 10× of the median `sigma_gk` for that contract (different estimators, related scale).
- Right-skewed distribution.

**Validation expectation:**
- Correlation with `sigma_gk` should be positive and moderate (r ≈ 0.5-0.8).
- If correlation > 0.95, Yang-Zhang adds no information beyond GK — flag as redundant but still include for LightGBM to evaluate.
- If correlation < 0.3, something is wrong — investigate before proceeding.

---

### Feature 9: `body_to_range` — Loss Tangent

**Formula:**

```
bar_range_t = high_t - low_t
bar_body_t = |close_t - open_t|

body_to_range_t = bar_body_t / bar_range_t   if bar_range_t > 0
                = 0                            if bar_range_t == 0 (zero-range bar)
```

**Implementation notes:**
- Straightforward per-bar computation, no rolling window.
- Bounded by construction in [0, 1].
- For zero-range bars (H == L), return 0 rather than NaN. This is consistent with the physics interpretation (no range → no deformation → no loss tangent measurable). Zero-range bars are common on RTY overnight sessions due to tick discreteness.

**Physics interpretation (context only):**
Maps to the loss tangent tan(δ) = G''/G' in viscoelastic materials. Near 0 = strong wicks, elastic/mean-reverting character. Near 1 = directional bar with no wicks, viscous/flowing character. Pure price-geometry feature, no volume classification.

**Edge case handling:**
- bar_range == 0: return 0.
- NaN open, high, low, or close: propagate NaN.
- Apply `is_valid_bar` filter at the end.

**Expected range:**
- [0, 1] by construction.
- Median per contract typically 0.30-0.55.
- Distribution approximately bimodal: mass near 0 (doji/indecision bars) and mass near 1 (strong directional bars).

**Validation expectation:**
- Correlation with `sign_concordance` should be low-to-moderate (|r| < 0.4). They're related (both capture "directional conviction") but sign_concordance uses sub-bar classification while body_to_range uses bar-level geometry. If correlation > 0.7, they're capturing the same thing and one should be dropped.
- Per-contract zero-rate (fraction of bars with body_to_range == 0): should be < 5% for ES and NQ, may be 5-15% for RTY due to Russell 2000 tick discreteness.

---

### Feature 10: `gel_fraction` — Network Participation

**Formula:**

```
bar_range_t = high_t - low_t
midrange_t = (high_t + low_t) / 2
close_deviation_t = |close_t - midrange_t|

gel_fraction_t = close_deviation_t / bar_range_t   if bar_range_t > 0
               = 0                                   if bar_range_t == 0
```

**Implementation notes:**
- Per-bar computation, no rolling window.
- Bounded by construction in [0, 0.5]. The maximum of |close - midrange|/(high - low) is 0.5, achieved when close = high or close = low.

**Physics interpretation (context only):**
Distinct from body_to_range. Measures whether close is near the bar's midrange (random/"sol-like" activity, gel_fraction → 0) or near its extremes (correlated/"gel-network" activity, gel_fraction → 0.5). Network participation indicator. A bar can have high body_to_range but low gel_fraction (directional move but closes mid-range on retracement) or low body_to_range but high gel_fraction (small body but closes at an extreme).

**Edge case handling:**
- bar_range == 0: return 0.
- NaN inputs: propagate NaN.
- Apply `is_valid_bar` filter at the end.

**Expected range:**
- [0, 0.5] by construction.
- Median per contract typically 0.20-0.30.
- Distribution: approximately uniform-ish with slight right skew.

**Validation expectation:**
- Correlation with `clv_mean` should be moderate (they both measure close location relative to bar extremes, but gel_fraction uses midrange reference while CLV uses range midpoint weighted differently). Expect r ≈ 0.5.
- Correlation with `body_to_range` should be low-to-moderate (r ≈ 0.3-0.5). They capture different aspects — body_to_range is body size, gel_fraction is closing-location extremity.

---

### Feature 11: `wick_asymmetry` — Transverse Anisotropy

**Formula:**

```
body_high_t = max(open_t, close_t)
body_low_t = min(open_t, close_t)

upper_wick_t = ln(high_t / body_high_t)      # ≥ 0 by construction
lower_wick_t = ln(body_low_t / low_t)        # ≥ 0 by construction

wick_asymmetry_t = upper_wick_t - lower_wick_t
```

**Implementation notes:**
- Per-bar computation, no rolling window.
- Both wicks are non-negative by construction (since high ≥ max(O,C) and low ≤ min(O,C)). Their difference can be either sign.
- Positive wick_asymmetry: upper wick longer than lower (upward rejection, selling into highs).
- Negative wick_asymmetry: lower wick longer than upper (downward rejection, buying into lows).

**Physics interpretation (context only):**
Captures transverse anisotropy from intrabar path geometry. Directional signal that doesn't rely on sub-bar flow classification. Candidate complementary feature to sign_concordance.

**Edge case handling:**
- If body_high == 0 or body_low == 0 or low == 0: log is undefined, produce NaN.
- If high == body_high (no upper wick) and low == body_low (no lower wick): wick_asymmetry = 0 - 0 = 0. Correct.
- Replace any inf/-inf with NaN.
- Apply `is_valid_bar` filter at the end.

**Expected range:**
- Centered near 0 by market symmetry.
- Typical magnitude: ±0.0005 to ±0.002 for 5-minute bars.
- Mean per contract should be very close to 0 (within ±0.0001). A structurally nonzero mean suggests either data issues or a real asymmetry (e.g., equity contracts tend to have slightly more downside wicks due to protective selling — small but nonzero).

**Validation expectation:**
- Distribution approximately symmetric around 0.
- Skewness should be small (|skew| < 0.5). Large skewness is a red flag for data quality issues.
- Per-contract mean: within ±0.0001 (essentially zero).

---

### Feature 12: `amihud` — Inverse Permeability (Illiquidity)

**Formula:**

```
log_ret_t = ln(close_t / close_{t-1})
abs_log_ret_t = |log_ret_t|

amihud_t = abs_log_ret_t / volume_t   if volume_t > 0 and abs_log_ret_t > 0
         = 0                            if volume_t > 0 and abs_log_ret_t == 0
         = NaN                          if volume_t == 0
```

**Implementation notes:**
- log_ret computed per `instrument_id` (do not compute across contract rolls).
- First bar of each contract has no log_ret (requires previous close) → NaN.
- The Amihud (2002) illiquidity measure. Higher values = more price impact per contract traded = less liquid.

**Physics interpretation (context only):**
The market's inverse permeability. Direct analog to Darcy's law in porous media — how much flow (volume) is required to produce a given pressure gradient (price movement). Contract-level liquidity context not captured by any of the seven BVC features.

**Edge case handling:**
- volume == 0: return NaN (undefined — no trading occurred).
- volume > 0 and log_ret == 0: return 0 (efficient trading, no price impact despite volume).
- NaN previous close: propagate NaN for the first bar of each contract.
- Apply `is_valid_bar` filter at the end.

**Expected range:**
- Highly right-skewed distribution, most mass near zero with long tail.
- Typical magnitude: 10^-9 to 10^-7 for liquid index futures.
- Raw values are fine for LightGBM (tree models handle skew natively). Log-transform is NOT required — LightGBM will learn appropriate splits on raw scale.

**Validation expectation:**
- Correlation with volume should be negative (high-volume bars have low Amihud).
- Correlation with |log_ret| should be positive (high-volatility bars have high Amihud).
- Fraction of NaN due to zero volume: should be < 5% on valid bars for ES and NQ, may be 5-15% for RTY overnight.
- If median is suspiciously close to 0 or very large (> 1e-4), check the volume and close_{t-1} columns.

---

### Feature 13: `ko_W` — Kyle-Obizhaeva Trading Activity

**Formula:**

```
gk_variance_t = sigma_gk_t ^ 2

ko_W_t = sqrt(gk_variance_t) * close_t * volume_t
      = sigma_gk_t * close_t * volume_t
```

**Implementation notes:**
- Per-bar computation, no rolling window.
- Pure product of three state variables: volatility (σ_GK), price (close), volume.
- The Kyle-Obizhaeva (2016) "trading activity" variable — their W, sometimes denoted ε in the market microstructure invariance literature.
- Not the same as Weissenberg number. "ko_W" stands for "Kyle-Obizhaeva W" — the market microstructure reference.

**Physics interpretation (context only):**
The market's "equation of state" variable. In the microstructure invariance framework, W is the fundamental coordinate — individual properties like σ, P, V are constrained to lie on a surface parameterized by W. Analogous to PV = nRT in thermodynamics where individual state variables are related through a single equation.

**Edge case handling:**
- If sigma_gk is NaN (warmup bar): ko_W is NaN.
- If sigma_gk is 0 (extremely rare): ko_W is 0.
- If volume is 0: ko_W is 0.
- Apply `is_valid_bar` filter at the end.

**Expected range:**
- Large absolute values by construction (product of three non-small quantities).
- For ES at ~4500, σ ~0.0003, volume ~100: W ~135. For RTY at ~2000, σ ~0.0004, volume ~30: W ~24.
- Highly right-skewed distribution.
- Raw values fine for LightGBM. Optionally compute `log1p(ko_W)` as an auxiliary feature if distributional shape matters for interpretability, but not required.

**Validation expectation:**
- Should correlate strongly with concurrent volume (raw correlation r > 0.7).
- Should correlate positively with concurrent |return|.
- Scale differs by roughly an order of magnitude between contracts — this is expected and correct.

---

### Feature 14: `polar_order_P` — Directional Persistence

**Formula:**

```
log_ret_t = ln(close_t / close_{t-1})
sign_ret_t = sign(log_ret_t)    # -1, 0, or +1
                                 # sign(0) = 0 (flat bars contribute 0)
                                 # NaN returns → sign = 0 via fillna

window = 50   # bars

rolling_sum_t = rolling sum of sign_ret over the past 50 bars (min_periods = 50)

polar_order_P_t = |rolling_sum_t| / 50
```

**Implementation notes:**
- Rolling operations computed per `instrument_id`.
- window = 50 bars chosen to match the approximate relaxation timescale of equity index futures (~4 hours on 5-min bars).
- Use `sign()` function that returns 0 for zero return (flat bars), not NaN.
- Rolling sum with min_periods = 50: first 50 bars of each contract have NaN output.

**Physics interpretation (context only):**
The polar order parameter from active matter physics. Near 0 = isotropic (random walk, equal up/down). Near 1 = fully aligned (all bars moved same direction). Measures directional coherence over the recent window. Complementary to sign_concordance (which measures sub-bar directional agreement within a single parent bar).

**Edge case handling:**
- NaN log returns (first bar of contract, or gaps): fillna(0) before sign computation so the sum is well-defined.
- First 50 bars of each contract: NaN output from rolling sum.
- Apply `is_valid_bar` filter at the end.

**Expected range:**
- [0, 1] by construction.
- For 5-minute bars on equity index futures, typical median 0.08-0.20 (markets are mostly random on this horizon).
- Values above 0.4 indicate strong trending. Above 0.6 is very rare.
- Right-skewed distribution.

**Validation expectation:**
- Distribution should be right-skewed with most mass below 0.3.
- If median > 0.3, either autocorrelation is higher than expected for this asset class (possible but surprising) or window is too short.
- Correlation with bar-level sign_concordance: low (|r| < 0.2). They measure different things — polar_order is cross-bar temporal, sign_concordance is within-bar sub-bar.

---

## 2. Implementation Path

### 2.1 Output Directory Structure

```
runs/2026-04-18_physics_feature_expansion/
├── phase2_features_expanded_ES.pkl
├── phase2_features_expanded_NQ.pkl
├── phase2_features_expanded_RTY.pkl
├── PHYSICS_FEATURE_VALIDATION.md
└── (validation figures / auxiliary outputs as needed)
```

### 2.2 Script to Create

Create `src/features/physics_features.py` implementing all seven feature functions per the specifications above. Architecture:

```python
"""
Phase 2 physics feature expansion.

Adds seven candidate features derived from OHLCV + existing BVC outputs.
Does not modify any baseline' artifacts.

Usage:
    python -m src.features.physics_features --contract ES
    python -m src.features.physics_features --contract NQ
    python -m src.features.physics_features --contract RTY
    python -m src.features.physics_features --all
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Input / output paths
PHASE2_INPUT_DIR = Path("results/regime_break")
OUTPUT_DIR = Path("results/stage1_volatility/physics_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_yang_zhang(df, window=20):
    """Yang-Zhang rolling volatility"""
    # Implementation per spec section "Feature 8"
    pass

def compute_body_to_range(df):
    """|close - open| / (high - low), clipped to [0, 1]"""
    # Implementation per spec section "Feature 9"
    pass

def compute_gel_fraction(df):
    """|close - midrange| / range, clipped to [0, 0.5]"""
    # Implementation per spec section "Feature 10"
    pass

def compute_wick_asymmetry(df):
    """ln(H/body_high) - ln(body_low/L)"""
    # Implementation per spec section "Feature 11"
    pass

def compute_amihud(df):
    """|log_return| / volume"""
    # Implementation per spec section "Feature 12"
    pass

def compute_ko_W(df):
    """sigma_gk * close * volume"""
    # Implementation per spec section "Feature 13"
    pass

def compute_polar_order(df, window=50):
    """|rolling_sum(sign(log_ret), 50)| / 50"""
    # Implementation per spec section "Feature 14"
    pass

def process_contract(contract):
    """Load Phase 2 locked pickle, compute all 7 features, save expanded pickle."""
    input_path = PHASE2_INPUT_DIR / f"phase2_features_cleaned_{contract}.pkl"
    output_path = OUTPUT_DIR / f"phase2_features_expanded_{contract}.pkl"

    df = pd.read_pickle(input_path)

    # Verify required columns present
    required_cols = ['ts_event', 'instrument_id', 'open', 'high', 'low', 'close',
                     'volume', 'imbalance_t', 'sigma_gk', 'is_valid_bar']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {contract}: {missing}")

    # Compute features in order (all OHLCV-derived)
    df['sigma_yz'] = compute_yang_zhang(df)
    df['body_to_range'] = compute_body_to_range(df)
    df['gel_fraction'] = compute_gel_fraction(df)
    df['wick_asymmetry'] = compute_wick_asymmetry(df)
    df['amihud'] = compute_amihud(df)
    df['ko_W'] = compute_ko_W(df)
    df['polar_order_P'] = compute_polar_order(df)

    # Apply is_valid_bar filter uniformly (set new features to NaN where not valid)
    new_feature_cols = ['sigma_yz', 'body_to_range', 'gel_fraction',
                        'wick_asymmetry', 'amihud', 'ko_W', 'polar_order_P']
    for col in new_feature_cols:
        df[col] = df[col].where(df['is_valid_bar'], np.nan)
    
    df.to_pickle(output_path)
    print(f"Saved {output_path} with {len(df)} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", choices=["ES", "NQ", "RTY"], default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    contracts = ["ES", "NQ", "RTY"] if args.all else [args.contract]
    for c in contracts:
        process_contract(c)
```

### 2.3 OHLCV Column Availability Check

**Before running the feature computation**, verify that the Phase 2 locked pickles contain the OHLCV columns (`open`, `high`, `low`, `close`, `volume`). If they don't, join from the source 1-minute OHLCV parquet files aggregated to 5-minute bars using the same aggregation logic as the original Phase 2 pipeline.

The join must preserve alignment: row N in the Phase 2 pickle must match row N in the OHLCV data (same timestamp, same instrument_id). Verify with a random-row spot check (5 random rows per contract) before computing features.

### 2.4 Computational Cost Estimate

- Per contract: 7 features × ~400K-800K bars = trivial for pandas/numpy on modern hardware.
- Total wall time: 1-3 minutes per contract.
- Memory: no concerns at this scale.

---

## 3. Validation Checks (All Six Must Pass)

After computing features, run these six checks and write the results to `PHYSICS_FEATURE_VALIDATION.md`.

### Check 1: No All-NaN Features

For each (contract, feature) pair, verify that not all values are NaN.

Report: a 3-contract × 7-feature table showing the count of non-NaN values and the total row count per contract.

### Check 2: Summary Statistics Match Expected Ranges

For each (contract, feature), compute mean, median, std, min, max, 1st percentile, 99th percentile. Compare against the "Expected range" and "Validation expectation" sections of each feature specification above.

Report: a 7-feature summary-statistics table per contract, with a "Flag" column indicating any deviation from expected range.

Specific expected values to verify:
- `body_to_range` median 0.30-0.55 per contract
- `gel_fraction` median 0.20-0.30 per contract, max ≤ 0.5
- `wick_asymmetry` mean within ±0.0001 per contract
- `polar_order_P` median 0.08-0.20 per contract, max ≤ 1.0

### Check 3: Full 14-Feature Correlation Matrix (Per Contract)

Compute Pearson correlation matrix on valid bars (`is_valid_bar == True` AND no NaN in any of the 14 features). Report the full 14×14 matrix per contract.

Flag any pair with |r| > 0.85 as potentially redundant. Specific pairs to watch:
- `body_to_range` vs `sign_concordance`: expect |r| < 0.4 (both measure directional conviction through different mechanisms)
- `gel_fraction` vs `clv_mean`: expect |r| around 0.5 (both measure close location vs bar extremes, related but not identical)
- `body_to_range` vs `gel_fraction`: expect |r| 0.3-0.5
- `polar_order_P` vs `imbalance_t`: expect |r| < 0.3
- `sigma_yz` vs `sigma_gk`: expect |r| 0.5-0.8 (if > 0.95, flag as redundant)
- `ko_W` vs `volume`: expect |r| > 0.7
- `amihud` vs `volume`: expect |r| negative, magnitude 0.2-0.5

### Check 4: Sign Concordance Orthogonality Preserved

The most important validation. The locked finding is that `sign_concordance` has |r| < 0.05 with `imbalance_t` on all three contracts. This survived four analytical transformations in the BVC pipeline and is the signature of a real structural property.

Verify that after adding the seven new features, this orthogonality is still intact on the expanded feature set. Specifically:
- `sign_concordance` vs `imbalance_t`: |r| < 0.05 per contract (locked baseline, must not change)
- `sign_concordance` vs all 7 new features: report correlations, flag if |r| > 0.4

If the sign_concordance vs imbalance_t correlation has changed since the baseline value, something in the pipeline is broken — diagnose before proceeding.

### Check 5: Feature Availability Rate

For each (contract, feature), compute `(count of non-NaN values where is_valid_bar == True) / (count of valid bars)`. This is the feature availability rate — what fraction of valid bars have a usable value for this feature.

Expected availability rates:
- `body_to_range`, `gel_fraction`, `wick_asymmetry`, `sigma_yz`, `ko_W`, `polar_order_P`: > 95% on all contracts
- `amihud`: > 90% on ES and NQ, > 85% on RTY (allowance for zero-volume bars)

Flag any feature with availability < 90% on any contract.

### Check 6: Temporal Stability

Split each contract's valid bars into two halves by time:
- First half: earliest 50% of bars
- Second half: latest 50% of bars

For each (contract, feature), compute mean and std on each half. Compute the normalized shift `(mean_second - mean_first) / std_combined`.

Flag any feature with |normalized shift| > 2.0 — this indicates meaningful non-stationarity that could hurt LightGBM generalization.

Note: for ES, the "first half" is post-2020 only (pre-2020 bars are gated out by the regime filter). This is fine — we're checking stability within the locked regime window.

Report: a 3-contract × 7-feature table of normalized shifts, with flags.

---

## 4. Report Format (`PHYSICS_FEATURE_VALIDATION.md`)

Required sections:

### Section 1: Executive Summary
- One paragraph: did all six checks pass?

### Section 2: Pipeline Verification
- Confirm OHLCV columns were present in Phase 2 pickles (or document the join if needed)
- Row counts and column counts for each contract before and after feature addition
- Spot check of 3 random bars per contract showing values of all 14 features

### Section 3: Check 1 Results
- All-NaN test results

### Section 4: Check 2 Results
- Summary statistics tables per contract
- Any flags

### Section 5: Check 3 Results
- Full 14×14 correlation matrix per contract (rendered as tables or heatmaps)
- Any |r| > 0.85 pairs flagged

### Section 6: Check 4 Results
- Sign concordance orthogonality values (locked baseline vs current)
- Sign concordance vs all 7 new features correlations

### Section 7: Check 5 Results
- Feature availability rate table

### Section 8: Check 6 Results
- Temporal stability table with normalized shifts

### Section 9: Decision
- Overall pass/fail recommendation
- Any features to investigate
- Any data quality issues surfaced

---

## 5. Summary

Seven candidate features with exact formulas, all computable from OHLCV + locked BVC outputs. Seven Python functions. One output pickle per contract. One validation report. Six validation checks.

End of specification.
