# Methodology Overview

This document is a standalone summary of the full pipeline. It mirrors
the structure of the paper at a level of detail sufficient for readers
to map results back to source modules.

## 1. Student-t Bulk Volume Classification

Bulk Volume Classification (BVC) classifies the volume of a bar into
buy/sell components using only bar-level information (no trade-level
aggressor flag). The standard Gaussian formulation

$$V_t^B = V_t \cdot \Phi(z_t), \qquad V_t^S = V_t \cdot (1 - \Phi(z_t))$$

uses $z_t = \Delta P_t / \sigma_t$, a standardized bar return. The
Student-t generalization replaces $\Phi$ with the Student-t CDF
$T_{\nu}(z)$, producing a heavy-tailed classifier calibrated to the
empirical tail of 5-minute equity index futures returns.

**Implementation:** `src/features/student_t_bvc.py` exposes
`aggregate_to_bars`, `compute_sigma_causal`, `compute_z`,
`fit_student_t`, and `pit_uniformity`. The core imbalance feature is

```
imbalance_t = 2 * t.cdf(z_t, df=nu, loc=loc, scale=scale) − 1
```

which lives on $[-1, +1]$ and has the interpretation of a signed
buy/sell fraction.

## 2. Causal calibration of $\nu$ and $\sigma$

Calibration is *strictly causal*: at every bar $t$, the Student-t
parameters used to compute $z_t$ and $\text{imbalance}_t$ are estimated
using only information from strictly before $t$. This eliminates a
subtle look-ahead that arises when $\sigma_t$ is computed as a
centered rolling estimate: a concurrent $\sigma$ absorbs the current
bar's surprise into the denominator and systematically
under-represents tail events, biasing $\nu$ upward.

Compared to the concurrent baseline, causal calibration lowers $\nu$
from roughly 6.8 to roughly 4.4 across ES/NQ/RTY on 1-year data, and
to 3.0–3.5 on 15-year data — i.e. the *honest* tail is substantially
heavier than the concurrent fit reports.

**Implementation:** `src/calibration/phase1_causal.py` (main driver),
`phase1_descriptive.py` (full-sample reference fit),
`phase4_final_validation.py` (final-lock diagnostics),
`sensitivity_grid.py` (span/warmup grid), and the `diagnostics/`
submodule (boundary, grid, per-year, reproducibility, convergence,
session-isolated tests).

PIT uniformity, shoulder-CDF residuals, and skewness-by-|z|-bucket
diagnostics are reported in `results/calibration_tables/`.

## 3. Regime break (2020)

A structural break in $\nu$ is detected around 2020. The robustness
of this finding is established across:

- **Break-date sensitivity** (`src/regime/es_breakdate.py`): varying
  the candidate break date within a ±6-month window preserves the
  post-break $\nu$ estimate within its own CI.
- **Extended window** (`es_extended.py`): extending the pre-break
  window to 2008 does not materially shift the pre-break $\nu$.
- **Monthly stabilization** (`es_monthly_stabilization.py`): monthly
  $\nu$ re-fits stabilize by mid-2020 and remain stable through 2025.
- **Rolling $\nu$** (`rolling_nu.py`): rolling one-year-window
  estimates show a single monotone transition rather than oscillation.
- **Training-window adoption** (`adopt_training_window.py`): the
  downstream analyses use a training window that begins at the
  identified break.

## 4. Physics-motivated features

Seven new features are added to the seven baseline BVC outputs,
producing a 14-feature set. Each new feature is derived from OHLCV
geometry or intra-bar flow and is designed to be orthogonal to the
existing BVC measures. See `docs/PHYSICS_FEATURES.md` for the
mathematical definitions and
`results/stage1_volatility/physics_features/PHYSICS_FEATURE_VALIDATION.md`
for the validation of availability, correlation structure, and
temporal stability. A candidate 15th feature (`v_star_C`) was pruned
after an initial validation surfaced redundancy with two existing
features (DER and `body_to_range`); the pruning reasoning is
documented in §10 of the validation report.

**Implementation:** `src/features/physics_features.py` and
`src/features/physics_validation.py`.

## 5. Stage-1 volatility prediction

Stage-1 is a walk-forward forecasting experiment for next-bar
realized volatility. The 14-feature set is trained via gradient
boosting on expanding windows and evaluated against three causal
baselines: a rolling Garman-Klass volatility, a heterogeneous
auto-regressive (HAR) model on log-RV, and a naive persistence
baseline.

**Implementation:** `src/volatility_prediction/stage1_training.py`
(training driver), `baselines.py` (GK/HAR/naive),
`metrics.py` (MSE, QLIKE, MZ coefficients),
`extensions.py` + `extensions_report.py` (extended robustness
tables), `validate_scaffold.py` (sanity thresholds),
`results_report.py` (final report generator).

## 6. Directional null result

A structured 72-cell test evaluates whether the sign of the BVC
imbalance at flow-event bars predicts the sign of the forward return
at $h \in \{1, 3, 6, 12\}$ bars ahead, across three gate
constructions and two conviction tiers. No configuration produces
accuracy meaningfully above the 48–49% lag-1 mean-reverting baseline.
The null result is explained structurally: at the identity level,
$\operatorname{sign}(\text{imbalance}_t) \equiv \operatorname{sign}(\text{log\_ret}_t)$
for essentially 100% of bars whenever the Student-t `loc` parameter
is near zero, so the gated subset inherits exactly the directional
property of the underlying log-return distribution.

**Implementation:** `src/directional_test/phase3c.py`. Full results:
`results/phase3c_directional/DIRECTIONAL_TEST_RESULTS.md`.

## 7. Reproducibility

All stochastic components use fixed seeds. All rolling statistics are
session-isolated: a new session begins at a change of `instrument_id`
or a `ts.diff() > 15 min` gap, and the first (window − 1) bars of
each session yield NaN. Walk-forward evaluations use frozen warmup
windows that end before the evaluation period begins. Re-running any
stage on the same input data and Python environment produces
byte-identical artifacts.
