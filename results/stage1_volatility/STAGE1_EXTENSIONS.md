# Stage 1 — Extensions (Quantile + Per-Contract SHAP)

**Generated:** 2026-04-18T12:54:07.047595+00:00
**Scope:** two diagnostic additions to the Stage 1 run. Locked artifacts (`predictions.parquet`, `fold_metrics.parquet`, `feature_importance.parquet`, `STAGE1_RESULTS.md`) were **not** modified.

This repository publishes Stage 1 only; no Stage 2 work is in scope.

## Extension A — Quantile regression at τ = 0.9

**Setup.** Same 31 folds, same 14 core + 3 HAR-analog + `contract` categorical features, same target (forward 12-bar log σ_rv). Only change: LightGBM `objective=quantile`, `alpha=0.9`. Per-fold internal-val split (last 10%), early stopping on quantile loss, same 5000-round cap. 585,202 pooled test rows.

**Artifact:** `predictions_q90.parquet` (sibling of `predictions.parquet`).

### A.1 Empirical coverage check (sanity on τ)

For a well-calibrated τ=0.9 quantile estimator, P(y ≤ ŷ) ≈ 0.90.

| scope | coverage P(y ≤ ŷ_q90) |
|:------|----------------------:|
| pooled | 0.901 |
| ES | 0.897 |
| NQ | 0.901 |
| RTY | 0.905 |

### A.2 Residual bias by target decile (pooled, deciles assigned within fold)

Residual = y − ŷ. For the mean model, positive residuals in high deciles indicate systematic underprediction of the top of the vol distribution. For the quantile (τ=0.9) model, E[resid] should be negative for deciles 1–9 and near zero for decile 10.

| decile | n | mean(target) | μ resid (mean model) | σ resid (mean) | μ resid (q90 model) | σ resid (q90) |
|------:|---:|-------------:|---------------------:|---------------:|--------------------:|--------------:|
| 1 | 58,531 | -7.221 | -0.5147 | 0.3773 | -1.1229 | 0.4224 |
| 2 | 58,519 | -6.790 | -0.3148 | 0.3304 | -0.8901 | 0.3586 |
| 3 | 58,517 | -6.554 | -0.2236 | 0.3128 | -0.7727 | 0.3305 |
| 4 | 58,519 | -6.365 | -0.1604 | 0.3026 | -0.6866 | 0.3135 |
| 5 | 58,524 | -6.198 | -0.0916 | 0.3027 | -0.6004 | 0.3058 |
| 6 | 58,509 | -6.038 | -0.0260 | 0.3006 | -0.5170 | 0.2953 |
| 7 | 58,517 | -5.872 | +0.0436 | 0.3116 | -0.4316 | 0.2962 |
| 8 | 58,519 | -5.684 | +0.1305 | 0.3276 | -0.3326 | 0.2994 |
| 9 | 58,517 | -5.452 | +0.2410 | 0.3571 | -0.2161 | 0.3144 |
| 10 | 58,530 | -5.011 | +0.5153 | 0.5033 | +0.0488 | 0.4454 |

### A.3 Top-decile bias test

- Mean-model top-decile μ resid: **+0.5153** (matches `STAGE1_RESULTS.md §6` "high-vol (top decile)" bias of +0.515)
- Quantile-model top-decile μ resid: **+0.0488**
- Threshold: top-decile q90 μ resid < +0.20 → **PASS**

**Finding.** The τ=0.9 quantile objective substantially reduces the systematic underprediction of high-vol bars. The remaining mean residual in the top decile is the expected behaviour of a correctly-calibrated 90th-percentile estimator: on average it sits at the 90th-percentile of the conditional target distribution, so conditionally on being in the top 10%, some positive residual is expected.

## Extension B — Per-contract SHAP on fold 15

**Setup.** Fold 15 (train window = anchor → 2023-07-02, test window = 2023-07-03 → 2023-09-02, train=385,966, test=18,218). Retrained the mean-regression model with identical params to the Stage 1 run (reproduced `best_iter=1198`), then computed TreeSHAP on the full test set via LightGBM native `pred_contrib=True`. No external `shap` dependency.

**Artifact:** `shap_fold15.parquet` (18,218 rows × 18 feature SHAP columns + expected value + 4 identifiers).

### B.1 Top-10 features per contract by mean |SHAP|

#### ES

| rank | feature | mean &#124;SHAP&#124; | share of total |
|---:|:--------|---------------------:|---------------:|
| 1 | `sigma_yz` | 0.4415 | 30.7% |
| 2 | `sigma_gk_short` | 0.2084 | 14.5% |
| 3 | `sigma_gk_lag1` | 0.1932 | 13.4% |
| 4 | `sigma_gk_long` | 0.1138 | 7.9% |
| 5 | `amihud` | 0.0713 | 5.0% |
| 6 | `z` | 0.0708 | 4.9% |
| 7 | `body_to_range` | 0.0583 | 4.1% |
| 8 | `imbalance_t` | 0.0544 | 3.8% |
| 9 | `der` | 0.0469 | 3.3% |
| 10 | `wick_asymmetry` | 0.0312 | 2.2% |

#### NQ

| rank | feature | mean &#124;SHAP&#124; | share of total |
|---:|:--------|---------------------:|---------------:|
| 1 | `sigma_yz` | 0.2964 | 28.1% |
| 2 | `sigma_gk_short` | 0.1344 | 12.7% |
| 3 | `sigma_gk_lag1` | 0.1202 | 11.4% |
| 4 | `z` | 0.0746 | 7.1% |
| 5 | `body_to_range` | 0.0594 | 5.6% |
| 6 | `imbalance_t` | 0.0585 | 5.6% |
| 7 | `sigma_gk_long` | 0.0553 | 5.2% |
| 8 | `der` | 0.0496 | 4.7% |
| 9 | `amihud` | 0.0420 | 4.0% |
| 10 | `ko_W` | 0.0343 | 3.3% |

#### RTY

| rank | feature | mean &#124;SHAP&#124; | share of total |
|---:|:--------|---------------------:|---------------:|
| 1 | `sigma_yz` | 0.3004 | 27.1% |
| 2 | `sigma_gk_short` | 0.1421 | 12.8% |
| 3 | `sigma_gk_lag1` | 0.1274 | 11.5% |
| 4 | `z` | 0.0682 | 6.2% |
| 5 | `sigma_gk_long` | 0.0612 | 5.5% |
| 6 | `body_to_range` | 0.0585 | 5.3% |
| 7 | `amihud` | 0.0535 | 4.8% |
| 8 | `imbalance_t` | 0.0514 | 4.6% |
| 9 | `der` | 0.0497 | 4.5% |
| 10 | `ko_W` | 0.0441 | 4.0% |

### B.2 Full feature × contract mean |SHAP| matrix

| feature | ES | NQ | RTY |
|:--------|---:|---:|----:|
| `sigma_yz` | 0.4415 | 0.2964 | 0.3004 |
| `sigma_gk_short` | 0.2084 | 0.1344 | 0.1421 |
| `sigma_gk_lag1` | 0.1932 | 0.1202 | 0.1274 |
| `sigma_gk_long` | 0.1138 | 0.0553 | 0.0612 |
| `z` | 0.0708 | 0.0746 | 0.0682 |
| `body_to_range` | 0.0583 | 0.0594 | 0.0585 |
| `amihud` | 0.0713 | 0.0420 | 0.0535 |
| `imbalance_t` | 0.0544 | 0.0585 | 0.0514 |
| `der` | 0.0469 | 0.0496 | 0.0497 |
| `ko_W` | 0.0268 | 0.0343 | 0.0441 |
| `wick_asymmetry` | 0.0312 | 0.0263 | 0.0280 |
| `contract` | 0.0252 | 0.0183 | 0.0333 |
| `clv_var` | 0.0273 | 0.0203 | 0.0278 |
| `subbar_imbalance` | 0.0237 | 0.0250 | 0.0205 |
| `gel_fraction` | 0.0197 | 0.0209 | 0.0207 |
| `clv_mean` | 0.0127 | 0.0111 | 0.0131 |
| `polar_order_P` | 0.0085 | 0.0056 | 0.0058 |
| `sign_concordance` | 0.0029 | 0.0025 | 0.0028 |

### B.3 RTY non-vol-state hypothesis

**Vol-state features** (4): `sigma_yz`, `sigma_gk_lag1`, `sigma_gk_short`, `sigma_gk_long`
**Non-vol-state features** (13): `z`, `imbalance_t`, `der`, `sign_concordance`, `clv_mean`, `clv_var`, `subbar_imbalance`, `body_to_range`, `gel_fraction`, `wick_asymmetry`, `amihud`, `ko_W`, `polar_order_P`

`contract` (categorical routing variable) is excluded from the share denominator so we compare information channels.

| contract | total mean &#124;SHAP&#124; | non-vol-state mean &#124;SHAP&#124; | share |
|:---------|---:|---:|---:|
| ES | 1.4116 | 0.4546 | 32.2% |
| NQ | 1.0365 | 0.4302 | 41.5% |
| RTY | 1.0751 | 0.4441 | 41.3% |

**Hypothesis:** RTY non-vol-state share > ES share **and** > NQ share.
- RTY = 41.3%, ES = 32.2%, NQ = 41.5%
- Result: **NOT CONFIRMED**

**Interpretation.** On fold 15, RTY does not show a systematically higher non-vol-state SHAP share than ES/NQ. The Andersen-Bondarenko pattern observed pooled in `STAGE1_RESULTS.md §5` (BVC never entering top-5 by gain) appears to hold approximately uniformly across the three contracts.

## Scope

Extensions complete. Locked Stage 1 artifacts are untouched. Stage 1 artifacts are documented here; this repository publishes Stage 1 only.
