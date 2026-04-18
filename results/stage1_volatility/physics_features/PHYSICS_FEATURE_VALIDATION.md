# Physics Feature Validation Report

**Date:** 2026-04-18
**Feature set:** 7 baseline BVC + 7 new physics features = 14 features (a candidate 15th feature, `v_star_C`, was pruned — see §10).

## 1. Executive Summary

**Overall status:** ALL SIX CHECKS PASS ✓

- **Check 1 — No all-NaN features:** PASS ✓
- **Check 2 — Summary stats within expected ranges:** PASS ✓
- **Check 3 — No novel |r| > 0.85 pairs (structural z/imb pairs excluded):** PASS ✓
- **Check 4 — sign_concordance ⟂ imbalance_t preserved (|r|<0.05):** PASS ✓
- **Check 5 — All feature availability ≥ 90%:** PASS ✓
- **Check 6 — Temporal stability (|norm shift| ≤ 2.0):** PASS ✓

## 2. Pipeline Verification

**OHLCV availability:** `open`, `high`, `low`, `close`, `volume` are present in every input bar. `imbalance_t`, `sigma_gk`, `is_valid_bar`, `instrument_id` are computed upstream by the baseline BVC pipeline.

**Row and column counts (input → output):**

| contract | total_rows | valid_rows | columns_in | columns_out | features_added |
|---|---|---|---|---|---|
| ES | 1,604,333 | 359,581 | 24 | 31 | 7 |
| NQ | 1,387,546 | 811,012 | 24 | 31 | 7 |
| RTY | 675,791 | 423,776 | 24 | 31 | 7 |

**Spot check — 3 random valid bars per contract (all 14 features):**

| contract | ts | instrument_id | z | imbalance_t | der | sign_concordance | clv_mean | clv_var | subbar_imbalance | sigma_yz | body_to_range | gel_fraction | wick_asymmetry | amihud | ko_W | polar_order_P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ES | 2023-09-22 14:40:00+00:00 | 314,863 | 0 | -0.0136 | 0 | 0.5000 | 0.0298 | 0.3310 | 0.0079 | 7.6509e-04 | 0 | 0.4333 | -7.4051e-04 | 0 | 6.1589e+04 | 0.0200 |
| ES | 2024-07-12 10:05:00+00:00 | 118 | 0.5278 | 0.4267 | 0.4000 | 0.6000 | 0.2000 | 0.7000 | 0.1026 | 1.6759e-04 | 0.5000 | 0 | 8.8578e-05 | 1.7235e-07 | 486.9650 | 0.1000 |
| ES | 2023-05-08 07:25:00+00:00 | 95,414 | -0.7843 | -0.6083 | 0.4444 | 0.8000 | -0.1733 | 0.5280 | -0.1861 | 2.7720e-04 | 0.8000 | 0.3000 | -6.0199e-05 | 3.0708e-07 | 1249.3570 | 0.0200 |
| NQ | 2023-05-12 16:00:00+00:00 | 3,522 | -2.0751 | -0.9359 | 0.8431 | 1.0000 | -0.5256 | 0.1275 | -0.3949 | 7.6354e-04 | 0.8958 | 0.4688 | 7.4498e-05 | 2.1630e-07 | 7.6827e+04 | 0.1200 |
| NQ | 2024-10-15 15:10:00+00:00 | 106,364 | 1.5249 | 0.8559 | 0.7155 | 0.8000 | 0.4372 | 0.2291 | 0.2538 | 0.0014 | 0.8341 | 0.5000 | -4.5408e-04 | 4.0275e-07 | 1.7279e+05 | 0.1200 |
| NQ | 2022-03-14 10:05:00+00:00 | 2,895 | -0.8147 | -0.6203 | 1.0000 | 0.8000 | -0.2568 | 0.2512 | -0.0965 | 0.0013 | 0.7143 | 0.3333 | -7.4944e-05 | 2.0239e-06 | 1.0011e+04 | 0 |
| RTY | 2019-06-03 05:15:00+00:00 | 1,504 | 0 | -0.0092 | 0 | 0.5000 | -0.4000 | 0.8000 | -0.0601 | 2.1373e-04 | 0 | 0 | -1.8783e-08 | 1.8521e-06 | 11.3910 | 0.0200 |
| RTY | 2024-01-30 12:45:00+00:00 | 7,062 | 0.3308 | 0.2658 | 0.1429 | 0.4000 | 0.1200 | 0.6276 | 0.0583 | 3.2502e-04 | 0.1538 | 0.1154 | 2.4751e-04 | 9.0047e-07 | 99.7584 | 0.1400 |
| RTY | 2021-07-09 19:55:00+00:00 | 817 | -0.2607 | -0.2272 | 0.1500 | 0.4000 | 0.3197 | 0.4108 | 0.0283 | 4.8563e-04 | 0.1429 | 0.1667 | 1.7561e-04 | 1.2946e-08 | 1.1720e+04 | 0.1200 |

## 3. Check 1 — No All-NaN Features

Non-NaN counts per (contract, feature) restricted to `is_valid_bar == True`.

| contract | feature | n_valid | n_nonan | all_nan |
|---|---|---|---|---|
| ES | sigma_yz | 359,581 | 359,581 | 0 |
| ES | body_to_range | 359,581 | 359,581 | 0 |
| ES | gel_fraction | 359,581 | 359,581 | 0 |
| ES | wick_asymmetry | 359,581 | 359,581 | 0 |
| ES | amihud | 359,581 | 359,581 | 0 |
| ES | ko_W | 359,581 | 359,581 | 0 |
| ES | polar_order_P | 359,581 | 359,581 | 0 |
| NQ | sigma_yz | 811,012 | 811,012 | 0 |
| NQ | body_to_range | 811,012 | 811,012 | 0 |
| NQ | gel_fraction | 811,012 | 811,012 | 0 |
| NQ | wick_asymmetry | 811,012 | 811,012 | 0 |
| NQ | amihud | 811,012 | 811,012 | 0 |
| NQ | ko_W | 811,012 | 811,012 | 0 |
| NQ | polar_order_P | 811,012 | 811,012 | 0 |
| RTY | sigma_yz | 423,776 | 423,776 | 0 |
| RTY | body_to_range | 423,776 | 423,776 | 0 |
| RTY | gel_fraction | 423,776 | 423,776 | 0 |
| RTY | wick_asymmetry | 423,776 | 423,776 | 0 |
| RTY | amihud | 423,776 | 423,776 | 0 |
| RTY | ko_W | 423,776 | 423,776 | 0 |
| RTY | polar_order_P | 423,776 | 423,776 | 0 |

**Check 1 result:** PASS — no all-NaN features.

## 4. Check 2 — Summary Statistics vs Expected Ranges

Statistics computed on valid bars only. `flag` column is non-empty when the realized statistic deviates from the spec §1 expectation for that feature.

### ES

| feature | mean | median | std | min | p01 | p99 | max | flag |
|---|---|---|---|---|---|---|---|---|
| sigma_yz | 6.3061e-04 | 4.8251e-04 | 5.5600e-04 | 0 | 1.0867e-04 | 0.0028 | 0.0100 |  |
| body_to_range | 0.4667 | 0.4737 | 0.2715 | 0 | 0 | 1.0000 | 1.0000 |  |
| gel_fraction | 0.2720 | 0.2778 | 0.1528 | 0 | 0 | 0.5000 | 0.5000 |  |
| wick_asymmetry | -1.0329e-05 | -3.1935e-08 | 4.1763e-04 | -0.0094 | -0.0012 | 0.0012 | 0.0175 |  |
| amihud | 1.6400e-06 | 1.0364e-07 | 1.3867e-05 | 0 | 0 | 3.1827e-05 | 0.0014 |  |
| ko_W | 2.6914e+04 | 4451.6610 | 6.5209e+04 | 0.4134 | 19.1877 | 2.6949e+05 | 4.2887e+06 |  |
| polar_order_P | 0.1030 | 0.0800 | 0.0784 | 0 | 0 | 0.3400 | 0.6000 |  |

### NQ

| feature | mean | median | std | min | p01 | p99 | max | flag |
|---|---|---|---|---|---|---|---|---|
| sigma_yz | 6.6948e-04 | 5.2969e-04 | 5.2268e-04 | 0 | 1.1704e-04 | 0.0026 | 0.0117 |  |
| body_to_range | 0.4689 | 0.4737 | 0.2723 | 0 | 0 | 1.0000 | 1.0000 |  |
| gel_fraction | 0.2721 | 0.2778 | 0.1530 | 0 | 0 | 0.5000 | 0.5000 |  |
| wick_asymmetry | -1.0754e-05 | -2.8519e-08 | 4.2355e-04 | -0.0120 | -0.0013 | 0.0012 | 0.0195 |  |
| amihud | 1.6770e-06 | 3.5531e-07 | 1.3055e-05 | 0 | 0 | 2.0896e-05 | 0.0031 |  |
| ko_W | 2.2699e+04 | 3168.2707 | 6.1237e+04 | 0.5221 | 10.8909 | 2.7849e+05 | 6.1763e+06 |  |
| polar_order_P | 0.1014 | 0.0800 | 0.0773 | 0 | 0 | 0.3200 | 0.6000 |  |

### RTY

| feature | mean | median | std | min | p01 | p99 | max | flag |
|---|---|---|---|---|---|---|---|---|
| sigma_yz | 8.1269e-04 | 6.4405e-04 | 6.2821e-04 | 0 | 1.4316e-04 | 0.0030 | 0.0132 |  |
| body_to_range | 0.4860 | 0.5000 | 0.2771 | 0 | 0 | 1.0000 | 1.0000 |  |
| gel_fraction | 0.2829 | 0.2949 | 0.1553 | 0 | 0 | 0.5000 | 0.5000 |  |
| wick_asymmetry | -8.7911e-06 | -3.2430e-08 | 5.1791e-04 | -0.0114 | -0.0015 | 0.0015 | 0.0220 |  |
| amihud | 3.1457e-06 | 1.1111e-06 | 8.2213e-06 | 0 | 0 | 2.8639e-05 | 0.0014 |  |
| ko_W | 2013.6460 | 387.1795 | 4997.3180 | 0.2027 | 3.0718 | 2.0700e+04 | 4.2330e+05 |  |
| polar_order_P | 0.1052 | 0.0800 | 0.0800 | 0 | 0 | 0.3400 | 0.6000 |  |

**No entries flagged.**

**Check 2 result:** PASS

## 5. Check 3 — Full 14-Feature Correlation Matrix

Pearson correlation on rows where `is_valid_bar == True` and none of the 14 features are NaN. Pairs with |r| > 0.85 are flagged.

### ES

| feature | z | imbalance_t | der | sign_concordance | clv_mean | clv_var | subbar_imbalance | sigma_yz | body_to_range | gel_fraction | wick_asymmetry | amihud | ko_W | polar_order_P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **z** | +1.000 | +0.901 | +0.006 | -0.011 | +0.617 | -0.000 | +0.867 | +0.001 | +0.001 | +0.017 | +0.059 | -0.001 | -0.008 | +0.019 |
| **imbalance_t** | +0.901 | +1.000 | +0.023 | -0.000 | +0.666 | -0.006 | +0.872 | -0.002 | +0.019 | +0.029 | +0.045 | -0.001 | -0.007 | +0.024 |
| **der** | +0.006 | +0.023 | +1.000 | +0.315 | +0.016 | -0.264 | +0.014 | -0.000 | +0.843 | +0.419 | +0.010 | +0.142 | +0.003 | -0.005 |
| **sign_concordance** | -0.011 | -0.000 | +0.315 | +1.000 | +0.008 | -0.269 | -0.004 | +0.147 | +0.432 | +0.262 | -0.007 | -0.073 | +0.129 | +0.016 |
| **clv_mean** | +0.617 | +0.666 | +0.016 | +0.008 | +1.000 | -0.018 | +0.632 | -0.008 | +0.012 | +0.022 | -0.085 | -0.001 | -0.008 | +0.016 |
| **clv_var** | -0.000 | -0.006 | -0.264 | -0.269 | -0.018 | +1.000 | -0.001 | -0.188 | -0.166 | -0.082 | +0.005 | +0.007 | -0.159 | -0.030 |
| **subbar_imbalance** | +0.867 | +0.872 | +0.014 | -0.004 | +0.632 | -0.001 | +1.000 | +0.007 | +0.011 | +0.017 | +0.095 | +0.003 | -0.002 | +0.017 |
| **sigma_yz** | +0.001 | -0.002 | -0.000 | +0.147 | -0.008 | -0.188 | +0.007 | +1.000 | +0.004 | +0.003 | +0.004 | +0.099 | +0.564 | +0.035 |
| **body_to_range** | +0.001 | +0.019 | +0.843 | +0.432 | +0.012 | -0.166 | +0.011 | +0.004 | +1.000 | +0.508 | +0.012 | +0.113 | +0.010 | -0.004 |
| **gel_fraction** | +0.017 | +0.029 | +0.419 | +0.262 | +0.022 | -0.082 | +0.017 | +0.003 | +0.508 | +1.000 | -0.001 | +0.086 | -0.002 | -0.003 |
| **wick_asymmetry** | +0.059 | +0.045 | +0.010 | -0.007 | -0.085 | +0.005 | +0.095 | +0.004 | +0.012 | -0.001 | +1.000 | +0.003 | +0.003 | +0.003 |
| **amihud** | -0.001 | -0.001 | +0.142 | -0.073 | -0.001 | +0.007 | +0.003 | +0.099 | +0.113 | +0.086 | +0.003 | +1.000 | -0.046 | -0.007 |
| **ko_W** | -0.008 | -0.007 | +0.003 | +0.129 | -0.008 | -0.159 | -0.002 | +0.564 | +0.010 | -0.002 | +0.003 | -0.046 | +1.000 | +0.034 |
| **polar_order_P** | +0.019 | +0.024 | -0.005 | +0.016 | +0.016 | -0.030 | +0.017 | +0.035 | -0.004 | -0.003 | +0.003 | -0.007 | +0.034 | +1.000 |

**Flagged |r|>0.85 pairs:**

| feature_a | feature_b | r | expected by construction |
|---|---|---|---|
| z | imbalance_t | +0.9013 | yes |
| z | subbar_imbalance | +0.8668 | yes |
| imbalance_t | subbar_imbalance | +0.8720 | yes |

### NQ

| feature | z | imbalance_t | der | sign_concordance | clv_mean | clv_var | subbar_imbalance | sigma_yz | body_to_range | gel_fraction | wick_asymmetry | amihud | ko_W | polar_order_P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **z** | +1.000 | +0.903 | +0.010 | +0.003 | +0.616 | -0.006 | +0.860 | -0.003 | +0.006 | +0.015 | +0.070 | +0.001 | -0.009 | +0.014 |
| **imbalance_t** | +0.903 | +1.000 | +0.023 | +0.012 | +0.662 | -0.010 | +0.862 | -0.003 | +0.022 | +0.025 | +0.061 | +0.001 | -0.006 | +0.017 |
| **der** | +0.010 | +0.023 | +1.000 | +0.307 | +0.015 | -0.261 | +0.015 | -0.026 | +0.849 | +0.413 | +0.011 | +0.132 | +0.000 | -0.007 |
| **sign_concordance** | +0.003 | +0.012 | +0.307 | +1.000 | +0.015 | -0.272 | +0.006 | +0.158 | +0.398 | +0.224 | -0.008 | -0.066 | +0.128 | +0.028 |
| **clv_mean** | +0.616 | +0.662 | +0.015 | +0.015 | +1.000 | -0.017 | +0.621 | -0.003 | +0.014 | +0.019 | -0.084 | -0.000 | -0.001 | +0.010 |
| **clv_var** | -0.006 | -0.010 | -0.261 | -0.272 | -0.017 | +1.000 | -0.005 | -0.182 | -0.155 | -0.070 | +0.003 | +0.013 | -0.144 | -0.033 |
| **subbar_imbalance** | +0.860 | +0.862 | +0.015 | +0.006 | +0.621 | -0.005 | +1.000 | -0.003 | +0.015 | +0.014 | +0.114 | +0.002 | -0.006 | +0.015 |
| **sigma_yz** | -0.003 | -0.003 | -0.026 | +0.158 | -0.003 | -0.182 | -0.003 | +1.000 | -0.011 | -0.015 | -0.015 | +0.046 | +0.591 | +0.049 |
| **body_to_range** | +0.006 | +0.022 | +0.849 | +0.398 | +0.014 | -0.155 | +0.015 | -0.011 | +1.000 | +0.498 | +0.013 | +0.111 | +0.008 | -0.005 |
| **gel_fraction** | +0.015 | +0.025 | +0.413 | +0.224 | +0.019 | -0.070 | +0.014 | -0.015 | +0.498 | +1.000 | -0.001 | +0.081 | -0.001 | -0.008 |
| **wick_asymmetry** | +0.070 | +0.061 | +0.011 | -0.008 | -0.084 | +0.003 | +0.114 | -0.015 | +0.013 | -0.001 | +1.000 | +0.003 | -0.007 | +0.000 |
| **amihud** | +0.001 | +0.001 | +0.132 | -0.066 | -0.000 | +0.013 | +0.002 | +0.046 | +0.111 | +0.081 | +0.003 | +1.000 | -0.041 | -0.004 |
| **ko_W** | -0.009 | -0.006 | +0.000 | +0.128 | -0.001 | -0.144 | -0.006 | +0.591 | +0.008 | -0.001 | -0.007 | -0.041 | +1.000 | +0.039 |
| **polar_order_P** | +0.014 | +0.017 | -0.007 | +0.028 | +0.010 | -0.033 | +0.015 | +0.049 | -0.005 | -0.008 | +0.000 | -0.004 | +0.039 | +1.000 |

**Flagged |r|>0.85 pairs:**

| feature_a | feature_b | r | expected by construction |
|---|---|---|---|
| z | imbalance_t | +0.9026 | yes |
| z | subbar_imbalance | +0.8597 | yes |
| imbalance_t | subbar_imbalance | +0.8617 | yes |

### RTY

| feature | z | imbalance_t | der | sign_concordance | clv_mean | clv_var | subbar_imbalance | sigma_yz | body_to_range | gel_fraction | wick_asymmetry | amihud | ko_W | polar_order_P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **z** | +1.000 | +0.888 | +0.004 | -0.001 | +0.602 | -0.000 | +0.815 | -0.001 | +0.003 | +0.009 | +0.058 | -0.000 | -0.002 | +0.008 |
| **imbalance_t** | +0.888 | +1.000 | +0.014 | +0.007 | +0.660 | -0.006 | +0.822 | -0.003 | +0.014 | +0.016 | +0.040 | +0.002 | -0.004 | +0.011 |
| **der** | +0.004 | +0.014 | +1.000 | +0.283 | +0.009 | -0.268 | +0.007 | -0.062 | +0.846 | +0.424 | +0.008 | +0.284 | -0.020 | -0.017 |
| **sign_concordance** | -0.001 | +0.007 | +0.283 | +1.000 | +0.012 | -0.268 | +0.002 | +0.193 | +0.406 | +0.234 | -0.005 | -0.059 | +0.128 | +0.035 |
| **clv_mean** | +0.602 | +0.660 | +0.009 | +0.012 | +1.000 | -0.012 | +0.617 | -0.005 | +0.009 | +0.012 | -0.085 | -0.001 | -0.004 | +0.006 |
| **clv_var** | -0.000 | -0.006 | -0.268 | -0.268 | -0.012 | +1.000 | -0.000 | -0.226 | -0.177 | -0.100 | +0.003 | +0.032 | -0.167 | -0.034 |
| **subbar_imbalance** | +0.815 | +0.822 | +0.007 | +0.002 | +0.617 | -0.000 | +1.000 | -0.003 | +0.007 | +0.008 | +0.100 | +0.002 | -0.002 | +0.008 |
| **sigma_yz** | -0.001 | -0.003 | -0.062 | +0.193 | -0.005 | -0.226 | -0.003 | +1.000 | -0.034 | -0.032 | -0.001 | -0.082 | +0.604 | +0.051 |
| **body_to_range** | +0.003 | +0.014 | +0.846 | +0.406 | +0.009 | -0.177 | +0.007 | -0.034 | +1.000 | +0.507 | +0.010 | +0.256 | -0.003 | -0.009 |
| **gel_fraction** | +0.009 | +0.016 | +0.424 | +0.234 | +0.012 | -0.100 | +0.008 | -0.032 | +0.507 | +1.000 | -0.002 | +0.167 | -0.009 | -0.009 |
| **wick_asymmetry** | +0.058 | +0.040 | +0.008 | -0.005 | -0.085 | +0.003 | +0.100 | -0.001 | +0.010 | -0.002 | +1.000 | +0.005 | +0.010 | +0.002 |
| **amihud** | -0.000 | +0.002 | +0.284 | -0.059 | -0.001 | +0.032 | +0.002 | -0.082 | +0.256 | +0.167 | +0.005 | +1.000 | -0.119 | -0.044 |
| **ko_W** | -0.002 | -0.004 | -0.020 | +0.128 | -0.004 | -0.167 | -0.002 | +0.604 | -0.003 | -0.009 | +0.010 | -0.119 | +1.000 | +0.033 |
| **polar_order_P** | +0.008 | +0.011 | -0.017 | +0.035 | +0.006 | -0.034 | +0.008 | +0.051 | -0.009 | -0.009 | +0.002 | -0.044 | +0.033 | +1.000 |

**Flagged |r|>0.85 pairs:**

| feature_a | feature_b | r | expected by construction |
|---|---|---|---|
| z | imbalance_t | +0.8883 | yes |

Expected-by-construction pairs (do not fail Check 3): `z ↔ imbalance_t`, `z ↔ subbar_imbalance`, `imbalance_t ↔ subbar_imbalance` — all three are monotonic / near-monotonic transforms of the same underlying standardized return z.

**Check 3 result:** PASS (no novel |r|>0.85 pairs)

## 6. Check 4 — sign_concordance Orthogonality Preserved

Locked baseline requirement: `|corr(sign_concordance, imbalance_t)| < 0.05` per contract. Secondary: `|corr(sign_concordance, new_feature)| < 0.4` for each of the eight new features.

| contract | n | r_sign_imbalance | orth_preserved |
|---|---|---|---|
| ES | 359,581 | -3.8741e-04 | 1 |
| NQ | 811,012 | 0.0123 | 1 |
| RTY | 423,776 | 0.0074 | 1 |

**sign_concordance vs each new feature:**

| contract | r_sign_sigma_yz | r_sign_body_to_range | r_sign_gel_fraction | r_sign_wick_asymmetry | r_sign_amihud | r_sign_ko_W | r_sign_polar_order_P |
|---|---|---|---|---|---|---|---|
| ES | 0.1469 | 0.4321 | 0.2622 | -0.0073 | -0.0734 | 0.1288 | 0.0158 |
| NQ | 0.1583 | 0.3978 | 0.2241 | -0.0081 | -0.0664 | 0.1283 | 0.0284 |
| RTY | 0.1933 | 0.4056 | 0.2337 | -0.0048 | -0.0585 | 0.1275 | 0.0347 |

**Secondary flags (|r|>0.4):**
- **ES:** body_to_range
- **RTY:** body_to_range

**Check 4 result:** PASS

## 7. Check 5 — Feature Availability Rate

Availability = `count(non-NaN) / count(is_valid_bar)`. Flag threshold: <0.90.

| contract | feature | n_valid | n_nonan | availability | min_expected | flag |
|---|---|---|---|---|---|---|
| ES | sigma_yz | 359,581 | 359,581 | 1.0000 | 0.9500 | 0 |
| ES | body_to_range | 359,581 | 359,581 | 1.0000 | 0.9500 | 0 |
| ES | gel_fraction | 359,581 | 359,581 | 1.0000 | 0.9500 | 0 |
| ES | wick_asymmetry | 359,581 | 359,581 | 1.0000 | 0.9500 | 0 |
| ES | amihud | 359,581 | 359,581 | 1.0000 | 0.9000 | 0 |
| ES | ko_W | 359,581 | 359,581 | 1.0000 | 0.9500 | 0 |
| ES | polar_order_P | 359,581 | 359,581 | 1.0000 | 0.9500 | 0 |
| NQ | sigma_yz | 811,012 | 811,012 | 1.0000 | 0.9500 | 0 |
| NQ | body_to_range | 811,012 | 811,012 | 1.0000 | 0.9500 | 0 |
| NQ | gel_fraction | 811,012 | 811,012 | 1.0000 | 0.9500 | 0 |
| NQ | wick_asymmetry | 811,012 | 811,012 | 1.0000 | 0.9500 | 0 |
| NQ | amihud | 811,012 | 811,012 | 1.0000 | 0.9000 | 0 |
| NQ | ko_W | 811,012 | 811,012 | 1.0000 | 0.9500 | 0 |
| NQ | polar_order_P | 811,012 | 811,012 | 1.0000 | 0.9500 | 0 |
| RTY | sigma_yz | 423,776 | 423,776 | 1.0000 | 0.9500 | 0 |
| RTY | body_to_range | 423,776 | 423,776 | 1.0000 | 0.9500 | 0 |
| RTY | gel_fraction | 423,776 | 423,776 | 1.0000 | 0.9500 | 0 |
| RTY | wick_asymmetry | 423,776 | 423,776 | 1.0000 | 0.9500 | 0 |
| RTY | amihud | 423,776 | 423,776 | 1.0000 | 0.8500 | 0 |
| RTY | ko_W | 423,776 | 423,776 | 1.0000 | 0.9500 | 0 |
| RTY | polar_order_P | 423,776 | 423,776 | 1.0000 | 0.9500 | 0 |

**Check 5 result:** PASS

## 8. Check 6 — Temporal Stability

Valid bars split in half by time. `norm_shift = (mean_second − mean_first) / std(full_valid)`. Flag when `|norm_shift| > 2.0`.

| contract | feature | mean_first | mean_second | std_first | std_second | norm_shift | flag |
|---|---|---|---|---|---|---|---|
| ES | sigma_yz | 7.6976e-04 | 4.9146e-04 | 6.4427e-04 | 4.0554e-04 | -0.5005 | 0 |
| ES | body_to_range | 0.4697 | 0.4638 | 0.2712 | 0.2717 | -0.0217 | 0 |
| ES | gel_fraction | 0.2739 | 0.2702 | 0.1524 | 0.1530 | -0.0241 | 0 |
| ES | wick_asymmetry | -9.1756e-06 | -1.1483e-05 | 4.9496e-04 | 3.2225e-04 | -0.0055 | 0 |
| ES | amihud | 2.1169e-06 | 1.1631e-06 | 1.5595e-05 | 1.1871e-05 | -0.0688 | 0 |
| ES | ko_W | 2.8424e+04 | 2.5405e+04 | 6.5443e+04 | 6.4939e+04 | -0.0463 | 0 |
| ES | polar_order_P | 0.1006 | 0.1054 | 0.0769 | 0.0798 | 0.0609 | 0 |
| NQ | sigma_yz | 5.7387e-04 | 7.6509e-04 | 4.0625e-04 | 6.0256e-04 | 0.3658 | 0 |
| NQ | body_to_range | 0.4797 | 0.4580 | 0.2824 | 0.2612 | -0.0796 | 0 |
| NQ | gel_fraction | 0.2799 | 0.2644 | 0.1598 | 0.1455 | -0.1010 | 0 |
| NQ | wick_asymmetry | -6.2201e-06 | -1.5287e-05 | 3.6569e-04 | 4.7436e-04 | -0.0214 | 0 |
| NQ | amihud | 1.5165e-06 | 1.8376e-06 | 6.3574e-06 | 1.7333e-05 | 0.0246 | 0 |
| NQ | ko_W | 5297.0508 | 4.0100e+04 | 1.4374e+04 | 8.1778e+04 | 0.5683 | 0 |
| NQ | polar_order_P | 0.0984 | 0.1044 | 0.0750 | 0.0793 | 0.0773 | 0 |
| RTY | sigma_yz | 8.2553e-04 | 7.9985e-04 | 6.6000e-04 | 5.9444e-04 | -0.0409 | 0 |
| RTY | body_to_range | 0.4881 | 0.4840 | 0.2791 | 0.2750 | -0.0148 | 0 |
| RTY | gel_fraction | 0.2850 | 0.2808 | 0.1567 | 0.1538 | -0.0271 | 0 |
| RTY | wick_asymmetry | -8.0692e-06 | -9.5131e-06 | 5.3616e-04 | 4.9899e-04 | -0.0028 | 0 |
| RTY | amihud | 3.6541e-06 | 2.6372e-06 | 9.9518e-06 | 5.9688e-06 | -0.1237 | 0 |
| RTY | ko_W | 1718.4652 | 2308.8268 | 4015.0908 | 5800.9718 | 0.1181 | 0 |
| RTY | polar_order_P | 0.1031 | 0.1074 | 0.0784 | 0.0816 | 0.0529 | 0 |

**No entries with |norm_shift| > 2.0.**

**Check 6 result:** PASS

## 9. Summary

All six checks pass on the 14-feature set. The expanded feature pickles are ready for downstream volatility-prediction work.

## 10. Feature Selection Decision

An initial 15-feature validation run surfaced a novel |r|>0.85 pair (`v_star_C ↔ body_to_range`) on all three contracts. After review, `v_star_C` was dropped from the candidate set before downstream training.

### Three reasons

1. **Structural redundancy with DER** — `corr(v_star_C, DER) ≈ 0.767` on all three contracts. DER is an existing directional-efficiency measure (`|bar_body| / sum_abs_subbar_returns`). `v_star_C` captures substantially overlapping information through a different normalization (session-EMA of `|imbalance_t|`), so the marginal information it adds over the existing feature is small.
2. **Near-redundancy with `body_to_range`** — `corr(v_star_C, body_to_range) ≈ 0.855` on all three contracts, exceeding the Check 3 flag threshold (|r|>0.85) and not on the spec §3 "expected pairs to watch" list. Keeping both would split tree-based-model importance between near-duplicates, introducing seed-dependent importance rankings.
3. **Interpretability preference for `body_to_range`** — when two features measure directional efficiency via different normalizations, `body_to_range = |close − open| / (high − low)` is preferred: it is self-contained per bar, uses only OHLC (no session-state, no EMA warmup), has a direct geometric interpretation on the candle, and does not require a baseline that resets at every contract roll. `v_star_C` depends on a per-instrument session-mean EMA that re-warms at every `instrument_id` change and has no additional physical interpretation once DER is already in the set.

### SHAP-based pruning rejected

An alternative option — "run with all 15 and use SHAP to prune" — was rejected. SHAP values on correlated features are unstable across seeds: the importance allocation between a correlated pair is essentially arbitrary, so SHAP-based pruning would not provide a reproducible selection rule.

### Final 14-feature set

| # | feature | source |
|---|---|---|
| 1 | z | baseline BVC |
| 2 | imbalance_t | baseline BVC |
| 3 | der | baseline BVC |
| 4 | sign_concordance | baseline BVC |
| 5 | clv_mean | baseline BVC |
| 6 | clv_var | baseline BVC |
| 7 | subbar_imbalance | baseline BVC |
| 8 | sigma_yz | new physics |
| 9 | body_to_range | new physics |
| 10 | gel_fraction | new physics |
| 11 | wick_asymmetry | new physics |
| 12 | amihud | new physics |
| 13 | ko_W | new physics |
| 14 | polar_order_P | new physics |

**Result on the 14-feature set:** all six checks pass cleanly without any gate override.
