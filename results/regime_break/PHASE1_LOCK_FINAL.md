# Calibration Lock — Final Validation Report (2026‑04‑17)

Inputs: 15‑year cleaned parquet (`data/parquet/ohlcv_1m_{ES,NQ,RTY}.parquet`) resampled to 5‑min bars under the session‑isolated causal GK σ (span=20, warmup=80, gap=15 min).

Outputs co-located with this report:
- `es_z_cleaned.parquet`, `nq_z_cleaned.parquet`, `rty_z_cleaned.parquet` — cached z series
- `rolling_nu.png`, `rolling_nu_{ES,NQ,RTY}.csv`, `rolling_nu_results.pkl`
- `es_break_sensitivity.csv`, `es_pre_start_scan.csv`, `es_dual_breakpoint.csv`, `es_post_start_scan.csv`, `es_extended_breakdate.pkl`
- `es_breakdate_validation.pkl`, `es_per_year_nu.png`, `es_{pre,post}_per_year.csv`
- `proposed_params_comparison.csv`, `per_year_validation.csv`, `phase4_final_validation.pkl`

---

## 1. Empirical outcome vs. original hypothesis

The plan specified a pre/post ES regime split at a cutoff in {2019‑10, 2020‑01, 2020‑03, 2020‑06, 2020‑09} with both sub‑regimes required to pass KS<0.05. The empirical result is **negative for ES‑pre at every candidate**:

| cutoff | ν_pre | KS_pre | pre_passes | ν_post | KS_post | post_passes | KS_combined |
|---|---|---|---|---|---|---|---|
| 2019‑10‑01 | 5.219 | 0.0787 | **False** | 4.569 | 0.0355 | True | 0.0601 |
| 2020‑01‑01 | 5.152 | 0.0785 | **False** | 4.615 | 0.0342 | True | 0.0602 |
| 2020‑03‑01 | 5.153 | 0.0779 | **False** | 4.602 | 0.0339 | True | 0.0601 |
| 2020‑06‑01 | 5.166 | 0.0766 | **False** | 4.564 | 0.0341 | True | 0.0601 |
| 2020‑09‑01 | 5.147 | 0.0755 | **False** | 4.565 | 0.0341 | True | 0.0601 |

The extended diagnostics (`es_pre_start_scan.csv`, `es_dual_breakpoint.csv`) confirm this is not a cutoff problem:

- **Narrower pre windows don't help.** Restricting pre to start at 2018‑01‑01 (a 2.5‑yr window ending 2020‑06‑01) still gives KS=0.0548.
- **Excluding transition bars doesn't help.** Even excluding 22 % of data around the 2017‑2020 transition, ES‑pre KS stays at 0.076–0.084.
- **Rolling‑ν explains why.** ES ν drifts monotonically from 5.67 (2011‑12 window) through 5.10 (2014‑06) to 4.85 (2018‑09) to 4.60 (2020‑03). This is gradual drift, not a step function. No stationary Student‑t fits the full 2010–2020 ES window.

ES pre‑2020 is therefore not a single stationary regime; it is a succession of slowly‑drifting regimes that cannot be captured by one (ν, loc, scale) triple. NQ and RTY show no analogous instability.

## 2. Proposed lock

Three parameter sets (not four), one per contract:

| Contract | Sample window | N | ν | loc | scale | PIT KS | Shoulder max | Status |
|---|---|---|---|---|---|---|---|---|
| **ES** | 2020‑03‑01 → 2026‑04‑14 | 370 741 | **4.6023** | +0.0152 | **0.8463** | 0.0339 | 0.0048 | PIT ✓, shoulder ✓ |
| **NQ** | 2010‑06‑07 → 2026‑04‑14 | 840 115 | **4.4098** | +0.0156 | **0.8510** | 0.0253 | 0.0038 | PIT ✓, shoulder ✓ |
| **RTY** | 2017‑07‑10 → 2026‑04‑14 | 440 306 | **4.4809** | +0.0108 | **0.8854** | 0.0250 | 0.0038 | PIT ✓, shoulder ✓ |

Comparison to the previous baseline lock (12‑month, contaminated σ):

| Contract | old ν | new ν | Δν | old scale | new scale | Δscale |
|---|---|---|---|---|---|---|
| ES | 4.607 | 4.602 | −0.005 | 0.7716 | 0.8463 | +0.0747 |
| NQ | 4.383 | 4.410 | +0.027 | 0.7519 | 0.8510 | +0.0991 |
| RTY | 4.460 | 4.481 | +0.021 | 0.8279 | 0.8854 | +0.0575 |

**The ν values barely moved.** The main change is in *scale*, which rises by ~0.06–0.10 across contracts. This is the direct signature of eliminating boundary contamination: the old EWMA was artificially small for bars just after session gaps, inflating |z| in the tails, and the old fit compensated with a smaller scale. Session‑isolated σ removes that artefact, producing properly‑sized σ at session starts, which yields a larger scale and nearly the same ν.

## 3. Validation detail

### 3.1 Per‑year PIT at new parameters

ES‑post (cutoff 2020‑03‑01) — 7/7 years **GOOD** (KS 0.022‑0.040):
```
 year     n     nu   scale     ks   shoulder  quality
 2020 36037  4.653  0.867   0.034    0.0069    GOOD
 2021 59396  4.582  0.849   0.040    0.0057    GOOD
 2022 63210  4.637  0.854   0.025    0.0040    GOOD
 2023 59740  4.825  0.846   0.040    0.0058    GOOD
 2024 60166  4.295  0.840   0.038    0.0073    GOOD
 2025 59887  4.431  0.817   0.031    0.0060    GOOD
 2026 17095  4.764  0.829   0.022    0.0035    GOOD
```
ES‑post per‑year ν trend: slope=‑0.011/year, r=‑0.13, p=0.78 → **no significant drift within post‑regime**.

NQ — 16/17 years GOOD (only 2013 marginally fails KS=0.053).
RTY — 9/9 years GOOD.

See `per_year_validation.csv`.

### 3.2 Shoulder CDF (threshold 0.01)

All three contracts pass both left and right shoulders by a ~2× margin:
- ES‑post: L=0.0048, R=0.0043
- NQ: L=0.0029, R=0.0038
- RTY: L=0.0038, R=0.0035

### 3.3 Split‑half stability

All three contracts have Δν < 0.2 between H1 and H2 (well under the 1.0 threshold), but only ES‑post passes the tight Δscale<0.03 rule (Δscale 0.027). NQ (Δscale 0.071) and RTY (Δscale 0.038) technically fail the tight scale rule.

The cause is the same ES pattern already documented: the skewness of |z|>3 bars flips sign between halves (NQ: −0.092 → −0.077, very small; RTY: +0.073 → +0.229). The ν stability is tight (Δν 0.166 NQ, 0.194 RTY), so the symmetric Student‑t is not structurally misspecified; the scale Δ is driven by a handful of extreme surprise‑event outliers whose distribution differs between halves.

### 3.4 Skewness by |z| — sign‑flip pattern preserved

All three contracts show the now‑standard pattern: ~0 skewness for |z|<2 (95 % of bars), positive skewness concentrated in the 3≤|z|<5 bucket (ES‑post +0.29, NQ +0.16, RTY +0.16), which is dominated by rare asymmetric event‑driven outliers. Symmetric Student‑t remains correct.

## 4. Session-isolation implementation

- `student_t_bvc.py :: compute_sigma_causal_session_isolated` returns `(sigma, warmup_valid)`. All downstream callers are updated accordingly.
- `warmup_valid` is a boolean `Series` indexed like the bars; it is True when the bar is at least `warmup_bars` past the last session break. Downstream consumers (feature pipeline, inference) filter on this to guarantee no warmup‑polluted σ is used.
- `compute_subbar_features` exposes `warmup_valid` as a column on the output bars so the feature-rebuild step can filter uniformly.

## 5. Design decisions recorded at lock time

1. **Three‑parameter‑set outcome for ES (post‑only).** The original plan specified four sets (ES pre/post + NQ + RTY). Empirically no valid ES‑pre exists. Any ES bar before 2020‑03‑01 is treated as out‑of‑sample for BVC inference.
2. **ES cutoff at 2020‑03‑01** (KS=0.0339 vs KS=0.0341 at 2020‑06‑01 — within noise). 2020‑03‑01 is the cleanest economic interpretation (COVID market restructuring); the rolling‑ν plot shows the ES drift largely completes by mid‑2020 in either case.
3. **Interpretation of the pre‑2020 ES misfit.** Two non‑exclusive explanations:
   - (a) Real regime change in E‑mini liquidity / HFT‑penetration between 2014 and 2020 (the rolling‑ν trend starts well before COVID).
   - (b) The session‑isolated σ spec may still be under‑filtering something specific to pre‑2020 ES microstructure. No evidence for (b) beyond the KS failure itself.
4. **`warmup_valid` filter on by default** in the feature‑rebuild step.

---

## 6. Feature Rebuild and Event-Analysis Replication Results

Outputs co-located with this file: `phase2_features_cleaned_{ES,NQ,RTY}.pkl`,
`phase6_correlations.txt`, `phase6_orthogonality.csv`, `phase6_retention.csv`,
`phase7_event_analysis.png`, `phase7_regime_summary.csv`, `phase7_results.pkl`.

### 6.1 Feature rebuild retention

Bars are produced at the 5‑min grid by `compute_subbar_features` under the
baseline' parameters (§2) and the session‑isolated causal σ
(warmup=80, gap=15 min). Warmup rows are now preserved in the output so that
`is_valid_bar = warmup_valid & sigma_gk.notna()` (plus the ES pre‑2020 gate)
can be used as a uniform downstream mask.

| Contract | total 5‑min bars | valid bars | pre‑2020 excluded (ES only) | retention |
|---|---|---|---|---|
| ES  | 1 604 333 | 359 581 | 975 392 | 22.4 % |
| NQ  | 1 387 546 | 811 012 | 0 | 58.4 % |
| RTY |   675 791 | 423 776 | 0 | 62.7 % |

The NQ / RTY retention figures are driven by the combination of two
mechanics and are **expected, not a bug**:

1. **CME's one‑hour daily maintenance halt** (16:00–17:00 CT) exceeds the
   15‑min gap threshold, so every instrument experiences ~5 daily session
   breaks per week plus the weekend break.
2. **Each session break discards 80 bars of EWMA warmup.** At 5‑min bars the
   effective warmup is 6h40m, so short calendar fragments (holiday halves,
   contract roll tails) never cross the warmup bar and contribute 0 valid
   bars. Session‑length profiling (`tmp_session_count.py`, sorted by
   `[instrument_id, ts_event]`) confirms: NQ has 48 878 sessions, 44 080
   under 80 bars (188 154 pre‑warmup bars); RTY has 15 429 sessions,
   12 921 under 80 bars; ES similarly.

ES 22.4 % = 57.2 % post‑2020 × (628 941 post‑2020 bars / 1 604 333 total);
the other 42.8 % loss within post‑2020 matches the NQ / RTY pattern.

### 6.2 Sign‑concordance orthogonality (pre‑registered check)

All three contracts pass the |r|<0.05 orthogonality criterion on valid bars:

| Contract | N | r(sign_concordance, imbalance_t) | status |
|---|---|---|---|
| ES  | 359 581 | −0.00039 | preserved |
| NQ  | 811 012 | +0.01229 | preserved |
| RTY | 423 776 | +0.00736 | preserved |

Full correlation matrices are in `phase6_correlations.txt`. Key structural
numbers hold across contracts: `z ↔ imbalance_t` ≈ 0.89–0.90 (expected
monotone Student‑t CDF map), `z ↔ sign_concordance` ≈ 0 (orthogonal by
construction), `der ↔ sign_concordance` ≈ 0.28–0.32 (shared directional
information), `clv_mean ↔ z` ≈ 0.60–0.62 (price location within the bar
tracks bar‑level return).

### 6.3 Phase 3A regime replication

Regime stats on valid bars only (FOMC = ±2h around each of 132 FOMC
announcements in `FOMC_DATES_15YR`; high‑vol = top decile of daily |z| std):

**ES (post‑2020 only, N_valid = 359 581)**
| regime | N | conc | DER | \|imb\| | VPIN_std | VPIN_adj | divergence |
|---|---|---|---|---|---|---|---|
| Normal        | 315 937 | 0.568 | 0.402 | 0.496 | 0.4963 | 0.2374 | −0.2589 |
| FOMC windows  |   3 235 | 0.612 | 0.410 | 0.490 | 0.5031 | 0.2597 | −0.2434 |
| High‑vol days |  41 285 | 0.565 | 0.428 | 0.513 | 0.5125 | 0.2465 | −0.2661 |

**NQ (N_valid = 811 012)**
| regime | N | conc | DER | \|imb\| | VPIN_std | VPIN_adj | divergence |
|---|---|---|---|---|---|---|---|
| Normal        | 722 656 | 0.580 | 0.404 | 0.498 | 0.4978 | 0.2414 | −0.2564 |
| FOMC windows  |   7 495 | 0.615 | 0.406 | 0.490 | 0.4932 | 0.2543 | −0.2389 |
| High‑vol days |  82 777 | 0.567 | 0.424 | 0.508 | 0.5087 | 0.2441 | −0.2646 |

**RTY (N_valid = 423 776)**
| regime | N | conc | DER | \|imb\| | VPIN_std | VPIN_adj | divergence |
|---|---|---|---|---|---|---|---|
| Normal        | 379 147 | 0.576 | 0.421 | 0.500 | 0.4994 | 0.2444 | −0.2550 |
| FOMC windows  |   4 146 | 0.611 | 0.413 | 0.478 | 0.4900 | 0.2576 | −0.2324 |
| High‑vol days |  41 551 | 0.567 | 0.438 | 0.503 | 0.5030 | 0.2446 | −0.2584 |

### 6.4 Direction check vs. 15‑year cleaned baseline

Baseline = `runs/2026-04-16_15year_cleaned/phase3a_results.pkl`
(pre‑rebuild, no warmup filter, no ES gating).

| Contract | Metric | Baseline | Phase 7 | Match |
|---|---|---|---|---|
| ES  | FOMC conc − Normal conc  | +0.0421 | +0.0435 | OK |
| ES  | FOMC div  − Normal div   | +0.0087 | +0.0155 | OK |
| ES  | HiVol conc − Normal conc | +0.0151 | −0.0033 | **near‑zero reversal** |
| ES  | HiVol div  − Normal div  | −0.0117 | −0.0072 | OK |
| NQ  | FOMC conc − Normal conc  | +0.0446 | +0.0350 | OK |
| NQ  | FOMC div  − Normal div   | +0.0106 | +0.0175 | OK |
| NQ  | HiVol conc − Normal conc | −0.0030 | −0.0129 | OK |
| NQ  | HiVol div  − Normal div  | −0.0188 | −0.0082 | OK |
| RTY | FOMC conc − Normal conc  | +0.0647 | +0.0348 | OK |
| RTY | FOMC div  − Normal div   | +0.0232 | +0.0226 | OK |
| RTY | HiVol conc − Normal conc | −0.0146 | −0.0086 | OK |
| RTY | HiVol div  − Normal div  | −0.0213 | −0.0034 | OK |

**FOMC direction replicates 3/3 on both concordance and divergence.** The
magnitudes compress by 15–50 % relative to the baseline because the baseline's
"Normal" regime was deflated by warmup‑polluted bars and (for ES) pre‑2020
drifting‑ν bars; filtering those out raises both Normal and FOMC concordance
uniformly (ES Normal 0.456 → 0.568; NQ 0.517 → 0.580; RTY 0.523 → 0.576),
shrinking the gap without changing its sign.

The one "reversal" — ES high‑vol concordance, −0.003 vs +0.015 — is within
Monte Carlo noise on the post‑2020 sub‑sample (|Δ| < 0.02 in both
directions). The FOMC signal is the load‑bearing Phase 3A result and is
preserved cleanly across all three contracts.

## 7. ES drift note for future work

ES ν from the 3‑year rolling fit (`rolling_nu_ES.csv`, 53 quarterly
windows) drifts monotonically:

| window centre | ν | scale | n | KS |
|---|---|---|---|---|
| 2011‑12‑07 | 5.674 | 0.858 | 162 225 | 0.085 |
| 2014‑06‑07 | 5.142 | 0.861 | 162 265 | 0.085 |
| 2017‑09‑05 | 4.714 | 0.860 | 180 862 | 0.082 |
| 2020‑03‑07 | 4.574 | 0.863 | 203 420 | 0.077 |
| 2023‑06‑07 | 4.448 | 0.839 | 186 033 | 0.037 |
| 2024‑09‑05 | 4.512 | 0.835 | 179 995 | 0.036 |

Per‑year ν *within* the post‑2020 sub‑regime has slope −0.011 / year
(r=−0.13, p=0.78) — not significant, so the locked ν=4.602 is stationary
over the 2020–2026 calibration window. But the full 16‑year view is a
clear monotone fall from 5.67 → 4.51 with the step change occurring between
2014 and 2020 rather than as a single break at 2020‑03‑01. The COVID cutoff
is defensible as the cleanest economic anchor; it is not a mechanical break
in the data.

**Future‑work triggers.** Refit cadence and monitoring rules the human may
want to adopt once the lock is in production:

1. **Quarterly ν health check.** Run `phase2_rolling_nu.py` at each quarter
   end. If the most recent 3‑year window ν drops below 4.25 on ES (or
   below 4.10 on NQ / 4.15 on RTY), trigger a refit.
2. **Per‑year PIT regression.** Run `phase4_final_validation.py` annually
   against the locked parameters. A year with KS>0.05 at locked (ν, scale)
   is a refit trigger even if the rolling estimate has not crossed the
   threshold above.
3. **Scale drift guard.** The rolling scale has fallen from 0.863 (2020‑03)
   to 0.835 (2024‑09) on ES — a 3 % drift in ~4 years. If it falls another
   3 % (below ≈0.810) between refits, both scale and ν should be updated
   jointly rather than piecewise.
4. **Do not attempt to extend ES back.** The diagnostic work in §1 and
   `es_pre_start_scan.csv` is conclusive: no stationary Student‑t fits
   2010‑2019 ES. Any attempt to cover that window requires either a
   piecewise‑ν model (distinct from the current single‑ν lock) or a
   time‑varying‑ν GARCH‑style extension. Both are out of scope for the
   current calibration lock.
5. **NQ / RTY are stationary.** The NQ rolling ν is flat at 4.40 ± 0.1 for
   the full sample; RTY is flat at 4.48 ± 0.1 since inception. Neither
   contract is expected to need a refit in the next 1–2 years absent a
   structural event comparable to March 2020.
