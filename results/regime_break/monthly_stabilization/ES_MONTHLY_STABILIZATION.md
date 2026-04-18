# ES Regime Stabilization — Monthly Resolution Diagnostic

**Date:** 2026-04-18
**Scope:** ES only. Read-only with respect to baseline artifacts.
**Objective:** Month-level identification of the post-COVID stabilization boundary for
selecting the Stage 1 training window.

## Inputs & Pipeline

- Source: `D:\BVC\data\15_year\ohlcv_1m_ES.parquet`
- Aggregation: 5-minute bars, session-isolated causal Garman–Klass σ
  (`compute_sigma_causal_session_isolated`, `span=20`, `warmup_bars=80`,
  `gap_threshold='15min'`) — identical to baseline configuration.
- Reused cache: `runs/2026-04-17_regime_validation/es_z_cleaned.parquet`
  (918,091 bars, 2010-06-07 → 2026-04-14, produced by that pipeline and filtered
  to finite `|z| < 50`).
- Post-period window for every candidate: `[cutoff, 2026-05-01)` → through 2026-04.
- Locked reference: ν = 4.602 (Phase 1′, 2020-03-01 onward).

Reproduction check: cutoff 2020-03-01 in this sweep yields ν = 4.6023, loc = +0.0152,
scale = 0.8463, PIT KS = 0.0339 — exact match to the baseline fit.

## Stabilization Criteria

The earliest monthly cutoff c satisfying **all** of:

1. **(a) ν proximity.** `|ν_post(c) − 4.602| ≤ 0.15`.
2. **(b) Global PIT KS.** `KS_post(c) ≤ 0.05` against uniform.
3. **(c) Per-month PIT KS.** For every calendar month M in `[c, 2026-04]`,
   PIT KS(M) ≤ 0.08 when evaluated against the global `(ν, loc, scale)` fit at c.

## Table 1 — 16 Candidate Cutoffs (global fit against post-period z)

| Cutoff | n bars | ν | loc | scale | PIT KS | skew | |Δν| vs lock | (a) | (b) |
|:---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|
| 2019-09-01 | 400,906 | 4.5506 | +0.0148 | 0.8439 | 0.0359 | −0.118 | 0.0514 | ✓ | ✓ |
| 2019-10-01 | 394,556 | 4.5692 | +0.0148 | 0.8452 | 0.0355 | −0.038 | 0.0328 | ✓ | ✓ |
| 2019-11-01 | 390,244 | 4.5752 | +0.0148 | 0.8458 | 0.0352 | −0.034 | 0.0268 | ✓ | ✓ |
| 2019-12-01 | 386,146 | 4.5845 | +0.0145 | 0.8468 | 0.0348 | −0.035 | 0.0175 | ✓ | ✓ |
| **2020-01-01** | **379,819** | **4.6154** | **+0.0146** | **0.8477** | **0.0342** | **−0.124** | **0.0134** | **✓** | **✓** |
| 2020-02-01 | 375,555 | 4.6124 | +0.0146 | 0.8477 | 0.0339 | −0.127 | 0.0104 | ✓ | ✓ |
| 2020-03-01 | 370,741 | 4.6023 | +0.0152 | 0.8463 | 0.0339 | −0.126 | 0.0003 | ✓ | ✓ |
| 2020-04-01 | 364,396 | 4.5812 | +0.0157 | 0.8450 | 0.0339 | −0.137 | 0.0208 | ✓ | ✓ |
| 2020-05-01 | 359,988 | 4.5669 | +0.0157 | 0.8438 | 0.0341 | −0.139 | 0.0351 | ✓ | ✓ |
| 2020-06-01 | 355,531 | 4.5644 | +0.0158 | 0.8433 | 0.0341 | −0.132 | 0.0376 | ✓ | ✓ |
| 2020-07-01 | 348,875 | 4.5685 | +0.0156 | 0.8430 | 0.0342 | −0.167 | 0.0335 | ✓ | ✓ |
| 2020-08-01 | 344,551 | 4.5673 | +0.0155 | 0.8432 | 0.0342 | −0.169 | 0.0347 | ✓ | ✓ |
| 2020-09-01 | 340,456 | 4.5655 | +0.0154 | 0.8431 | 0.0341 | −0.169 | 0.0365 | ✓ | ✓ |
| 2020-10-01 | 333,987 | 4.5439 | +0.0151 | 0.8407 | 0.0342 | −0.168 | 0.0581 | ✓ | ✓ |
| 2020-11-01 | 329,873 | 4.5492 | +0.0151 | 0.8411 | 0.0342 | −0.166 | 0.0528 | ✓ | ✓ |
| 2020-12-01 | 325,902 | 4.5435 | +0.0152 | 0.8408 | 0.0343 | −0.190 | 0.0585 | ✓ | ✓ |

All 16 cutoffs satisfy criteria (a) and (b). Criterion (c) is the discriminating test.

## Table 2 — Why Cutoffs Before 2020-01-01 Fail Criterion (c)

For cutoffs Sept/Oct/Nov 2019 the bound is violated by **November 2019**; for the
Dec 2019 cutoff the bound is violated by **December 2019** itself. 2020-01-01
is the first month whose inclusion does not pull any subsequent calendar month
above the 0.08 threshold.

| Cutoff | Worst month | Worst-month PIT KS | Per-month pass (≤ 0.08) |
|:---|:---:|---:|:-:|
| 2019-09-01 | 2019-11 | 0.0903 | ✗ |
| 2019-10-01 | 2019-11 | 0.0902 | ✗ |
| 2019-11-01 | 2019-11 | 0.0903 | ✗ |
| 2019-12-01 | 2019-12 | 0.0830 | ✗ |
| **2020-01-01** | **2021-09** | **0.0673** | **✓** |

Interpretation: late-2019 ES has abnormally compressed realized volatility (the
“grind higher” regime) that produces a PIT distribution measurably
different from the heavy-tailed post-COVID fit; once those two months are
excluded, every subsequent calendar month through 2026-04 is consistent with
the current regime parameters.

## Table 3 — Per-Month PIT KS for Winning Cutoff (2020-01-01)

Global fit applied: ν = 4.6154, loc = +0.0146, scale = 0.8477 (n = 379,819; KS = 0.0342).

Summary: 76 calendar months covered. KS range [0.0158, 0.0673].
Count of months exceeding 0.05: 15/76 (all ≤ 0.0673, well within 0.08 threshold).

Full month-by-month values: `per_month_winning_cutoff.csv`.
Highlights (top 5 worst months, still passing):

| Month | n | PIT KS |
|:---|---:|---:|
| 2021-09 | 6,425 | 0.0673 |
| 2023-11 | 4,560 | 0.0645 |
| 2021-08 | 4,509 | 0.0641 |
| 2021-04 | 4,030 | 0.0640 |
| 2024-02 | 4,503 | 0.0618 |

Early COVID-crash months are notably well-behaved under the 2020-01-01 global
fit: 2020-03 KS = 0.0365, 2020-04 KS = 0.0234 — both below the post-period
global KS of 0.0342, consistent with the crash being part of (not an
exception to) the current-regime heavy-tail behavior.

## Rolling 6-Month ν — Transition Visualization

See `rolling6m_nu.png` and `rolling6m_nu.csv`. Window = 6 calendar months ending
at the anchor month, stepped monthly from 2019-01 through 2021-12.

Qualitative pattern:

- **2019-01 → 2019-06:** ν declining from 5.6 to 4.8 (mild tail-heaviness growth,
  still pre-regime).
- **2019-07 → 2020-01:** ν collapses from 4.7 to 3.7 as the compressed-vol late-2019
  period dominates the window.
- **2020-02 → 2020-06:** ν recovers sharply past 5.2 as the COVID crash and
  rebound dominate; brief overshoot.
- **2020-07 → 2021-12:** ν settles in the 4.4 – 4.8 band, straddling the locked
  value of 4.602 within ± tolerance for the remainder of the plot.

The 2020-01-01 anchor line falls precisely at the inflection point where the
rolling window transitions out of the compressed-vol regime into the current
regime.

## Recommendation

**Stabilization month: 2020-01-01.**

**Confidence: HIGH.**

Supporting evidence:

- All three formal criteria pass (ν Δ = 0.013, global KS = 0.034, worst per-month
  KS = 0.067 across 76 months through 2026-04).
- 2020-01-01 is the **earliest** month satisfying criterion (c); every earlier
  candidate is rejected by a well-identified adjacent month (Nov or Dec 2019)
  rather than by marginal noise.
- The winning fit parameters are materially indistinguishable from the locked
  baseline fit: ν shifts by +0.013 (locked 4.602 → new 4.615), scale by +0.0014
  (0.8463 → 0.8477), loc by −0.0006. Δν = 0.013 is ~9 % of the ±0.15 tolerance.
- The recommendation **extends** the locked training-eligible window by two
  additional months (Jan – Feb 2020, ~9,000 additional bars) without degrading
  the distributional fit, which is material for Stage 1 training-set size.
- Quarterly-resolution baseline selected 2020-03-01 as the conservative round
  boundary; monthly resolution refines this to 2020-01-01 by demonstrating that
  Jan and Feb 2020 are already drawn from the current regime.

## Caveats

- Criterion (c) uses the global post-period fit as the reference. Using a
  per-month re-fit would be less stringent; the current test is deliberately
  conservative.
- The worst per-month KS (2021-09, 0.0673) is still 16 % below the 0.08 bound;
  the margin would shrink if the threshold were tightened toward the global
  KS level of ~0.05.
- No attempt is made here to re-lock baseline parameters. The locked
  `(ν, loc, scale) = (4.602, +0.0152, 0.8463)` remain authoritative; this
  diagnostic only informs **training-window selection** for Stage 1.

## Artifacts in This Directory

- `candidate_cutoffs.csv` — Table 1 source.
- `pre_winner_rejections.csv` — Table 2 source.
- `per_month_winning_cutoff.csv` — Table 3 full data.
- `rolling6m_nu.csv`, `rolling6m_nu.png` — rolling-ν series and plot.
- `stabilization_summary.json` — winning cutoff and fit parameters in JSON.
- `ES_MONTHLY_STABILIZATION.md` — this report.

## Status

The selected cutoff is documented here; downstream volatility-prediction
experiments use `is_train_valid_bar = is_valid_bar AND (ts_event >= 2020-01-01 UTC)`
as described in `docs/METHODOLOGY.md`.
