# Stage 1 Training-Window Adoption

**Date:** 2026-04-18
**Status:** Adopted. Training-window gate applied via `is_train_valid_bar` column
in the expanded feature pickles. Baseline calibration artifacts and
`PHASE1_LOCK_FINAL.md` are unchanged.

## 1. Scope

This document records the Stage 1 training-window adoption decision. It is
supplementary to — and does **not** supersede — `PHASE1_LOCK_FINAL.md`.

- Original calibration lock (authoritative, unchanged):
  `runs/2026-04-17_regime_validation/PHASE1_LOCK_FINAL.md`
- Monthly stabilization diagnostic (supporting evidence):
  `runs/2026-04-18_es_monthly_stabilization/ES_MONTHLY_STABILIZATION.md`

## 2. Adopted Training Windows

| Contract | Training-window start | Basis |
|:---|:---|:---|
| ES  | 2020-03-01 | Baseline regime boundary (unchanged). |
| NQ  | 2020-01-01 | Monthly stabilization cutoff. |
| RTY | 2020-01-01 | Monthly stabilization cutoff. |

All three contracts train on post-2020 data. ES excludes Jan–Feb 2020 because
the locked `is_valid_bar` column already gates ES at 2020-03-01 (the economic
regime boundary).

Mechanism: a new boolean column `is_train_valid_bar` was added to each
`runs/2026-04-18_physics_feature_expansion/phase2_features_expanded_{ES,NQ,RTY}.pkl`,
computed as

    is_train_valid_bar = is_valid_bar AND (ts_event >= 2020-01-01 UTC)

No feature values or existing columns were modified. The input pickles
in `runs/2026-04-17_regime_validation/` were not touched.

## 3. Pipeline Consistency Check

The mental model of `is_valid_bar` semantics was verified against the locked ES
pickle:

```
predicted = warmup_valid AND sigma_gk.notna() AND (ts >= 2020-03-01)
actual    = is_valid_bar  (from phase2_features_cleaned_ES.pkl)
```

Result on all 1,604,333 ES rows:

| n total | n is_valid_bar=True | n predicted=True | n disagreement | status |
|---:|---:|---:|---:|:---:|
| 1,604,333 | 359,581 | 359,581 | 0 | **PASS** |

Agreement is exact (100.000000 %). No additional Phase 6 gating logic is
present beyond warmup, σ availability, and the 2020-03-01 ES regime cut.

## 4. Bar Counts in the Training Window

Read from `training_window_counts.json` (sibling file in this directory):

| Contract | n total rows | n is_valid_bar | **n is_train_valid_bar** |
|:---|---:|---:|---:|
| ES  | 1,604,333 | 359,581 | **359,581** |
| NQ  | 1,387,546 | 811,012 | **346,624** |
| RTY |   675,791 | 423,776 | **324,875** |
| **Total training set** | | | **1,031,080** |

For ES, `is_train_valid_bar == is_valid_bar` exactly (ES is already gated at
2020-03-01 ≥ 2020-01-01). For NQ and RTY the new flag restricts the
full-history locked pickles to post-2020-01-01.

## 5. Locked Parameters Retained Unchanged

The baseline Student-t parameters are retained verbatim:

| Contract | Regime | ν | loc | scale | PIT KS | Bars (locked) |
|:---|:---|---:|---:|---:|---:|---:|
| ES  | 2020-03-01 onward | 4.602 | +0.0152 | 0.8463 | 0.0339 | 359,581 |
| NQ  | Full sample       | 4.410 | +0.0156 | 0.8510 | 0.0253 | 811,012 |
| RTY | Full sample       | 4.481 | +0.0108 | 0.8854 | 0.0250 | 423,776 |

Parameter-shift check (post-2020-01-01 refit on ES cache): ν moves 4.602 → 4.615
(Δν = 0.013), loc 0.0152 → 0.0146, scale 0.8463 → 0.8477 — well inside the
split-half stability tolerances documented in `PHASE1_LOCK_FINAL.md`. Student-t
MLE was not rerun; the locked triplet is used unchanged for BVC inference.

## 6. Note on Spec Calibration — NQ Density

The initial adoption spec expected NQ to have ~450–500K valid bars
post-2020-01-01. Actual NQ count is 346,624, which is ~27 % below the low end
of that estimate. This is not a pipeline defect: the 346K figure corresponds
to ~4,225 valid bars per month, which is consistent with the NQ per-month
density across the entire locked pickle (811,012 valid bars over 192 months of
2010-06 → 2026-04, ≈ 4,225/month).

The ~4,225 bars-per-month density is driven by the baseline warmup
filter (80-bar warmup discarded after each session boundary — instrument
roll or ≥ 15-minute gap) applied on top of the CME trading schedule (one
60-minute halt per day and the weekend close). The 450–500K expectation in
the adoption spec was a miscalculation; 346K is the correct NQ count.

The ES (359,581) and RTY (324,875) counts are within the spec's tolerance
(−5.4 % and −8.5 % respectively vs their expected midpoints).

## 7. Interpretation of the Monthly Diagnostic

The monthly-resolution sweep (16 cutoffs from 2019-09-01 through 2020-12-01,
see `ES_MONTHLY_STABILIZATION.md`) establishes that post-break ES tail index
is **insensitive** to boundary choice in Jan–Mar 2020:

| Cutoff | ν | |Δν| vs locked |
|:---|---:|---:|
| 2020-01-01 | 4.6154 | 0.013 |
| 2020-02-01 | 4.6124 | 0.010 |
| 2020-03-01 | 4.6023 | 0.000 (reference) |

Range of ν across all eight candidate cutoffs in [2019-10-01, 2020-05-01]:
4.5669 – 4.6154, a spread of 0.0485. The parameters are effectively constant
inside this window.

**Methodological reading.** An insensitivity of post-break ν to the exact
boundary choice is consistent with a **discrete regime break**, not a
gradual transition. Under a gradual transition, sweeping the boundary over
a five-month window would produce monotonic drift in ν of comparable
magnitude to the pre/post gap itself. The observed variation (< 0.05) is
an order of magnitude smaller than the pre/post gap (pre ≈ 5.5, post ≈ 4.6).

**Economic reading.** 2020-03-01 is the economically justified boundary:
CME circuit-breaker halts, Fed emergency rate cuts, the corporate-credit
facility announcement, and the structural derivatives-flow shift all
cluster in late February through mid-March 2020. The monthly diagnostic
does not move the boundary; it provides statistical support for the
discrete-break hypothesis that motivated the boundary choice in the first
place.

## 8. Paper Notes (do not investigate further now)

- **Publishable methodological finding.** The insensitivity of post-COVID ES
  ν to boundary choice in Jan–Mar 2020 is a small standalone result worth
  including when the BVC paper is written — it distinguishes a discrete
  regime break from a gradual distributional shift.
- **Structural driver candidates**, flagged for paper discussion only:
  0DTE options growth, HFT / market-maker share shifts, the 2019–2020
  zero-commission retail rollout, index-volume migration, and the
  COVID volatility shock itself. No further investigation at this stage.

## 9. Scope of This Adoption

This adoption is strictly additive: it introduces the
`is_train_valid_bar` column in the expanded per-contract pickles (31 → 32
columns) without modifying existing calibration parameters, existing
feature values, or the baseline `is_valid_bar` column. The
`ES_POST_CUT = 2020-03-01` constant used by the final-validation stage
is retained with provenance.

## 10. Artifacts Produced by This Adoption

- `src/regime/adopt_training_window.py` — additive gate-writer.
- Added column `is_train_valid_bar` in all three expanded per-contract
  feature pickles (31 → 32 columns).
- `training_window_counts.json` — machine-readable bar counts.
- `PHASE1_LOCK_FINAL_v2_TRAINING_WINDOW.md` — this file.
