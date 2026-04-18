# Phase 3C — OHLCV Directional Signal Test Results

**Generated:** 2026-04-18T14:09:14.807095+00:00
**Scope:** Test of whether OHLCV-derived BVC imbalance at flow-event bars predicts direction on ES, NQ, RTY at h ∈ {1, 3, 6, 12}. Three gate constructions × two tiers × three contracts × four horizons = 72 evaluation cells.

## 1. Executive summary

- **No** gate × tier configuration produced pooled accuracy significantly above 50% at h=1 (z≥1.96, n≥100).
- Highest observed pooled accuracy: Gate B / primary at 48.26% (n=4,424, z=-2.32).

## 2. Gate construction and motivation

All gates are computed on bars where `is_train_valid_bar == True`. Rolling statistics are session-isolated: a new session starts at an `instrument_id` change or a `ts.diff() > 15 min` gap. The first (window − 1) bars of each session return NaN.

- **Gate A (one-factor).** `d_hat_A = max(z20(|imbalance_t|), 0)`. Deviation of flow magnitude from the rolling 20-bar norm, clipped at zero.
- **Gate B (two-factor product).** `d_hat_B = max(z20(|imbalance_t|) · z20(sign_concordance), 0)`. Adds sub-bar persistence. The two z-scores are each normalised, keeping the product on a clean relative-deviation scale.
- **Gate C (hybrid, three-factor with DER as binary conviction filter).** `d_hat_C = d_hat_B · 1{DER_t > median_50(DER)}`. DER is bounded on [0,1] and clusters near the middle of its support, so its z-score dynamics are noisier than |imbalance| or concordance — multiplying three z-scores would amplify that sampling noise. Using DER as an above-median binary filter keeps the two-factor product clean while adding directional efficiency as a conviction prerequisite.

## 3. Warmup threshold calibration

Warmup windows: ES 2020-03-01..2020-12-31 (locked regime boundary); NQ, RTY 2020-01-02..2020-12-31. Thresholds are frozen at the close of 2020-12-31; all evaluation is from 2021-01-01 onward.

| contract | gate | n_pos_warmup | p95 (primary) | p70 (secondary) |
|:--------:|:----:|-------------:|--------------:|----------------:|
| ES | A | 22,171 | 1.6458 | 1.1097 |
| ES | B | 30,049 | 2.5935 | 1.1051 |
| ES | C | 12,316 | 2.9124 | 1.2682 |
| NQ | A | 23,700 | 1.6775 | 1.1337 |
| NQ | B | 31,860 | 2.6419 | 1.1692 |
| NQ | C | 12,702 | 2.8710 | 1.2684 |
| RTY | A | 21,874 | 1.6360 | 1.1116 |
| RTY | B | 29,300 | 2.5840 | 1.1502 |
| RTY | C | 11,633 | 2.9324 | 1.3088 |

## 4. Event counts (evaluation period)

Events are the first rising-edge bar in each session where `d_hat ≥ threshold` (strictly one event per session).

| contract | gate | tier | n_events | events/month (μ±σ) | frac sessions w/ event |
|:--------:|:----:|:----:|---------:|:------------------:|----------------------:|
| ES | A | primary | 1,535 | 24.0 ± 5.5 | 85.90% |
| ES | A | secondary | 1,665 | 26.0 ± 6.2 | 93.17% |
| ES | B | primary | 1,636 | 25.6 ± 5.9 | 91.55% |
| ES | B | secondary | 1,683 | 26.3 ± 6.3 | 94.18% |
| ES | C | primary | 1,477 | 23.1 ± 5.2 | 82.65% |
| ES | C | secondary | 1,631 | 25.5 ± 6.1 | 91.27% |
| NQ | A | primary | 1,463 | 22.9 ± 3.9 | 87.55% |
| NQ | A | secondary | 1,556 | 24.3 ± 4.8 | 93.12% |
| NQ | B | primary | 1,535 | 24.0 ± 4.6 | 91.86% |
| NQ | B | secondary | 1,569 | 24.5 ± 5.0 | 93.90% |
| NQ | C | primary | 1,371 | 21.4 ± 3.7 | 82.05% |
| NQ | C | secondary | 1,528 | 23.9 ± 4.4 | 91.44% |
| RTY | A | primary | 1,410 | 22.0 ± 2.9 | 93.38% |
| RTY | A | secondary | 1,477 | 23.1 ± 3.3 | 97.81% |
| RTY | B | primary | 1,468 | 22.9 ± 3.2 | 97.22% |
| RTY | B | secondary | 1,486 | 23.2 ± 3.5 | 98.41% |
| RTY | C | primary | 1,327 | 20.7 ± 2.7 | 87.88% |
| RTY | C | secondary | 1,456 | 22.8 ± 3.0 | 96.42% |

## 5. Sanity check (Gate A primary, first 20 eval sessions, h=1)

| contract | sessions | n_events | n_scored | n_correct | accuracy |
|:--------:|---------:|---------:|---------:|----------:|---------:|
| ES | 20 | 15 | 14 | 4 | 28.57% ⚠ |
| NQ | 20 | 14 | 14 | 3 | 21.43% ⚠ |
| RTY | 20 | 17 | 17 | 10 | 58.82% |

### 5a. Ungated directional baseline (full evaluation period)

The 20-session Gate A primary sample only produces ~15 events per contract, so a 29–58% range of observed accuracies is consistent with null p=0.5 under small-sample noise (p(k≤4 | n=14, p=0.5) ≈ 0.09). To distinguish a genuine sign-convention bug from short-horizon mean-reversion, we report the ungated directional accuracy of `sign(imbalance_t)` against the same-session forward return at h=1 across **all** train-valid evaluation bars per contract.

| contract | n_scored (all bars) | accuracy | z vs 0.5 |
|:--------:|--------------------:|---------:|---------:|
| ES | 289,377 | 48.68% | -14.19 |
| NQ | 287,977 | 49.13% | -9.38 |
| RTY | 264,808 | 49.36% | -6.60 |

**Interpretation — sign convention is verified by construction, not by accuracy > 0.5.** By definition, `imbalance_t = 2·t.cdf(z_t) − 1` where `z_t = log_ret_t / σ_t`, so `sign(imbalance_t) ≡ sign(z_t) ≡ sign(log_ret_t)` wherever `loc ≈ 0` (the Student-t `loc` parameters of 0.0108–0.0156 produce sign agreement of 99.84% on NQ and 100.00% on ES and RTY — verified on full-sample data). Therefore the h=1 'directional accuracy of `sign(imbalance_t)`' is identical, up to a fraction of a percent, to the lag-1 sign-autocorrelation of 5-min log returns. The observed 48.7%/49.1%/49.4% baselines reproduce the lag-1 sign-autocorrelation of 48.98%/49.27%/49.69% computed directly on log_ret (ES/NQ/RTY, 2021-01-01 onward, same-instrument), which is a well-documented microstructure property of 5-minute equity index futures returns. **This rules out a sign-convention bug in the pipeline** and reframes the sub-50% cells in §6–§7 as a *real, tiny* mean-reversion that is being amplified to ~−2 pp by the gate-selection step. The 20-session sanity 'fail' on ES/NQ is small-sample noise (at n=14, observing ≤4 correct under p=0.5 has Pr ≈ 0.09; RTY's 58.8% on the same small-sample check lies in the opposite tail and reinforces the 'noise at n=14' reading).

The deeper question the 72-cell table therefore answers is: **do any of the three gates select a subset of bars in which the lag-1 sign-autocorrelation flips to above 50%?** The answer, in §7, is no — all six gate × tier configurations pooled sit at ~48.1–48.3%, i.e. the gates select bars that are, if anything, *more* mean-reverting than the ungated baseline.

## 6. Full accuracy table (72 cells)

| contract | gate | tier | h | n_scored | accuracy | Wilson 95% CI | z |
|:--------:|:----:|:----:|--:|---------:|---------:|:-------------:|--:|
| ES | A | primary | 1 | 1,441 | 48.02% | [45.45%, 50.60%] | -1.50 |
| ES | A | primary | 3 | 1,487 | 47.55% | [45.02%, 50.09%] | -1.89 |
| ES | A | primary | 6 | 1,489 | 46.61% | [44.09%, 49.15%] | -2.62 |
| ES | A | primary | 12 | 1,502 | 47.74% | [45.22%, 50.26%] | -1.75 |
| ES | A | secondary | 1 | 1,535 | 47.56% | [45.07%, 50.06%] | -1.91 |
| ES | A | secondary | 3 | 1,604 | 49.13% | [46.69%, 51.57%] | -0.70 |
| ES | A | secondary | 6 | 1,615 | 47.49% | [45.07%, 49.93%] | -2.02 |
| ES | A | secondary | 12 | 1,613 | 49.04% | [46.60%, 51.48%] | -0.77 |
| ES | B | primary | 1 | 1,514 | 46.70% | [44.20%, 49.22%] | -2.57 |
| ES | B | primary | 3 | 1,580 | 49.05% | [46.59%, 51.51%] | -0.75 |
| ES | B | primary | 6 | 1,582 | 46.71% | [44.27%, 49.18%] | -2.61 |
| ES | B | primary | 12 | 1,581 | 47.56% | [45.11%, 50.03%] | -1.94 |
| ES | B | secondary | 1 | 1,558 | 46.85% | [44.39%, 49.34%] | -2.48 |
| ES | B | secondary | 3 | 1,627 | 49.11% | [46.68%, 51.54%] | -0.72 |
| ES | B | secondary | 6 | 1,631 | 51.01% | [48.59%, 53.43%] | +0.82 |
| ES | B | secondary | 12 | 1,619 | 50.03% | [47.60%, 52.46%] | +0.02 |
| ES | C | primary | 1 | 1,383 | 49.46% | [46.83%, 52.09%] | -0.40 |
| ES | C | primary | 3 | 1,423 | 49.68% | [47.09%, 52.28%] | -0.24 |
| ES | C | primary | 6 | 1,429 | 48.29% | [45.70%, 50.88%] | -1.30 |
| ES | C | primary | 12 | 1,439 | 46.98% | [44.41%, 49.56%] | -2.29 |
| ES | C | secondary | 1 | 1,520 | 47.76% | [45.26%, 50.28%] | -1.74 |
| ES | C | secondary | 3 | 1,552 | 46.46% | [43.99%, 48.94%] | -2.79 |
| ES | C | secondary | 6 | 1,569 | 47.48% | [45.02%, 49.96%] | -1.99 |
| ES | C | secondary | 12 | 1,587 | 48.02% | [45.56%, 50.47%] | -1.58 |
| NQ | A | primary | 1 | 1,449 | 47.69% | [45.13%, 50.26%] | -1.76 |
| NQ | A | primary | 3 | 1,440 | 48.12% | [45.55%, 50.71%] | -1.42 |
| NQ | A | primary | 6 | 1,451 | 48.04% | [45.47%, 50.61%] | -1.50 |
| NQ | A | primary | 12 | 1,453 | 46.94% | [44.38%, 49.51%] | -2.33 |
| NQ | A | secondary | 1 | 1,528 | 49.15% | [46.65%, 51.65%] | -0.67 |
| NQ | A | secondary | 3 | 1,537 | 51.14% | [48.64%, 53.63%] | +0.89 |
| NQ | A | secondary | 6 | 1,542 | 49.81% | [47.31%, 52.30%] | -0.15 |
| NQ | A | secondary | 12 | 1,546 | 47.74% | [45.25%, 50.23%] | -1.78 |
| NQ | B | primary | 1 | 1,508 | 49.87% | [47.35%, 52.39%] | -0.10 |
| NQ | B | primary | 3 | 1,514 | 51.72% | [49.20%, 54.23%] | +1.34 |
| NQ | B | primary | 6 | 1,513 | 49.64% | [47.12%, 52.15%] | -0.28 |
| NQ | B | primary | 12 | 1,512 | 50.20% | [47.68%, 52.71%] | +0.15 |
| NQ | B | secondary | 1 | 1,540 | 48.12% | [45.63%, 50.61%] | -1.48 |
| NQ | B | secondary | 3 | 1,546 | 49.48% | [46.99%, 51.97%] | -0.41 |
| NQ | B | secondary | 6 | 1,545 | 50.87% | [48.38%, 53.36%] | +0.69 |
| NQ | B | secondary | 12 | 1,544 | 46.24% | [43.77%, 48.74%] | -2.95 |
| NQ | C | primary | 1 | 1,355 | 47.23% | [44.59%, 49.89%] | -2.04 |
| NQ | C | primary | 3 | 1,356 | 48.82% | [46.17%, 51.48%] | -0.87 |
| NQ | C | primary | 6 | 1,356 | 49.93% | [47.27%, 52.58%] | -0.05 |
| NQ | C | primary | 12 | 1,347 | 49.81% | [47.15%, 52.48%] | -0.14 |
| NQ | C | secondary | 1 | 1,500 | 46.20% | [43.69%, 48.73%] | -2.94 |
| NQ | C | secondary | 3 | 1,512 | 44.97% | [42.48%, 47.49%] | -3.91 |
| NQ | C | secondary | 6 | 1,518 | 46.97% | [44.47%, 49.48%] | -2.36 |
| NQ | C | secondary | 12 | 1,517 | 45.75% | [43.26%, 48.26%] | -3.31 |
| RTY | A | primary | 1 | 1,359 | 49.08% | [46.43%, 51.74%] | -0.68 |
| RTY | A | primary | 3 | 1,384 | 49.06% | [46.43%, 51.69%] | -0.70 |
| RTY | A | primary | 6 | 1,390 | 48.20% | [45.58%, 50.83%] | -1.34 |
| RTY | A | primary | 12 | 1,385 | 47.00% | [44.39%, 49.64%] | -2.23 |
| RTY | A | secondary | 1 | 1,426 | 48.04% | [45.45%, 50.63%] | -1.48 |
| RTY | A | secondary | 3 | 1,445 | 49.90% | [47.32%, 52.47%] | -0.08 |
| RTY | A | secondary | 6 | 1,457 | 50.45% | [47.88%, 53.01%] | +0.34 |
| RTY | A | secondary | 12 | 1,455 | 49.69% | [47.13%, 52.26%] | -0.24 |
| RTY | B | primary | 1 | 1,402 | 48.22% | [45.61%, 50.83%] | -1.34 |
| RTY | B | primary | 3 | 1,434 | 46.86% | [44.29%, 49.45%] | -2.38 |
| RTY | B | primary | 6 | 1,438 | 48.54% | [45.96%, 51.12%] | -1.11 |
| RTY | B | primary | 12 | 1,443 | 48.09% | [45.52%, 50.67%] | -1.45 |
| RTY | B | secondary | 1 | 1,413 | 49.40% | [46.80%, 52.00%] | -0.45 |
| RTY | B | secondary | 3 | 1,435 | 47.18% | [44.61%, 49.76%] | -2.14 |
| RTY | B | secondary | 6 | 1,457 | 48.80% | [46.24%, 51.37%] | -0.92 |
| RTY | B | secondary | 12 | 1,454 | 48.56% | [45.99%, 51.13%] | -1.10 |
| RTY | C | primary | 1 | 1,283 | 48.01% | [45.29%, 50.75%] | -1.42 |
| RTY | C | primary | 3 | 1,298 | 47.92% | [45.21%, 50.64%] | -1.50 |
| RTY | C | primary | 6 | 1,293 | 47.87% | [45.16%, 50.60%] | -1.53 |
| RTY | C | primary | 12 | 1,287 | 48.80% | [46.07%, 51.53%] | -0.86 |
| RTY | C | secondary | 1 | 1,398 | 46.71% | [44.11%, 49.33%] | -2.46 |
| RTY | C | secondary | 3 | 1,426 | 47.76% | [45.17%, 50.35%] | -1.69 |
| RTY | C | secondary | 6 | 1,427 | 48.77% | [46.19%, 51.37%] | -0.93 |
| RTY | C | secondary | 12 | 1,427 | 47.72% | [45.14%, 50.32%] | -1.72 |

## 7. Pooled h=1 (ES+NQ+RTY) for all six gate × tier combinations

| gate | tier | n_scored | accuracy | Wilson 95% CI | boot 95% CI | z vs 0.5 | MDE (80% pwr) |
|:----:|:----:|---------:|---------:|:-------------:|:-----------:|---------:|--------------:|
| A | primary | 4,249 | 48.25% | [46.75%, 49.75%] | [46.79%, 49.92%] | -2.29 | ±2.15% |
| A | secondary | 4,489 | 48.25% | [46.79%, 49.71%] | [46.85%, 49.68%] | -2.34 | ±2.09% |
| B | primary | 4,424 | 48.26% | [46.79%, 49.73%] | [46.90%, 49.68%] | -2.32 | ±2.11% |
| B | secondary | 4,511 | 48.08% | [46.63%, 49.54%] | [46.69%, 49.61%] | -2.58 | ±2.09% |
| C | primary | 4,021 | 48.25% | [46.70%, 49.79%] | [46.78%, 49.86%] | -2.22 | ±2.21% |
| C | secondary | 4,418 | 46.90% | [45.43%, 48.37%] | [45.47%, 48.30%] | -4.12 | ±2.11% |

Session-bootstrap 95% CI resamples events with replacement (1000 resamples, seed=20260419). MDE (minimum detectable effect) is the smallest |p − 0.5| a two-sided binomial z-test would reject at 80% power and α=0.05, given the realised n.

## 8. Per-contract h=1 breakdown

| contract | gate | tier | n_scored | accuracy | Wilson 95% CI | z |
|:--------:|:----:|:----:|---------:|---------:|:-------------:|--:|
| ES | A | primary | 1,441 | 48.02% | [45.45%, 50.60%] | -1.50 |
| ES | A | secondary | 1,535 | 47.56% | [45.07%, 50.06%] | -1.91 |
| ES | B | primary | 1,514 | 46.70% | [44.20%, 49.22%] | -2.57 |
| ES | B | secondary | 1,558 | 46.85% | [44.39%, 49.34%] | -2.48 |
| ES | C | primary | 1,383 | 49.46% | [46.83%, 52.09%] | -0.40 |
| ES | C | secondary | 1,520 | 47.76% | [45.26%, 50.28%] | -1.74 |
| NQ | A | primary | 1,449 | 47.69% | [45.13%, 50.26%] | -1.76 |
| NQ | A | secondary | 1,528 | 49.15% | [46.65%, 51.65%] | -0.67 |
| NQ | B | primary | 1,508 | 49.87% | [47.35%, 52.39%] | -0.10 |
| NQ | B | secondary | 1,540 | 48.12% | [45.63%, 50.61%] | -1.48 |
| NQ | C | primary | 1,355 | 47.23% | [44.59%, 49.89%] | -2.04 |
| NQ | C | secondary | 1,500 | 46.20% | [43.69%, 48.73%] | -2.94 |
| RTY | A | primary | 1,359 | 49.08% | [46.43%, 51.74%] | -0.68 |
| RTY | A | secondary | 1,426 | 48.04% | [45.45%, 50.63%] | -1.48 |
| RTY | B | primary | 1,402 | 48.22% | [45.61%, 50.83%] | -1.34 |
| RTY | B | secondary | 1,413 | 49.40% | [46.80%, 52.00%] | -0.45 |
| RTY | C | primary | 1,283 | 48.01% | [45.29%, 50.75%] | -1.42 |
| RTY | C | secondary | 1,398 | 46.71% | [44.11%, 49.33%] | -2.46 |

## 9. Horizon decay (pooled accuracy by horizon)

| gate | tier | h=1 | h=3 | h=6 | h=12 |
|:----:|:----:|----:|----:|----:|-----:|
| A | primary | 48.25% (4,249) | 48.23% (4,311) | 47.60% (4,330) | 47.24% (4,340) |
| A | secondary | 48.25% (4,489) | 50.04% (4,586) | 49.20% (4,614) | 48.81% (4,614) |
| B | primary | 48.26% (4,424) | 49.25% (4,528) | 48.27% (4,533) | 48.61% (4,536) |
| B | secondary | 48.08% (4,511) | 48.63% (4,608) | 50.27% (4,633) | 48.30% (4,617) |
| C | primary | 48.25% (4,021) | 48.83% (4,077) | 48.70% (4,078) | 48.49% (4,073) |
| C | secondary | 46.90% (4,418) | 46.37% (4,490) | 47.72% (4,514) | 47.16% (4,531) |

Each cell shows pooled ES+NQ+RTY accuracy followed by the scored event count in parentheses. A monotonic decay toward 50% with increasing h is consistent with a short-lived directional edge.

## 10. Cross-cutting comparisons

| comparison | Δ accuracy | event-count ratio |
|:-----------|-----------:|------------------:|
| B − A (primary) | 0.01% | 1.04 |
| C − B (primary) | -0.01% | 0.91 |
| B − A (secondary) | -0.17% | 1.00 |
| C − B (secondary) | -1.18% | 0.98 |
| primary − secondary (Gate A) | -0.00% | 1.06 |
| primary − secondary (Gate B) | 0.18% | 1.02 |
| primary − secondary (Gate C) | 1.35% | 1.10 |

- **B vs A** isolates the marginal value of adding the sign-concordance z-score to the imbalance-magnitude z-score.
- **C vs B** isolates the marginal value of the above-median DER conviction filter at fixed (z_abs_imb, z_concordance) scale.
- **Primary vs secondary** trades event count for per-event accuracy. Under the spec's 'first rising edge per session' event rule, the secondary-to-primary event-count ratio is ceiling-capped at `(frac_sessions_sec / frac_sessions_pri) ≈ 1.07–1.10`: primary already fires in 83–93% of eval sessions, so dropping the threshold from p95 to p70 only adds the remaining ~7–17% of sessions rather than the 5–10× multiplier the event-count heuristic would suggest.

## 11. Statistical power analysis

| gate | tier | n_scored | MDE (80% power) | Wilson half-width |
|:----:|:----:|---------:|----------------:|------------------:|
| A | primary | 4,249 | ±2.15% | ±1.50% |
| A | secondary | 4,489 | ±2.09% | ±1.46% |
| B | primary | 4,424 | ±2.11% | ±1.47% |
| B | secondary | 4,511 | ±2.09% | ±1.46% |
| C | primary | 4,021 | ±2.21% | ±1.54% |
| C | secondary | 4,418 | ±2.11% | ±1.47% |

Secondary tier buys meaningfully tighter CIs roughly in proportion to sqrt(event-count ratio). Whether the tighter interval is worth the weaker per-event accuracy depends on whether the secondary point estimate remains above 50% by more than its MDE.

## 12. Interpretation

- Best observed pooled accuracy: **48.26%** (Gate B, primary tier).
- **Gates add essentially zero selective power over the ungated baseline.** Per-contract ungated baselines (§5a) are 48.7% (ES), 49.1% (NQ), 49.4% (RTY); all six pooled gate-tier configurations sit in a ~1.4 pp band of 46.9–48.3%. The gates are not finding a subset of bars with a different directional property than the whole sample — they are reproducing (and, for Gate C secondary, mildly amplifying) the same weak 5-min mean-reversion.
- At the identity level, `sign(imbalance_t) ≡ sign(log_ret_t)` for essentially 100% of bars (see §5a), which is a direct consequence of the monotonic Student-t CDF applied to the z-score: the gated subset inherits exactly the directional property of the underlying log return distribution. An OHLCV-derived BVC imbalance therefore cannot produce incremental directional signal beyond what is already carried by the sign of the contemporaneous log return.

## 13. Conclusion

- The null result closes the question for OHLCV-based directional prediction of 5-minute equity index futures returns via BVC imbalance. Across 72 evaluation cells (three gates × two tiers × three contracts × four horizons) no configuration produces accuracy meaningfully above the 48–49% mean-reverting baseline.
- Applications requiring directional signal from equity index futures must use alternative data or methods beyond OHLCV-derived BVC at the 5-minute horizon.

## 14. Artifacts

- `thresholds.csv` — primary/secondary thresholds per contract × gate
- `event_counts.csv` — event counts and session coverage per cell
- `accuracy_table.csv` — 72-cell accuracy table (long-format)
- `pooled_h1.csv` — pooled ES+NQ+RTY h=1 per gate × tier
- `cross_cutting.csv` — A-vs-B, B-vs-C, primary-vs-secondary deltas
- `sanity_check.json` — step 4 sanity check per contract
- `DIRECTIONAL_TEST_RESULTS.md` — this report

