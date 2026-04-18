"""phase2_physics_validation.py — Six validation checks for physics features.

Per PHYSICS_FEATURE_SPECIFICATION.md §3-§4. Produces
    runs/2026-04-18_physics_feature_expansion/PHYSICS_FEATURE_VALIDATION.md

Checks:
  1. No all-NaN features per contract.
  2. Summary statistics vs expected ranges.
  3. Full 15-feature correlation matrix per contract (flag |r| > 0.85 pairs).
  4. sign_concordance orthogonality with imbalance_t preserved (|r| < 0.05) and
     vs all 8 new features (flag |r| > 0.4).
  5. Feature availability rate per (contract, feature).
  6. Temporal stability — first vs second half normalized mean shift.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

ROOT = Path("runs/2026-04-18_physics_feature_expansion")
REPORT_PATH = ROOT / "PHYSICS_FEATURE_VALIDATION.md"
CONTRACTS = ["ES", "NQ", "RTY"]

# Seven locked BVC + seven new physics = 14 features.
# v_star_C was computed and evaluated in the initial 15-feature run but pruned
# 2026-04-18 after initial validation review (structural redundancy with DER at
# r=0.767 and body_to_range at r=0.855 on all three contracts). See
# PHYSICS_FEATURE_VALIDATION.md §10 "Feature Selection Decision".
BVC_FEATURES = ['z', 'imbalance_t', 'der', 'sign_concordance',
                'clv_mean', 'clv_var', 'subbar_imbalance']
NEW_FEATURES = ['sigma_yz', 'body_to_range', 'gel_fraction',
                'wick_asymmetry', 'amihud', 'ko_W', 'polar_order_P']
ALL_FEATURES = BVC_FEATURES + NEW_FEATURES

# Expected-range flags for Check 2 (spec §3 Check 2 + §1 per-feature expectations).
EXPECTED = {
    'body_to_range':  {'median_min': 0.30, 'median_max': 0.55,
                       'min': 0.0, 'max': 1.0},
    'gel_fraction':   {'median_min': 0.20, 'median_max': 0.30,
                       'min': 0.0, 'max': 0.5},
    'wick_asymmetry': {'mean_center': 0.0, 'mean_tol': 1e-4},
    'polar_order_P':  {'median_min': 0.08, 'median_max': 0.20,
                       'min': 0.0, 'max': 1.0},
}


def load_contracts():
    out = {}
    for c in CONTRACTS:
        with open(ROOT / f"phase2_features_expanded_{c}.pkl", 'rb') as f:
            out[c] = pickle.load(f)
    return out


def valid_mask(df):
    return df['is_valid_bar'].astype(bool)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check1_allnan(dfs):
    rows = []
    for c, df in dfs.items():
        vm = valid_mask(df)
        for feat in NEW_FEATURES:
            s = df.loc[vm, feat]
            rows.append({'contract': c, 'feature': feat,
                         'n_valid': int(vm.sum()),
                         'n_nonan': int(s.notna().sum()),
                         'all_nan': bool(s.notna().sum() == 0)})
    t = pd.DataFrame(rows)
    passed = not t['all_nan'].any()
    return t, passed


def check2_stats(dfs):
    rows = []
    for c, df in dfs.items():
        vm = valid_mask(df)
        for feat in NEW_FEATURES:
            s = df.loc[vm, feat].dropna()
            flag = ''
            exp = EXPECTED.get(feat, {})
            if 'mean_center' in exp:
                if abs(s.mean() - exp['mean_center']) > exp['mean_tol']:
                    flag = (f"mean {s.mean():+.4g} outside "
                            f"{exp['mean_center']} ± {exp['mean_tol']:g}")
            if not flag and 'median_min' in exp:
                m = s.median()
                if not (exp['median_min'] <= m <= exp['median_max']):
                    flag = (f"median {m:.3g} outside "
                            f"[{exp['median_min']}, {exp['median_max']}]")
            if not flag and 'max' in exp and s.max() > exp['max'] + 1e-9:
                flag = f"max {s.max():.4g} > expected {exp['max']}"
            if not flag and 'min' in exp and s.min() < exp['min'] - 1e-9:
                flag = f"min {s.min():.4g} < expected {exp['min']}"
            rows.append({
                'contract': c, 'feature': feat,
                'mean': s.mean(), 'median': s.median(), 'std': s.std(),
                'min': s.min(), 'p01': s.quantile(0.01),
                'p99': s.quantile(0.99), 'max': s.max(),
                'flag': flag,
            })
    t = pd.DataFrame(rows)
    passed = (t['flag'] == '').all()
    return t, passed


# Pairs known-by-construction to have |r|>0.85 — not grounds for failing Check 3.
# imbalance_t is a monotonic transform of z; subbar_imbalance is the
# volume-weighted sub-bar BVC aggregation, also using the same Student-t CDF of
# z, so it co-moves with both z and imbalance_t.
EXPECTED_HIGH_CORR_PAIRS = {
    ('z', 'imbalance_t'),
    ('z', 'subbar_imbalance'),
    ('imbalance_t', 'subbar_imbalance'),
}


def check3_corr(dfs):
    mats, flags_all, flags_novel = {}, {}, {}
    for c, df in dfs.items():
        vm = valid_mask(df)
        sub = df.loc[vm, ALL_FEATURES].dropna()
        cm = sub.corr()
        mats[c] = cm
        pairs_all, pairs_novel = [], []
        cols = cm.columns.tolist()
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = cm.loc[a, b]
                if abs(r) > 0.85:
                    key = tuple(sorted([a, b]))
                    expected = key in {tuple(sorted(p)) for p in EXPECTED_HIGH_CORR_PAIRS}
                    pairs_all.append((a, b, float(r), expected))
                    if not expected:
                        pairs_novel.append((a, b, float(r)))
        flags_all[c] = pairs_all
        flags_novel[c] = pairs_novel
    # Check passes if no *novel* |r|>0.85 pairs appear. Expected structural pairs
    # (z ↔ imbalance_t family) are reported but do not fail the check.
    passed = all(len(v) == 0 for v in flags_novel.values())
    return mats, flags_all, flags_novel, passed


def check4_orthogonality(dfs):
    rows = []
    for c, df in dfs.items():
        vm = valid_mask(df)
        sub = df.loc[vm, ['sign_concordance', 'imbalance_t'] + NEW_FEATURES].dropna()
        r_bvc = sub['sign_concordance'].corr(sub['imbalance_t'])
        rec = {'contract': c, 'n': len(sub), 'r_sign_imbalance': r_bvc,
               'orth_preserved': bool(abs(r_bvc) < 0.05)}
        for feat in NEW_FEATURES:
            rec[f'r_sign_{feat}'] = sub['sign_concordance'].corr(sub[feat])
        rows.append(rec)
    t = pd.DataFrame(rows)
    passed = bool(t['orth_preserved'].all())
    # Flag |r|>0.4 vs new features
    new_flags = {}
    for _, row in t.iterrows():
        bad = [f for f in NEW_FEATURES if abs(row[f'r_sign_{f}']) > 0.4]
        new_flags[row['contract']] = bad
    return t, new_flags, passed


def check5_availability(dfs):
    rows = []
    for c, df in dfs.items():
        vm = valid_mask(df)
        nv = int(vm.sum())
        for feat in NEW_FEATURES:
            nn = int(df.loc[vm, feat].notna().sum())
            rate = nn / nv if nv > 0 else float('nan')
            min_rate = 0.85 if (feat == 'amihud' and c == 'RTY') else \
                       (0.90 if feat == 'amihud' else 0.95)
            rows.append({'contract': c, 'feature': feat,
                         'n_valid': nv, 'n_nonan': nn,
                         'availability': rate,
                         'min_expected': min_rate,
                         'flag': rate < 0.90})
    t = pd.DataFrame(rows)
    passed = not t['flag'].any()
    return t, passed


def check6_stability(dfs):
    rows = []
    for c, df in dfs.items():
        vm = valid_mask(df)
        sub = df.loc[vm].sort_index()
        mid = len(sub) // 2
        first, second = sub.iloc[:mid], sub.iloc[mid:]
        for feat in NEW_FEATURES:
            a = first[feat].dropna()
            b = second[feat].dropna()
            m1, m2 = a.mean(), b.mean()
            s1, s2 = a.std(), b.std()
            combined_std = sub[feat].std()
            if combined_std and not np.isnan(combined_std) and combined_std > 0:
                shift = (m2 - m1) / combined_std
            else:
                shift = float('nan')
            rows.append({
                'contract': c, 'feature': feat,
                'mean_first': m1, 'mean_second': m2,
                'std_first': s1, 'std_second': s2,
                'norm_shift': shift,
                'flag': bool(abs(shift) > 2.0) if not np.isnan(shift) else False,
            })
    t = pd.DataFrame(rows)
    passed = not t['flag'].any()
    return t, passed


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(v, prec=4):
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        if v == 0:
            return "0"
        av = abs(v)
        if av >= 1e4 or av < 1e-3:
            return f"{v:.{prec}e}"
        return f"{v:.{prec}f}"
    return str(v)


def df_to_md(df, cols=None, precision=4):
    if cols is None:
        cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], precision) for c in cols) + " |")
    return "\n".join(lines)


def corr_to_md(cm, precision=3):
    cols = cm.columns.tolist()
    header = "| feature | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for r in cols:
        cells = []
        for c in cols:
            v = cm.loc[r, c]
            cells.append(f"{v:+.{precision}f}" if not np.isnan(v) else "NaN")
        lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def status_line(name, passed):
    return f"- **{name}:** {'PASS ✓' if passed else 'FAIL ✗'}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dfs):
    results = {}
    results['c1_table'], results['c1_pass'] = check1_allnan(dfs)
    results['c2_table'], results['c2_pass'] = check2_stats(dfs)
    (results['c3_mats'], results['c3_flags_all'],
     results['c3_flags_novel'], results['c3_pass']) = check3_corr(dfs)
    results['c4_table'], results['c4_flags'], results['c4_pass'] = check4_orthogonality(dfs)
    results['c5_table'], results['c5_pass'] = check5_availability(dfs)
    results['c6_table'], results['c6_pass'] = check6_stability(dfs)
    return results


def spot_check(dfs, n=3, seed=20260418):
    rng = np.random.default_rng(seed)
    rows = []
    for c, df in dfs.items():
        vm = valid_mask(df)
        sub = df.loc[vm]
        idxs = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
        for i in idxs:
            row = sub.iloc[i]
            rec = {'contract': c, 'ts': str(sub.index[i]),
                   'instrument_id': int(row['instrument_id'])}
            for feat in ALL_FEATURES:
                rec[feat] = row[feat]
            rows.append(rec)
    return pd.DataFrame(rows)



def row_counts(dfs):
    rows = []
    for c, df in dfs.items():
        vm = valid_mask(df)
        rows.append({
            'contract': c,
            'total_rows': len(df),
            'valid_rows': int(vm.sum()),
            'columns_in': 24,
            'columns_out': int(df.shape[1]),
            'features_added': len(NEW_FEATURES),
        })
    return pd.DataFrame(rows)


def render_report(dfs, results, spot, counts):
    r = results
    c1_overall = r['c1_pass']
    c2_overall = r['c2_pass']
    c3_overall = r['c3_pass']
    c4_overall = r['c4_pass']
    c5_overall = r['c5_pass']
    c6_overall = r['c6_pass']
    all_pass = all([c1_overall, c2_overall, c3_overall,
                    c4_overall, c5_overall, c6_overall])

    md = []
    md.append("# Physics Feature Validation Report")
    md.append("")
    md.append("**Date:** 2026-04-18  ")
    md.append("**Input:** `runs/2026-04-17_regime_validation/phase2_features_cleaned_{ES,NQ,RTY}.pkl` (baseline)  ")
    md.append("**Output:** `runs/2026-04-18_physics_feature_expansion/phase2_features_expanded_{ES,NQ,RTY}.pkl`  ")
    md.append("**Spec:** `PHYSICS_FEATURE_SPECIFICATION.md`  ")
    md.append("**Feature set:** 7 locked BVC + 7 new physics = 14 features "
              "(v_star_C pruned 2026-04-18 — see §10).")
    md.append("")

    # Section 1 — executive summary
    md.append("## 1. Executive Summary")
    md.append("")
    md.append(f"**Overall status:** {'ALL SIX CHECKS PASS ✓' if all_pass else 'ONE OR MORE CHECKS FLAGGED ✗'}")
    md.append("")
    md.append(status_line("Check 1 — No all-NaN features", c1_overall))
    md.append(status_line("Check 2 — Summary stats within expected ranges", c2_overall))
    md.append(status_line("Check 3 — No novel |r| > 0.85 pairs (structural z/imb pairs excluded)", c3_overall))
    md.append(status_line("Check 4 — sign_concordance ⟂ imbalance_t preserved (|r|<0.05)", c4_overall))
    md.append(status_line("Check 5 — All feature availability ≥ 90%", c5_overall))
    md.append(status_line("Check 6 — Temporal stability (|norm shift| ≤ 2.0)", c6_overall))
    md.append("")
    md.append("")

    # Section 2 — pipeline verification
    md.append("## 2. Pipeline Verification")
    md.append("")
    md.append("**OHLCV availability:** `open`, `high`, `low`, `close`, `volume` are already "
              "present in every input bar. No join "
              "from raw parquet was required. `imbalance_t`, `sigma_gk`, `is_valid_bar`, "
              "`instrument_id` are also already present.")
    md.append("")
    md.append("**Row and column counts (input → output):**")
    md.append("")
    md.append(df_to_md(counts))
    md.append("")
    md.append("**Spot check — 3 random valid bars per contract (all 14 features):**")
    md.append("")
    md.append(df_to_md(spot))
    md.append("")

    # Section 3 — check 1
    md.append("## 3. Check 1 — No All-NaN Features")
    md.append("")
    md.append("Non-NaN counts per (contract, feature) restricted to `is_valid_bar == True`.")
    md.append("")
    md.append(df_to_md(r['c1_table']))
    md.append("")
    md.append(f"**Check 1 result:** {'PASS — no all-NaN features.' if c1_overall else 'FAIL'}")
    md.append("")

    # Section 4 — check 2
    md.append("## 4. Check 2 — Summary Statistics vs Expected Ranges")
    md.append("")
    md.append("Statistics computed on valid bars only. `flag` column is non-empty when the "
              "realized statistic deviates from the spec §1 expectation for that feature.")
    md.append("")
    for c in CONTRACTS:
        md.append(f"### {c}")
        md.append("")
        sub = r['c2_table'][r['c2_table']['contract'] == c].drop(columns=['contract'])
        md.append(df_to_md(sub))
        md.append("")
    flagged = r['c2_table'][r['c2_table']['flag'] != '']
    if len(flagged):
        md.append("**Flagged entries:**")
        md.append("")
        md.append(df_to_md(flagged[['contract', 'feature', 'flag']]))
    else:
        md.append("**No entries flagged.**")
    md.append("")
    md.append(f"**Check 2 result:** {'PASS' if c2_overall else 'FAIL'}")
    md.append("")

    # Section 5 — check 3
    md.append("## 5. Check 3 — Full 14-Feature Correlation Matrix")
    md.append("")
    md.append("Pearson correlation on rows where `is_valid_bar == True` and none of the 14 "
              "features are NaN. Pairs with |r| > 0.85 are flagged.")
    md.append("")
    for c in CONTRACTS:
        md.append(f"### {c}")
        md.append("")
        cm = r['c3_mats'][c]
        md.append(corr_to_md(cm))
        md.append("")
        pairs = r['c3_flags_all'][c]
        if pairs:
            md.append("**Flagged |r|>0.85 pairs:**")
            md.append("")
            md.append("| feature_a | feature_b | r | expected by construction |")
            md.append("|---|---|---|---|")
            for a, b, rv, exp in pairs:
                md.append(f"| {a} | {b} | {rv:+.4f} | {'yes' if exp else 'no'} |")
        else:
            md.append("**No pairs with |r| > 0.85.**")
        md.append("")
    md.append("Expected-by-construction pairs (do not fail Check 3): "
              "`z ↔ imbalance_t`, `z ↔ subbar_imbalance`, `imbalance_t ↔ subbar_imbalance` — "
              "all three are monotonic / near-monotonic transforms of the same underlying "
              "standardized return z.")
    md.append("")
    md.append(f"**Check 3 result:** {'PASS (no novel |r|>0.85 pairs)' if c3_overall else 'FAIL — novel pairs flagged; see tables above.'}")
    md.append("")

    # Section 6 — check 4
    md.append("## 6. Check 4 — sign_concordance Orthogonality Preserved")
    md.append("")
    md.append("Locked baseline requirement: `|corr(sign_concordance, imbalance_t)| < 0.05` "
              "per contract. Secondary: `|corr(sign_concordance, new_feature)| < 0.4` for "
              "each of the eight new features.")
    md.append("")
    t4 = r['c4_table']
    core_cols = ['contract', 'n', 'r_sign_imbalance', 'orth_preserved']
    md.append(df_to_md(t4[core_cols]))
    md.append("")
    md.append("**sign_concordance vs each new feature:**")
    md.append("")
    detail_cols = ['contract'] + [f'r_sign_{f}' for f in NEW_FEATURES]
    md.append(df_to_md(t4[detail_cols]))
    md.append("")
    any_secondary = any(len(v) > 0 for v in r['c4_flags'].values())
    if any_secondary:
        md.append("**Secondary flags (|r|>0.4):**")
        for c, bad in r['c4_flags'].items():
            if bad:
                md.append(f"- **{c}:** {', '.join(bad)}")
    else:
        md.append("**No |r|>0.4 secondary correlations.**")
    md.append("")
    md.append(f"**Check 4 result:** {'PASS' if c4_overall else 'FAIL'}")
    md.append("")

    # Section 7 — check 5
    md.append("## 7. Check 5 — Feature Availability Rate")
    md.append("")
    md.append("Availability = `count(non-NaN) / count(is_valid_bar)`. Flag threshold: <0.90.")
    md.append("")
    md.append(df_to_md(r['c5_table']))
    md.append("")
    md.append(f"**Check 5 result:** {'PASS' if c5_overall else 'FAIL'}")
    md.append("")

    # Section 8 — check 6
    md.append("## 8. Check 6 — Temporal Stability")
    md.append("")
    md.append("Valid bars split in half by time. `norm_shift = (mean_second − mean_first) / "
              "std(full_valid)`. Flag when `|norm_shift| > 2.0`.")
    md.append("")
    md.append(df_to_md(r['c6_table']))
    md.append("")
    flagged = r['c6_table'][r['c6_table']['flag']]
    if len(flagged):
        md.append("**Flagged entries:**")
        md.append("")
        md.append(df_to_md(flagged[['contract', 'feature', 'mean_first',
                                    'mean_second', 'norm_shift']]))
    else:
        md.append("**No entries with |norm_shift| > 2.0.**")
    md.append("")
    md.append(f"**Check 6 result:** {'PASS' if c6_overall else 'FAIL'}")
    md.append("")

    # Section 9 — decision
    md.append("## 9. Decision and Next Steps")
    md.append("")
    if all_pass:
        md.append("All six checks pass on the 14-feature set. The expanded feature pickles "
                  "are ready for downstream volatility-prediction work.")
    else:
        md.append("One or more checks surfaced flags. See the sections above.")
    md.append("")

    # Section 10 — feature selection decision
    md.append("## 10. Feature Selection Decision")
    md.append("")
    md.append("An initial 15-feature validation run surfaced a novel |r|>0.85 pair "
              "(`v_star_C ↔ body_to_range`) on all three contracts. After review, "
              "`v_star_C` was dropped from the candidate set before downstream training.")
    md.append("")
    md.append("### Three reasons")
    md.append("")
    md.append("1. **Structural redundancy with DER** — `corr(v_star_C, DER) ≈ 0.767` on all "
              "three contracts. DER is an existing directional-efficiency measure "
              "(`|bar_body| / sum_abs_subbar_returns`). `v_star_C` captures "
              "substantially overlapping information through a different normalization "
              "(session-EMA of `|imbalance_t|`), so the marginal information it adds over "
              "the existing feature is small.")
    md.append("2. **Near-redundancy with `body_to_range`** — `corr(v_star_C, body_to_range) "
              "≈ 0.855` on all three contracts, exceeding the Check 3 flag threshold "
              "(|r|>0.85) and not on the spec §3 \"expected pairs to watch\" list. "
              "Keeping both would split tree-based-model importance between "
              "near-duplicates, introducing seed-dependent importance rankings.")
    md.append("3. **Interpretability preference for `body_to_range`** — when two features "
              "measure directional efficiency via different normalizations, "
              "`body_to_range = |close − open| / (high − low)` is preferred: it is "
              "self-contained per bar, uses only OHLC (no session-state, no EMA warmup), "
              "has a direct geometric interpretation on the candle, and does not require a "
              "baseline that resets at every contract roll. `v_star_C` depends on a "
              "per-instrument session-mean EMA that re-warms at every `instrument_id` "
              "change and has no additional physical interpretation once DER is already in "
              "the set.")
    md.append("")
    md.append("### SHAP-based pruning rejected")
    md.append("")
    md.append("An alternative option — \"run with all 15 and use SHAP to prune\" — was "
              "rejected. SHAP values on correlated features are unstable across seeds: "
              "the importance allocation between a correlated pair is essentially "
              "arbitrary, so SHAP-based pruning would not provide a reproducible "
              "selection rule.")
    md.append("")
    md.append("### Final 14-feature set")
    md.append("")
    md.append("| # | feature | source |")
    md.append("|---|---|---|")
    md.append("| 1 | z | baseline BVC |")
    md.append("| 2 | imbalance_t | baseline BVC |")
    md.append("| 3 | der | baseline BVC |")
    md.append("| 4 | sign_concordance | baseline BVC |")
    md.append("| 5 | clv_mean | baseline BVC |")
    md.append("| 6 | clv_var | baseline BVC |")
    md.append("| 7 | subbar_imbalance | baseline BVC |")
    md.append("| 8 | sigma_yz | new physics |")
    md.append("| 9 | body_to_range | new physics |")
    md.append("| 10 | gel_fraction | new physics |")
    md.append("| 11 | wick_asymmetry | new physics |")
    md.append("| 12 | amihud | new physics |")
    md.append("| 13 | ko_W | new physics |")
    md.append("| 14 | polar_order_P | new physics |")
    md.append("")
    md.append(f"**Result on the 14-feature set:** "
              f"{'all six checks pass cleanly without any gate override.' if all_pass else 'one or more checks still flag — see sections above.'}")
    md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    dfs = load_contracts()
    counts = row_counts(dfs)
    spot = spot_check(dfs)
    results = run(dfs)
    report = render_report(dfs, results, spot, counts)
    REPORT_PATH.write_text(report, encoding='utf-8')
    print(f"Wrote {REPORT_PATH}  ({len(report):,} chars)")
    # Brief pass/fail summary to stdout
    for name, key in [("Check 1", 'c1_pass'), ("Check 2", 'c2_pass'),
                      ("Check 3", 'c3_pass'), ("Check 4", 'c4_pass'),
                      ("Check 5", 'c5_pass'), ("Check 6", 'c6_pass')]:
        print(f"  {name}: {'PASS' if results[key] else 'FAIL'}")
