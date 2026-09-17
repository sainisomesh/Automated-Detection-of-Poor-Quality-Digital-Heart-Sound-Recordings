#!/usr/bin/env python3
"""
Bootstrap hypothesis testing + significance comparison: AST-QA vs. the Tang
et al. and Giordano et al. baselines, per Reviewer #1 Comment 1's request
for "bootstrap hypothesis testing (B=1000) and paired Student's t-tests...
to prove statistically significant superiority."

METHODOLOGY NOTE -- read before trusting these numbers (must stay visible,
not buried): there is no prior bootstrap/significance-testing code anywhere
in this repo (checked directly, 2026-09-12) -- this is a fresh implementation
of the reviewer's requested method, not a port of something that already
existed. It reuses compute_metrics.py's exact metric definitions and CI
convention (Student's t across folds) for consistency with the rest of the
codebase.

More importantly: this is explicitly framed as an UNPAIRED (independent
two-sample) comparison, not a literal "paired" test, because the two sides
were evaluated on DIFFERENT fold structures:
  - AST-QA (Table 2):    10-fold patient-level CV (published results)
  - Tang / Giordano:      5-fold patient-level CV (this revision's documented
                          deviation, CLAUDE.md Sec 9.4)
A paired test requires the same test items on both sides. Since the fold
membership differs, samples can't be paired 1:1. Instead:
  1. Pool each method's raw held-out predictions across all its own folds
     (valid because every sample is held out exactly once per method's CV).
  2. Bootstrap (B=1000): independently resample-with-replacement from each
     method's pooled predictions, compute AUROC each draw, take the
     difference. Report the empirical 95% CI and two-sided p-value of that
     difference.
  3. As a secondary, complementary check: an independent-samples (Welch's)
     t-test on the two methods' per-fold AUROC values (10 folds vs 5 folds).
Do not describe either of these as a "paired t-test on matched folds" in
any writeup -- that would misrepresent what was actually computed.

Usage:
    python compute_significance.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
AST_QA_DIR = REPO_ROOT / "reproducibility" / "results" / "per_lambda_cv" / "raw_predictions"
TANG_DIR = REPO_ROOT / "PAPER_REVISIONS" / "reviewer1_baselines_tang_leal" / "results" / "tang_full" / "raw_predictions"
GIORDANO_DIR = REPO_ROOT / "PAPER_REVISIONS" / "reviewer1_baselines_tang_leal" / "results" / "giordano_full" / "raw_predictions"
OUTPUT_DIR = REPO_ROOT / "PAPER_REVISIONS" / "reviewer1_baselines_tang_leal" / "results"

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
B = 1000
SEED = 42


def pool_predictions(base_dir: Path, lam: float, ytrue_col="y_true", prob_col="probs"):
    """Pool raw predictions across all fold subdirectories for one lambda.
    Only y_true/probs columns are read -- the filename column's name differs
    between AST-QA's CSVs ("filenames") and ours ("filename"), but it's
    never used in this computation, so that difference doesn't matter."""
    lam_dir = base_dir / f"lambda_{lam}"
    if not lam_dir.exists():
        return None, None
    y_true_all, probs_all = [], []
    for fold_dir in sorted(lam_dir.glob("fold_*")):
        csv_path = fold_dir / "predictions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        y_true_all.append(df[ytrue_col].values.astype(float))
        probs_all.append(df[prob_col].values.astype(float))
    if not y_true_all:
        return None, None
    return np.concatenate(y_true_all), np.concatenate(probs_all)


def per_fold_auroc(base_dir: Path, lam: float, ytrue_col="y_true", prob_col="probs"):
    """AUROC computed separately per fold (for the secondary Welch's t-test)."""
    lam_dir = base_dir / f"lambda_{lam}"
    if not lam_dir.exists():
        return []
    aurocs = []
    for fold_dir in sorted(lam_dir.glob("fold_*")):
        csv_path = fold_dir / "predictions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        try:
            aurocs.append(roc_auc_score(df[ytrue_col].values, df[prob_col].values))
        except ValueError:
            continue
    return aurocs


def bootstrap_auroc_diff(y_true_a, probs_a, y_true_b, probs_b, B=1000, seed=42):
    """Independent bootstrap: resample each method's pooled predictions
    separately (NOT paired -- different underlying test sets), compute
    AUROC each draw, return the array of (A - B) differences."""
    rng = np.random.default_rng(seed)
    n_a, n_b = len(y_true_a), len(y_true_b)
    diffs = np.empty(B)
    for i in range(B):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        try:
            auroc_a = roc_auc_score(y_true_a[idx_a], probs_a[idx_a])
        except ValueError:
            auroc_a = 0.5
        try:
            auroc_b = roc_auc_score(y_true_b[idx_b], probs_b[idx_b])
        except ValueError:
            auroc_b = 0.5
        diffs[i] = auroc_a - auroc_b
    return diffs


def compare_method(method_name, method_dir, ast_ytrue_col="y_true", ast_prob_col="probs",
                    method_ytrue_col="y_true", method_prob_col="probs"):
    results = {}
    for lam in LAMBDAS:
        ast_y, ast_p = pool_predictions(AST_QA_DIR, lam, ast_ytrue_col, ast_prob_col)
        meth_y, meth_p = pool_predictions(method_dir, lam, method_ytrue_col, method_prob_col)
        if ast_y is None or meth_y is None:
            print(f"  [skip] lambda={lam}: missing predictions (AST={ast_y is not None}, {method_name}={meth_y is not None})")
            continue

        ast_auroc = roc_auc_score(ast_y, ast_p)
        meth_auroc = roc_auc_score(meth_y, meth_p)

        diffs = bootstrap_auroc_diff(ast_y, ast_p, meth_y, meth_p, B=B, seed=SEED)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        frac_le0 = np.mean(diffs <= 0)
        frac_ge0 = np.mean(diffs >= 0)
        p_value_bootstrap = 2 * min(frac_le0, frac_ge0)
        p_value_bootstrap = min(p_value_bootstrap, 1.0)

        ast_fold_aurocs = per_fold_auroc(AST_QA_DIR, lam, ast_ytrue_col, ast_prob_col)
        meth_fold_aurocs = per_fold_auroc(method_dir, lam, method_ytrue_col, method_prob_col)
        if len(ast_fold_aurocs) >= 2 and len(meth_fold_aurocs) >= 2:
            t_stat, t_pvalue = stats.ttest_ind(ast_fold_aurocs, meth_fold_aurocs, equal_var=False)
        else:
            t_stat, t_pvalue = np.nan, np.nan

        results[lam] = {
            "ast_qa_auroc": float(ast_auroc),
            f"{method_name}_auroc": float(meth_auroc),
            "diff": float(ast_auroc - meth_auroc),
            "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
            "bootstrap_p_value": float(p_value_bootstrap),
            "welch_t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
            "welch_p_value": float(t_pvalue) if not np.isnan(t_pvalue) else None,
            "ast_qa_n_folds": len(ast_fold_aurocs),
            f"{method_name}_n_folds": len(meth_fold_aurocs),
        }

        sig = "***" if p_value_bootstrap < 0.001 else ("*" if p_value_bootstrap < 0.05 else "ns")
        print(f"  lambda={lam:>6}: AST-QA={ast_auroc:.4f} vs {method_name}={meth_auroc:.4f} "
              f"| diff={ast_auroc-meth_auroc:+.4f} | bootstrap p={p_value_bootstrap:.4g} {sig}")

    return results


def main():
    print("=" * 90)
    print("  AST-QA vs Tang et al. -- UNPAIRED bootstrap comparison (10-fold vs 5-fold)")
    print("=" * 90)
    tang_results = compare_method("tang", TANG_DIR)

    print()
    print("=" * 90)
    print("  AST-QA vs Giordano et al. -- UNPAIRED bootstrap comparison (10-fold vs 5-fold)")
    print("=" * 90)
    giordano_results = compare_method("giordano", GIORDANO_DIR)

    out = {
        "methodology_note": (
            "UNPAIRED comparison -- AST-QA (10-fold, published Table 2 raw predictions) vs "
            "Tang/Giordano (5-fold, this revision). Not a paired test on matched folds; see "
            "module docstring for why and exactly what was computed instead."
        ),
        "bootstrap_iterations": B,
        "seed": SEED,
        "tang_vs_ast_qa": tang_results,
        "giordano_vs_ast_qa": giordano_results,
    }
    out_path = OUTPUT_DIR / "significance_vs_ast_qa.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
