#!/usr/bin/env python3
"""
UNPAIRED significance comparison of AST-QA against the Tang et al. and
Giordano et al. baselines, at every noise intensity lambda.

Which AST-QA model this compares against
----------------------------------------
The originally published, frozen-backbone AST-QA evaluated with 10-fold
patient-level CV (the source of the published per-lambda table), read from
`../../../results/per_lambda_cv/raw_predictions/`.

For the paired comparison against the fully fine-tuned, 5-fold AST-QA model,
see `compute_significance_paired.py`; the two scripts answer different
questions and write to different output files. Neither supersedes the other.

Why unpaired
------------
The two sides of this comparison were evaluated under different fold
structures -- 10 patient-level folds for the published AST-QA results, 5 for
the baselines in this package. A paired test requires the same held-out items
on both sides, which fold membership here does not provide, so samples cannot
be matched 1:1. The comparison is therefore an independent two-sample one:

  1. Pool each method's raw held-out predictions across its own folds. This is
     valid because each sample is held out exactly once within a method's CV.
  2. Bootstrap (B = 1000): resample each method's pooled predictions
     independently with replacement, compute AUROC for each draw, and take the
     difference. Report the empirical 95% interval and two-sided p-value of
     that difference distribution.
  3. As a secondary check, run an independent-samples (Welch's) t-test on the
     two methods' per-fold AUROC values (10 values vs 5).

Neither test is a paired t-test on matched folds, and results from this script
should not be described as one.

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

# Paths are resolved relative to this file, which lives at
# <package>/revision/src/reviewer1_baselines/, so the script runs from any
# working directory.
REVISION_ROOT = Path(__file__).resolve().parents[2]
REPRODUCIBILITY_ROOT = REVISION_ROOT.parent
AST_QA_DIR = REPRODUCIBILITY_ROOT / "results" / "per_lambda_cv" / "raw_predictions"
TANG_DIR = REVISION_ROOT / "results" / "reviewer1_baselines" / "tang_full" / "raw_predictions"
GIORDANO_DIR = REVISION_ROOT / "results" / "reviewer1_baselines" / "giordano_full" / "raw_predictions"
OUTPUT_DIR = REVISION_ROOT / "results" / "reviewer1_baselines"

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
B = 1000          # bootstrap iterations
SEED = 42


def pool_predictions(base_dir: Path, lam: float, ytrue_col="y_true", prob_col="probs"):
    """Concatenate raw held-out predictions across every fold for one lambda.

    Only the label and score columns are read. The filename column is named
    "filenames" in the published AST-QA prediction CSVs and "filename" in the
    baselines', but it plays no part in an unpaired comparison. Returns
    (None, None) if the lambda directory or its prediction files are absent.
    """
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
    """AUROC per fold for one lambda, used by the secondary Welch's t-test.
    Folds whose predictions contain a single class are skipped."""
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
    """Unpaired bootstrap of the AUROC difference between two methods.

    Each iteration resamples the two methods' pooled predictions
    independently, since they come from different held-out sets, and records
    AUROC(A) - AUROC(B). Returns the array of `B` differences, from which the
    percentile interval and p-value are derived.
    """
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
    """Compare AST-QA against one baseline across all lambdas.

    Per lambda: pooled AUROC for each method, the unpaired bootstrap interval
    and p-value for their difference, and the secondary Welch's t-test on
    per-fold AUROCs. Lambdas with predictions missing on either side are
    reported as skipped and omitted from the results.
    """
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
        # Two-sided p-value: twice the smaller tail mass of the bootstrap
        # difference distribution. Its resolution is limited to 1/B, so a
        # reported 0.0 means "below the 1/B = 0.001 resolution of this
        # bootstrap", not an exact zero.
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "significance_vs_ast_qa.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
