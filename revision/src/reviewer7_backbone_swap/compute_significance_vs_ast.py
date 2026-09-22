#!/usr/bin/env python3
"""
Significance testing of each swapped backbone (PANNs CNN14 / YAMNet / HuBERT)
against the originally published AST-QA model, both with a frozen encoder and
both trained with the Variable-Noise (noise_0_10) strategy.

The reference here is deliberately AST-QA's own noise_0_10 strategy from
../../../results/three_strategies_cv/, not the per-lambda matched benchmark of
the paper's Table 2. The question being asked is whether AST's advantage is
architectural, which requires every model to have been trained under the same
noise strategy; the matched per-lambda benchmark trains a separate model at
each noise level and is therefore the wrong reference for this comparison.
(The classical baselines in ../reviewer1_baselines/ use that other reference
because they are compared against the published per-lambda numbers.)

This script is the frozen-versus-frozen comparison against the published
model. Its companion, compute_significance_full_paired.py, instead compares
the fully fine-tuned backbones against the fully fine-tuned AST from
../reviewer1_unfreezing_ablation/ and, because those two experiments share
identical per-fold test sets, can use a genuinely paired test. The two are
complementary and should not be conflated.

Test construction and its limitations
-------------------------------------
The comparison performed here is UNPAIRED: each side's held-out predictions
are pooled across its own folds and then resampled independently. It must not
be described as a paired test on matched folds. Two properties of the
underlying data make pairing impossible:

  1. The two sides were run with different numbers of folds. The published
     AST-QA noise_0_10 results are 10-fold; the backbone-swap runs in this
     package use the fold count documented in ../../README.md. Fold k of one
     experiment is therefore not the same test set as fold k of the other.
  2. Only part of AST-QA's raw per-fold prediction CSVs survives. Of the ten
     published folds, only fold_1, fold_2 and fold_10 exist in
     ../../../results/three_strategies_cv/raw_predictions/, and fold_2 is
     itself partial: its 'noise_10' directory is empty, 'noise_0_10' contains
     only lambda_0.0.csv, and that file carries one row whose filename,
     y_true and probs are all NaN (an artifact of the original cloud run,
     dropped on load). In practice two AST-QA folds are usable at every
     lambda except 0.0, where three are. The aggregated numbers in
     final_results.json were computed from the complete 10-fold run and are
     unaffected, but the raw CSVs needed for a resampling test cannot be
     regenerated, as ../../README.md records.

Consequently, the AST-QA side of every comparison below is pooled from only
about two folds rather than the published ten. pool_ast_qa() emits a warning
whenever fewer than ten folds are pooled and records the actual count in the
`ast_qa_n_folds` field of every result entry; any use of these numbers should
state both that count and the unpaired construction.

Directory layouts differ between the two sides:
  - AST-QA (three_strategies_cv):    fold_<K>/noise_0_10/lambda_<L>.csv
  - Each backbone (this experiment): lambda_<L>/fold_<K>/predictions.csv

Writes results/reviewer7_backbone_swap/significance_vs_ast_qa.json.

Usage:
    python compute_significance_vs_ast.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# All inputs and outputs are resolved relative to this file's location
# (reproducibility/revision/src/reviewer7_backbone_swap/), so the script can be
# run from any working directory. REVISION_ROOT is reproducibility/revision/;
# REPRODUCIBILITY_ROOT is the original package alongside it, which holds the
# published three_strategies_cv results used as the AST-QA reference.
REVISION_ROOT = Path(__file__).resolve().parents[2]
REPRODUCIBILITY_ROOT = REVISION_ROOT.parent
AST_QA_DIR = REPRODUCIBILITY_ROOT / "results" / "three_strategies_cv" / "raw_predictions"
BACKBONE_RESULTS_DIR = REVISION_ROOT / "results" / "reviewer7_backbone_swap"
OUTPUT_DIR = REVISION_ROOT / "results" / "reviewer7_backbone_swap"

BACKBONES = ["panns", "yamnet", "hubert"]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
B = 1000
SEED = 42


def pool_ast_qa(lam: float):
    """Pool AST-QA's noise_0_10 predictions at one lambda.

    Returns (y_true, probs, n_folds_used) concatenated over every fold that
    has a readable raw-predictions CSV for this lambda, or (None, None, 0) if
    none does. Rows with NaN in y_true or probs are dropped.

    The returned fold count is not constant across the sweep, because the
    surviving raw CSVs are incomplete: see the module docstring. Callers are
    expected to report n_folds_used alongside any statistic derived from this
    pool. The gap is specific to three_strategies_cv; the per_lambda_cv
    results used by the classical baselines are complete at 10 folds x 10
    lambdas.
    """
    y_true_all, probs_all = [], []
    n_folds_used = 0
    for fold_dir in sorted(AST_QA_DIR.glob("fold_*")):
        csv_path = fold_dir / "noise_0_10" / f"lambda_{lam}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).dropna(subset=["y_true", "probs"])
        if df.empty:
            continue
        y_true_all.append(df["y_true"].values.astype(float))
        probs_all.append(df["probs"].values.astype(float))
        n_folds_used += 1
    if not y_true_all:
        return None, None, 0
    return np.concatenate(y_true_all), np.concatenate(probs_all), n_folds_used


def pool_ast_qa_per_fold_auroc(lam: float):
    """AST-QA's AUROC computed separately within each available fold, at one
    lambda. These per-fold values are the observations fed to the Welch
    t-test; folds whose AUROC is undefined (single-class) are skipped."""
    aurocs = []
    for fold_dir in sorted(AST_QA_DIR.glob("fold_*")):
        csv_path = fold_dir / "noise_0_10" / f"lambda_{lam}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).dropna(subset=["y_true", "probs"])
        if df.empty:
            continue
        try:
            aurocs.append(roc_auc_score(df["y_true"].values, df["probs"].values))
        except ValueError:
            continue
    return aurocs


def pool_backbone(backbone: str, lam: float):
    """Pool one backbone's frozen-mode predictions at one lambda, across all
    of its own folds. Returns (y_true, probs), or (None, None) if this
    backbone has no results for that lambda."""
    lam_dir = BACKBONE_RESULTS_DIR / backbone / "raw_predictions" / f"lambda_{lam}"
    if not lam_dir.exists():
        return None, None
    y_true_all, probs_all = [], []
    for fold_dir in sorted(lam_dir.glob("fold_*")):
        csv_path = fold_dir / "predictions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        y_true_all.append(df["y_true"].values.astype(float))
        probs_all.append(df["probs"].values.astype(float))
    if not y_true_all:
        return None, None
    return np.concatenate(y_true_all), np.concatenate(probs_all)


def pool_backbone_per_fold_auroc(backbone: str, lam: float):
    """One backbone's per-fold AUROCs at one lambda, as the second sample for
    the Welch t-test."""
    lam_dir = BACKBONE_RESULTS_DIR / backbone / "raw_predictions" / f"lambda_{lam}"
    if not lam_dir.exists():
        return []
    aurocs = []
    for fold_dir in sorted(lam_dir.glob("fold_*")):
        csv_path = fold_dir / "predictions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        try:
            aurocs.append(roc_auc_score(df["y_true"].values, df["probs"].values))
        except ValueError:
            continue
    return aurocs


def bootstrap_auroc_diff(y_true_a, probs_a, y_true_b, probs_b, B=1000, seed=42):
    """Two-sample (unpaired) bootstrap of the AUROC difference.

    Each side's pooled predictions are resampled with replacement
    independently, because the two sides come from different test sets and
    different numbers of folds and cannot be matched case by case. Returns
    the B resampled differences (side A minus side B); an undefined AUROC in
    a resample contributes the chance value 0.5.
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


def compare_backbone(backbone: str):
    """Compare AST-QA against one backbone at every lambda in the sweep.

    Per lambda, reports the pooled AUROC of each side, their difference, a
    95% percentile interval and two-sided p-value from the unpaired bootstrap,
    a Welch t-test over the two sets of per-fold AUROCs (reported only when
    both sides have at least two usable folds), and the number of folds each
    side contributed.
    """
    results = {}
    for lam in LAMBDAS:
        ast_y, ast_p, ast_n_folds_pooled = pool_ast_qa(lam)
        bb_y, bb_p = pool_backbone(backbone, lam)
        if ast_y is None or bb_y is None:
            print(f"  [skip] lambda={lam}: missing predictions (AST-QA={ast_y is not None}, {backbone}={bb_y is not None})")
            continue
        if ast_n_folds_pooled < 10:
            print(f"  [WARNING] lambda={lam}: AST-QA pooled from only {ast_n_folds_pooled}/10 folds "
                  f"(known local data gap in three_strategies_cv/fold_2 -- see pool_ast_qa docstring)")

        ast_auroc = roc_auc_score(ast_y, ast_p)
        bb_auroc = roc_auc_score(bb_y, bb_p)

        diffs = bootstrap_auroc_diff(ast_y, ast_p, bb_y, bb_p, B=B, seed=SEED)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        frac_le0 = np.mean(diffs <= 0)
        frac_ge0 = np.mean(diffs >= 0)
        p_value_bootstrap = min(2 * min(frac_le0, frac_ge0), 1.0)

        ast_fold_aurocs = pool_ast_qa_per_fold_auroc(lam)
        bb_fold_aurocs = pool_backbone_per_fold_auroc(backbone, lam)
        if len(ast_fold_aurocs) >= 2 and len(bb_fold_aurocs) >= 2:
            t_stat, t_pvalue = stats.ttest_ind(ast_fold_aurocs, bb_fold_aurocs, equal_var=False)
        else:
            t_stat, t_pvalue = np.nan, np.nan

        results[lam] = {
            "ast_qa_auroc": float(ast_auroc),
            f"{backbone}_auroc": float(bb_auroc),
            "diff": float(ast_auroc - bb_auroc),
            "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
            "bootstrap_p_value": float(p_value_bootstrap),
            "welch_t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
            "welch_p_value": float(t_pvalue) if not np.isnan(t_pvalue) else None,
            "ast_qa_n_folds": len(ast_fold_aurocs),
            f"{backbone}_n_folds": len(bb_fold_aurocs),
        }

        sig = "***" if p_value_bootstrap < 0.001 else ("*" if p_value_bootstrap < 0.05 else "ns")
        print(f"  lambda={lam:>6}: AST-QA={ast_auroc:.4f} vs {backbone}={bb_auroc:.4f} "
              f"| diff={ast_auroc - bb_auroc:+.4f} | bootstrap p={p_value_bootstrap:.4g} {sig}")

    return results


def main():
    out = {
        "methodology_note": (
            "UNPAIRED comparison. The AST-QA side is the published variable-noise "
            "(noise_0_10) model evaluated at 10-fold, read from "
            "results/three_strategies_cv/; each alternative backbone was evaluated at "
            "5-fold. Because the two sides use different fold structures, the held-out "
            "items do not correspond and this is not a paired test. A further limitation: "
            "only part of the published model's raw per-fold prediction CSVs survives "
            "(fold_1, fold_2 and fold_10, with fold_2 itself incomplete), so the AST-QA "
            "side of each comparison below is pooled from the folds that are available "
            "rather than all 10. The 'ast_qa_n_folds' field on each entry records how "
            "many were actually used. For the paired, same-fold-count comparison against "
            "the fully fine-tuned model, see compute_significance_full_paired.py."
        ),
        "bootstrap_iterations": B,
        "seed": SEED,
    }
    any_ran = False
    for backbone in BACKBONES:
        lam_dir_exists = (BACKBONE_RESULTS_DIR / backbone / "raw_predictions").exists()
        print("=" * 90)
        print(f"  AST-QA (noise_0_10) vs {backbone} -- UNPAIRED bootstrap comparison (AST-QA 10-fold [~2 pooled] vs {backbone} 5-fold)")
        print("=" * 90)
        if not lam_dir_exists:
            print(f"  [skip] no raw_predictions found for '{backbone}' yet -- run "
                  f"train_backbone_swap_cv.py --backbone {backbone} first.")
            out[f"{backbone}_vs_ast_qa"] = None
            continue
        out[f"{backbone}_vs_ast_qa"] = compare_backbone(backbone)
        any_ran = True
        print()

    out_path = OUTPUT_DIR / "significance_vs_ast_qa.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    if any_ran:
        print(f"Saved to {out_path}")
    else:
        print(f"No backbone results found yet -- wrote a placeholder to {out_path}. "
              f"Re-run this after the real training run completes.")


if __name__ == "__main__":
    main()
