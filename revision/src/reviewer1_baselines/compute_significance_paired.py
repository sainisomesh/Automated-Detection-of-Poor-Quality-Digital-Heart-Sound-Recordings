#!/usr/bin/env python3
"""
Paired significance comparison of AST-QA against the Tang et al. and Giordano
et al. baselines at every noise level lambda.

The AST-QA model is the fully fine-tuned, variable-noise-trained model
evaluated with 5-fold patient-level CV, read from
`../../results/reviewer1_unfreezing_ablation/full_5fold_variable/`.

Why rows can be paired
The baselines and this AST-QA model use the same 5-fold construction, so
fold i holds out the same recordings in all three pipelines:

- Patient ID is `f.name.split('_')[0]`; IDs are sorted and split with
  `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Noise corpora are listed the same way (`sorted(list(data_root.rglob(...)))`)
  for ICBHI 2017 and ESC-50 + UrbanSound8K.
- Test samples use the same per-sample noise seed `random.Random(42 + seed_idx)`,
  with seed_idx 0..N-1 for positives and N..2N-1 for negatives in a fold of
  N heart recordings.
- Composite-noise and RMS-mixing formulas are the same.

So row i is the same mixed audio in each pipeline. Noise-only rows have no
real filename, so they are matched by position; `merged_fold` asserts that
row counts, basenames and labels agree.

Method
  1. Per lambda and fold, align each baseline's predictions with AST-QA's by
     row position (after the checks above).
  2. Pool the aligned pairs across the 5 folds (each sample is held out once).
  3. Paired bootstrap (B = 1000): the same resample indices are applied to
     both methods in each iteration.
  4. Paired Student's t-test (`scipy.stats.ttest_rel`) on the per-fold AUROCs.

Usage:
    python compute_significance_paired.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# Paths are relative to this file (revision/src/reviewer1_baselines/), so the
# script runs from any working directory.
REVISION_ROOT = Path(__file__).resolve().parents[2]
AST_QA_DIR = REVISION_ROOT / "results" / "reviewer1_unfreezing_ablation" / "full_5fold_variable" / "raw_predictions"
TANG_DIR = REVISION_ROOT / "results" / "reviewer1_baselines" / "tang_full" / "raw_predictions"
GIORDANO_DIR = REVISION_ROOT / "results" / "reviewer1_baselines" / "giordano_full" / "raw_predictions"
OUTPUT_DIR = REVISION_ROOT / "results" / "reviewer1_baselines"

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
N_FOLDS = 5
B = 1000          # bootstrap iterations
SEED = 42


def load_fold(base_dir: Path, lam: float, fold: int) -> pd.DataFrame:
    """Read one fold's predictions for one lambda. Positive rows store a full
    path, so only the basename is kept; negative rows are "noise"."""
    csv_path = base_dir / f"lambda_{lam}" / f"fold_{fold}" / "predictions.csv"
    df = pd.read_csv(csv_path)
    df["basename"] = df["filename"].apply(lambda p: Path(p).name if p != "noise" else "noise")
    return df[["basename", "y_true", "probs"]]


def merged_fold(lam: float, fold: int, method_dir: Path):
    """Row-align one fold of AST-QA and baseline predictions for one lambda.

    Alignment is by row position because all negative rows share the name
    "noise". Row count, basenames and labels are checked first so a mismatch
    in sample order fails here instead of producing mispaired results.

    Returns a frame of (y_true_ast, probs_ast, probs_meth), one row per
    held-out sample.
    """
    ast_df = load_fold(AST_QA_DIR, lam, fold).reset_index(drop=True)
    meth_df = load_fold(method_dir, lam, fold).reset_index(drop=True)
    assert len(ast_df) == len(meth_df), (
        f"lambda={lam} fold={fold}: row-count mismatch "
        f"(ast={len(ast_df)}, meth={len(meth_df)}); the two pipelines did not "
        f"evaluate the same number of held-out samples, so pairing is invalid"
    )
    assert (ast_df["basename"] == meth_df["basename"]).all(), (
        f"lambda={lam} fold={fold}: filenames differ at some row position; the "
        f"two pipelines' test-set iteration order does not match, so positional "
        f"pairing is invalid"
    )
    assert (ast_df["y_true"] == meth_df["y_true"]).all(), (
        f"lambda={lam} fold={fold}: labels differ at some row position; sample "
        f"construction diverged between the two pipelines"
    )
    merged = pd.DataFrame({
        "y_true_ast": ast_df["y_true"],
        "probs_ast": ast_df["probs"],
        "probs_meth": meth_df["probs"],
    })
    return merged


def paired_bootstrap_auroc_diff(y_true, probs_ast, probs_meth, B=1000, seed=42):
    """Paired bootstrap of the AUROC difference between two methods.

    Each iteration applies one set of resample indices to both methods.
    Returns `B` values of AUROC(AST-QA) - AUROC(baseline).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        try:
            auroc_ast = roc_auc_score(yt, probs_ast[idx])
        except ValueError:
            auroc_ast = 0.5
        try:
            auroc_meth = roc_auc_score(yt, probs_meth[idx])
        except ValueError:
            auroc_meth = 0.5
        diffs[i] = auroc_ast - auroc_meth
    return diffs


def compare_method(method_name, method_dir):
    """Compare AST-QA against one baseline across all lambdas.

    Per lambda: per-fold and pooled AUROC for each method, the paired
    bootstrap interval and p-value for their difference, and the paired
    t-test on the per-fold AUROCs.
    """
    results = {}
    for lam in LAMBDAS:
        fold_frames = [merged_fold(lam, fold, method_dir) for fold in range(1, N_FOLDS + 1)]
        fold_aurocs_ast = [roc_auc_score(f["y_true_ast"], f["probs_ast"]) for f in fold_frames]
        fold_aurocs_meth = [roc_auc_score(f["y_true_ast"], f["probs_meth"]) for f in fold_frames]

        pooled = pd.concat(fold_frames, ignore_index=True)
        y_true = pooled["y_true_ast"].values.astype(float)
        probs_ast = pooled["probs_ast"].values.astype(float)
        probs_meth = pooled["probs_meth"].values.astype(float)

        ast_auroc = roc_auc_score(y_true, probs_ast)
        meth_auroc = roc_auc_score(y_true, probs_meth)

        diffs = paired_bootstrap_auroc_diff(y_true, probs_ast, probs_meth, B=B, seed=SEED)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        # Two-sided p-value: twice the smaller tail of the bootstrap
        # differences. Resolution is 1/B, so 0.0 means p < 0.001.
        frac_le0 = np.mean(diffs <= 0)
        frac_ge0 = np.mean(diffs >= 0)
        p_value_bootstrap = min(2 * min(frac_le0, frac_ge0), 1.0)

        t_stat, t_pvalue = stats.ttest_rel(fold_aurocs_ast, fold_aurocs_meth)

        results[lam] = {
            "ast_qa_unfrozen_auroc": float(ast_auroc),
            f"{method_name}_auroc": float(meth_auroc),
            "diff": float(ast_auroc - meth_auroc),
            "paired_bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
            "paired_bootstrap_p_value": float(p_value_bootstrap),
            "paired_t_statistic": float(t_stat),
            "paired_t_p_value": float(t_pvalue),
            "n_folds": N_FOLDS,
            "n_pooled_samples": int(len(pooled)),
            "ast_qa_unfrozen_fold_aurocs": [float(x) for x in fold_aurocs_ast],
            f"{method_name}_fold_aurocs": [float(x) for x in fold_aurocs_meth],
        }

        sig = "***" if p_value_bootstrap < 0.001 else ("*" if p_value_bootstrap < 0.05 else "ns")
        print(f"  lambda={lam:>6}: AST-QA(unfrozen)={ast_auroc:.4f} vs {method_name}={meth_auroc:.4f} "
              f"| diff={ast_auroc - meth_auroc:+.4f} | paired bootstrap p={p_value_bootstrap:.4g} {sig} "
              f"| paired t p={t_pvalue:.4g}")

    return results


def main():
    print("=" * 90)
    print("  AST-QA (full-unfreeze, variable-noise, 5-fold) vs Tang et al., paired comparison")
    print("=" * 90)
    tang_results = compare_method("tang", TANG_DIR)

    print()
    print("=" * 90)
    print("  AST-QA (full-unfreeze, variable-noise, 5-fold) vs Giordano et al., paired comparison")
    print("=" * 90)
    giordano_results = compare_method("giordano", GIORDANO_DIR)

    out = {
        "methodology_note": (
            "Paired comparison against AST-QA with a fully fine-tuned backbone, trained with "
            "variable noise (U[0,10]), 5-fold "
            "(revision/results/reviewer1_unfreezing_ablation/full_5fold_variable/). "
            "Tang, Giordano and AST-QA score the same held-out files per fold and lambda, so "
            "predictions are paired row by row: paired bootstrap (same resample indices for "
            "both methods) and paired t-test (scipy.stats.ttest_rel) on per-fold AUROCs."
        ),
        "ast_qa_source": "revision/results/reviewer1_unfreezing_ablation/full_5fold_variable/",
        "bootstrap_iterations": B,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "tang_vs_ast_qa_unfrozen": tang_results,
        "giordano_vs_ast_qa_unfrozen": giordano_results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "significance_vs_ast_qa_unfrozen_paired.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
