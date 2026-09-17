#!/usr/bin/env python3
"""
PAIRED significance comparison of AST-QA against the Tang et al. and Giordano
et al. baselines, at every noise intensity lambda.

Which AST-QA model this compares against
----------------------------------------
The fully fine-tuned (unfrozen backbone), variable-noise-trained AST-QA model
evaluated with 5-fold patient-level CV, read from
`../../results/reviewer1_unfreezing_ablation/full_5fold_variable/`.

This is a different model, and a different statistical test, from
`compute_significance.py`, which runs an UNPAIRED comparison against the
originally published frozen-backbone, 10-fold AST-QA results. The two scripts
write separate output files and neither supersedes the other: one reports
against the published model, this one against the fully fine-tuned model.

Why pairing is valid here
-------------------------
The baselines and this AST-QA model both run 5-fold patient-level CV with the
same construction, so fold i holds out the same recordings in all three
pipelines:

- Identical patient-level fold construction: patient ID taken as
  `f.name.split('_')[0]`, IDs sorted, and `KFold(n_splits=5, shuffle=True,
  random_state=42)` over those IDs.
- Identical noise corpora enumeration (`sorted(list(data_root.rglob(...)))`)
  for ICBHI 2017 and ESC-50 + UrbanSound8K.
- Identical per-sample noise seeding for evaluation samples:
  `random.Random(42 + seed_idx)`, with seed_idx running 0..N-1 over the
  positives and N..2N-1 over the negatives for a fold of N heart recordings.
- Identical composite-noise and RMS-mixing formulas.

The same seed, the same candidate file lists, and the same PRNG therefore draw
the same noise clip at the same row position in every pipeline, so row i is
the same synthesized audio in each. Negative ("noise"-only) rows carry no
real filename in any of the pipelines, so they cannot be matched by name;
positional alignment is what makes them pairable, and the asserts in
`merged_fold` fail loudly if that construction order ever diverges.

Method
------
  1. Per lambda and fold, align each baseline's predictions with AST-QA's by
     row position, after asserting equal row counts, basenames, and labels.
  2. Pool the aligned pairs across all 5 folds. Each sample is held out
     exactly once, and pairing survives pooling because rows stay aligned
     within a fold before concatenation.
  3. Paired bootstrap (B = 1000): draw one set of resample indices per
     iteration and apply it to both methods, which preserves the sample-level
     correlation between their scores on the same held-out cases. This is the
     substantive difference from `compute_significance.py`, where the two
     pools are resampled independently.
  4. Paired Student's t-test (`scipy.stats.ttest_rel`) on the per-fold AUROCs,
     which is meaningful here because fold i denotes the same held-out
     patients for both methods.

Usage:
    python compute_significance_paired.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# Paths are resolved relative to this file, which lives at
# <package>/revision/src/reviewer1_baselines/, so the script runs from any
# working directory.
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
    """Read one fold's predictions for one lambda, reduced to the columns
    needed for pairing. Positive rows store an absolute source path, so the
    basename is extracted for comparison across pipelines; negative rows are
    written as the literal "noise" by every pipeline and are left as-is."""
    csv_path = base_dir / f"lambda_{lam}" / f"fold_{fold}" / "predictions.csv"
    df = pd.read_csv(csv_path)
    df["basename"] = df["filename"].apply(lambda p: Path(p).name if p != "noise" else "noise")
    return df[["basename", "y_true", "probs"]]


def merged_fold(lam: float, fold: int, method_dir: Path):
    """Row-align one fold of AST-QA and baseline predictions for one lambda.

    Alignment is by row position rather than a name-based merge: the negative
    rows all share the key "noise" and so cannot be merged by name. See the
    module docstring for why identical construction order makes positional
    alignment correct rather than merely convenient. Row count, basenames, and
    labels are asserted first, so any divergence in construction order surfaces
    as a failure here instead of as silently mispaired results.

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

    Each iteration draws a single set of resample indices and applies it to
    both methods' aligned score arrays, so every draw compares the two methods
    on the same resampled cases. Returns the array of `B` values of
    AUROC(AST-QA) - AUROC(baseline).
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
        # Two-sided p-value: twice the smaller tail mass of the bootstrap
        # difference distribution. Its resolution is limited to 1/B, so a
        # reported 0.0 means "below the 1/B = 0.001 resolution of this
        # bootstrap", not an exact zero.
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
    print("  AST-QA (full-unfreeze, variable-noise, 5-fold) vs Tang et al. -- PAIRED comparison")
    print("=" * 90)
    tang_results = compare_method("tang", TANG_DIR)

    print()
    print("=" * 90)
    print("  AST-QA (full-unfreeze, variable-noise, 5-fold) vs Giordano et al. -- PAIRED comparison")
    print("=" * 90)
    giordano_results = compare_method("giordano", GIORDANO_DIR)

    out = {
        "methodology_note": (
            "PAIRED comparison -- AST-QA here is the fully fine-tuned (unfrozen backbone), "
            "variable-noise (U[0,10]) trained, 5-fold model from revision/results/"
            "reviewer1_unfreezing_ablation/full_5fold_variable/, not the originally published "
            "frozen 10-fold model (see compute_significance.py / significance_vs_ast_qa.json "
            "for that unpaired comparison, which is kept separately and not superseded). "
            "Tang/Giordano and this AST-QA model score the same held-out files per fold per "
            "lambda, which enables row-level pairing: a paired bootstrap (the same resample "
            "indices applied to both methods) and a paired Student's t-test "
            "(scipy.stats.ttest_rel) on matched per-fold AUROCs."
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
