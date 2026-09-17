#!/usr/bin/env python3
"""
Paired bootstrap significance: AST versus PANNs CNN14, YAMNet and HuBERT,
compared within each unfreezing mode (frozen against frozen, fully fine-tuned
against fully fine-tuned) under the identical variable-noise training
protocol.

How this differs from compute_significance_vs_ast.py
----------------------------------------------------
compute_significance_vs_ast.py compares each backbone against the
originally published, frozen, 10-fold AST-QA noise_0_10 model. That
comparison is necessarily unpaired -- the two sides have different fold
counts, and only part of the published model's raw per-fold CSVs survives --
and its own docstring documents those limitations. It is kept as the
comparison against the published model, and its output
(results/reviewer7_backbone_swap/significance_vs_ast_qa.json) is independent
of this script.

This script instead uses the AST runs from
../reviewer1_unfreezing_ablation/ as the reference, which is the same AST
comparator cited in the manuscript's backbone-comparison table. Because that
experiment and this one build their folds identically -- the same sorted
patient and noise file lists, the same seed, the same KFold construction, and
the same per-index noise seeding for the evaluation sets -- row i of a given
lambda/fold CSV is the same test case on both sides. merged_fold() asserts
that row counts, test-file basenames and labels agree position by position
before pairing, so the pairing is checked at runtime rather than assumed.
This is the same approach as
../reviewer1_baselines/compute_significance_paired.py.

Pairing enables a paired bootstrap (resample test cases once and score both
models on the same resample) and a paired t-test over per-fold AUROCs, which
removes between-fold variance from the comparison. Note that the t-test is
computed over N_FOLDS observations, so its power is limited; the bootstrap
interval over pooled test cases is the primary statistic.

Both sides must have been run with the same fold count, which N_FOLDS below
fixes; see ../../README.md for the fold counts of the checked-in results.

Writes results/reviewer7_backbone_swap/significance_full_paired.json.

Usage:
    python compute_significance_full_paired.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# Inputs and outputs are resolved relative to this file's location
# (reproducibility/revision/src/reviewer7_backbone_swap/), so the script can be
# run from any working directory.
REVISION_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = REVISION_ROOT / "results" / "reviewer1_unfreezing_ablation"
BACKBONE_DIR = REVISION_ROOT / "results" / "reviewer7_backbone_swap"
OUTPUT_DIR = BACKBONE_DIR

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
N_FOLDS = 3
B = 1000
SEED = 42

# Result directory for each backbone in each mode: frozen-mode runs use the
# bare backbone name, fully fine-tuned runs carry a "_full" suffix.
BACKBONES = {
    "panns": {"frozen": "panns", "full": "panns_full"},
    "yamnet": {"frozen": "yamnet", "full": "yamnet_full"},
    "hubert": {"frozen": "hubert", "full": "hubert_full"},
}


def load_fold(base_dir: Path, lam: float, fold: int) -> pd.DataFrame:
    """Load one method's raw predictions for a single (lambda, fold).

    Positives are reduced to their file basename so that the two experiments
    can be aligned even though they recorded different absolute paths;
    negatives are already the literal string "noise" and are left as is.
    """
    csv_path = base_dir / f"lambda_{lam}" / f"fold_{fold}" / "predictions.csv"
    df = pd.read_csv(csv_path)
    df["basename"] = df["filename"].apply(lambda p: Path(p).name if p != "noise" else "noise")
    return df[["basename", "y_true", "probs"]]


def merged_fold(ast_dir: Path, meth_dir: Path, lam: float, fold: int):
    """Align AST's and one method's predictions for a single (lambda, fold).

    Pairing is positional, which is only valid if both experiments evaluated
    the same test cases in the same order. The three assertions below verify
    that precondition -- equal row counts, identical test-file basenames at
    every position, identical labels at every position -- and fail loudly
    rather than silently producing a meaningless paired statistic.
    """
    ast_df = load_fold(ast_dir, lam, fold).reset_index(drop=True)
    meth_df = load_fold(meth_dir, lam, fold).reset_index(drop=True)
    assert len(ast_df) == len(meth_df), (
        f"lambda={lam} fold={fold} row-count mismatch (ast={len(ast_df)}, meth={len(meth_df)})"
    )
    assert (ast_df["basename"] == meth_df["basename"]).all(), (
        f"lambda={lam} fold={fold} filenames diverge at some row position -- "
        f"positional pairing is not valid here"
    )
    assert (ast_df["y_true"] == meth_df["y_true"]).all(), (
        f"lambda={lam} fold={fold} has mismatched y_true at the same row position"
    )
    return pd.DataFrame({
        "y_true_ast": ast_df["y_true"],
        "probs_ast": ast_df["probs"],
        "probs_meth": meth_df["probs"],
    })


def paired_bootstrap_auroc_diff(y_true, probs_ast, probs_meth, B=1000, seed=42):
    """Paired bootstrap of the AUROC difference (AST minus method).

    Each iteration resamples test-case indices once and scores both models on
    that same resample, so the shared test-set variance cancels. Returns the
    B resampled differences; an undefined AUROC in a resample contributes the
    chance value 0.5.
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


def compare(ast_dir: Path, meth_dir: Path, name: str):
    """Compare AST against one method at every lambda in the sweep.

    Per lambda: per-fold AUROCs for the paired t-test, then the folds pooled
    into a single test set for the pooled AUROCs, the paired-bootstrap 95%
    percentile interval, and a two-sided bootstrap p-value.
    """
    results = {}
    for lam in LAMBDAS:
        fold_frames = [merged_fold(ast_dir, meth_dir, lam, fold) for fold in range(1, N_FOLDS + 1)]
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
        frac_le0, frac_ge0 = np.mean(diffs <= 0), np.mean(diffs >= 0)
        p_value_bootstrap = min(2 * min(frac_le0, frac_ge0), 1.0)

        t_stat, t_pvalue = stats.ttest_rel(fold_aurocs_ast, fold_aurocs_meth)

        results[lam] = {
            "ast_auroc": float(ast_auroc),
            f"{name}_auroc": float(meth_auroc),
            "diff": float(ast_auroc - meth_auroc),
            "paired_bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
            "paired_bootstrap_p_value": float(p_value_bootstrap),
            "paired_t_p_value": float(t_pvalue),
        }
        sig = "***" if p_value_bootstrap < 0.001 else ("*" if p_value_bootstrap < 0.05 else "ns")
        print(f"  lambda={lam:>6}: AST={ast_auroc:.4f} vs {name}={meth_auroc:.4f} "
              f"diff={ast_auroc - meth_auroc:+.4f} p={p_value_bootstrap:.4g} {sig}")
    return results


def main():
    out = {
        "methodology_note": (
            "PAIRED comparison of AST against each alternative backbone, within matching "
            "unfreezing modes, using the 3-fold runs on both sides: "
            "results/reviewer1_unfreezing_ablation/{frozen,full}_3fold/ for AST and "
            "results/reviewer7_backbone_swap/{panns,yamnet,hubert}[_full]/ for the "
            "alternatives. Both sides construct folds, file ordering and synthesized "
            "negatives identically, so predictions correspond row by row; the loader "
            "asserts matching row counts, basenames and labels before pairing. This "
            "enables a paired bootstrap and a paired t-test on per-fold AUROCs. Distinct "
            "from compute_significance_vs_ast.py, which compares against the published "
            "frozen 10-fold model and is unpaired."
        ),
        "bootstrap_iterations": B, "n_folds": N_FOLDS, "seed": SEED,
    }
    for mode in ("frozen", "full"):
        # The unfreezing ablation's results are stored per fold count, as
        # {mode}_3fold and {mode}_5fold. This comparison reads the 3fold
        # directories because pairing requires the AST reference to have been
        # run with the same fold count as the backbone-swap results (N_FOLDS).
        ast_dir = ABLATION_DIR / f"{mode}_3fold" / "raw_predictions"
        for backbone, dirs in BACKBONES.items():
            meth_dir = BACKBONE_DIR / dirs[mode] / "raw_predictions"
            print(f"=== AST ({mode}) vs {backbone} ({mode}) ===")
            out[f"{backbone}_{mode}_vs_ast_{mode}"] = compare(ast_dir, meth_dir, backbone)
            print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "significance_full_paired.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
