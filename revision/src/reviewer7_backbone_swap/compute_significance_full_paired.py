#!/usr/bin/env python3
"""
Paired bootstrap significance tests: AST versus PANNs CNN14, YAMNet and
HuBERT, within each unfreezing mode (frozen vs frozen, fully fine-tuned vs
fully fine-tuned), all trained with variable noise and evaluated with 5-fold
patient-level CV.

The AST reference is the 5-fold runs of the unfreezing ablation
(../reviewer1_unfreezing_ablation/, directories frozen_5fold/ and
full_5fold_variable/). That experiment and this one build folds the same way
(same sorted patient and noise file lists, seed, KFold split and per-index
noise seeding), so row i of a lambda/fold CSV is the same test case on both
sides. merged_fold() checks row counts, basenames and labels before pairing.
Same approach as ../reviewer1_baselines/compute_significance_paired.py.

For each lambda we report a paired bootstrap of the pooled AUROC difference
(resample test cases once, score both models on the same resample) and a
paired t-test over per-fold AUROCs. The t-test has only N_FOLDS observations,
so the bootstrap interval is the main statistic.

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

# Paths are relative to this file, so the script runs from any directory.
REVISION_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = REVISION_ROOT / "results" / "reviewer1_unfreezing_ablation"
BACKBONE_DIR = REVISION_ROOT / "results" / "reviewer7_backbone_swap"
OUTPUT_DIR = BACKBONE_DIR

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
N_FOLDS = 5
B = 1000
SEED = 42

# Result directory per backbone and mode ("_full" = fully fine-tuned).
BACKBONES = {
    "panns": {"frozen": "panns", "full": "panns_full"},
    "yamnet": {"frozen": "yamnet", "full": "yamnet_full"},
    "hubert": {"frozen": "hubert", "full": "hubert_full"},
}

# AST runs from the unfreezing ablation. For full fine-tuning we use
# full_5fold_variable, the run trained with variable noise like the backbones.
ABLATION_MODE_DIRS = {"frozen": "frozen_5fold", "full": "full_5fold_variable"}


def load_fold(base_dir: Path, lam: float, fold: int) -> pd.DataFrame:
    """Load one method's raw predictions for a single (lambda, fold).

    Positive filenames are reduced to basenames because the two experiments
    recorded different absolute paths. Negatives are stored as "noise".
    """
    csv_path = base_dir / f"lambda_{lam}" / f"fold_{fold}" / "predictions.csv"
    df = pd.read_csv(csv_path)
    df["basename"] = df["filename"].apply(lambda p: Path(p).name if p != "noise" else "noise")
    return df[["basename", "y_true", "probs"]]


def merged_fold(ast_dir: Path, meth_dir: Path, lam: float, fold: int):
    """Align AST's and one method's predictions for a single (lambda, fold).

    Pairing is by row position, so both runs must have evaluated the same
    test cases in the same order. The asserts check row counts, basenames and
    labels.
    """
    ast_df = load_fold(ast_dir, lam, fold).reset_index(drop=True)
    meth_df = load_fold(meth_dir, lam, fold).reset_index(drop=True)
    assert len(ast_df) == len(meth_df), (
        f"lambda={lam} fold={fold} row-count mismatch (ast={len(ast_df)}, meth={len(meth_df)})"
    )
    assert (ast_df["basename"] == meth_df["basename"]).all(), (
        f"lambda={lam} fold={fold} filenames diverge at some row position: "
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

    Each iteration draws one resample of test cases and scores both models on
    it. Returns the B differences. If a resample has only one class, its
    AUROC is set to 0.5.
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
    """Compare AST with one method at every lambda.

    Per lambda: paired t-test on per-fold AUROCs, and on the pooled folds the
    AUROCs, 95% percentile bootstrap interval of the difference and a
    two-sided bootstrap p-value.
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
            "Paired comparison of AST against each alternative backbone in the same "
            "unfreezing mode, 5-fold on both sides: "
            "results/reviewer1_unfreezing_ablation/{frozen_5fold,full_5fold_variable}/ for "
            "AST and results/reviewer7_backbone_swap/{panns,yamnet,hubert}[_full]/ for the "
            "other backbones. Folds, file order and noise-only negatives are built the same "
            "way in both, so predictions are paired row by row (row counts, basenames and "
            "labels are checked). Paired bootstrap on pooled AUROC and paired t-test on "
            "per-fold AUROCs."
        ),
        "bootstrap_iterations": B, "n_folds": N_FOLDS, "seed": SEED,
    }
    for mode in ("frozen", "full"):
        ast_dir = ABLATION_DIR / ABLATION_MODE_DIRS[mode] / "raw_predictions"
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
