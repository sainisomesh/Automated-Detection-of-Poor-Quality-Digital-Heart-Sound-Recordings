#!/usr/bin/env python3
"""
Genuinely PAIRED bootstrap significance: AST (full fine-tuned) vs. PANNs
CNN14, YAMNet, and HuBERT (each full fine-tuned), at 3-fold, under the
identical variable-noise training protocol.

WHY THIS SCRIPT EXISTS, AND HOW IT DIFFERS FROM compute_significance_vs_ast.py
-------------------------------------------------------------------------
compute_significance_vs_ast.py compares each backbone against AST-QA's
originally-published, FROZEN, 10-fold noise_0_10 strategy -- and its own
docstring discloses a serious data-quality problem: only ~2 of the 10
published AST-QA folds exist locally, and the comparison is unpaired
(different fold counts, 3 vs. 10). That script and its output
(results/significance_vs_ast_qa.json) are untouched here.

This script instead compares against reviewer1_unfreezing_ablation's own
`full/` (and `frozen/`) 3-fold run -- the exact same experiment already used
as the AST-QA reference in the main paper's backbone-comparison table
(Table~\ref{tab:backbone-swap}). VERIFIED (2026-09-16), not assumed: for
every fold and lambda checked, the ordered sequence of test-file basenames
(positives AND "noise"-labeled negatives) is IDENTICAL between the
ablation's `full`/`frozen` raw predictions and each backbone's `_full`/
plain raw predictions -- both experiments share the same sorted patient/
noise-file lists, same seed, same KFold construction, so position i is the
literal same test case in both. This makes row-position pairing valid, same
reasoning as reviewer1_baselines_tang_leal/src/compute_significance_paired.py.

Usage:
    python compute_significance_full_paired.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
ABLATION_DIR = REPO_ROOT / "PAPER_REVISIONS" / "reviewer1_unfreezing_ablation" / "results"
BACKBONE_DIR = REPO_ROOT / "PAPER_REVISIONS" / "reviewer7_backbone_swap" / "results"
OUTPUT_DIR = BACKBONE_DIR

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
N_FOLDS = 3
B = 1000
SEED = 42

BACKBONES = {
    "panns": {"frozen": "panns", "full": "panns_full"},
    "yamnet": {"frozen": "yamnet", "full": "yamnet_full"},
    "hubert": {"frozen": "hubert", "full": "hubert_full"},
}


def load_fold(base_dir: Path, lam: float, fold: int) -> pd.DataFrame:
    csv_path = base_dir / f"lambda_{lam}" / f"fold_{fold}" / "predictions.csv"
    df = pd.read_csv(csv_path)
    df["basename"] = df["filename"].apply(lambda p: Path(p).name if p != "noise" else "noise")
    return df[["basename", "y_true", "probs"]]


def merged_fold(ast_dir: Path, meth_dir: Path, lam: float, fold: int):
    ast_df = load_fold(ast_dir, lam, fold).reset_index(drop=True)
    meth_df = load_fold(meth_dir, lam, fold).reset_index(drop=True)
    assert len(ast_df) == len(meth_df), (
        f"BUG: lambda={lam} fold={fold} row-count mismatch (ast={len(ast_df)}, meth={len(meth_df)})"
    )
    assert (ast_df["basename"] == meth_df["basename"]).all(), (
        f"BUG: lambda={lam} fold={fold} filenames diverge at some row position -- "
        f"positional pairing is NOT valid here"
    )
    assert (ast_df["y_true"] == meth_df["y_true"]).all(), (
        f"BUG: lambda={lam} fold={fold} has mismatched y_true at the same row position"
    )
    return pd.DataFrame({
        "y_true_ast": ast_df["y_true"],
        "probs_ast": ast_df["probs"],
        "probs_meth": meth_df["probs"],
    })


def paired_bootstrap_auroc_diff(y_true, probs_ast, probs_meth, B=1000, seed=42):
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
            "PAIRED comparison, AST vs. each backbone, at their matching 3-fold runs "
            "(reviewer1_unfreezing_ablation/results/{frozen,full}/ vs. "
            "reviewer7_backbone_swap/results/{panns,yamnet,hubert}[_full]/). Verified "
            "byte-for-byte (2026-09-16) that both sides share identical per-fold, "
            "per-lambda test-file ordering (positives and synthesized negatives), enabling "
            "row-position pairing. Distinct from compute_significance_vs_ast.py, which "
            "compares against the published frozen 10-fold model and discloses its own "
            "unpaired, partial-data limitations."
        ),
        "bootstrap_iterations": B, "n_folds": N_FOLDS, "seed": SEED,
    }
    for mode in ("frozen", "full"):
        ast_dir = ABLATION_DIR / mode / "raw_predictions"
        for backbone, dirs in BACKBONES.items():
            meth_dir = BACKBONE_DIR / dirs[mode] / "raw_predictions"
            print(f"=== AST ({mode}) vs {backbone} ({mode}) ===")
            out[f"{backbone}_{mode}_vs_ast_{mode}"] = compare(ast_dir, meth_dir, backbone)
            print()

    out_path = OUTPUT_DIR / "significance_full_paired.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
