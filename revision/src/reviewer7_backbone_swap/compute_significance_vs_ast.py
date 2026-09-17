#!/usr/bin/env python3
"""
Bootstrap significance testing: each backbone (PANNs / YAMNet / HuBERT) vs.
AST-QA's own Variable-Noise (noise_0_10) strategy. This is the actual
comparison Reviewer #7 Comment 2 asks for: "freeze each backbone, attach an
identical head, train on the identical Variable-Noise strategy, evaluate
across all 10 lambda levels" -- i.e. AST-QA vs. everything else under the
SAME training strategy, not against Table 2's per-lambda matched-benchmark
upper bound (that comparison is what reviewer1_baselines_tang_leal/ uses for
Tang/Giordano, and is the wrong reference point here).

METHODOLOGY NOTE -- same disclosed limitation as
reviewer1_baselines_tang_leal/src/compute_significance.py, for the same
reason: AST-QA's noise_0_10 strategy was published at 10-fold
(reproducibility/results/three_strategies_cv/), while the backbone-swap
experiments ACTUALLY RAN at 3-fold, not the 5-fold this docstring used to
say. Corrected 2026-09-13 after pulling the real completed Vertex AI results
from GCS and checking `final_results.json`'s own `n_folds` field directly
(=3) rather than trusting this file's stale prose. This 3-fold run is the
CLAUDE.md Sec 9.4 "settled exception": a deliberate first pass on real cloud
hardware to confirm all three backbones train end-to-end before committing
budget to the real 5-fold run -- which is STILL PENDING, not something this
comparison should be mistaken for. Whatever this script reports is provisional
on that basis, not the final Reviewer #7 Comment 2 number. On top of the
10-fold-vs-3-fold mismatch, this is an UNPAIRED (independent two-sample)
bootstrap comparison: pool each side's own held-out predictions across its
own folds, then resample each side independently. See the sibling script's
docstring for the full reasoning; it applies here unchanged. Do not describe
this as a "paired test on matched folds" in any writeup.

Directory layouts differ between the two sides being compared:
  - AST-QA (three_strategies_cv):  fold_X/noise_0_10/lambda_Y.csv
  - Each backbone (this experiment): lambda_Y/fold_X/predictions.csv

SECOND, LARGER DISCLOSED LIMITATION (found + accepted 2026-09-13): on top of
the 3-fold-vs-10-fold mismatch above, the LOCAL copy of AST-QA's noise_0_10
raw predictions is itself incomplete -- only fold_1, fold_2, and fold_10 of
the published 10 exist on this machine at all (folds 3-9 are simply absent
locally, most likely never fully synced down from the original Vertex
AI/GCS run given this project's recurring local-disk constraints), and
fold_2 is further broken (missing 'noise_10' entirely, missing 9/10 lambda
files for 'noise_0_10', and the one file it does have contains a corrupted
NaN row). Net effect, verified directly: only 2 usable AST-QA folds are
available at every lambda except 0.0 (which has 3). `final_results.json`'s
aggregated numbers still look like genuine 10-fold statistics, so the
complete data likely still exists on GCS -- recovering it was explicitly
declined (2026-09-13) in favor of proceeding with whatever is available
locally, same spirit as the 5-vs-10-fold tradeoff already accepted for
Tang/Giordano. `pool_ast_qa()` logs a WARNING whenever fewer than 10 folds
are pooled and records the actual count in every result entry
(`ast_qa_n_folds`) so this is never silently glossed over. **Any writeup
using these numbers must state AST-QA's side of this comparison is pooled
from only ~2 folds, not the full published 10** -- on top of stating the
comparison is unpaired.

Usage:
    python compute_significance_vs_ast.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
AST_QA_DIR = REPO_ROOT / "reproducibility" / "results" / "three_strategies_cv" / "raw_predictions"
BACKBONE_RESULTS_DIR = REPO_ROOT / "PAPER_REVISIONS" / "reviewer7_backbone_swap" / "results"
OUTPUT_DIR = REPO_ROOT / "PAPER_REVISIONS" / "reviewer7_backbone_swap" / "results"

BACKBONES = ["panns", "yamnet", "hubert"]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
B = 1000
SEED = 42


def pool_ast_qa(lam: float):
    """AST-QA's noise_0_10 strategy, pooled across whatever folds actually
    have a raw predictions file for this lambda.

    KNOWN DATA GAP (found 2026-09-13, not something this script caused):
    the local copy of reproducibility/results/three_strategies_cv/raw_predictions
    is INCOMPLETE for fold_2 -- 'noise_10' has zero files and 'noise_0_10' has
    only lambda_0.0.csv (which itself has one corrupted row: filename/y_true/
    probs all NaN, with a leftover '/gcs/heart-quality-training-78e/...' path,
    i.e. a real artifact from wherever this was originally run on Vertex AI,
    not something introduced locally). reproducibility/results/per_lambda_cv
    (used for the Tang/Giordano comparison) was checked and is fully complete
    at 10 folds x 10 lambdas -- this gap is specific to three_strategies_cv.
    Until this is resolved (recovering a complete copy, or explicitly
    accepting fold_2 as missing), the number of AST-QA folds contributing to
    each lambda is NOT constant across the sweep -- this function logs that
    count every call so it's never silently inconsistent.
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
    """A backbone's predictions, pooled across all its own (5) folds."""
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
    """Independent bootstrap: resample each side's pooled predictions
    separately (NOT paired -- different underlying test sets/fold counts)."""
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
            "UNPAIRED comparison -- AST-QA's noise_0_10 (Variable-Noise) strategy, "
            "published at 10-fold in reproducibility/results/three_strategies_cv/, vs. "
            "each backbone at 3-fold -- a first-pass real-hardware validation run per "
            "CLAUDE.md Sec 9.4's settled exception, NOT the final 5-fold Reviewer #7 number "
            "(that run is still pending). Corrected from a stale '5-fold' claim this docstring "
            "Sec 9.4). Not a paired test on matched folds. ADDITIONALLY: the local copy "
            "of AST-QA's raw predictions only has fold_1/fold_2/fold_10 of the published "
            "10 (fold_2 itself partial) -- AST-QA's side of every comparison below is "
            "pooled from ~2 folds (3 at lambda=0.0), not the full 10. Check each entry's "
            "'ast_qa_n_folds' field. Accepted explicitly on 2026-09-13 rather than "
            "recovering the complete data from GCS -- see module docstring. Also note: "
            "the backbone side of this comparison is the 3-fold first-pass validation run, "
            "not the pending final 5-fold run -- see module docstring."
        ),
        "bootstrap_iterations": B,
        "seed": SEED,
    }
    any_ran = False
    for backbone in BACKBONES:
        lam_dir_exists = (BACKBONE_RESULTS_DIR / backbone / "raw_predictions").exists()
        print("=" * 90)
        print(f"  AST-QA (noise_0_10) vs {backbone} -- UNPAIRED bootstrap comparison (AST-QA 10-fold [~2 pooled] vs {backbone} 3-fold first-pass)")
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
