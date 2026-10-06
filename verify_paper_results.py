#!/usr/bin/env python3
"""Recompute every reported number from the checked-in raw per-fold
prediction CSVs and compare it to the reference value.

Runs on CPU in about a minute, needs no GPU and no audio data, only
numpy / pandas / scikit-learn. This is the fast path for checking that the
paper's tables follow from the released predictions; the run_all.sh scripts
are the slow path that regenerates those predictions by retraining.

For each reported cell the script recomputes the per-fold metrics at the
fixed 0.5 decision threshold, aggregates them exactly as the training scripts
do (mean across folds, 95% half-width = 1.96 * SD / sqrt(n_folds)), rounds to
two decimals and checks equality with the reference. Significance markers
are checked against the paired-bootstrap p-values in the checked-in
significance JSONs.

Usage:
    python verify_paper_results.py            # from reproducibility/
    python verify_paper_results.py --verbose  # also print every passing cell

Exit status is 0 if every check passes and 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
REV = ROOT / "revision" / "results"
LAMBDAS = ["0.0", "0.25", "0.5", "1.0", "5.0", "10.0", "25.0", "50.0", "75.0", "100.0"]
METRICS = ["auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity"]


# Metric recomputation

def fold_metrics(csv_path):
    df = pd.read_csv(csv_path)
    y, p = df["y_true"].astype(int).values, df["probs"].values
    pred = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def aggregate(lambda_dir, expected_folds):
    """Mean and 1.96*SD/sqrt(n) half-width, in percent, over a lambda's folds."""
    csvs = sorted(Path(lambda_dir).glob("fold_*/predictions.csv"))
    if not csvs:
        csvs = sorted(Path(lambda_dir).glob("fold_*.csv"))
    if len(csvs) != expected_folds:
        raise FileNotFoundError(f"{lambda_dir}: expected {expected_folds} folds, found {len(csvs)}")
    per_fold = [fold_metrics(c) for c in csvs]
    n = len(per_fold)
    out = {}
    for m in METRICS:
        vals = np.array([f[m] for f in per_fold])
        out[m] = (100 * vals.mean(), 100 * 1.96 * vals.std() / np.sqrt(n))
    return out


_cache = {}


def result(raw_dir, lam, expected_folds):
    key = (str(raw_dir), lam)
    if key not in _cache:
        _cache[key] = aggregate(Path(raw_dir) / f"lambda_{lam}", expected_folds)
    return _cache[key]


# Checking

class Checker:
    def __init__(self, verbose):
        self.verbose = verbose
        self.n_pass = 0
        self.failures = []

    def section(self, title):
        print(f"\n== {title}")

    def value(self, label, got, expected):
        ok = round(got + 1e-9, 2) == expected
        self._record(ok, f"{label}: paper {expected:.2f}, recomputed {got:.4f}")

    def mean_ci(self, label, got, expected):
        (gm, gc), (em, ec) = got, expected
        ok = round(gm + 1e-9, 2) == em and (ec is None or round(gc + 1e-9, 2) == ec)
        exp = f"{em:.2f}" + (f" ± {ec:.2f}" if ec is not None else "")
        self._record(ok, f"{label}: paper {exp}, recomputed {gm:.4f} ± {gc:.4f}")

    def stars(self, label, p_value, expected_stars):
        got = "***" if p_value < 0.001 else "*" if p_value < 0.05 else ""
        self._record(got == expected_stars,
                     f"{label}: paper '{expected_stars}', p = {p_value:.3f} -> '{got}'")

    def claim(self, label, ok, detail):
        self._record(ok, f"{label}: {detail}")

    def _record(self, ok, msg):
        if ok:
            self.n_pass += 1
            if self.verbose:
                print(f"  ok    {msg}")
        else:
            self.failures.append(msg)
            print(f"  FAIL  {msg}")


def parse_cell(cell):
    """'99.97$\\pm$0.03$^{***}$' -> (99.97, 0.03, '***'); '--' -> (None, None, '')."""
    if cell == "--":
        return None, None, ""
    stars = "***" if "{***}" in cell else "*" if "{*}" in cell else ""
    cell = cell.split("$^")[0]
    if "$\\pm$" in cell:
        m, c = cell.split("$\\pm$")
        return float(m), float(c), stars
    return float(cell), None, stars


def parse_rows(block):
    rows = {}
    for line in block.strip().splitlines():
        cells = [c.strip() for c in line.split("&")]
        rows[cells[0]] = [parse_cell(c) for c in cells[1:]]
    return rows


# Reference values.

TABLE2 = """
0.0   & 100.00$\\pm$0.00 & 100.00$\\pm$0.00 & 99.98$\\pm$0.03 & 99.98$\\pm$0.03 & 99.97$\\pm$0.05 & 100.00$\\pm$0.00
0.25  & 100.00$\\pm$0.00 & 100.00$\\pm$0.00 & 99.67$\\pm$0.47 & 99.67$\\pm$0.48 & 99.97$\\pm$0.05 & 99.37$\\pm$0.97
0.5   & 99.98$\\pm$0.02  & 99.99$\\pm$0.02  & 99.23$\\pm$0.52 & 99.23$\\pm$0.53 & 99.55$\\pm$0.54 & 98.90$\\pm$1.22
1.0   & 99.96$\\pm$0.02  & 99.97$\\pm$0.02  & 97.65$\\pm$1.76 & 97.64$\\pm$1.77 & 97.70$\\pm$3.03 & 97.58$\\pm$3.32
5.0   & 96.91$\\pm$0.73  & 97.27$\\pm$0.70  & 90.20$\\pm$1.38 & 89.94$\\pm$1.67 & 92.17$\\pm$2.59 & 87.71$\\pm$4.95
10.0  & 88.91$\\pm$1.25  & 90.05$\\pm$1.35  & 79.86$\\pm$1.40 & 80.59$\\pm$0.91 & 77.20$\\pm$4.05 & 83.97$\\pm$3.45
25.0  & 55.63$\\pm$6.11  & 56.46$\\pm$6.19  & 50.59$\\pm$9.92 & 53.14$\\pm$4.82 & 53.13$\\pm$22.34 & 53.15$\\pm$26.33
50.0  & 50.15$\\pm$2.30  & 50.19$\\pm$2.03  & 26.73$\\pm$28.58 & 50.02$\\pm$0.03 & 40.03$\\pm$42.92 & 60.00$\\pm$42.94
75.0  & 48.23$\\pm$1.31  & 48.79$\\pm$1.02  & 40.00$\\pm$28.63 & 50.00$\\pm$0.00 & 60.00$\\pm$42.94 & 40.00$\\pm$42.94
100.0 & 49.33$\\pm$1.24  & 49.77$\\pm$1.19  & 52.60$\\pm$23.08 & 49.62$\\pm$0.66 & 77.65$\\pm$34.26 & 21.60$\\pm$34.47
"""

TABLE3 = """
0.0   & 99.97$\\pm$0.03 & 99.94$\\pm$0.06 & 99.95$\\pm$0.02 & 92.17$\\pm$0.64$^{***}$
0.25  & 99.93$\\pm$0.03 & 99.88$\\pm$0.08 & 99.90$\\pm$0.05 & 89.25$\\pm$0.95$^{***}$
0.5   & 99.90$\\pm$0.04 & 99.79$\\pm$0.12 & 99.59$\\pm$0.22$^{***}$ & 84.72$\\pm$1.24$^{***}$
1.0   & 99.78$\\pm$0.08 & 99.52$\\pm$0.24 & 98.00$\\pm$0.48$^{***}$ & 74.44$\\pm$1.69$^{***}$
5.0   & 96.02$\\pm$0.47 & 91.54$\\pm$1.11 & 60.82$\\pm$1.16$^{***}$ & 51.65$\\pm$1.51$^{***}$
10.0  & 88.49$\\pm$0.80 & 77.72$\\pm$1.47 & 51.30$\\pm$0.90$^{***}$ & 50.35$\\pm$1.17$^{***}$
25.0  & 70.48$\\pm$1.06 & 60.50$\\pm$0.95 & 50.43$\\pm$0.78$^{***}$ & 50.24$\\pm$1.03$^{***}$
50.0  & 58.93$\\pm$1.38 & 54.61$\\pm$0.83 & 50.39$\\pm$0.82$^{***}$ & 50.19$\\pm$1.14$^{***}$
75.0  & 54.75$\\pm$1.52 & 52.99$\\pm$0.74 & 50.42$\\pm$0.82$^{***}$ & 50.24$\\pm$1.10$^{***}$
100.0 & 52.86$\\pm$1.61 & 52.30$\\pm$0.70 & 50.50$\\pm$0.92 & 50.23$\\pm$1.11$^{*}$
"""

TABLE4 = """
0.0   & 99.94$\\pm$0.06 & 99.97$\\pm$0.03 & 99.87$\\pm$0.18 & 99.99$\\pm$0.01$^{***}$ & 99.88$\\pm$0.02 & 87.25$\\pm$13.00$^{***}$ & 99.93$\\pm$0.08 & 99.95$\\pm$0.05
0.25  & 99.88$\\pm$0.08 & 99.93$\\pm$0.03 & 99.44$\\pm$0.31 & 99.94$\\pm$0.03$^{*}$ & 98.80$\\pm$0.23 & 81.99$\\pm$14.18$^{***}$ & 99.88$\\pm$0.08 & 99.93$\\pm$0.04
0.5   & 99.79$\\pm$0.12 & 99.90$\\pm$0.04 & 98.84$\\pm$0.39 & 99.90$\\pm$0.02$^{*}$ & 97.15$\\pm$0.42 & 80.30$\\pm$13.22$^{***}$ & 99.71$\\pm$0.08 & 99.91$\\pm$0.03
1.0   & 99.52$\\pm$0.24 & 99.78$\\pm$0.08 & 97.04$\\pm$0.48 & 99.75$\\pm$0.08 & 93.67$\\pm$0.69 & 78.07$\\pm$11.47$^{***}$ & 98.77$\\pm$0.23 & 99.75$\\pm$0.12$^{*}$
5.0   & 91.54$\\pm$1.11 & 96.02$\\pm$0.47 & 74.38$\\pm$1.32 & 94.94$\\pm$0.65 & 68.89$\\pm$1.14 & 60.19$\\pm$3.01$^{***}$ & 80.70$\\pm$0.73 & 95.73$\\pm$1.58$^{***}$
10.0  & 77.72$\\pm$1.47 & 88.49$\\pm$0.80 & 61.20$\\pm$1.65 & 85.51$\\pm$0.58$^{*}$ & 58.61$\\pm$0.72 & 51.57$\\pm$3.05$^{***}$ & 66.71$\\pm$1.01 & 87.13$\\pm$2.81$^{***}$
25.0  & 60.50$\\pm$0.95 & 70.48$\\pm$1.06 & 53.79$\\pm$1.59 & 67.28$\\pm$0.62$^{*}$ & 53.09$\\pm$0.76 & 49.38$\\pm$2.89$^{***}$ & 55.69$\\pm$1.06 & 68.28$\\pm$3.18$^{***}$
50.0  & 54.61$\\pm$0.83 & 58.93$\\pm$1.38 & 52.01$\\pm$1.57 & 57.39$\\pm$0.67 & 51.51$\\pm$0.69 & 49.67$\\pm$2.67$^{***}$ & 52.39$\\pm$0.99 & 57.39$\\pm$2.48$^{*}$
75.0  & 52.99$\\pm$0.74 & 54.75$\\pm$1.52 & 51.52$\\pm$1.52 & 53.91$\\pm$0.99 & 51.04$\\pm$0.71 & 49.82$\\pm$2.54$^{***}$ & 51.50$\\pm$0.97 & 53.85$\\pm$1.91
100.0 & 52.30$\\pm$0.70 & 52.86$\\pm$1.61 & 51.26$\\pm$1.60 & 52.55$\\pm$1.11 & 50.82$\\pm$0.72 & 49.89$\\pm$2.47$^{*}$ & 51.09$\\pm$0.96 & 52.50$\\pm$1.68
"""

TABLE5 = """
0.0   & 99.94 & 99.96 & 99.98 & 99.97
0.25  & 99.88 & 99.94 & 99.97 & 99.93
0.5   & 99.79 & 99.89 & 99.93 & 99.90
1.0   & 99.52 & 99.75 & 99.83 & 99.78
5.0   & 91.54 & 95.45 & 96.13 & 96.02
10.0  & 77.72 & 85.66 & 86.39 & 88.49
25.0  & 60.50 & 67.30 & 67.29 & 70.48
50.0  & 54.61 & 57.93 & 57.59 & 58.93
75.0  & 52.99 & 54.83 & 54.49 & 54.75
100.0 & 52.30 & 53.43 & 53.07 & 52.86
"""

TABLE6_FULL = """
0.0   & 100.00 & 100.00 & 97.24 & 99.90 & 99.97
0.25  & 99.69  & 99.73  & 92.46 & 99.57 & 99.93
0.5   & 98.45  & 98.57  & 89.70 & 99.14 & 99.90
1.0   & 93.65  & 93.94  & 85.67 & 97.53 & 99.78
5.0   & 54.49  & 55.27  & 64.82 & 66.28 & 96.02
10.0  & 49.63  & 50.08  & 56.72 & 53.27 & 88.49
25.0  & 49.55  & 49.77  & 52.72 & 50.17 & 70.48
50.0  & 49.63  & 49.80  & 51.84 & 49.97 & 58.93
75.0  & 49.65  & 49.82  & 51.60 & 49.97 & 54.75
100.0 & 49.65  & 49.82  & 51.51 & 49.94 & 52.86
"""

# "--" cells are filled from the frozen denoiser run (revision/run_all.sh Step 4a).
TABLE6_FROZEN = """
0.0   & -- & -- & -- & -- & 99.94
0.25  & -- & -- & -- & -- & 99.88
0.5   & -- & -- & -- & -- & 99.79
1.0   & -- & -- & -- & -- & 99.52
5.0   & -- & -- & -- & -- & 91.54
10.0  & -- & -- & -- & -- & 77.72
25.0  & -- & -- & -- & -- & 60.50
50.0  & -- & -- & -- & -- & 54.61
75.0  & -- & -- & -- & -- & 52.99
100.0 & -- & -- & -- & -- & 52.30
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true", help="print passing checks too")
    args = ap.parse_args()
    ck = Checker(args.verbose)

    abl = REV / "reviewer1_unfreezing_ablation"
    bb = REV / "reviewer7_backbone_swap"
    base = REV / "reviewer1_baselines"
    den = REV / "reviewer7_denoiser_benchmark"
    full_var = abl / "full_5fold_variable" / "raw_predictions"
    frozen = abl / "frozen_5fold" / "raw_predictions"

    # Table 2: per-lambda matched benchmark, fully fine-tuned, 5-fold.
    ck.section("Table 2  per-lambda matched benchmark (fine-tuned, 5-fold)")
    for lam, cells in parse_rows(TABLE2).items():
        raw = abl / "per_lambda_unfrozen" / "full" / f"lambda_{lam}" / "raw_predictions"
        r = result(raw, lam, 5)
        for m, (em, ec, _) in zip(METRICS, cells):
            ck.mean_ci(f"lambda={lam} {m}", r[m], (em, ec))

    # Table 3: baselines, with paired significance against the fine-tuned model.
    ck.section("Table 3  comparison with published baselines (5-fold)")
    sig = json.load(open(base / "significance_vs_ast_qa_unfrozen_paired.json"))
    cols = [("AST fine-tuned", full_var, None),
            ("AST frozen", frozen, None),
            ("Tang", base / "tang_full" / "raw_predictions", "tang_vs_ast_qa_unfrozen"),
            ("Giordano", base / "giordano_full" / "raw_predictions", "giordano_vs_ast_qa_unfrozen")]
    for lam, cells in parse_rows(TABLE3).items():
        for (name, raw, sig_key), (em, ec, st) in zip(cols, cells):
            ck.mean_ci(f"lambda={lam} {name} AUROC", result(raw, lam, 5)["auroc"], (em, ec))
            if sig_key:
                ck.stars(f"lambda={lam} {name} significance", sig[sig_key][lam]["paired_bootstrap_p_value"], st)
    for name, key, lam, p in [("Tang", "tang_vs_ast_qa_unfrozen", "100.0", 0.058),
                              ("Giordano", "giordano_vs_ast_qa_unfrozen", "100.0", 0.006)]:
        got = sig[key][lam]["paired_bootstrap_p_value"]
        ck.claim(f"text: {name} p at lambda={lam}", round(got, 3) == p, f"paper p={p}, json p={got}")

    # Table 4: backbone swap, frozen (fz) and fully fine-tuned (fl).
    ck.section("Table 4  backbone swap (5-fold)")
    bsig = json.load(open(bb / "significance_full_paired.json"))
    cols = [("AST fz", frozen, None), ("AST fl", full_var, None)]
    for b in ["panns", "yamnet", "hubert"]:
        cols += [(f"{b} fz", bb / b / "raw_predictions", None),
                 (f"{b} fl", bb / f"{b}_full" / "raw_predictions", f"{b}_full_vs_ast_full")]
    for lam, cells in parse_rows(TABLE4).items():
        for (name, raw, sig_key), (em, ec, st) in zip(cols, cells):
            ck.mean_ci(f"lambda={lam} {name} AUROC", result(raw, lam, 5)["auroc"], (em, ec))
            if sig_key:
                ck.stars(f"lambda={lam} {name} significance", bsig[sig_key][lam]["paired_bootstrap_p_value"], st)

    # Table 5: backbone adaptation ablation.
    ck.section("Table 5  backbone fine-tuning ablation (5-fold)")
    cols = [("frozen", frozen), ("top-K=2", abl / "topk2_5fold" / "raw_predictions"),
            ("top-K=4", abl / "topk4_5fold" / "raw_predictions"), ("full", full_var)]
    for lam, cells in parse_rows(TABLE5).items():
        for (name, raw), (em, _, _) in zip(cols, cells):
            ck.value(f"lambda={lam} {name} AUROC", result(raw, lam, 5)["auroc"][0], em)
    conv = {n: pd.read_csv(abl / d / "convergence_log.csv").epoch_seconds.sum()
            for n, d in [("frozen", "frozen_5fold"), ("full", "full_5fold_variable"),
                         ("topk2", "topk2_5fold"), ("topk4", "topk4_5fold")]}
    ratio = conv["full"] / conv["frozen"]
    ck.claim("text: full fine-tuning ~2.6x frozen training compute", round(ratio, 1) == 2.6,
             f"{ratio:.2f}x ({conv['full'] / 3600:.2f} h vs {conv['frozen'] / 3600:.2f} h)")
    ck.claim("text: top-K training compute falls between frozen and full",
             conv["frozen"] < conv["topk2"] < conv["full"] and conv["frozen"] < conv["topk4"] < conv["full"],
             f"top-K=2 {conv['topk2'] / 3600:.2f} h, top-K=4 {conv['topk4'] / 3600:.2f} h")

    # Table 6: denoise-then-classify.
    ck.section("Table 6  denoise-then-classify")
    conds = ["no_denoise", "denoise_wavelet", "denoise_wavelet_leveldep", "denoise_lunet"]
    for lam, cells in parse_rows(TABLE6_FULL).items():
        for cond, (em, _, _) in zip(conds, cells[:4]):
            raw = den / "denoiser_comparison_full_5fold" / "raw_predictions" / cond
            ck.value(f"fine-tuned lambda={lam} {cond} AUROC", result(raw, lam, 5)["auroc"][0], em)
        ck.value(f"fine-tuned lambda={lam} variable-noise AUROC", result(full_var, lam, 5)["auroc"][0], cells[4][0])
    frozen_den = den / "denoiser_comparison_frozen_5fold" / "raw_predictions"
    for lam, cells in parse_rows(TABLE6_FROZEN).items():
        for cond, (em, _, _) in zip(conds, cells[:4]):
            label = f"frozen lambda={lam} {cond} AUROC"
            if not (frozen_den / cond / f"lambda_{lam}").is_dir():
                ck.claim(label, False, "5-fold frozen denoiser results missing, rerun revision/run_all.sh Step 4a")
            elif em is None:
                got = result(frozen_den / cond, lam, 5)["auroc"][0]
                ck.claim(label, False, f"no reference value yet, recomputed {got:.4f}")
            else:
                ck.value(label, result(frozen_den / cond, lam, 5)["auroc"][0], em)
        ck.value(f"frozen lambda={lam} variable-noise AUROC", result(frozen, lam, 5)["auroc"][0], cells[4][0])
    # Denoiser checkpoints are the same clean-only model whether or not a
    # denoiser is applied, so no_denoise must equal the clean-only run.
    for tag, a, b, k in [("fine-tuned", "clean_only_full_5fold", "denoiser_comparison_full_5fold", 5),
                         ("frozen", "clean_only_frozen_5fold", "denoiser_comparison_frozen_5fold", 5)]:
        if not (den / a).is_dir() or not (den / b).is_dir():
            ck.claim(f"{tag}: reloaded checkpoint reproduces clean-only run", False,
                     f"{a}/ or {b}/ missing, rerun the denoiser benchmark")
            continue
        for lam in LAMBDAS:
            x = result(den / a / "raw_predictions" / "no_denoise", lam, k)["auroc"][0]
            y = result(den / b / "raw_predictions" / "no_denoise", lam, k)["auroc"][0]
            ck.claim(f"{tag} lambda={lam}: reloaded checkpoint reproduces clean-only run", abs(x - y) < 1e-9,
                     f"{x:.4f} vs {y:.4f}")
    for cond, expected in [("denoise_wavelet", 0.22), ("denoise_lunet", 18.36), ("denoise_wavelet_leveldep", 31.09)]:
        raw = den / "denoiser_comparison_full_5fold" / "raw_predictions" / cond
        ck.value(f"text: lambda=5.0 {cond} sensitivity", result(raw, "5.0", 5)["sensitivity"][0], expected)

    # Figure 3 and its text: the three training strategies under the fine-tuned backbone.
    ck.section("Figure 3 / Results text  training strategies (fine-tuned, 5-fold)")
    clean = abl / "full_5fold_clean" / "raw_predictions"
    fixed = abl / "full_5fold_fixed10" / "raw_predictions"
    text_claims = [
        (clean, "0.0", "auroc", 100.00), (clean, "0.0", "f1", 99.95), (clean, "0.0", "sensitivity", 99.91),
        (clean, "1.0", "sensitivity", 18.95), (clean, "5.0", "sensitivity", 0.19),
        (full_var, "0.0", "auroc", 99.97), (full_var, "5.0", "auroc", 96.02), (full_var, "10.0", "auroc", 88.49),
        (fixed, "0.0", "auroc", 99.96), (fixed, "5.0", "auroc", 95.72), (fixed, "10.0", "auroc", 89.36),
        (fixed, "10.0", "sensitivity", 68.23), (full_var, "10.0", "sensitivity", 59.67),
        (fixed, "10.0", "f1", 76.92), (full_var, "10.0", "f1", 70.50),
        (full_var, "0.25", "auroc", 99.93), (full_var, "0.25", "f1", 95.98),
    ]
    for raw, lam, m, expected in text_claims:
        ck.value(f"{raw.parent.name} lambda={lam} {m}", result(raw, lam, 5)[m][0], expected)

    # Abstract and Discussion claims that summarise the tables.
    ck.section("Abstract / Discussion")
    pl = {lam: result(abl / "per_lambda_unfrozen" / "full" / f"lambda_{lam}" / "raw_predictions", lam, 5)
          for lam in LAMBDAS}
    low = [pl[lam]["auroc"][0] for lam in ["0.0", "0.25", "0.5", "1.0"]]
    ck.claim("abstract: per-lambda AUROC 99.96%-100.00% for lambda<=1",
             round(min(low), 2) == 99.96 and round(max(low), 2) == 100.00, f"range {min(low):.4f}-{max(low):.4f}")
    lowf1 = min(pl[lam]["f1"][0] for lam in ["0.0", "0.25", "0.5", "1.0"])
    ck.claim("discussion: per-lambda F1 > 97.6% for lambda<=1", lowf1 > 97.6, f"min F1 {lowf1:.4f}")
    gain10 = result(full_var, "10.0", 5)["auroc"][0] - result(frozen, "10.0", 5)["auroc"][0]
    gain25 = result(full_var, "25.0", 5)["auroc"][0] - result(frozen, "25.0", 5)["auroc"][0]
    gain5 = result(full_var, "5.0", 5)["auroc"][0] - result(frozen, "5.0", 5)["auroc"][0]
    ck.claim("section 3.1: fine-tuning gain at lambda=5.0 is 4.5 points", round(gain5, 1) == 4.5, f"{gain5:.2f}")
    ck.claim("section 3.1: fine-tuning gain at lambda=10.0 is 10.8 points", round(gain10, 1) == 10.8, f"{gain10:.2f}")
    ck.claim("section 3.4: fine-tuning gain at lambda=25.0 is +10.0 points", round(gain25, 1) == 10.0, f"{gain25:.2f}")

    print(f"\n{ck.n_pass} checks passed, {len(ck.failures)} failed.")
    if ck.failures:
        print("\nFailures:")
        for f in ck.failures:
            print(f"  - {f}")
    return 1 if ck.failures else 0


if __name__ == "__main__":
    sys.exit(main())
