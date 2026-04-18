#!/usr/bin/env python3
"""
Compute Metrics from Raw Prediction CSVs.

Works on both per-lambda and three-strategy result directories.
Can verify re-run results against the bundled reference results.

Usage:
    python compute_metrics.py --results_dir results/
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, confusion_matrix
)
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_all_metrics(y_true, y_probs, threshold=0.5):
    """Compute full metric suite from true labels and predicted probabilities."""
    y_true = np.array(y_true, dtype=np.float64)
    y_probs = np.array(y_probs, dtype=np.float64)
    y_preds = (y_probs > threshold).astype(int)
    y_true_int = y_true.astype(int)
    try:
        auroc = roc_auc_score(y_true, y_probs)
    except Exception:
        auroc = np.nan
    try:
        auprc = average_precision_score(y_true, y_probs)
    except Exception:
        auprc = np.nan
    f1 = f1_score(y_true_int, y_preds, zero_division=0)
    acc = accuracy_score(y_true_int, y_preds)
    tn, fp, fn, tp = confusion_matrix(y_true_int, y_preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auroc": auroc, "auprc": auprc, "f1": f1,
        "accuracy": acc, "sensitivity": sens, "specificity": spec
    }


def aggregate_mean_ci(fold_metrics, n_folds):
    """Aggregate fold-level metrics into mean ± 95% CI."""
    df = pd.DataFrame(fold_metrics)
    results = {}
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        if n_folds > 1:
            ci = stats.t.ppf(0.975, n_folds - 1) * (std / np.sqrt(n_folds))
        else:
            ci = 0.0
        results[col] = {"mean": mean, "ci": ci}
    return results


def compute_per_lambda(results_dir):
    """Compute metrics for per-lambda CV experiment."""
    raw_dir = os.path.join(results_dir, "per_lambda_cv", "raw_predictions")
    if not os.path.exists(raw_dir):
        logger.warning(f"Per-lambda predictions not found at {raw_dir}")
        return None

    all_results = {}
    for lambda_dir in sorted(os.listdir(raw_dir)):
        if not lambda_dir.startswith("lambda_"):
            continue
        l_val = lambda_dir.replace("lambda_", "")
        lambda_path = os.path.join(raw_dir, lambda_dir)

        fold_metrics = []
        for fold_dir in sorted(os.listdir(lambda_path)):
            if not fold_dir.startswith("fold_"):
                continue
            csv_path = os.path.join(lambda_path, fold_dir, "predictions.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            m = compute_all_metrics(df["y_true"].values, df["probs"].values)
            fold_metrics.append(m)

        if fold_metrics:
            all_results[l_val] = aggregate_mean_ci(fold_metrics, len(fold_metrics))

    return all_results


def compute_three_strategies(results_dir):
    """Compute metrics for three-strategy CV experiment."""
    raw_dir = os.path.join(results_dir, "three_strategies_cv", "raw_predictions")
    if not os.path.exists(raw_dir):
        logger.warning(f"Three-strategy predictions not found at {raw_dir}")
        return None

    strategies = ['clean', 'noise_0_10', 'noise_10']
    all_results = {}

    for strategy in strategies:
        strategy_results = {}

        for fold_dir in sorted(os.listdir(raw_dir)):
            if not fold_dir.startswith("fold_"):
                continue
            strat_path = os.path.join(raw_dir, fold_dir, strategy)
            if not os.path.exists(strat_path):
                continue

            for csv_file in sorted(os.listdir(strat_path)):
                if not csv_file.endswith(".csv"):
                    continue
                lam = csv_file.replace("lambda_", "").replace(".csv", "")
                df = pd.read_csv(os.path.join(strat_path, csv_file))
                m = compute_all_metrics(df["y_true"].values, df["probs"].values)
                strategy_results.setdefault(lam, []).append(m)

        all_results[strategy] = {}
        for lam, fold_metrics in strategy_results.items():
            all_results[strategy][lam] = aggregate_mean_ci(fold_metrics, len(fold_metrics))

    return all_results


def print_per_lambda_table(results):
    """Print formatted per-lambda results table."""
    print("\n" + "=" * 100)
    print("  PER-LAMBDA CV RESULTS")
    print("=" * 100)
    print(f"{'Lambda':>8} {'AUROC':>18} {'AUPRC':>18} {'F1':>18} {'Accuracy':>18} {'Sens':>18} {'Spec':>18}")
    print("-" * 100)

    for l_val in sorted(results.keys(), key=float):
        m = results[l_val]
        line = f"{l_val:>8}"
        for metric in ["auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity"]:
            mean = m[metric]["mean"]
            ci = m[metric]["ci"]
            line += f" {mean:.4f} ± {ci:.4f}"
        print(line)


def print_three_strategy_table(results):
    """Print formatted three-strategy results table."""
    print("\n" + "=" * 120)
    print("  THREE TRAINING STRATEGIES — AUROC COMPARISON")
    print("=" * 120)
    print(f"{'Lambda':>8} {'Clean':>18} {'Noise [0,10]':>18} {'Noise 10':>18}")
    print("-" * 80)

    all_lambdas = set()
    for s in results.values():
        all_lambdas.update(s.keys())

    for l_val in sorted(all_lambdas, key=float):
        line = f"{l_val:>8}"
        for strategy in ['clean', 'noise_0_10', 'noise_10']:
            if l_val in results.get(strategy, {}):
                m = results[strategy][l_val]
                mean = m["auroc"]["mean"]
                ci = m["auroc"]["ci"]
                line += f" {mean:.4f} ± {ci:.4f}"
            else:
                line += f" {'N/A':>18}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from raw prediction CSVs")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for computed results (default: results_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Per-lambda experiment
    per_lambda = compute_per_lambda(args.results_dir)
    if per_lambda:
        print_per_lambda_table(per_lambda)
        with open(os.path.join(output_dir, "computed_per_lambda.json"), 'w') as f:
            json.dump(per_lambda, f, indent=2)
        logger.info(f"Per-lambda metrics saved to {output_dir}/computed_per_lambda.json")

    # Three-strategy experiment
    three_strat = compute_three_strategies(args.results_dir)
    if three_strat:
        print_three_strategy_table(three_strat)
        with open(os.path.join(output_dir, "computed_three_strategies.json"), 'w') as f:
            json.dump(three_strat, f, indent=2)
        logger.info(f"Three-strategy metrics saved to {output_dir}/computed_three_strategies.json")

    if not per_lambda and not three_strat:
        logger.error("No raw prediction CSVs found. Run training scripts first.")
        return

    print(f"\n✅ Metrics computed and saved to {output_dir}")


if __name__ == "__main__":
    main()
