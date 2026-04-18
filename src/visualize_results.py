#!/usr/bin/env python3
"""
Visualize Phase 5 Results — Generate All Paper Figures.

Reads pre-computed results (JSON/CSV) and generates:
  1. AUROC vs λ for three training strategies (main paper figure)
  2. Stress degradation curves (accuracy, AUROC, F1 vs λ)
  3. Sensitivity/Specificity trade-off plots
  4. Summary metrics CSV table

Usage:
    python visualize_results.py --results_dir results/ --output_dir results/figures/
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────── FIGURE 1: Three-Strategy Robustness Comparison ────────

def plot_three_strategies(results, output_dir):
    """AUROC vs λ for clean / noise_0_10 / noise_10 strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategy_config = {
        'clean': {'color': '#2196F3', 'label': 'Clean Training Only', 'marker': 'o'},
        'noise_0_10': {'color': '#4CAF50', 'label': 'Noise Augmented (λ ~ U[0,10])', 'marker': 's'},
        'noise_10': {'color': '#F44336', 'label': 'Noise Augmented (λ = 10)', 'marker': '^'},
    }

    for s_name, cfg in strategy_config.items():
        if s_name not in results:
            continue
        lambdas = sorted([float(l) for l in results[s_name].keys()])
        means = [results[s_name][str(l)]['mean']['auroc'] for l in lambdas]
        cis = [results[s_name][str(l)]['ci']['auroc'] for l in lambdas]

        ax.errorbar(lambdas, means, yerr=cis, label=cfg['label'],
                    color=cfg['color'], capsize=5, marker=cfg['marker'],
                    linewidth=2, markersize=7)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, label='Chance (0.50)')
    ax.set_xscale('symlog', linthresh=1.0)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Noise Level (λ_test)', fontsize=13)
    ax.set_ylabel('Mean AUROC', fontsize=13)
    ax.set_title('Robustness Comparison — Training Strategies', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "robustness_auroc_vs_noise.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {save_path}")


# ──────── FIGURE 2: Stress Degradation Curves ────────

def plot_stress_degradation(df, output_dir):
    """Three-panel plot: Accuracy, AUROC, F1 vs λ for both mixing modes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    metrics = [("accuracy", "Accuracy"), ("auroc", "AUROC"), ("f1", "F1 Score")]
    colors = {"rms": "#2196F3", "paper": "#FF5722"}
    labels = {"rms": "RMS Mixing (Ours)", "paper": "Paper Mixing (0.2× Amplitude)"}

    for ax, (col, title) in zip(axes, metrics):
        for mode in ["rms", "paper"]:
            mode_df = df[df["mixing_mode"] == mode].sort_values("lambda")
            if mode_df.empty:
                continue
            ax.plot(mode_df["lambda"], mode_df[col],
                    marker="o", color=colors[mode], label=labels[mode],
                    linewidth=2, markersize=5)

            # Breaking point annotation
            below = mode_df[mode_df[col] < 0.55]
            if not below.empty:
                bp = below.iloc[0]["lambda"]
                val = below.iloc[0][col]
                ax.axvline(bp, color=colors[mode], linestyle="--", alpha=0.4)
                ax.annotate(f"λ={bp}", (bp, val), textcoords="offset points",
                            xytext=(10, -15), fontsize=8, color=colors[mode])

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Chance (0.50)")
        ax.set_xlabel("Noise Level (λ)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.suptitle("AST Heart Quality — Extreme Stress Degradation Curve",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "stress_degradation_curve.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {save_path}")


# ──────── FIGURE 3: Sensitivity / Specificity ────────

def plot_sensitivity_specificity(df, output_dir):
    """Show how sensitivity and specificity trade off as noise increases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"rms": "#2196F3", "paper": "#FF5722"}
    labels = {"rms": "RMS Mixing", "paper": "Paper Mixing"}

    for ax, (col, title) in zip(axes, [("sensitivity", "Sensitivity"), ("specificity", "Specificity")]):
        for mode in ["rms", "paper"]:
            mode_df = df[df["mixing_mode"] == mode].sort_values("lambda")
            if mode_df.empty:
                continue
            ax.plot(mode_df["lambda"], mode_df[col],
                    marker="s", color=colors[mode], label=labels[mode],
                    linewidth=2, markersize=5)

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("Noise Level (λ)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    save_path = os.path.join(output_dir, "stress_sensitivity_specificity.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {save_path}")


# ──────── TABLE: Summary Metrics ────────

def generate_summary_table(results, output_dir):
    """Generate a CSV summary table from three-strategy results."""
    rows = []
    for s_name in results:
        for l_val in sorted(results[s_name].keys(), key=float):
            m = results[s_name][l_val]
            row = {"strategy": s_name, "lambda": float(l_val)}
            for metric in ["auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity"]:
                row[f"{metric}_mean"] = m["mean"][metric]
                row[f"{metric}_ci"] = m["ci"][metric]
                row[metric] = f"{m['mean'][metric]:.4f} ± {m['ci'][metric]:.4f}"
            rows.append(row)

    df = pd.DataFrame(rows)
    save_path = os.path.join(output_dir, "average_metrics_table.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"Saved → {save_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from results")
    parser.add_argument("--results_dir", type=str, required=True, help="Root results directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures (default: results_dir/figures/)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Three-strategy results (JSON)
    three_strat_path = os.path.join(args.results_dir, "three_strategies_cv", "final_results.json")
    if os.path.exists(three_strat_path):
        with open(three_strat_path) as f:
            three_strat = json.load(f)
        plot_three_strategies(three_strat, output_dir)
        generate_summary_table(three_strat, output_dir)
    else:
        logger.warning(f"Three-strategy results not found at {three_strat_path}")

    # 2. Extreme stress results (CSV)
    stress_csv = os.path.join(args.results_dir, "extreme_stress", "extreme_stress_results.csv")
    if os.path.exists(stress_csv):
        df = pd.read_csv(stress_csv)
        plot_stress_degradation(df, output_dir)
        plot_sensitivity_specificity(df, output_dir)
    else:
        logger.warning(f"Extreme stress CSV not found at {stress_csv}")

    print(f"\n✅ All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
