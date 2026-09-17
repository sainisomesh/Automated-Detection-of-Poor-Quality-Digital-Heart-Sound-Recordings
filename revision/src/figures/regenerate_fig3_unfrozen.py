"""
Regenerates Fig3_*.pdf (the 6-panel metrics-vs-noise figure + legend) using the
new full-unfreeze, 5-fold data, replacing the original frozen-backbone figures
from the pre-revision manuscript. Styling (colors, markers, layout, fonts) is
copied verbatim from ../../plot_separate_metrics.py so the figure looks
identical in style to the original -- only the underlying data source changes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.formatter.use_mathtext'] = True

REPO_ROOT = Path(__file__).resolve().parents[2]
ABL_RESULTS = REPO_ROOT / "PAPER_REVISIONS" / "reviewer1_unfreezing_ablation" / "results"
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
METRICS = ['AUROC', 'AUPRC', 'F1-score', 'Accuracy', 'Sensitivity', 'Specificity']
JSON_KEYS = ['auroc', 'auprc', 'f1', 'accuracy', 'sensitivity', 'specificity']


def load_strategy(path, results_key_is_str_lambda=True):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for lam in LAMBDAS:
        key = str(lam) if results_key_is_str_lambda else lam
        m = d['results'][key]['mean']
        c = d['results'][key]['ci']
        row = {'lambda': lam}
        for metric, jkey in zip(METRICS, JSON_KEYS):
            row[f'{metric}_mean'] = m[jkey]
            row[f'{metric}_ci'] = c[jkey]
        rows.append(row)
    return pd.DataFrame(rows)


def load_per_lambda():
    rows = []
    for lam in LAMBDAS:
        with open(ABL_RESULTS / "per_lambda_unfrozen" / "full" / f"lambda_{lam}" / "per_lambda_unfrozen_final_results.json") as f:
            d = json.load(f)
        m, c = d['results']['mean'], d['results']['ci']
        row = {'lambda': lam}
        for metric, jkey in zip(METRICS, JSON_KEYS):
            row[f'{metric}_mean'] = m[jkey]
            row[f'{metric}_ci'] = c[jkey]
        rows.append(row)
    return pd.DataFrame(rows)


dfs = {
    'clean': load_strategy(ABL_RESULTS / "full_5fold_clean" / "unfreezing_ablation_final_results.json"),
    'noise_0_10': load_strategy(ABL_RESULTS / "full_5fold_variable" / "unfreezing_ablation_final_results.json"),
    'noise_10': load_strategy(ABL_RESULTS / "full_5fold_fixed10" / "unfreezing_ablation_final_results.json"),
    'per_lambda': load_per_lambda(),
}

lambdas = dfs['clean']['lambda'].values
x_indices = np.arange(len(lambdas))

color_map = {
    'clean': '#1A365D',
    'noise_0_10': '#9B2226',
    'noise_10': '#2D6A4F',
    'per_lambda': '#E07A5F'
}
label_map = {
    'clean': 'Clean',
    'noise_0_10': r'Variable Noise (0-10 $\lambda$)',
    'noise_10': r'Noise at 10 $\lambda$',
    'per_lambda': 'Per-Lambda'
}
marker_map = {
    'clean': 'o',
    'noise_0_10': 's',
    'noise_10': '^',
    'per_lambda': 'D'
}

order = ['clean', 'noise_0_10', 'noise_10', 'per_lambda']
OUT_DIR = Path(__file__).resolve().parent

for metric in METRICS:
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')
    ax.set_facecolor('white')

    for strat in order:
        df = dfs[strat]
        mean_vals = df[f'{metric}_mean'] * 100
        ci_vals = df[f'{metric}_ci'] * 100
        ax.errorbar(x_indices, mean_vals, yerr=ci_vals,
                    label=label_map[strat], color=color_map[strat], marker=marker_map[strat],
                    linestyle='-', linewidth=2.5, markersize=8, capsize=4, elinewidth=1.5)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(l) for l in lambdas], fontsize=12)
    ax.set_xlabel(r'Evaluation Noise ($\lambda$)', fontsize=14, labelpad=10)
    display_title = metric.replace('F1-score', 'F1 Score')
    ax.set_title(display_title, fontsize=18, fontweight='heavy', pad=6)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Score (%)', fontsize=14, labelpad=10)

    ax.set_ylim(-2, 102)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(axis='y', color='#E5E5E5', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(axis='x', visible=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')

    plt.tight_layout()
    safe_name = metric.replace('-', '_').lower()
    save_path = OUT_DIR / f'Fig3_{safe_name}_vs_noise.pdf'
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"Generated {save_path}")

fig_leg, ax_leg = plt.subplots(figsize=(10, 1), facecolor='white')
ax_leg.axis('off')
handles, labels = [], []
for strat in order:
    line, = ax_leg.plot([], [], label=label_map[strat], color=color_map[strat], marker=marker_map[strat],
                        linestyle='-', linewidth=2.5, markersize=8)
    handles.append(line)
    labels.append(label_map[strat])

leg = fig_leg.legend(handles, labels, loc='center', ncol=4,
                     title='Training Strategy', title_fontproperties={'weight': 'bold', 'size': 14},
                     frameon=False, fontsize=12)

save_path_leg = OUT_DIR / 'Fig3_legend_only.pdf'
plt.savefig(save_path_leg, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig_leg)
print(f"Generated {save_path_leg}")
