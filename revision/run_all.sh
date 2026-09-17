#!/bin/bash
# ============================================================================
# RUN ALL — Revision Experiments
# ============================================================================
#
# Reproduces the four additional experiments reported in the revised
# manuscript:
#
#   Step 2  Comparison against two published PCG quality-assessment methods
#           (Tang et al. 2021, Giordano et al. 2021), 5-fold, CPU only.
#   Step 3  Backbone adaptation ablation: frozen, fully fine-tuned, and
#           top-K-layer unfreezing, at 3-fold (3a) and 5-fold (3b), plus the
#           per-lambda matched benchmark for the fully fine-tuned model.
#   Step 4  Backbone swap: PANNs CNN14, YAMNet and HuBERT substituted for the
#           AST encoder, each frozen and fully fine-tuned, 3-fold.
#   Step 5  Denoise-then-classify comparison: a clean-only classifier
#           evaluated on corrupted audio with and without each denoiser, run
#           for both the fully fine-tuned model (5-fold) and the frozen model
#           (10-fold).
#
# Every step is optional and prompts before running; results for all of them
# are already checked in under results/, so a fresh run is a verification
# rather than a prerequisite. Each step prints the fold count it uses.
#
# ../run_all.sh is the separate entry point for the original per-lambda and
# three-strategies experiments, and is unaffected by this script.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Reuses ../dataset and ../mixed_dataset, the same source data the sibling
# package uses, rather than keeping a second copy.
DATA_DIR="$SCRIPT_DIR/../dataset"
MIXED_DIR="$SCRIPT_DIR/../mixed_dataset"

echo "============================================"
echo "  Revision — Reproducibility Suite"
echo "============================================"
echo ""

# ── Step 0: Dataset (downloads from Zenodo if not already present) ─
# Fetches only the raw source dataset (~3 GB). The pre-mixed lambda sweep that
# Step 5 needs is generated locally from it rather than downloaded separately.
bash "$SCRIPT_DIR/../download_data.sh" "$DATA_DIR"
echo ""

# ── Step 1: Install dependencies ──────────────────────────────────
echo "── Step 1: Installing dependencies ──"
pip install -r requirements.txt
echo ""

# ── Step 2: Published baselines (Tang, Giordano) — 5-fold, CPU ────
echo "── Step 2: Published baselines (Tang et al., Giordano et al.) ──"
echo "5-fold patient-level CV, CPU only. Results are already in results/reviewer1_baselines/;"
echo "rerunning regenerates them in place."
read -p "Rerun baselines? [y/N]: " run_baselines
if [[ "$run_baselines" =~ ^[Yy]$ ]]; then
    cd src/reviewer1_baselines
    python train_tang_baseline_cv.py \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_baselines/tang_full/ \
        --n_folds 5 --seed 42
    python train_giordano_baseline_cv.py \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_baselines/giordano_full/ \
        --n_folds 5 --seed 42
    echo "Significance testing against AST-QA is computed by two separate scripts:"
    echo "  compute_significance.py        unpaired comparison vs. the frozen-backbone model"
    echo "  compute_significance_paired.py paired comparison vs. the fully fine-tuned model"
    python compute_significance.py
    python compute_significance_paired.py
    cd "$SCRIPT_DIR"
    echo "✓ Baselines complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

# ── Step 3: Backbone adaptation ablation ──────────────────────────
echo "── Step 3a: Backbone adaptation ablation, 3-fold ──"
echo "Results in results/reviewer1_unfreezing_ablation/{frozen,full,topk2,topk4}_3fold/."
echo "These provide the AST reference values for the 3-fold backbone-swap comparison in Step 4."
read -p "Rerun 3-fold unfreezing ablation (all 4 conditions)? [y/N]: " run_unfreeze3
if [[ "$run_unfreeze3" =~ ^[Yy]$ ]]; then
    cd src/reviewer1_unfreezing_ablation
    for mode_arg in "frozen:frozen_3fold" "full:full_3fold" "topk:topk2_3fold" "topk:topk4_3fold"; do
        mode="${mode_arg%%:*}"; outdir="${mode_arg##*:}"
        extra=""
        [[ "$outdir" == topk2_3fold ]] && extra="--topk_layers 2"
        [[ "$outdir" == topk4_3fold ]] && extra="--topk_layers 4"
        [[ "$mode" == full ]] && extra="--backbone_lr 5e-5 --grad_checkpointing"
        python train_unfreezing_ablation_cv.py --unfreeze_mode "$mode" $extra \
            --data_dir "$DATA_DIR/" --output_dir "../../results/reviewer1_unfreezing_ablation/$outdir/" \
            --n_folds 3 --epochs 5 --seed 42
    done
    cd "$SCRIPT_DIR"
    echo "✓ 3-fold unfreezing ablation complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

echo "── Step 3b: Backbone adaptation ablation, 5-fold, plus per-lambda benchmark ──"
echo "Results in results/reviewer1_unfreezing_ablation/{frozen,full,topk2,topk4}_5fold/,"
echo "full_5fold_{clean,fixed10,variable}/ and per_lambda_unfrozen/. These are the ablation"
echo "table, the per-lambda table and the metrics-vs-noise figure in the manuscript."
read -p "Rerun 5-fold unfreezing ablation (7 conditions)? [y/N]: " run_unfreeze5
if [[ "$run_unfreeze5" =~ ^[Yy]$ ]]; then
    cd src/reviewer1_unfreezing_ablation
    python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/frozen_5fold/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 2 \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/topk2_5fold/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/topk4_5fold/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_variable/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
        --noise_strategy clean \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_clean/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
        --noise_strategy fixed_10 \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_fixed10/ \
        --n_folds 5 --epochs 5 --seed 42
    echo "-- per-lambda matched benchmark, fully fine-tuned model, one lambda per run --"
    for lam in 0.0 0.25 0.5 1.0 5.0 10.0 25.0 50.0 75.0 100.0; do
        python train_per_lambda_unfrozen_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
            --lambda_val "$lam" \
            --data_dir "$DATA_DIR/" --output_dir "../../results/reviewer1_unfreezing_ablation/per_lambda_unfrozen/full/lambda_$lam/" \
            --n_folds 5 --epochs 5 --seed 42
    done
    cd "$SCRIPT_DIR"
    echo "✓ 5-fold unfreezing ablation complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

# ── Step 4: Backbone swap — 3-fold, frozen and fine-tuned, GPU ────
echo "── Step 4: Backbone swap (PANNs / YAMNet / HuBERT, frozen and fully fine-tuned) ──"
echo "3-fold patient-level CV. Results in"
echo "results/reviewer7_backbone_swap/{panns,yamnet,hubert}[_full]/."
read -p "Rerun backbone swap (all 3 backbones x both modes)? [y/N]: " run_backbone
if [[ "$run_backbone" =~ ^[Yy]$ ]]; then
    cd src/reviewer7_backbone_swap
    for bb in panns yamnet hubert; do
        python train_backbone_swap_cv.py --backbone "$bb" --unfreeze_mode frozen \
            --data_dir "$DATA_DIR/" --output_dir "../../results/reviewer7_backbone_swap/$bb/" \
            --n_folds 3 --epochs 5 --seed 42
        python train_backbone_swap_cv.py --backbone "$bb" --unfreeze_mode full \
            --data_dir "$DATA_DIR/" --output_dir "../../results/reviewer7_backbone_swap/${bb}_full/" \
            --n_folds 3 --epochs 5 --backbone_lr 5e-5 --seed 42
    done
    cd "$SCRIPT_DIR"
    echo "✓ Backbone swap complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

# ── Step 5: Denoise-then-classify comparison ──────────────────────
echo "── Step 5a: Denoise-then-classify, frozen model, 10-fold ──"
echo "Results in results/reviewer7_denoiser_benchmark/{clean_only,denoiser_comparison}_10fold/."
echo "Requires the pre-mixed lambda sweep, generated locally into $MIXED_DIR on first use."
read -p "Rerun 10-fold denoiser benchmark (Step 1 + Step 2)? [y/N]: " run_denoiser10
if [[ "$run_denoiser10" =~ ^[Yy]$ ]]; then
    if [ ! -d "$MIXED_DIR/lambda_0.0" ]; then
        echo "Generating pre-mixed lambda-sweep dataset locally (same source data, no download)..."
        python "$SCRIPT_DIR/../src/generate_mixed_datasets.py" \
            --data_dir "$DATA_DIR/" --output_dir "$MIXED_DIR/" --seed 42
    fi
    cd src/reviewer7_denoiser_benchmark
    python run_denoiser_benchmark_cv.py \
        --data_dir "$DATA_DIR/" --mixed_dir "$MIXED_DIR/" \
        --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/ \
        --n_folds 10 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
    python run_denoiser_benchmark_cv.py \
        --data_dir "$DATA_DIR/" --mixed_dir "$MIXED_DIR/" \
        --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_10fold/ \
        --n_folds 10 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \
        --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/checkpoints/ \
        --seed 42
    cd "$SCRIPT_DIR"
    echo "✓ 10-fold denoiser benchmark complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

echo "── Step 5b: Denoise-then-classify, fully fine-tuned model, 5-fold ──"
echo "Results in results/reviewer7_denoiser_benchmark/{clean_only,denoiser_comparison}_full_5fold/."
read -p "Rerun 5-fold fully-fine-tuned denoiser benchmark (Step 1 + Step 2)? [y/N]: " run_denoiser5
if [[ "$run_denoiser5" =~ ^[Yy]$ ]]; then
    if [ ! -d "$MIXED_DIR/lambda_0.0" ]; then
        echo "Generating pre-mixed lambda-sweep dataset locally (same source data, no download)..."
        python "$SCRIPT_DIR/../src/generate_mixed_datasets.py" \
            --data_dir "$DATA_DIR/" --output_dir "$MIXED_DIR/" --seed 42
    fi
    cd src/reviewer7_denoiser_benchmark
    python run_denoiser_benchmark_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
        --data_dir "$DATA_DIR/" --mixed_dir "$MIXED_DIR/" \
        --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/ \
        --n_folds 5 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
    python run_denoiser_benchmark_cv.py --unfreeze_mode full \
        --data_dir "$DATA_DIR/" --mixed_dir "$MIXED_DIR/" \
        --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_full_5fold/ \
        --n_folds 5 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \
        --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/checkpoints/ \
        --seed 42
    cd "$SCRIPT_DIR"
    echo "✓ 5-fold fully-fine-tuned denoiser benchmark complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

# ── Done ──────────────────────────────────────────────────────────
echo "============================================"
echo "  RUN COMPLETE (see above for what actually ran vs. was skipped)"
echo "============================================"
echo ""
echo "Results, where generated, are under results/<experiment>/. See README.md's"
echo "'Verifying a fresh run' section for how to compare them against the"
echo "checked-in reference values."
