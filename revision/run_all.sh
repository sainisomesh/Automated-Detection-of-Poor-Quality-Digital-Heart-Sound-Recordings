#!/usr/bin/env bash
# ============================================================================
# RUN ALL: revision experiments
# ============================================================================
#
# Reproduces the four additional experiments reported in the revised
# manuscript:
#
#   Step 1  Comparison against two published PCG quality-assessment methods
#           (Tang et al. 2021, Giordano et al. 2021), 5-fold, CPU only.
#   Step 2  Backbone adaptation ablation: frozen, fully fine-tuned, and
#           top-K-layer unfreezing, 5-fold, plus the per-lambda matched
#           benchmark for the fully fine-tuned model.
#   Step 3  Backbone swap: PANNs CNN14, YAMNet and HuBERT substituted for the
#           AST encoder, each frozen and fully fine-tuned, 5-fold, matched
#           against Step 2's frozen/full AST results for a paired comparison.
#   Step 4  Denoise-then-classify comparison: a clean-only classifier
#           evaluated on corrupted audio with and without each denoiser, run
#           for both the fully fine-tuned model (5-fold) and the frozen model
#           (10-fold).
#
# Every step is optional and prompts before running; results for all of them
# are already checked in under results/, so a fresh run is a verification
# rather than a prerequisite. Each step prints the fold count it uses.
#
# Normally reached from ../run_all.sh (Step 2), but can be run on its own.
#
# Usage:
#   ./run_all.sh          ask before each experiment
#   ./run_all.sh --yes    run every experiment without asking
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/../env_setup.sh"
export ASSUME_YES="${ASSUME_YES:-0}"
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# Reuses ../dataset and ../mixed_dataset, the same source data the sibling
# package uses, rather than keeping a second copy.
DATA_DIR="$SCRIPT_DIR/../dataset"
MIXED_DIR="$SCRIPT_DIR/../mixed_dataset"

echo "============================================"
echo "  Revision Experiments"
echo "============================================"
echo ""

# Dependencies and the raw source dataset (~3 GB, from Zenodo) are set up the
# first time an experiment is selected, so answering "n" everywhere costs
# nothing. The pre-mixed lambda sweep that Step 4 needs is generated locally.
setup_python
PREPARED=0
prepare() {
    if [ "$PREPARED" = "0" ]; then
        echo "── Installing dependencies and fetching the dataset (first time only) ──"
        install_requirements "$SCRIPT_DIR/requirements.txt"
        python "$SCRIPT_DIR/../download_data.py" "$DATA_DIR"
        PREPARED=1
    fi
}
echo ""

# ── Step 1: Published baselines (Tang, Giordano), 5-fold, CPU ────
echo "── Step 1: Published baselines (Tang et al., Giordano et al.) ──"
echo "5-fold patient-level CV, CPU only. Results are already in results/reviewer1_baselines/;"
echo "rerunning regenerates them in place."
if ask "Rerun baselines?"; then
    prepare
    cd src/reviewer1_baselines
    python train_tang_baseline_cv.py \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_baselines/tang_full/ \
        --n_folds 5 --seed 42
    python train_giordano_baseline_cv.py \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_baselines/giordano_full/ \
        --n_folds 5 --seed 42
    python compute_significance_paired.py
    cd "$SCRIPT_DIR"
    echo "✓ Baselines complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

# ── Step 2: Backbone adaptation ablation ──────────────────────────
echo "── Step 2: Backbone adaptation ablation, 5-fold, plus per-lambda benchmark ──"
echo "Results in results/reviewer1_unfreezing_ablation/{frozen,full,topk2,topk4}_5fold/,"
echo "full_5fold_{clean,fixed10,variable}/ and per_lambda_unfrozen/. These are the ablation"
echo "table, the per-lambda table and the metrics-vs-noise figure in the manuscript."
if ask "Rerun 5-fold unfreezing ablation (7 conditions)?"; then
    prepare
    cd src/reviewer1_unfreezing_ablation
    python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/frozen_5fold/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 2 --grad_checkpointing \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_unfreezing_ablation/topk2_5fold/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 --grad_checkpointing \
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

# ── Step 3: Backbone swap, 5-fold, frozen and fine-tuned, GPU ────
echo "── Step 3: Backbone swap (PANNs / YAMNet / HuBERT, frozen and fully fine-tuned) ──"
echo "5-fold patient-level CV, matched against Step 2's frozen_5fold/full_5fold_variable AST"
echo "results for a paired comparison. Results in results/reviewer7_backbone_swap/{panns,yamnet,hubert}[_full]/."
if ask "Rerun backbone swap (all 3 backbones x both modes)?"; then
    prepare
    cd src/reviewer7_backbone_swap
    for bb in panns yamnet hubert; do
        python train_backbone_swap_cv.py --backbone "$bb" --unfreeze_mode frozen \
            --data_dir "$DATA_DIR/" --output_dir "../../results/reviewer7_backbone_swap/$bb/" \
            --n_folds 5 --epochs 5 --seed 42
        python train_backbone_swap_cv.py --backbone "$bb" --unfreeze_mode full \
            --data_dir "$DATA_DIR/" --output_dir "../../results/reviewer7_backbone_swap/${bb}_full/" \
            --n_folds 5 --epochs 5 --backbone_lr 5e-5 --seed 42
    done
    python compute_significance_full_paired.py
    cd "$SCRIPT_DIR"
    echo "✓ Backbone swap complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

# ── Step 4: Denoise-then-classify comparison ──────────────────────
echo "── Step 4a: Denoise-then-classify, frozen model, 10-fold ──"
echo "Results in results/reviewer7_denoiser_benchmark/{clean_only,denoiser_comparison}_10fold/."
echo "Requires the pre-mixed lambda sweep, generated locally into $MIXED_DIR on first use."
if ask "Rerun 10-fold denoiser benchmark (train, then evaluate each denoiser)?"; then
    prepare
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

echo "── Step 4b: Denoise-then-classify, fully fine-tuned model, 5-fold ──"
echo "Results in results/reviewer7_denoiser_benchmark/{clean_only,denoiser_comparison}_full_5fold/."
if ask "Rerun 5-fold fully-fine-tuned denoiser benchmark (train, then evaluate each denoiser)?"; then
    prepare
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
