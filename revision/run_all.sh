#!/bin/bash
# ============================================================================
# RUN ALL — Revision Reproducibility Suite
# ============================================================================
#
# Reproduces the NEW experiments built for the revision response to Reviewer
# #1 and Reviewer #7, INCLUDING the 2026-09-15 PI-directed pivot to a fully
# fine-tuned (not frozen) AST backbone as the primary/deployed model, whose
# numbers are what main.tex actually reports as of 2026-09-17. Sibling to
# ../run_all.sh, which reproduces the original published Table 2 / Figure 3
# (frozen backbone, 10-fold) and is untouched by this script.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# IMPORTANT — fold counts and model variants differ BETWEEN steps in this
# script, on purpose. See README.md "What's final vs. historical" before
# reporting any number next to a different step's results:
#   - reviewer1_baselines:                5-fold, vs. BOTH the frozen (historical)
#                                          and fully fine-tuned (final, in main.tex) AST-QA
#   - reviewer1_unfreezing_ablation:      3-fold (frozen/full/topk2/topk4, historical
#                                          ablation) AND 5-fold (frozen/full/topk2/topk4/
#                                          full-clean/full-fixed10 + per-lambda-full,
#                                          the numbers actually in main.tex)
#   - reviewer7_backbone_swap:            3-fold, frozen AND fully fine-tuned, all final
#   - reviewer7_denoiser_benchmark:       10-fold frozen-model comparison (historical,
#                                          validates against the published clean-only
#                                          numbers) AND 5-fold fully-fine-tuned-model
#                                          comparison (final, in main.tex)
#
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Reuses ../dataset and ../mixed_dataset -- the same source data as ../run_all.sh's
# package, not a separate copy. If you've already run ../run_all.sh, this is a no-op.
DATA_DIR="$SCRIPT_DIR/../dataset"
MIXED_DIR="$SCRIPT_DIR/../mixed_dataset"

echo "============================================"
echo "  Revision — Reproducibility Suite"
echo "============================================"
echo ""

# ── Step 0: Dataset (auto-downloads from Zenodo if missing) ────────
# Only the raw source dataset (~3 GB). The pre-mixed lambda sweep needed by the
# denoiser benchmark steps below is generated locally from this, not downloaded --
# see Step 5 below.
bash "$SCRIPT_DIR/../download_data.sh" "$DATA_DIR"
echo ""

# ── Step 1: Install dependencies ──────────────────────────────────
echo "── Step 1: Installing dependencies ──"
pip install -r requirements.txt
echo ""

# ── Step 2: Reviewer #1 baselines (Tang, Giordano) — 5-fold, CPU ──
echo "── Step 2: Reviewer #1 baselines (Tang et al., Giordano et al.) ──"
echo "5-fold, CPU only. Already complete in results/reviewer1_baselines/ --"
echo "rerunning will overwrite it with a fresh (should be identical) copy."
read -p "Rerun baselines? [y/N]: " run_baselines
if [[ "$run_baselines" =~ ^[Yy]$ ]]; then
    cd src/reviewer1_baselines
    python train_tang_baseline_cv.py \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_baselines/tang_full/ \
        --n_folds 5 --seed 42
    python train_giordano_baseline_cv.py \
        --data_dir "$DATA_DIR/" --output_dir ../../results/reviewer1_baselines/giordano_full/ \
        --n_folds 5 --seed 42
    echo "Significance vs. AST-QA: two variants are checked in --"
    echo "  significance_vs_ast_qa.json            (unpaired, vs. the published frozen 10-fold model)"
    echo "  significance_vs_ast_qa_unfrozen_paired.json (paired, vs. the fully fine-tuned 5-fold"
    echo "                                          model -- this is what main.tex cites)"
    echo "Regenerating these requires the AST-QA prediction CSVs referenced in each script's"
    echo "own --help; see compute_significance.py / compute_significance_paired.py."
    cd "$SCRIPT_DIR"
    echo "✓ Baselines complete"
else
    echo "Skipping (results already checked in)"
fi
echo ""

# ── Step 3: Reviewer #1 unfreezing ablation — 3-fold (historical) ────
echo "── Step 3a: Reviewer #1 unfreezing ablation, 3-fold (historical ablation) ──"
echo "Already complete in results/reviewer1_unfreezing_ablation/{frozen,full,topk2,topk4}_3fold/."
echo "NOTE: these 3-fold numbers are still a live input elsewhere -- reviewer7_backbone_swap's"
echo "AST comparator column reads directly from these folders (see its own README). Do not"
echo "delete or renumber them even though Step 3b below supersedes them for the ablation table."
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

echo "── Step 3b: Reviewer #1 unfreezing ablation, 5-fold (FINAL — cited in main.tex) ──"
echo "Already complete in results/reviewer1_unfreezing_ablation/{frozen,full,topk2,topk4}_5fold/"
echo "and full_5fold_{clean,fixed10,variable}/. This is the actual ablation table and Figure 3"
echo "'variable-noise (full)' curve in the current manuscript."
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
    echo "-- per-lambda unfrozen benchmark (Table 2 analog, fully fine-tuned model) --"
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

# ── Step 4: Reviewer #7 backbone swap — 3-fold, frozen + full, GPU ──
echo "── Step 4: Reviewer #7 backbone swap (PANNs/YAMNet/HuBERT, frozen + full) ──"
echo "3-fold (settled exception, see README.md). Both frozen and full-unfreeze are"
echo "already complete in results/reviewer7_backbone_swap/{panns,yamnet,hubert}[_full]/."
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

# ── Step 5: Reviewer #7 denoiser benchmark ─────────────────────────
echo "── Step 5a: Reviewer #7 denoiser benchmark, 10-fold frozen model (historical) ──"
echo "Validates the pipeline against the published clean-only Table 2/Figure 3 numbers."
echo "Already complete in results/reviewer7_denoiser_benchmark/{clean_only,denoiser_comparison}_10fold/."
echo "Needs the pre-mixed lambda sweep, generated locally into $MIXED_DIR on first use."
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

echo "── Step 5b: Reviewer #7 denoiser benchmark, 5-fold fully fine-tuned model (FINAL — cited in main.tex) ──"
echo "Already complete in results/reviewer7_denoiser_benchmark/{clean_only,denoiser_comparison}_full_5fold/."
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
echo "Results, where generated, are under results/<experiment>/. Compare against"
echo "the checked-in reference results the same way -- see README.md's"
echo "'Verifying a fresh run' section."
