#!/bin/bash
# ============================================================================
# RUN ALL — Reproduce AST Heart Quality Noise Robustness Experiments
# ============================================================================
#
# This script runs the complete reproducibility pipeline:
#   1. Installs dependencies
#   2. (Optional) Generates pre-mixed datasets for each λ
#   3. Runs Per-Lambda 10-Fold CV
#   4. Runs Three Training Strategies 10-Fold CV
#   5. Computes metrics from raw predictions
#   6. Generates all paper figures
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Requirements:
#   - Python 3.8+
#   - CUDA-enabled GPU recommended (~8GB VRAM)
#   - ~4 GB disk for datasets, ~2 GB for results
#
# Estimated runtime on A100 GPU:
#   Per-Lambda CV:        ~24 hours (10 λ × 10 folds × 5 epochs)
#   Three Strategies CV:  ~36 hours (3 strategies × 10 folds × 5 epochs × 10 λ eval)
#
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  AST Heart Quality — Reproducibility Suite"
echo "============================================"
echo ""

# ── Step 0: Check dataset ──────────────────────────────────────────
if [ ! -d "dataset/PhysioNet2022" ]; then
    echo "ERROR: Dataset not found!"
    echo ""
    echo "Please download the dataset from Zenodo and extract it:"
    echo "  1. Download ast-heart-quality-dataset.zip from [ZENODO_LINK]"
    echo "  2. Unzip into this directory:"
    echo "     unzip ast-heart-quality-dataset.zip -d dataset/"
    echo ""
    echo "Expected structure:"
    echo "  dataset/"
    echo "    PhysioNet2022/training_data/*.wav  (3,163 files)"
    echo "    ICBHI2017/*.wav                    (174 files)"
    echo "    ESC-50/audio/*.wav                 (2,000 files)"
    echo "    UrbanSound8K/*.wav                 (8,732 files)"
    exit 1
fi

echo "✓ Dataset found"
echo "  PhysioNet2022: $(find dataset/PhysioNet2022 -name '*.wav' | wc -l | tr -d ' ') files"
echo "  ICBHI2017:     $(find dataset/ICBHI2017 -name '*.wav' | wc -l | tr -d ' ') files"
echo "  ESC-50:        $(find dataset/ESC-50 -name '*.wav' | wc -l | tr -d ' ') files"
echo "  UrbanSound8K:  $(find dataset/UrbanSound8K -name '*.wav' | wc -l | tr -d ' ') files"
echo ""

# ── Step 1: Install dependencies ──────────────────────────────────
echo "── Step 1: Installing dependencies ──"
pip install -r requirements.txt
echo ""

# ── Step 2: (Optional) Generate pre-mixed datasets ────────────────
echo "── Step 2: Generate pre-mixed datasets (optional) ──"
read -p "Generate pre-mixed WAV files for each λ? (~19 GB, can skip) [y/N]: " gen_mixed
if [[ "$gen_mixed" =~ ^[Yy]$ ]]; then
    echo "Generating mixed datasets..."
    python src/generate_mixed_datasets.py \
        --data_dir dataset/ \
        --output_dir mixed_dataset/ \
        --seed 42
    echo "✓ Mixed datasets generated"
else
    echo "Skipping pre-mixed generation (mixing will happen on-the-fly during training)"
fi
echo ""

# ── Step 3: Per-Lambda CV ─────────────────────────────────────────
echo "── Step 3: Per-Lambda Cross-Validation ──"
echo "Training separate models for each λ value (10 λ × 10 folds × 5 epochs)"
echo "This will take several hours on GPU..."
echo ""

cd src/
python train_per_lambda_cv.py \
    --data_dir ../dataset/ \
    --output_dir ../results/per_lambda_cv/ \
    --n_folds 10 \
    --epochs 5 \
    --seed 42

echo "✓ Per-Lambda CV complete"
echo ""

# ── Step 4: Three Training Strategies CV ──────────────────────────
echo "── Step 4: Three Training Strategies Cross-Validation ──"
echo "Training clean / noise_0_10 / noise_10 strategies (3 × 10 folds × 5 epochs)"
echo "This will take several hours on GPU..."
echo ""

python train_three_strategies_cv.py \
    --data_dir ../dataset/ \
    --output_dir ../results/three_strategies_cv/ \
    --n_folds 10 \
    --epochs 5 \
    --seed 42

echo "✓ Three Strategies CV complete"
echo ""

# ── Step 5: Compute metrics ───────────────────────────────────────
echo "── Step 5: Computing metrics from raw predictions ──"
python compute_metrics.py --results_dir ../results/
echo ""

# ── Step 6: Generate figures ──────────────────────────────────────
echo "── Step 6: Generating paper figures ──"
python visualize_results.py \
    --results_dir ../results/ \
    --output_dir ../results/figures/

cd ..
echo ""

# ── Done ──────────────────────────────────────────────────────────
echo "============================================"
echo "  ✅ ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  results/per_lambda_cv/          Per-lambda predictions + metrics"
echo "  results/three_strategies_cv/    Three-strategy predictions + metrics"
echo "  results/figures/                Paper figures (PNG + CSV)"
echo ""
echo "To verify against reference results, compare:"
echo "  results/per_lambda_cv/per_lambda_progress.json"
echo "  results/three_strategies_cv/final_results.json"
echo "with the files in results/ (pre-computed from Vertex AI)."
