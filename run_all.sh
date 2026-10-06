#!/usr/bin/env bash
# Reproduce the results of "Automated Detection of Poor-Quality Digital Heart
# Sounds via Noise Augmentation".
#
# Step 1  Verify (minutes, CPU only, no data download). Re-runs the paired
#         significance tests on the checked-in per-fold predictions, checks
#         they match the checked-in JSONs, and regenerates Figure 3.
# Step 2  Retrain the experiments in revision/ (GPU, optional). Hands off to
#         revision/run_all.sh, which prompts per experiment.
# Step 3  Retrain the frozen-backbone per-lambda and three-strategies
#         experiments in src/ (GPU, optional).
#
# All cross-validation is 5-fold at the patient level, seed 42.
#
# Usage:
#   ./run_all.sh                 verify, then ask before any retraining
#   ./run_all.sh --verify-only   verify and stop
#   ./run_all.sh --yes           verify and retrain everything without asking
#
# Requires bash (Linux, macOS, or Git Bash / WSL on Windows) and Python
# 3.9-3.13. A virtual environment is created in .venv/ automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/env_setup.sh"

VERIFY_ONLY=0
export ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --verify-only) VERIFY_ONLY=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (see --help)" >&2; exit 1 ;;
    esac
done

echo "AST heart quality reproducibility"
echo ""

setup_python
echo ""

# Step 1: check the significance tests and regenerate Figure 3
echo "[Step 1] Checking results from the checked-in predictions"
install_requirements requirements-verify.txt
echo ""

echo "Re-running the paired significance tests (deterministic, B=1000)..."
SIG_A=revision/results/reviewer1_baselines/significance_vs_ast_qa_unfrozen_paired.json
SIG_B=revision/results/reviewer7_backbone_swap/significance_full_paired.json
BACKUP="$(mktemp -d)"
cp "$SIG_A" "$BACKUP/a.json"
cp "$SIG_B" "$BACKUP/b.json"
(cd revision/src/reviewer1_baselines && python compute_significance_paired.py > /dev/null)
(cd revision/src/reviewer7_backbone_swap && python compute_significance_full_paired.py > /dev/null)
SIG_OK=0
python - "$BACKUP/a.json" "$SIG_A" "$BACKUP/b.json" "$SIG_B" <<'PY' || SIG_OK=1
import json, math, sys

def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b

args = sys.argv[1:]
ok = all(same(json.load(open(args[i])), json.load(open(args[i + 1]))) for i in range(0, len(args), 2))
sys.exit(0 if ok else 1)
PY
cp "$BACKUP/a.json" "$SIG_A"
cp "$BACKUP/b.json" "$SIG_B"
rm -rf "$BACKUP"
if [ "$SIG_OK" = "0" ]; then
    echo "Significance tests reproduce the checked-in JSONs."
else
    echo "ERROR: regenerated significance results differ from the checked-in JSONs." >&2
    exit 1
fi

echo ""
echo "Regenerating Figure 3..."
python revision/src/figures/regenerate_fig3_unfrozen.py > /dev/null
echo "Figure 3 panels written to revision/results/figures/"
echo ""

if [ "$VERIFY_ONLY" = "1" ]; then
    echo "Verification complete (--verify-only, no retraining)."
    exit 0
fi

# Step 2: revision experiments
echo "[Step 2] Retrain the experiments in revision/ (GPU)"
echo "Overwrites the checked-in results under revision/results/ in place."
if ask "Continue into the revision experiments?"; then
    bash "$SCRIPT_DIR/revision/run_all.sh"
else
    echo "Skipping"
fi
echo ""

# Step 3: frozen-backbone experiments in src/
echo "[Step 3] Frozen-backbone per-lambda and three-strategies experiments (5-fold, GPU)"
echo "Per-lambda CV (10 lambdas x 5 folds) and three training strategies (3 x 5 folds)."
echo "Roughly 30 GPU-hours on an A100. Writes into results/."
if ask "Retrain the frozen-backbone experiments in src/?"; then
    install_requirements requirements.txt
    python download_data.py dataset

    if ask "Also write the pre-mixed WAV files for each lambda (~19 GB, for inspection only)?"; then
        python src/generate_mixed_datasets.py --data_dir dataset/ --output_dir mixed_dataset/ --seed 42
    fi

    cd src
    python train_per_lambda_cv.py \
        --data_dir ../dataset/ --output_dir ../results/per_lambda_cv/ \
        --n_folds 5 --epochs 5 --seed 42
    python train_three_strategies_cv.py \
        --data_dir ../dataset/ --output_dir ../results/three_strategies_cv/ \
        --n_folds 5 --epochs 5 --seed 42
    python compute_metrics.py --results_dir ../results/
    cd ..
    echo "Frozen-backbone experiments complete."
else
    echo "Skipping"
fi
echo ""

echo "Done."
