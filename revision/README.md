# Revision Experiments

Reproducibility package for the four additional experiments reported in the revised
manuscript of **Automated Detection of Poor-Quality Digital Heart Sounds via Noise
Augmentation** ([SSRN 6749564](https://ssrn.com/abstract=6749564)).

The sibling package in `../` reproduces the per-lambda and three-strategies experiments
with a frozen backbone and 10-fold cross-validation. This package is self-contained and
does not modify it; both use the same source datasets.

All results reported below are checked in under `results/`, so the scripts can be read
and verified without rerunning anything.

## Experiments

| Experiment | Directory | Folds | What it measures |
|---|---|---|---|
| Published baselines | `reviewer1_baselines` | 5 | AST-QA against two prior PCG quality-assessment methods: Tang et al. (2021), an SVM over ten hand-crafted features, and Giordano et al. (2021), an SNR-threshold method. |
| Backbone adaptation ablation | `reviewer1_unfreezing_ablation` | 3 and 5 | Frozen backbone vs. full fine-tuning vs. unfreezing only the top K transformer layers, plus the per-lambda matched benchmark for the fully fine-tuned model. |
| Backbone swap | `reviewer7_backbone_swap` | 3 | PANNs CNN14, YAMNet and HuBERT substituted for the AST encoder, each evaluated frozen and fully fine-tuned, to test whether the advantage is architectural. |
| Denoise-then-classify | `reviewer7_denoiser_benchmark` | 5 and 10 | A clean-only classifier evaluated on corrupted audio with and without three denoisers, compared against noise-aware training. |

Every experiment evaluates across the same ten noise-intensity levels used throughout the
study, $\lambda \in \{0, 0.25, 0.5, 1, 5, 10, 25, 50, 75, 100\}$, and uses patient-level
splits with a fixed seed of 42.

## Frozen and fine-tuned results

The manuscript reports a fully fine-tuned AST backbone as its primary configuration, and
retains the frozen-backbone results alongside it for comparison. Both are included here:

- `reviewer1_unfreezing_ablation`: `{frozen,topk2,topk4}_5fold/` and `full_5fold_variable/`
  are the ablation itself. `full_5fold_clean/` and `full_5fold_fixed10/` are the other two
  training strategies under the fine-tuned backbone, and `per_lambda_unfrozen/` is the
  per-lambda matched benchmark. The `*_3fold/` directories are the same four conditions at
  3-fold, and supply the AST reference values for the backbone-swap comparison so that all
  four architectures there are compared at an identical fold count.
- `reviewer7_backbone_swap`: `{panns,yamnet,hubert}/` are frozen, `*_full/` are fully
  fine-tuned.
- `reviewer7_denoiser_benchmark`: `*_full_5fold/` use the fine-tuned backbone,
  `*_10fold/` use the frozen backbone.
- `reviewer1_baselines`: `compute_significance_paired.py` compares the baselines against
  the fine-tuned model on matched folds; `compute_significance.py` compares them against
  the frozen model.

## Layout

```
revision/
├── run_all.sh                      # interactive, one prompt per experiment
├── requirements.txt
├── fold_assignments/
│   ├── patient_folds_5fold.csv     # 942 patients, 3163 recordings
│   ├── patient_folds_3fold.csv
│   └── generate_5fold_assignments.py
├── src/
│   ├── models/ast_qa.py                    # the published frozen-backbone model
│   ├── reviewer1_baselines/                # Tang and Giordano features, training, significance
│   ├── reviewer1_unfreezing_ablation/      # frozen/fine-tuned/top-K model wrapper and training
│   ├── reviewer7_backbone_swap/            # PANNs, YAMNet and HuBERT wrappers and training
│   ├── reviewer7_denoiser_benchmark/       # wavelet and LU-Net denoisers, training and evaluation
│   └── figures/regenerate_fig3_unfrozen.py # regenerates the metrics-vs-noise figure
└── results/                                # aggregated metrics and raw per-fold predictions
```

## Setup

```bash
cd reproducibility/revision
pip install -r requirements.txt
```

`run_all.sh` downloads the source dataset automatically on first run. The pre-mixed
lambda sweep required by the denoise-then-classify experiment is generated locally from
that dataset the first time Step 5 runs.

## Reproduction commands

`run_all.sh` runs everything below interactively, one prompt per step. To run a single
experiment directly, use the commands in this section. Each training script also accepts
`--gcs_bucket` and related flags for cloud execution; they default to off and can be
ignored for local reproduction.

### Published baselines (5-fold, CPU only)

```bash
cd src/reviewer1_baselines
python train_tang_baseline_cv.py \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_baselines/tang_full/ \
    --n_folds 5 --seed 42
python train_giordano_baseline_cv.py \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_baselines/giordano_full/ \
    --n_folds 5 --seed 42
python compute_significance.py          # vs. frozen backbone, unpaired
python compute_significance_paired.py   # vs. fine-tuned backbone, paired
```

### Backbone adaptation ablation

3-fold conditions, which also serve as the AST reference for the backbone swap:

```bash
cd src/reviewer1_unfreezing_ablation
python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/frozen_3fold/ \
    --n_folds 3 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_3fold/ \
    --n_folds 3 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 2 \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk2_3fold/ \
    --n_folds 3 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk4_3fold/ \
    --n_folds 3 --epochs 5 --seed 42
```

5-fold conditions, the three training strategies under the fine-tuned backbone, and the
per-lambda matched benchmark:

```bash
cd src/reviewer1_unfreezing_ablation
for mode in frozen; do
  python train_unfreezing_ablation_cv.py --unfreeze_mode $mode \
      --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/${mode}_5fold/ \
      --n_folds 5 --epochs 5 --seed 42
done
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 2 \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk2_5fold/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk4_5fold/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_variable/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --noise_strategy clean \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_clean/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --noise_strategy fixed_10 \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_fixed10/ \
    --n_folds 5 --epochs 5 --seed 42
for lam in 0.0 0.25 0.5 1.0 5.0 10.0 25.0 50.0 75.0 100.0; do
  python train_per_lambda_unfrozen_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
      --lambda_val $lam \
      --data_dir ../../../dataset/ \
      --output_dir ../../results/reviewer1_unfreezing_ablation/per_lambda_unfrozen/full/lambda_$lam/ \
      --n_folds 5 --epochs 5 --seed 42
done
```

### Backbone swap (3-fold)

```bash
cd src/reviewer7_backbone_swap
for bb in panns yamnet hubert; do
  python train_backbone_swap_cv.py --backbone $bb --unfreeze_mode frozen \
      --data_dir ../../../dataset/ --output_dir ../../results/reviewer7_backbone_swap/$bb/ \
      --n_folds 3 --epochs 5 --seed 42
  python train_backbone_swap_cv.py --backbone $bb --unfreeze_mode full \
      --data_dir ../../../dataset/ --output_dir ../../results/reviewer7_backbone_swap/${bb}_full/ \
      --n_folds 3 --epochs 5 --backbone_lr 5e-5 --seed 42
done
python compute_significance_vs_ast.py
python compute_significance_full_paired.py
```

### Denoise-then-classify

Each configuration runs in two stages: train the clean-only classifier and save its
per-fold weights, then evaluate those saved weights under each denoiser without
retraining. `--unfreeze_mode full` saves and loads the whole model, since the encoder
differs per fold; frozen mode saves only the classification head, which is the only part
that differs.

Frozen backbone, 10-fold:

```bash
cd src/reviewer7_denoiser_benchmark
python run_denoiser_benchmark_cv.py \
    --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/ \
    --n_folds 10 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
python run_denoiser_benchmark_cv.py \
    --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_10fold/ \
    --n_folds 10 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \
    --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/checkpoints/ \
    --seed 42
```

Fully fine-tuned backbone, 5-fold:

```bash
cd src/reviewer7_denoiser_benchmark
python run_denoiser_benchmark_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/ \
    --n_folds 5 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
python run_denoiser_benchmark_cv.py --unfreeze_mode full \
    --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_full_5fold/ \
    --n_folds 5 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \
    --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/checkpoints/ \
    --seed 42
```

### Figure

```bash
cd src/figures
python regenerate_fig3_unfrozen.py
```

## Verifying a fresh run

Every training script writes an aggregated `*_final_results.json` with the mean and 95%
confidence interval per metric per lambda, plus raw per-fold `predictions.csv` files.
With the same seed and fold count, a rerun reproduces the checked-in values up to GPU
non-determinism.

The significance scripts are fully deterministic and read only the checked-in prediction
CSVs, so they reproduce their JSON outputs exactly.

## Data notes

The raw per-fold prediction CSVs for the sibling package's three-strategies experiment are
incomplete for folds 3 through 9; only the aggregated metrics survive for those. This
affects one comparison, `compute_significance_vs_ast.py`, which pools the folds that are
available and reports how many it used in both its console output and its JSON output. All
other comparisons use complete data.
