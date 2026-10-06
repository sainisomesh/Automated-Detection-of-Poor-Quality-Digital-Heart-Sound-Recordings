# Revision Experiments

Code and results for four experiments of **Automated Detection of Poor-Quality Digital
Heart Sounds via Noise Augmentation** ([SSRN 6749564](https://ssrn.com/abstract=6749564)).

The parent directory `../` holds the frozen-backbone per-lambda and three-strategies
experiments and the single entry point `../run_all.sh`. Both use the same source datasets
and the same 5-fold patient splits.

All results are checked in under `results/`. `../run_all.sh` (or
`python ../verify_paper_results.py`) recomputes every reported number from them without
retraining.

## Experiments

| Experiment | Directory | Folds | What it measures |
|---|---|---|---|
| Published baselines | `reviewer1_baselines` | 5 | AST-QA against two prior PCG quality-assessment methods: Tang et al. (2021), an SVM over ten hand-crafted features, and Giordano et al. (2021), an SNR-threshold method. |
| Backbone adaptation ablation | `reviewer1_unfreezing_ablation` | 5 | Frozen backbone vs. full fine-tuning vs. unfreezing only the top K transformer layers, plus the per-lambda matched benchmark for the fully fine-tuned model. |
| Backbone swap | `reviewer7_backbone_swap` | 5 | PANNs CNN14, YAMNet and HuBERT substituted for the AST encoder, each evaluated frozen and fully fine-tuned, to test whether the advantage is architectural. |
| Denoise-then-classify | `reviewer7_denoiser_benchmark` | 5 | A clean-only classifier evaluated on corrupted audio with and without three denoisers, compared against noise-aware training. |

Every experiment evaluates across the same ten noise-intensity levels used throughout the
study, $\lambda \in \{0, 0.25, 0.5, 1, 5, 10, 25, 50, 75, 100\}$, and uses patient-level
splits with a fixed seed of 42.

## Frozen and fine-tuned results

The fully fine-tuned AST backbone is the primary configuration, with frozen-backbone
results alongside it for comparison:

- `reviewer1_unfreezing_ablation`: `{frozen,topk2,topk4}_5fold/` and `full_5fold_variable/`
  are the ablation itself. `full_5fold_clean/` and `full_5fold_fixed10/` are the other two
  training strategies under the fine-tuned backbone, and `per_lambda_unfrozen/` is the
  per-lambda matched benchmark. `frozen_5fold/` and `full_5fold_variable/` also supply the
  AST reference values for the backbone-swap comparison below, since both experiments use
  the same fold construction (seed 42, `KFold(n_splits=5, ...)` over the same sorted patient
  list) and so evaluate the same held-out patients fold for fold.
- `reviewer7_backbone_swap`: `{panns,yamnet,hubert}/` are frozen, `*_full/` are fully
  fine-tuned. `compute_significance_full_paired.py` pairs each against the matching AST
  condition above.
- `reviewer7_denoiser_benchmark`: `*_full_5fold/` use the fine-tuned backbone,
  `*_frozen_5fold/` use the frozen backbone. The noise-aware reference for each is the
  matching variable-noise run above (`full_5fold_variable/`, `frozen_5fold/`).
- `reviewer1_baselines`: `compute_significance_paired.py` compares the baselines against
  the fine-tuned model on matched folds.

## Layout

```
revision/
├── run_all.sh                      # interactive, one prompt per experiment
├── requirements.txt
├── fold_assignments/
│   ├── patient_folds_5fold.csv     # 942 patients, 3163 recordings
│   └── generate_5fold_assignments.py
├── src/
│   ├── models/ast_qa.py                    # AST model with the binary QA head
│   ├── reviewer1_baselines/                # Tang and Giordano features, training, significance
│   ├── reviewer1_unfreezing_ablation/      # frozen/fine-tuned/top-K model wrapper and training
│   ├── reviewer7_backbone_swap/            # PANNs, YAMNet and HuBERT wrappers and training
│   ├── reviewer7_denoiser_benchmark/       # wavelet and LU-Net denoisers, training and evaluation
│   └── figures/regenerate_fig3_unfrozen.py # regenerates the metrics-vs-noise figure
└── results/                                # aggregated metrics and raw per-fold predictions
```

## Setup

`run_all.sh` creates a virtual environment (`../.venv/`), installs `requirements.txt`
and downloads the source dataset into `../dataset/` the first time an experiment is
selected. To set up by hand instead:

```bash
cd reproducibility
python -m pip install -r requirements.txt
python download_data.py dataset
```

The pre-mixed lambda sweep required by the denoise-then-classify experiment is generated
locally from that dataset the first time Step 4 runs.

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
python compute_significance_paired.py   # paired, vs. the fine-tuned AST-QA model
```

### Backbone adaptation ablation

All four conditions, the three training strategies under the fine-tuned backbone, and the
per-lambda matched benchmark, all at 5-fold. `frozen_5fold/` and `full_5fold_variable/` also
serve as the AST reference for the backbone swap below:

```bash
cd src/reviewer1_unfreezing_ablation
python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/frozen_5fold/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 2 --grad_checkpointing \
    --data_dir ../../../dataset/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk2_5fold/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 --grad_checkpointing \
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

### Backbone swap (5-fold)

`compute_significance_full_paired.py` reads its AST reference from
`results/reviewer1_unfreezing_ablation/{frozen_5fold,full_5fold_variable}/`, so that step
must have produced those two directories first (either from this run or from the checked-in
results) before it can pair fold for fold.

```bash
cd src/reviewer7_backbone_swap
for bb in panns yamnet hubert; do
  python train_backbone_swap_cv.py --backbone $bb --unfreeze_mode frozen \
      --data_dir ../../../dataset/ --output_dir ../../results/reviewer7_backbone_swap/$bb/ \
      --n_folds 5 --epochs 5 --seed 42
  python train_backbone_swap_cv.py --backbone $bb --unfreeze_mode full \
      --data_dir ../../../dataset/ --output_dir ../../results/reviewer7_backbone_swap/${bb}_full/ \
      --n_folds 5 --epochs 5 --backbone_lr 5e-5 --seed 42
done
python compute_significance_full_paired.py    # paired, vs. the matched 5-fold AST results
```

### Denoise-then-classify

Each configuration runs in two stages: train the clean-only classifier and save its
per-fold weights, then evaluate those saved weights under each denoiser without
retraining. `--unfreeze_mode full` saves and loads the whole model, since the encoder
differs per fold; frozen mode saves only the classification head, which is the only part
that differs.

Frozen backbone, 5-fold:

```bash
cd src/reviewer7_denoiser_benchmark
python run_denoiser_benchmark_cv.py \
    --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_frozen_5fold/ \
    --n_folds 5 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
python run_denoiser_benchmark_cv.py \
    --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_frozen_5fold/ \
    --n_folds 5 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \
    --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_frozen_5fold/checkpoints/ \
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
python src/figures/regenerate_fig3_unfrozen.py   # writes results/figures/Fig3_*.pdf
```

## Verifying a fresh run

Every training script writes raw per-fold `predictions.csv` files plus aggregated
metrics (mean and 95% half-width, 1.96 · SD / sqrt(n_folds), per lambda). With the same
seed and fold count, a rerun reproduces the checked-in values up to GPU non-determinism.
`python ../verify_paper_results.py` compares whatever predictions are in `results/` with
the reference values.

The significance scripts are deterministic and read only the prediction CSVs, so they
reproduce their JSON outputs exactly.
