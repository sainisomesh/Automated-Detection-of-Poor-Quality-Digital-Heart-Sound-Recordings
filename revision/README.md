# Revision Reproducibility Package

Public reproducibility package for the "Accepted with Major Revision" response to
Reviewer #1 and Reviewer #7 on **Automated Detection of Poor-Quality Digital Heart Sounds
via Noise Augmentation** ([SSRN 6749564](https://ssrn.com/abstract=6749564)). Sibling to
`../` (the original package reproducing the published Table 2 / Figure 3, frozen backbone,
10-fold) — that package's contents are untouched by this one; nothing here overwrites or
depends on regenerating it.

**Status (2026-09-17): all experiments below are complete and their results are checked in.**
On 2026-09-15 the PI reviewed `reviewer1_unfreezing_ablation`'s results and decided a fully
fine-tuned (not frozen) AST backbone should become the primary/deployed model. That pivot has
downstream effects on three other experiments (baselines, backbone swap's AST comparator,
denoiser benchmark), all completed by 2026-09-17. **`PAPER_REVISIONS/paper_latex/main.tex` is
the ground truth for which specific result set is "final"** — where an experiment has more
than one completed variant (see table below), the one cited in `main.tex` is the one to trust;
older variants are kept for validation/history, not deleted, and are labeled as such.

## What's final vs. historical, per experiment

| Experiment | Final (cited in `main.tex`) | Historical / kept for a different reason |
|---|---|---|
| `reviewer1_baselines` (Tang, Giordano) | `results/significance_vs_ast_qa_unfrozen_paired.json` — paired bootstrap vs. the fully fine-tuned 5-fold AST-QA | `results/significance_vs_ast_qa.json` — unpaired, vs. the originally published frozen 10-fold AST-QA. Kept because its own JSON `methodology_note` explicitly says so, not superseded. |
| `reviewer1_unfreezing_ablation` | `results/{frozen,topk2,topk4}_5fold/`, `full_5fold_variable/` (the 4-way ablation table), `full_5fold_clean/`, `full_5fold_fixed10/`, `per_lambda_unfrozen/full/lambda_*/` | `results/{frozen,full,topk2,topk4}_3fold/` — the *original* 3-fold ablation. **Not dead**: `reviewer7_backbone_swap`'s AST comparator column in `main.tex` reads directly from these 3-fold folders (see its own README footnote), so they stay even though the 5-fold results superseded them for the ablation table itself. |
| `reviewer7_backbone_swap` | `results/{panns,yamnet,hubert}/` (frozen) and `results/{panns,yamnet,hubert}_full/` (fully fine-tuned), `significance_vs_ast_qa.json`, `significance_full_paired.json` | None — this experiment doesn't depend on which AST variant is "the" deployed model (frozen-vs-frozen and full-vs-full are both self-contained comparisons), so nothing here was superseded by the pivot. |
| `reviewer7_denoiser_benchmark` | `results/clean_only_full_5fold/`, `results/denoiser_comparison_full_5fold/` — the fully fine-tuned clean-only model vs. each denoiser, 5-fold | `results/clean_only_10fold/`, `results/denoiser_comparison_10fold/` — the *original* frozen, 10-fold run. Kept deliberately: this is what validated the pipeline against the already-published clean-only Table 2/Figure 3 numbers before trusting it on anything new (CLAUDE.md §10 guideline 7). T-BiLSTM (a 4th candidate) was attempted and canceled after a silent GPU-fallback bug produced no usable weights; its code lives in `PAPER_REVISIONS/reviewer7_denoiser_benchmark/src/attempted_not_used/` and was intentionally not vendored here since it produced no results. |

## Fold count policy — read this before comparing across tables

Carried over from `CLAUDE.md` §9.4 and §10.6, and the single most important caveat here:

- The original published Table 2 / Figure 3 (in `../`) used **10-fold** patient-level CV.
  That package is untouched.
- **New experiments in this package default to 5-fold**, not 10-fold — a deliberate,
  documented trade of rigor for GPU budget on a $300-credit account, using the fixed splits
  in `fold_assignments/patient_folds_5fold.csv`.
- **`reviewer1_unfreezing_ablation` (both variants) and `reviewer7_backbone_swap` use
  3-fold**, not the 5-fold default — using `fold_assignments/patient_folds_3fold.csv`. A
  further deliberate call (2026-09-13) to fit all conditions of each ablation within budget.
  `reviewer1_unfreezing_ablation`'s 5-fold results are a *separate*, later addition (built
  specifically for the pivot) that reuses the 5-fold split instead.
- `reviewer7_denoiser_benchmark` mixes both on purpose: 10-fold for the historical
  pipeline-validation run, 5-fold for the final comparison — see table above.
- **Never present any number from this package next to the original 10-fold Table 2 /
  Figure 3 numbers, or across a 3-fold/5-fold boundary within this package, without saying
  so explicitly.**

## Directory layout

```
revision/
├── README.md
├── run_all.sh                      # interactive, one prompt per step, matches this README
├── requirements.txt
├── fold_assignments/
│   ├── patient_folds_5fold.csv
│   ├── patient_folds_3fold.csv
│   └── generate_5fold_assignments.py
├── src/
│   ├── models/ast_qa.py                    # vendored, byte-identical copy of the published frozen model
│   ├── reviewer1_baselines/                # Tang + Giordano features, training, both significance scripts
│   ├── reviewer1_unfreezing_ablation/      # frozen/full/topk train script + per-lambda-unfrozen + model wrapper
│   ├── reviewer7_backbone_swap/            # PANNs/YAMNet/HuBERT wrappers, training, both significance scripts
│   ├── reviewer7_denoiser_benchmark/       # wavelet + LU-Net denoisers, training/eval (T-BiLSTM excluded, see above)
│   └── figures/regenerate_fig3_unfrozen.py # regenerates the manuscript's Figure 3 from the unfrozen 5-fold results
└── results/
    ├── reviewer1_baselines/
    ├── reviewer1_unfreezing_ablation/
    ├── reviewer7_backbone_swap/
    └── reviewer7_denoiser_benchmark/
```

## Setup

```bash
cd reproducibility/revision
pip install -r requirements.txt
```

Dataset: same source data as `../` — `PhysioNet2022`, `ICBHI2017`, `ESC-50`, `UrbanSound8K`,
laid out exactly as described in `../README.md`'s dataset section. `reviewer7_denoiser_benchmark`
additionally needs the pre-mixed `zenodo_mixed/lambda_*/` evaluation sets
(`ast-heart-quality-mixed-lambdas.zip` on Zenodo) for its λ-sweep evaluation.

## Reproduction commands

All scripts below run locally (CPU or GPU) and take a `--data_dir` pointing at the dataset
described above. Every script also supports `--gcs_bucket` (and related `--data_prefix` /
`--output_prefix` flags) for the private Vertex AI pipeline this package's own results were
produced on — **ignore those flags entirely for local reproduction**, they default to off.
`run_all.sh` runs all of the below interactively, one y/N prompt per step.

### Reviewer #1 baselines (Tang et al., Giordano et al.) — 5-fold, CPU only

```bash
cd src/reviewer1_baselines
python train_tang_baseline_cv.py \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_baselines/tang_full/ \
    --n_folds 5 --seed 42
python train_giordano_baseline_cv.py \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_baselines/giordano_full/ \
    --n_folds 5 --seed 42
```

### Reviewer #1 unfreezing ablation, 3-fold (historical)

```bash
cd src/reviewer1_unfreezing_ablation
python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/frozen_3fold/ \
    --n_folds 3 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode full --grad_checkpointing \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_3fold/ \
    --n_folds 3 --epochs 5 --backbone_lr 5e-5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 2 \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk2_3fold/ \
    --n_folds 3 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk4_3fold/ \
    --n_folds 3 --epochs 5 --seed 42
```

### Reviewer #1 unfreezing ablation, 5-fold (FINAL — cited in `main.tex`)

```bash
cd src/reviewer1_unfreezing_ablation
python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/frozen_5fold/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 2 \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk2_5fold/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/topk4_5fold/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_variable/ \
    --n_folds 5 --epochs 5 --seed 42
# Same fully fine-tuned model, the other two Figure-3 training strategies:
python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --noise_strategy clean \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_clean/ \
    --n_folds 5 --epochs 5 --seed 42
python train_unfreezing_ablation_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --noise_strategy fixed_10 \
    --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer1_unfreezing_ablation/full_5fold_fixed10/ \
    --n_folds 5 --epochs 5 --seed 42
# Table-2 analog for the fully fine-tuned model, one lambda per invocation:
for lam in 0.0 0.25 0.5 1.0 5.0 10.0 25.0 50.0 75.0 100.0; do
  python train_per_lambda_unfrozen_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
      --lambda_val $lam \
      --data_dir ../../../../data_processed/ \
      --output_dir ../../results/reviewer1_unfreezing_ablation/per_lambda_unfrozen/full/lambda_$lam/ \
      --n_folds 5 --epochs 5 --seed 42
done
```

### Reviewer #7 backbone swap — 3-fold, frozen + fully fine-tuned

```bash
cd src/reviewer7_backbone_swap
for bb in panns yamnet hubert; do
  python train_backbone_swap_cv.py --backbone $bb --unfreeze_mode frozen \
      --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer7_backbone_swap/$bb/ \
      --n_folds 3 --epochs 5 --seed 42
  python train_backbone_swap_cv.py --backbone $bb --unfreeze_mode full \
      --data_dir ../../../../data_processed/ --output_dir ../../results/reviewer7_backbone_swap/${bb}_full/ \
      --n_folds 3 --epochs 5 --backbone_lr 5e-5 --seed 42
done
```

### Reviewer #7 denoiser benchmark, 10-fold frozen model (historical validation)

```bash
cd src/reviewer7_denoiser_benchmark
python run_denoiser_benchmark_cv.py \
    --data_dir ../../../../data_processed/ --mixed_dir ../../../../zenodo_mixed/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/ \
    --n_folds 10 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
python run_denoiser_benchmark_cv.py \
    --data_dir ../../../../data_processed/ --mixed_dir ../../../../zenodo_mixed/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_10fold/ \
    --n_folds 10 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \
    --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/checkpoints/ \
    --seed 42
```

### Reviewer #7 denoiser benchmark, 5-fold fully fine-tuned model (FINAL — cited in `main.tex`)

```bash
cd src/reviewer7_denoiser_benchmark
python run_denoiser_benchmark_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \
    --data_dir ../../../../data_processed/ --mixed_dir ../../../../zenodo_mixed/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/ \
    --n_folds 5 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
python run_denoiser_benchmark_cv.py --unfreeze_mode full \
    --data_dir ../../../../data_processed/ --mixed_dir ../../../../zenodo_mixed/ \
    --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_full_5fold/ \
    --n_folds 5 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \
    --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/checkpoints/ \
    --seed 42
```

Note: `--unfreeze_mode full` checkpoints save/load the *entire* fine-tuned model (`fold_N_full.pth`),
not just the head, since the encoder itself differs across folds in this mode — unlike the
frozen-mode `fold_N_qa_classifier.pth` checkpoints, which omit the encoder because it's identical
across every fold. This is handled automatically by `--unfreeze_mode`; no separate flag needed.

## Known, permanent, disclosed gaps (not TODOs)

- **No checkpoint for the fully fine-tuned (now primary) model exists anywhere in this repo
  outside `clean_only_full_5fold/checkpoints/`** (and that's the clean-only variant, not the
  variable-noise deployed model). The single-recording inference recipe in the top-level
  `CLAUDE.md` §8.6 still loads the old frozen `phase6_paper_quality/results/best_quality_model.pth`
  — that checkpoint reflects the superseded frozen model, not the model reported in the current
  manuscript. Producing a deployable variable-noise, fully fine-tuned checkpoint would need a
  new training run with `--save_checkpoints`; not done as part of this consolidation.
- The originally published `reproducibility/results/three_strategies_cv/raw_predictions/` is
  permanently missing 7 of its 10 folds (only folds 1, 2, 10 exist anywhere) — the source GCS
  project was confirmed permanently deleted. This doesn't affect any aggregated number already
  reported, but that specific raw-CSV gap cannot be regenerated.

## Verifying a fresh run against the checked-in results

Every script above writes an aggregated `*_final_results.json` (mean + 95% CI per λ per
metric) and raw per-fold `predictions.csv` files. Compare your regenerated numbers against the
checked-in ones the same way `../README.md` describes for the original package — matching
seeds (`42`) and fold counts should reproduce them exactly (barring documented GPU
non-determinism), per `CLAUDE.md` §10.6's "if a result can't be regenerated byte-for-byte, it
doesn't go in the rebuttal or the paper" rule.
