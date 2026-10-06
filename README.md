# Automated Detection of Poor-Quality Digital Heart Sounds via Noise Augmentation

Code, results and reproduction scripts for the paper of the same title
(SSRN: https://ssrn.com/abstract=6749564). Data: https://doi.org/10.5281/zenodo.19638493

## Description

The model classifies 10-second recordings as containing a heart sound, clean or
with added noise (label 1), or as noise without a heart sound (label 0). Log-Mel
spectrograms are passed through the Audio Spectrogram Transformer (AST),
initialized from the `MIT/ast-finetuned-audioset-10-10-0.4593` checkpoint on
Hugging Face, with a binary classification head (768 -> 128 -> 1). The whole
backbone is fine-tuned together with the head.

Noise is added with RMS-matched mixing controlled by an intensity λ:

```
mixed = heart + λ × (noise × (rms_heart / rms_noise))
where noise = lung_sound + 0.5 × environmental_sound
```

Every model is evaluated at λ ∈ {0, 0.25, 0.5, 1, 5, 10, 25, 50, 75, 100} with
5-fold patient-level cross-validation (seed 42).

## Quick start

```
./run_all.sh
```

On Windows, run this from Git Bash or WSL. The only prerequisite is Python
3.9-3.13 on PATH (as python3, python or py); the script creates a virtual
environment in .venv/ and installs everything else itself.

The first step takes about a minute on a laptop CPU and needs no GPU and no data
download. It recomputes every table, significance marker and quoted number from
the per-fold predictions checked in under revision/results/, re-runs the paired
significance tests, and regenerates Figure 3 into revision/results/figures/. It
then asks before starting any retraining.

```
./run_all.sh --verify-only   only the verification step
./run_all.sh --yes           verification, then all retraining without prompts
```

verify_paper_results.py can also be run on its own (needs numpy, pandas,
scikit-learn; see requirements-verify.txt).

## Where each result comes from

| Result | Directory |
|---|---|
| Table 2, per-λ matched benchmark | `revision/results/reviewer1_unfreezing_ablation/per_lambda_unfrozen/` |
| Figure 3, training strategies | `revision/results/reviewer1_unfreezing_ablation/full_5fold_{clean,variable,fixed10}/` |
| Table 3, published baselines | `revision/results/reviewer1_baselines/` |
| Table 4, backbone swap | `revision/results/reviewer7_backbone_swap/` |
| Table 5, fine-tuning ablation | `revision/results/reviewer1_unfreezing_ablation/{frozen,topk2,topk4}_5fold/`, `full_5fold_variable/` |
| Table 6, denoise-then-classify | `revision/results/reviewer7_denoiser_benchmark/`, with `frozen_5fold/` and `full_5fold_variable/` as the variable-noise reference |

All experiments use the patient splits in
`revision/fold_assignments/patient_folds_5fold.csv` (a copy is in
`fold_assignments/`). See revision/README.md for details of each experiment.

## Retraining

Retraining is optional, because every reported number already follows from the
checked-in predictions. A CUDA GPU is needed in practice; full fine-tuning of the
backbone needs about 3 GB of GPU memory with gradient checkpointing, and the
complete set of experiments takes several days of single-GPU time. The published
baselines (revision Step 1) run on CPU.

- Step 2 of run_all.sh runs revision/run_all.sh, with one prompt per experiment.
- Step 3 of run_all.sh runs the frozen-backbone per-λ and three-strategies
  experiments in src/.

The first retraining step installs requirements.txt and downloads the raw dataset
from Zenodo (~3 GB) into dataset/. Retraining writes into the same results
directories, so afterwards run

```
python verify_paper_results.py
```

to compare the new predictions with the reference values. Expect small
differences from GPU non-determinism.

## Data availability

Zenodo: https://doi.org/10.5281/zenodo.19638493

1. `ast-heart-quality-dataset.zip` (~3 GB), the four source datasets resampled to
   16 kHz mono WAV. download_data.py fetches and extracts it:

   ```
   dataset/
     PhysioNet2022/training_data/   3,163 heart sound recordings
     ICBHI2017/                     174 respiratory sound recordings
     ESC-50/audio/                  environmental sounds
     UrbanSound8K/                  urban sounds
   ```

2. `ast-heart-quality-mixed-lambdas.zip` (~19 GB), heart sounds mixed with noise
   at each λ. Not needed: the denoiser benchmark generates the same files locally
   with src/generate_mixed_datasets.py when it first runs.

Sources:

- PhysioNet 2022 CirCor: https://physionet.org/content/circor-heart-sound/1.0.3/
- ICBHI 2017: https://bhichallenge.med.auth.gr/
- ESC-50: https://github.com/karolpiczak/ESC-50
- UrbanSound8K: https://urbansounddataset.weebly.com/

## Layout

```
run_all.sh                 single entry point (see Quick start)
verify_paper_results.py    recomputes the reported numbers from predictions
download_data.py           fetches the source dataset from Zenodo
env_setup.sh               virtual environment setup shared by both run_all.sh
requirements.txt           full dependencies for retraining
requirements-verify.txt    minimal dependencies for verification
fold_assignments/          5-fold patient splits
src/                       frozen-backbone per-λ and three-strategies experiments
results/                   output of src/ when retrained (Step 3)
revision/                  ablation, baselines, backbone swap and denoiser experiments
```
