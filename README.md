AST HEART QUALITY — NOISE ROBUSTNESS EVALUATION
=================================================

DESCRIPTION
-----------
This repository contains the code and data to reproduce the noise
robustness evaluation of the AST-QA (Audio Spectrogram Transformer
for Quality Assurance) model for digital heart sound quality detection.

The AST-QA model classifies 10-second audio recordings as either
containing a valid heart sound (label=1) or being noise-contaminated
(label=0). It uses Mel-spectrograms processed through MIT's pretrained
Audio Spectrogram Transformer (AST) with a custom binary classification
head.

Noise contamination is controlled by a parameter λ using RMS-based mixing:

    mixed = heart + λ × (noise × (rms_heart / rms_noise))
    where noise = lung_sound + 0.5 × environmental_sound

Two experiments are included:

  1. Per-Lambda Cross-Validation
     Trains a separate model at each noise level (λ), 10-fold
     patient-level CV. Tests how well a model trained at a specific
     noise level performs at that same level.
     λ ∈ {0, 0.25, 0.5, 1, 5, 10, 25, 50, 75, 100}

  2. Three Training Strategies
     Compares three noise-aware training approaches, each evaluated
     across all 10 λ test levels, 10-fold patient-level CV:
       - Clean:      Trained on clean heart sounds only
       - Noise [0,10]: Trained with random noise λ ~ Uniform[0, 10]
       - Noise 10:   Trained with fixed noise λ = 10


REQUIREMENTS
------------
  Python 3.8+
  CUDA-enabled GPU recommended (~8 GB VRAM minimum)

  Install dependencies:
    pip install -r requirements.txt


DATA AVAILABILITY
-----------------
  All datasets and pre-generated mixed audio are available on Zenodo:

    DOI: https://doi.org/10.5281/zenodo.19638493
    Download: https://zenodo.org/records/19638493

  The Zenodo archive contains:

  1. RAW SOURCE DATASETS (~3.5 GB)
     The four source audio datasets used in our experiments,
     resampled to 16 kHz mono WAV format:

     dataset/
       PhysioNet2022/training_data/   3,163 heart sound recordings
       ICBHI2017/                     174 respiratory/lung sounds
       ESC-50/audio/                  2,000 environmental sounds
       UrbanSound8K/                  8,732 urban noise sounds

  2. PRE-MIXED DATASETS (~19 GB)
     Heart sounds mixed with structured noise at each λ level,
     ready for direct use. Each λ folder contains balanced
     mixed heart+noise (label=1) and noise-only (label=0) files:

     mixed_dataset/
       lambda_0.0/                    Clean heart sounds + noise-only
         manifest.csv                 File labels and metadata
         {patient}_{loc}_mixed.wav    Heart + noise at λ=0
         noise_{N}.wav                Noise-only samples
       lambda_0.25/
       lambda_0.5/
       lambda_1.0/
       lambda_5.0/
       lambda_10.0/
       lambda_25.0/
       lambda_50.0/
       lambda_75.0/
       lambda_100.0/

  Sources:
    - PhysioNet 2022 CirCor:  https://physionet.org/content/circor-heart-sound/1.0.3/
    - ICBHI 2017:             https://bhichallenge.med.auth.gr/
    - ESC-50:                 https://github.com/karolpiczak/ESC-50
    - UrbanSound8K:           https://urbansounddataset.weebly.com/

  NOTE ON MIXING: The training scripts mix noise on-the-fly during
  training (stochastic noise selection per epoch). The pre-mixed
  datasets use deterministic seeding for reproducibility and audio
  inspection. To regenerate the mixed datasets locally:

    python src/generate_mixed_datasets.py \
      --data_dir dataset/ --output_dir mixed_dataset/


HOW TO RUN
----------
  Option A — Run everything with one command:

    chmod +x run_all.sh
    ./run_all.sh

  Option B — Run experiments individually:

    cd src/

    # Experiment 1: Per-Lambda CV
    python train_per_lambda_cv.py \
      --data_dir ../dataset/ \
      --output_dir ../results/per_lambda_cv/ \
      --n_folds 10 --epochs 5 --seed 42

    # Experiment 2: Three Training Strategies
    python train_three_strategies_cv.py \
      --data_dir ../dataset/ \
      --output_dir ../results/three_strategies_cv/ \
      --n_folds 10 --epochs 5 --seed 42

    # Compute metrics from raw prediction CSVs
    python compute_metrics.py --results_dir ../results/

    # Generate paper figures
    python visualize_results.py \
      --results_dir ../results/ \
      --output_dir ../results/figures/

    cd ..

  Option C — Generate pre-mixed datasets (for inspection or upload):

    cd src/

    # Full generation (~19 GB, all λ values)
    python generate_mixed_datasets.py \
      --data_dir ../dataset/ --output_dir ../mixed_dataset/

    # Demo mode (10 examples per λ, ~200 MB)
    python generate_mixed_datasets.py \
      --data_dir ../dataset/ --output_dir ../demo_samples/ --demo

    cd ..


OUTPUT
------
  results/
    per_lambda_cv/
      per_lambda_progress.json          Aggregated AUROC/F1 per λ
      per_lambda_metrics.csv            Summary table
      raw_predictions/                  100 CSVs (10 λ × 10 folds)
        lambda_{λ}/fold_{N}/predictions.csv

    three_strategies_cv/
      final_results.json                Aggregated results for all strategies
      raw_predictions/                  300 CSVs (3 strategies × 10 folds × 10 λ)
        fold_{N}/{strategy}/lambda_{λ}.csv

    extreme_stress/
      extreme_stress_results.csv        Single-model stress test results

    figures/
      robustness_auroc_vs_noise.png     Main paper figure
      stress_degradation_curve.png      Stress degradation curves
      stress_sensitivity_specificity.png
      average_metrics_table.csv


VERIFICATION
------------
  Pre-computed results from Vertex AI (NVIDIA A100) runs are included
  in the results/ directory. After re-running, compare your results
  with these reference values. Minor differences (< 0.01 AUROC) are
  expected due to GPU/hardware non-determinism.


CITATION
--------
  [To be added upon publication]
