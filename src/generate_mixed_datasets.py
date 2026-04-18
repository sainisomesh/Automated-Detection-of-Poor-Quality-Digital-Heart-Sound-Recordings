#!/usr/bin/env python3
"""
Generate Pre-Mixed Datasets for Each Noise Level (λ).

Creates WAV files with heart sounds mixed with noise at each λ value,
using the exact same RMS mixing formula and deterministic seeds as
the training scripts.

NOTE: The training scripts (train_per_lambda_cv.py, train_three_strategies_cv.py)
already mix noise ON-THE-FLY during training — you do NOT need to run this
script before training. This script is provided for:
  1. Generating demo/example files for audio inspection
  2. Pre-generating full mixed datasets if desired
  3. Zenodo upload of mixed datasets for data availability

Modes:
  --demo     Generate 10 example files per λ (quick, ~200 MB total)
  (default)  Generate ALL files per λ (~19 GB total)

Output structure:
    mixed_dataset/
        lambda_0.0/
            {patientID}_{loc}_mixed.wav    (heart + noise, label=1)
            noise_{N}.wav                  (noise-only, label=0)
            manifest.csv
        lambda_0.25/
            ...

Usage:
    python generate_mixed_datasets.py --data_dir dataset/ --output_dir mixed_dataset/
    python generate_mixed_datasets.py --data_dir dataset/ --output_dir demo_samples/ --demo
"""

import argparse
import os
import random
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, sosfilt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SR = 16000
DURATION = 10
MAX_LENGTH = TARGET_SR * DURATION
DEFAULT_LAMBDAS = [0, 0.25, 0.5, 1, 5, 10, 25, 50, 75, 100]


def load_audio(path):
    """Load and preprocess a single audio file (matches training pipeline exactly)."""
    try:
        wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception:
        return None
    if len(wav) < 100 or np.max(np.abs(wav)) < 1e-6:
        return None
    wav = wav - np.mean(wav)
    nyquist = TARGET_SR / 2
    sos_hp = butter(2, 20 / nyquist, btype='high', output='sos')
    wav = sosfilt(sos_hp, wav)
    sos_lp = butter(5, 1000 / nyquist, btype='low', output='sos')
    wav = sosfilt(sos_lp, wav)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak
    if len(wav) > MAX_LENGTH:
        wav = wav[:MAX_LENGTH]
    elif len(wav) < MAX_LENGTH:
        n_repeats = int(np.ceil(MAX_LENGTH / len(wav)))
        wav = np.tile(wav, n_repeats)[:MAX_LENGTH]
    return wav


def mix_rms(heart, noise, lam):
    """Mix heart sound with noise at λ using RMS-based scaling.

    Formula: mixed = heart + λ × (noise × (rms_heart / rms_noise))
    """
    if lam == 0:
        return heart
    rms_h = np.sqrt(np.mean(heart**2))
    rms_n = np.sqrt(np.mean(noise**2))
    if rms_h == 0 or rms_n == 0:
        return heart
    scale = rms_h / rms_n
    mixed = heart + lam * (noise * scale)
    m_peak = np.max(np.abs(mixed))
    if m_peak > 1.0:
        mixed = mixed / m_peak
    return mixed


def generate_noise(icbhi_files, env_files, rng):
    """Generate structured noise: lung + 0.5 × environmental."""
    for _ in range(10):
        lung = load_audio(rng.choice(icbhi_files))
        env = load_audio(rng.choice(env_files))
        if lung is not None and env is not None:
            combined = lung + 0.5 * env
            peak = np.max(np.abs(combined))
            if peak > 0:
                combined = combined / peak
            return combined
    return np.zeros(MAX_LENGTH)


def main():
    parser = argparse.ArgumentParser(description="Generate pre-mixed datasets for each λ")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing dataset/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for mixed WAVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true",
                        help="Generate only 10 example files per λ (for inspection/demo)")
    parser.add_argument("--n_demo", type=int, default=10,
                        help="Number of demo files per λ (used with --demo)")
    parser.add_argument("--lambdas", type=str, default=",".join(str(l) for l in DEFAULT_LAMBDAS),
                        help="Comma-separated lambda values")
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]

    # Discover files
    data_root = Path(args.data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))
    icbhi_files = sorted(list(data_root.rglob("ICBHI2017/**/*.wav")))
    env_files = sorted(list(data_root.rglob("ESC-50/**/*.wav")) + list(data_root.rglob("UrbanSound8K/**/*.wav")))

    logger.info(f"Found {len(heart_files)} heart, {len(icbhi_files)} lung, {len(env_files)} env files")

    if args.demo:
        logger.info(f"DEMO MODE: generating {args.n_demo} examples per λ")
        heart_files = heart_files[:args.n_demo]

    total_files = 0
    for lam in lambdas:
        logger.info(f"=== Generating mixed dataset for λ = {lam} ===")
        out_dir = os.path.join(args.output_dir, f"lambda_{lam}")
        os.makedirs(out_dir, exist_ok=True)

        manifest_rows = []

        # Generate mixed heart+noise samples (label=1)
        for i, hf in enumerate(tqdm(heart_files, desc=f"λ={lam} hearts")):
            heart_wav = load_audio(hf)
            if heart_wav is None:
                continue

            # Deterministic noise selection per file
            file_rng = random.Random(args.seed + i)
            noise_wav = generate_noise(icbhi_files, env_files, file_rng)
            mixed = mix_rms(heart_wav, noise_wav, lam)

            out_name = f"{hf.stem}_mixed.wav"
            sf.write(os.path.join(out_dir, out_name), mixed, TARGET_SR)
            manifest_rows.append({
                "filename": out_name, "label": 1,
                "source_heart": hf.name, "lambda": lam
            })

        # Generate noise-only samples (label=0), balanced count
        n_noise = len(manifest_rows)
        for i in tqdm(range(n_noise), desc=f"λ={lam} noise"):
            noise_rng = random.Random(args.seed + len(heart_files) + i)
            noise_wav = generate_noise(icbhi_files, env_files, noise_rng)

            out_name = f"noise_{i:04d}.wav"
            sf.write(os.path.join(out_dir, out_name), noise_wav, TARGET_SR)
            manifest_rows.append({
                "filename": out_name, "label": 0,
                "source_heart": "none", "lambda": lam
            })

        # Save manifest
        pd.DataFrame(manifest_rows).to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
        total_files += len(manifest_rows)
        logger.info(f"  Saved {len(manifest_rows)} files to {out_dir}")

    logger.info(f"Done! Generated {total_files} total files across {len(lambdas)} lambda values.")


if __name__ == "__main__":
    main()
