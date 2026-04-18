#!/usr/bin/env python3
"""
Three Training Strategies — 10-Fold Cross-Validation.

Compares three noise-aware training approaches for AST heart quality detection,
each evaluated across 10 noise test levels (λ):

  1. clean:      Trained on clean heart sounds only (no noise added)
  2. noise_0_10: Trained with random noise λ ~ Uniform[0, 10]
  3. noise_10:   Trained with fixed noise λ = 10

All three models are tested at λ_test ∈ {0, 0.25, 0.5, 1, 5, 10, 25, 50, 75, 100}
using 10-fold patient-level cross-validation.

Usage:
    python train_three_strategies_cv.py --data_dir dataset/ --output_dir results/three_strategies_cv/
"""

import argparse
import os
import sys
import random
import json
import numpy as np
import librosa
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.model_selection import KFold
from scipy.signal import butter, sosfilt

from models.ast_qa import ASTHeartQA
from transformers import ASTFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_SWEEP = [0.0, 0.25, 0.50, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]

# Audio processing constants
TARGET_SR = 16000
DURATION = 10
MAX_LENGTH = TARGET_SR * DURATION


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio(path):
    """Load and preprocess a single audio file."""
    try:
        wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception:
        return None
    if len(wav) < 100 or np.max(np.abs(wav)) < 1e-6:
        return None
    wav = wav - np.mean(wav)
    # Clinical bandpass: 20-1000 Hz
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
    """Mix heart sound with noise at a given λ using RMS-based scaling."""
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


class ThreeStrategyDataset(Dataset):
    """Dataset supporting three training strategies and evaluation."""

    def __init__(self, heart_files, icbhi_files, env_files, strategy, processor, lambda_val=None):
        """
        Args:
            strategy: 'clean', 'noise_0_10', 'noise_10', or 'eval'
            lambda_val: Used for 'eval' mode (fixed test lambda)
        """
        self.heart_files = heart_files
        self.icbhi_files = icbhi_files
        self.env_files = env_files
        self.strategy = strategy
        self.processor = processor
        self.lambda_val = lambda_val

        self.data = []
        if strategy == 'eval':
            n = len(heart_files)
            for f in heart_files:
                self.data.append((f, 1, 'eval'))
            for _ in range(n):
                self.data.append((None, 0, 'noise'))
        else:
            # Training: 1 clean + 1 noisy per heart file
            for f in heart_files:
                self.data.append((f, 1, 'clean'))
                self.data.append((f, 1, 'mixed'))
            # Balanced with noise-only
            for _ in range(len(heart_files) * 2):
                self.data.append((None, 0, 'noise'))

        random.shuffle(self.data)

    def get_noise(self, idx=None):
        """Generate structured noise (lung + 0.5 × environmental)."""
        if idx is not None:
            local_rng = random.Random(42 + idx)
            l = load_audio(local_rng.choice(self.icbhi_files))
            e = load_audio(local_rng.choice(self.env_files))
        else:
            l = load_audio(random.choice(self.icbhi_files))
            e = load_audio(random.choice(self.env_files))
        if l is not None and e is not None:
            combined = l + 0.5 * e
            peak = np.max(np.abs(combined))
            if peak > 0:
                combined = combined / peak
            return combined
        return np.zeros(MAX_LENGTH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, s_type = self.data[idx]

        if s_type == 'noise':
            wav = self.get_noise(idx if self.strategy == 'eval' else None)
        elif s_type == 'clean':
            wav = load_audio(path)
        elif s_type == 'mixed':
            h = load_audio(path)
            n = self.get_noise()
            if self.strategy == 'noise_0_10':
                lam = random.uniform(0, 10)
            elif self.strategy == 'noise_10':
                lam = 10.0
            else:
                lam = 0.0
            wav = mix_rms(h, n, lam)
        elif s_type == 'eval':
            h = load_audio(path)
            n = self.get_noise(idx)
            wav = mix_rms(h, n, self.lambda_val)

        if wav is None:
            wav = np.zeros(MAX_LENGTH)
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
            "filename": str(path) if path else "noise"
        }


def train_model(train_loader, strategy_name, fold, device, epochs=5, lr=1e-4):
    """Train a fresh AST-QA model for one strategy/fold."""
    logger.info(f"  --> [Train] Strategy={strategy_name}, Fold={fold}")
    model = ASTHeartQA(freeze_base=True).to(device)
    optimizer = torch.optim.AdamW(model.qa_classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"F{fold} {strategy_name} E{epoch+1}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)
            _, logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"  {strategy_name} | Fold {fold} | Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
    return model


def evaluate_model(model, hearts, icbhi, env, processor, device, batch_size=16):
    """Evaluate a trained model across all λ test values."""
    results = {}
    for lam in LAMBDA_SWEEP:
        ds = ThreeStrategyDataset(hearts, icbhi, env, 'eval', processor, lambda_val=lam)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        model.eval()
        y_true, y_probs, filenames = [], [], []
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input_values"].to(device)
                _, logits = model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                y_true.extend(batch["labels"].numpy())
                y_probs.extend(probs)
                filenames.extend(batch["filename"])

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_preds = (y_probs > 0.5).astype(int)

        auroc = roc_auc_score(y_true, y_probs)
        auprc = average_precision_score(y_true, y_probs)
        f1 = f1_score(y_true, y_preds, zero_division=0)
        acc = accuracy_score(y_true, y_preds)
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        results[lam] = {
            "metrics": {
                "auroc": auroc, "auprc": auprc, "f1": f1, "accuracy": acc,
                "sensitivity": sens, "specificity": spec
            },
            "predictions": {
                "filenames": filenames,
                "y_true": y_true.tolist(),
                "probs": y_probs.tolist()
            }
        }
        logger.info(f"  λ={lam}: AUROC={auroc:.4f}, F1={f1:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Three Training Strategies — 10-Fold CV")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing dataset/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover audio files
    data_root = Path(args.data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))
    icbhi_files = sorted(list(data_root.rglob("ICBHI2017/**/*.wav")))
    env_files = sorted(list(data_root.rglob("ESC-50/**/*.wav")) + list(data_root.rglob("UrbanSound8K/**/*.wav")))

    logger.info(f"Found {len(heart_files)} heart, {len(icbhi_files)} lung, {len(env_files)} environmental files")

    # Patient-level grouping
    patient_map = {}
    for f in heart_files:
        pid = f.name.split('_')[0]
        patient_map.setdefault(pid, []).append(f)
    pids = sorted(list(patient_map.keys()))

    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_mel_bins=128, max_length=1024, sampling_rate=16000, f_min=0, f_max=8000
    )

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    strategies = ['clean', 'noise_0_10', 'noise_10']
    all_fold_results = {s: [] for s in strategies}

    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        logger.info(f"=== FOLD {fold+1}/{args.n_folds} ===")
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]

        train_hearts = [f for p in train_pids for f in patient_map[p]]
        test_hearts = [f for p in test_pids for f in patient_map[p]]

        # Split noise deterministically per fold
        rng = random.Random(args.seed + fold)
        tr_icbhi = sorted(list(icbhi_files))
        tr_env = sorted(list(env_files))
        rng.shuffle(tr_icbhi)
        rng.shuffle(tr_env)

        te_icbhi = tr_icbhi[int(len(tr_icbhi)*0.8):]
        tr_icbhi = tr_icbhi[:int(len(tr_icbhi)*0.8)]
        te_env = tr_env[int(len(tr_env)*0.8):]
        tr_env = tr_env[:int(len(tr_env)*0.8)]

        for s_name in strategies:
            logger.info(f"  Strategy: {s_name}")
            ds = ThreeStrategyDataset(train_hearts, tr_icbhi, tr_env, s_name, processor)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            model = train_model(loader, s_name, fold+1, DEVICE, epochs=args.epochs)

            fold_metrics_data = evaluate_model(model, test_hearts, te_icbhi, te_env, processor, DEVICE)

            fold_lam_metrics = {}
            for lam, data in fold_metrics_data.items():
                fold_lam_metrics[lam] = data["metrics"]
                # Save raw predictions
                preds_dir = os.path.join(args.output_dir, "raw_predictions", f"fold_{fold+1}", s_name)
                os.makedirs(preds_dir, exist_ok=True)
                pd.DataFrame(data["predictions"]).to_csv(
                    os.path.join(preds_dir, f"lambda_{lam}.csv"), index=False
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_fold_results[s_name].append(fold_lam_metrics)

            # Save progress
            with open(os.path.join(args.output_dir, "partial_results.json"), 'w') as f:
                json.dump(all_fold_results, f, indent=2)

    # Final aggregation
    final_results = {}
    for s_name in strategies:
        final_results[s_name] = {}
        for lam in LAMBDA_SWEEP:
            metrics = [fold[lam] for fold in all_fold_results[s_name]]
            means = {m: np.mean([f[m] for f in metrics]) for m in metrics[0]}
            stds = {m: np.std([f[m] for f in metrics]) for m in metrics[0]}
            cis = {m: 1.96 * stds[m] / np.sqrt(args.n_folds) for m in metrics[0]}
            final_results[s_name][lam] = {"mean": means, "ci": cis}

    with open(os.path.join(args.output_dir, "final_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
