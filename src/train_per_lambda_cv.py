#!/usr/bin/env python3
"""
Per-Lambda Cross-Validation for AST Heart Quality Model.

For each noise level λ, trains a fresh AST-QA model and evaluates using
10-fold patient-level cross-validation. This measures how well a model
can distinguish heart sounds from noise when BOTH trained and tested at
the same noise contamination level.

λ values: 0, 0.25, 0.5, 1, 5, 10, 25, 50, 75, 100

Noise mixing formula (RMS-based):
    mixed = heart + λ × (noise × (rms_heart / rms_noise))
    where noise = lung_sound + 0.5 × environmental_sound

Usage:
    python train_per_lambda_cv.py --data_dir dataset/ --output_dir results/per_lambda_cv/
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
    # DC offset removal
    wav = wav - np.mean(wav)
    # Clinical bandpass: 20-1000 Hz
    nyquist = TARGET_SR / 2
    sos_hp = butter(2, 20 / nyquist, btype='high', output='sos')
    wav = sosfilt(sos_hp, wav)
    sos_lp = butter(5, 1000 / nyquist, btype='low', output='sos')
    wav = sosfilt(sos_lp, wav)
    # Peak normalize
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak
    # Loop/crop to fixed length
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


class PerLambdaDataset(Dataset):
    """Dataset that mixes heart sounds with noise at a fixed λ."""

    def __init__(self, heart_files, icbhi_files, env_files, lambda_val, processor, is_train=True):
        self.heart_files = heart_files
        self.icbhi_files = icbhi_files
        self.env_files = env_files
        self.lambda_val = lambda_val
        self.processor = processor
        self.is_train = is_train

        self.data = []
        for f in heart_files:
            self.data.append((f, 1, 'mixed'))
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
        return len(self.data) * 2  # Balanced: heart+noise and noise-only

    def __getitem__(self, idx):
        if idx >= len(self.data):
            # Noise-only sample (label 0)
            wav = self.get_noise(idx if not self.is_train else None)
            label = 0
            filename = "noise"
        else:
            path, label, _ = self.data[idx]
            h = load_audio(path)
            n = self.get_noise(idx if not self.is_train else None)
            wav = mix_rms(h, n, self.lambda_val)
            filename = str(path)

        if wav is None:
            wav = np.zeros(MAX_LENGTH)
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
            "filename": filename
        }


def train_one_fold(train_loader, lambda_val, fold, device, epochs=5):
    """Train a fresh AST-QA model on one fold."""
    logger.info(f"  --> [Train] Initializing ASTHeartQA for Lambda={lambda_val}, Fold={fold}")
    model = ASTHeartQA(freeze_base=True).to(device)
    optimizer = torch.optim.AdamW(model.qa_classifier.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"L={lambda_val} F{fold} E{epoch+1}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)
            _, logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"  Lambda={lambda_val} | Fold {fold} | Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
    return model


def evaluate_fold(model, test_loader, device):
    """Evaluate model on one fold, return metrics + raw predictions."""
    model.eval()
    y_true, y_probs, filenames = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_values"].to(device)
            _, logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_true.extend(batch["labels"].numpy())
            y_probs.extend(probs)
            filenames.extend(batch["filename"])

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    return {
        "metrics": {
            "auroc": roc_auc_score(y_true, y_probs),
            "f1": f1_score(y_true, (y_probs > 0.5).astype(int), zero_division=0)
        },
        "predictions": {
            "filenames": filenames,
            "y_true": y_true.tolist(),
            "probs": y_probs.tolist()
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Per-Lambda 10-Fold CV for AST Heart Quality")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing dataset/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,1,5,10,25,50,75,100",
                        help="Comma-separated lambda values")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    # Discover audio files
    data_root = Path(args.data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))
    icbhi_files = sorted(list(data_root.rglob("ICBHI2017/**/*.wav")))
    env_files = sorted(list(data_root.rglob("ESC-50/**/*.wav")) + list(data_root.rglob("UrbanSound8K/**/*.wav")))

    logger.info(f"Found {len(heart_files)} heart, {len(icbhi_files)} lung, {len(env_files)} environmental files")
    if not heart_files:
        raise ValueError(f"No heart audio files found in {args.data_dir}")

    # Patient-level grouping for cross-validation
    patient_map = {}
    for f in heart_files:
        pid = f.name.split('_')[0]
        patient_map.setdefault(pid, []).append(f)
    pids = sorted(list(patient_map.keys()))
    logger.info(f"Found {len(pids)} unique patients")

    # AST feature extractor
    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_mel_bins=128, max_length=1024, sampling_rate=16000, f_min=0, f_max=8000
    )

    final_results = {}

    for l_val in lambdas:
        logger.info(f"=== Starting 10-Fold CV for Lambda={l_val} ===")
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        lambda_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
            tr_hearts = [f for i in train_idx for f in patient_map[pids[i]]]
            te_hearts = [f for i in test_idx for f in patient_map[pids[i]]]

            # Split noise sources deterministically per fold
            rng = random.Random(args.seed + fold)
            tr_icbhi = sorted(list(icbhi_files))
            tr_env = sorted(list(env_files))
            rng.shuffle(tr_icbhi)
            rng.shuffle(tr_env)

            te_icbhi = tr_icbhi[int(len(tr_icbhi)*0.8):]
            tr_icbhi = tr_icbhi[:int(len(tr_icbhi)*0.8)]
            te_env = tr_env[int(len(tr_env)*0.8):]
            tr_env = tr_env[:int(len(tr_env)*0.8)]

            logger.info(f"  Fold {fold+1}: {len(tr_hearts)} train / {len(te_hearts)} test hearts")

            train_ds = PerLambdaDataset(tr_hearts, tr_icbhi, tr_env, l_val, processor)
            test_ds = PerLambdaDataset(te_hearts, te_icbhi, te_env, l_val, processor, is_train=False)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            model = train_one_fold(train_loader, l_val, fold+1, DEVICE, epochs=args.epochs)
            res = evaluate_fold(model, test_loader, DEVICE)
            lambda_metrics.append(res["metrics"])
            logger.info(f"  Fold {fold+1}: AUROC={res['metrics']['auroc']:.4f}")

            # Save raw predictions
            preds_dir = os.path.join(args.output_dir, "raw_predictions", f"lambda_{l_val}", f"fold_{fold+1}")
            os.makedirs(preds_dir, exist_ok=True)
            pd.DataFrame(res["predictions"]).to_csv(os.path.join(preds_dir, "predictions.csv"), index=False)

            # Memory cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate
        avg_auroc = np.mean([m['auroc'] for m in lambda_metrics])
        ci_auroc = 1.96 * np.std([m['auroc'] for m in lambda_metrics]) / np.sqrt(args.n_folds)
        avg_f1 = np.mean([m['f1'] for m in lambda_metrics])
        ci_f1 = 1.96 * np.std([m['f1'] for m in lambda_metrics]) / np.sqrt(args.n_folds)

        final_results[l_val] = {
            "auroc": f"{avg_auroc:.4f} \u00b1 {ci_auroc:.4f}",
            "f1": f"{avg_f1:.4f} \u00b1 {ci_f1:.4f}"
        }

        # Save progress after each lambda
        with open(os.path.join(args.output_dir, "per_lambda_progress.json"), 'w') as f:
            json.dump(final_results, f, indent=2)

    # Final CSV table
    df = pd.DataFrame.from_dict(final_results, orient='index')
    df.index.name = "Lambda"
    df.to_csv(os.path.join(args.output_dir, "per_lambda_metrics.csv"))
    logger.info("Done! Per-lambda CV results saved.")


if __name__ == "__main__":
    main()
