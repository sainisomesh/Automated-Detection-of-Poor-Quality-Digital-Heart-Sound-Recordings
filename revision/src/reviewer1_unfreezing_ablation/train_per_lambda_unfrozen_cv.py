#!/usr/bin/env python3
"""
Per-lambda matched benchmark with a fully fine-tuned AST backbone.

Unfrozen-backbone version of the paper's matched per-lambda benchmark
(src/train_per_lambda_cv.py), where a fresh model is trained and tested at the
same noise level, giving an upper bound at each lambda. The published benchmark
uses ASTHeartQA(freeze_base=True); this script runs the same protocol with
ASTHeartQAUnfreeze, by default with end-to-end fine-tuning. It is a separate
script so the code that reproduces the published table stays unchanged.

One lambda per invocation, so the (lambda, fold) grid can run as separate
jobs. --lambda_val is required and sets both the training and test noise
level.

--unfreeze_mode is "full" (default, used for the reported benchmark) or
"frozen". The top-K schedule is only in train_unfreezing_ablation_cv.py.

Splits are patient-level: recordings are grouped by the patient id in the
filename (`13918_AV.wav` -> `13918`) and whole patients are assigned to folds.
Folds come from KFold(n_splits=--n_folds, shuffle=True, random_state=--seed)
over the sorted patient ids; with the defaults (5 folds, seed 42) this gives
the assignments in revision/fold_assignments/patient_folds_5fold.csv. The CSV
is a reference copy and is not read at runtime.

load_audio / mix_rms and PerLambdaDataset are copied from
src/train_per_lambda_cv.py; build_optimizer is the same two-learning-rate setup
as in train_unfreezing_ablation_cv.py.

Outputs (under --output_dir):
  raw_predictions/lambda_<L>/fold_<N>/predictions.csv
  per_lambda_unfrozen_progress.json         per-fold metrics, written each fold
  per_lambda_unfrozen_final_results.json    aggregated mean + CI
  convergence_log.csv                       per-epoch loss, time, peak GPU memory

Fine-tuning the 86M-parameter backbone needs a GPU; combine --limit_patients
with small --n_folds / --epochs for a local smoke test.

Usage:
    python train_per_lambda_unfrozen_cv.py --lambda_val 5.0 \\
        --data_dir ../../../dataset/ \\
        --output_dir ../../results/reviewer1_unfreezing_ablation/per_lambda_unfrozen/full/lambda_5.0/ \\
        --n_folds 5 --epochs 5 --backbone_lr 5e-5 --grad_checkpointing --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unfreeze_ast_qa import ASTHeartQAUnfreeze

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000
DURATION = 10
MAX_LENGTH = TARGET_SR * DURATION


def set_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Audio pipeline and dataset, copied from src/train_per_lambda_cv.py
def load_audio(path):
    """Load one file as a fixed-length, normalized 16 kHz mono waveform.

    Resamples to TARGET_SR, removes the DC offset, band-passes to 20-1000 Hz
    (2nd-order high-pass, 5th-order low-pass), and peak-normalizes. The result
    is trimmed or loop-padded (np.tile) to MAX_LENGTH samples (10 s).

    Returns None for unreadable, too-short (<100 samples) or effectively
    silent files, which callers replace with a zero-filled waveform.
    """
    try:
        wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception:
        return None
    if len(wav) < 100 or np.max(np.abs(wav)) < 1e-6:
        return None
    wav = wav - np.mean(wav)
    nyquist = TARGET_SR / 2
    sos_hp = butter(2, 20 / nyquist, btype="high", output="sos")
    wav = sosfilt(sos_hp, wav)
    sos_lp = butter(5, 1000 / nyquist, btype="low", output="sos")
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
    """Add noise to a heart sound at RMS-matched intensity lambda.

    The noise is rescaled to the heart sound's RMS before weighting by lambda,
    so lambda = 1 is 0 dB SNR and lambda = 10 is ten times the cardiac RMS.
    The mixture is rescaled if it would clip beyond +/-1.0.
    """
    if lam == 0:
        return heart
    rms_h = np.sqrt(np.mean(heart ** 2))
    rms_n = np.sqrt(np.mean(noise ** 2))
    if rms_h == 0 or rms_n == 0:
        return heart
    scale = rms_h / rms_n
    mixed = heart + lam * (noise * scale)
    m_peak = np.max(np.abs(mixed))
    if m_peak > 1.0:
        mixed = mixed / m_peak
    return mixed


class PerLambdaDataset(Dataset):
    """Class-balanced dataset at one fixed noise intensity.

    Indices [0, n) are heart recordings mixed at `lambda_val` (label 1) and
    indices [n, 2n) are noise-only clips (label 0).

    Args:
        is_train: If True, noise comes from the global RNG on every access
            (new mixtures each epoch). If False, noise comes from an
            index-seeded RNG, so each test index gets the same noise across
            lambdas, folds and models.
    """

    def __init__(self, heart_files, icbhi_files, env_files, lambda_val, processor, is_train=True):
        self.heart_files = heart_files
        self.icbhi_files = icbhi_files
        self.env_files = env_files
        self.lambda_val = lambda_val
        self.processor = processor
        self.is_train = is_train

        self.data = []
        for f in heart_files:
            self.data.append((f, 1, "mixed"))
        random.shuffle(self.data)

    def get_noise(self, idx=None):
        """Build one composite interference waveform: lung + 0.5 * environmental.

        A respiratory recording (ICBHI 2017) is summed with a half-weighted
        environmental clip (ESC-50 / UrbanSound8K) and peak-normalized. If
        `idx` is given, clips are drawn from random.Random(42 + idx).
        """
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
        return len(self.data) * 2  # balanced: heart+noise and noise-only

    def __getitem__(self, idx):
        if idx >= len(self.data):
            wav = self.get_noise(idx if not self.is_train else None)
            label = 0
            filename = "noise"
        else:
            path, label, _ = self.data[idx]
            h = load_audio(path)
            n = self.get_noise(idx if not self.is_train else None)
            wav = mix_rms(h, n, self.lambda_val) if (h is not None and n is not None) else None
            filename = str(path)

        if wav is None:
            wav = np.zeros(MAX_LENGTH)
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
            "filename": filename,
        }


# Optional Google Cloud Storage I/O for --gcs_bucket; not used in local runs.
def download_from_gcs(bucket_name, prefix, local_dir):
    """Mirror every object under gs://bucket/prefix into local_dir."""
    from google.cloud import storage
    logger.info(f"Downloading from gs://{bucket_name}/{prefix} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    count = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix):].lstrip("/")
        local = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        blob.download_to_filename(local)
        count += 1
        if count % 500 == 0:
            logger.info(f"  Downloaded {count} files...")
    logger.info(f"Downloaded {count} files")


def upload_to_gcs(bucket_name, local_dir, prefix):
    """Upload local_dir recursively to gs://bucket/prefix."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, local_dir)
            blob_path = f"{prefix}/{rel}"
            bucket.blob(blob_path).upload_from_filename(local_path)
    logger.info(f"Uploaded {local_dir} -> gs://{bucket_name}/{prefix}")


def build_optimizer(model, phase, head_lr, backbone_lr):
    """Build the AdamW optimizer for the given training phase.

    Frozen encoder: one group, the QA head at `head_lr`. Otherwise two groups:
    the head at `head_lr` and the unfrozen encoder parameters at the lower
    `backbone_lr`. Same as build_optimizer in train_unfreezing_ablation_cv.py.
    """
    if phase == "frozen":
        return torch.optim.AdamW(model.qa_classifier.parameters(), lr=head_lr)
    backbone_params = model.trainable_backbone_parameters()
    return torch.optim.AdamW([
        {"params": model.qa_classifier.parameters(), "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ])


def train_one_fold(unfreeze_mode, train_loader, lambda_val, fold, device, args, convergence_rows):
    """Train a fresh model on one fold at this job's fixed lambda.

    Appends one convergence row per epoch (mean loss, wall-clock time, peak
    GPU memory or None on CPU) and returns the trained model.
    """
    logger.info(f"  --> [Train] mode={unfreeze_mode} Lambda={lambda_val}, Fold={fold}")
    model = ASTHeartQAUnfreeze(unfreeze_mode=unfreeze_mode, topk_layers=0).to(device)
    if args.grad_checkpointing and unfreeze_mode == "full":
        # use_reentrant=False, as in train_unfreezing_ablation_cv.py (reentrant
        # checkpointing drops parameter gradients of a block whose input does
        # not require grad).
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logger.info(f"  trainable params = {model.num_trainable_parameters():,} / {model.num_total_parameters():,}")
    optimizer = build_optimizer(model, unfreeze_mode, args.head_lr, args.backbone_lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        epoch_loss = 0.0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"L={lambda_val} F{fold} E{epoch + 1}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)
            _, logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        elapsed = time.time() - t0
        mean_loss = epoch_loss / len(train_loader)
        peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                       if torch.cuda.is_available() else None)
        logger.info(
            f"  Lambda={lambda_val} | Fold {fold} | Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {mean_loss:.4f} | {elapsed:.1f}s"
            + (f" | peak_mem={peak_mem_mb:.0f}MB" if peak_mem_mb is not None else "")
        )
        convergence_rows.append({
            "lambda": lambda_val, "fold": fold, "epoch": epoch + 1,
            "loss": mean_loss, "epoch_seconds": elapsed, "peak_gpu_mem_mb": peak_mem_mb,
        })
    return model


def evaluate_fold(model, test_loader, device):
    """Score the held-out fold and return metrics plus raw per-clip outputs.

    The six reported metrics at a 0.5 threshold; AUROC and AUPRC fall back to
    0.5 / 0.0 if only one class is present (tiny debug runs). The caller
    writes the predictions to disk.
    """
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
    y_preds = (y_probs > 0.5).astype(int)
    try:
        auroc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, y_probs)
    except ValueError:
        auprc = 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
    metrics = {
        "auroc": round(float(auroc), 6),
        "auprc": round(float(auprc), 6),
        "f1": round(float(f1_score(y_true, y_preds, zero_division=0)), 6),
        "accuracy": round(float(accuracy_score(y_true, y_preds)), 6),
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 6),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 6),
    }
    return {
        "metrics": metrics,
        "predictions": {"filenames": filenames, "y_true": y_true.tolist(), "probs": y_probs.tolist()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-lambda matched benchmark with a fine-tuned AST backbone, patient-level CV")
    parser.add_argument("--lambda_val", type=float, required=True,
                         help="Noise intensity used for BOTH training and testing in this job "
                              "(one invocation per lambda)")
    parser.add_argument("--unfreeze_mode", type=str, default="full", choices=["frozen", "full"],
                         help="Backbone adaptation policy; defaults to end-to-end fine-tuning. "
                              "Top-K unfreezing is specific to the ablation and lives in "
                              "train_unfreezing_ablation_cv.py.")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Dataset root containing PhysioNet2022/, ICBHI2017/, ESC-50/ and "
                              "UrbanSound8K/. Required unless --gcs_bucket is set.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Local output dir. Defaults to a temp dir when --gcs_bucket is set.")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                         help="If set, download data from gs://<bucket>/<data_prefix> before "
                              "training and upload --output_dir to gs://<bucket>/<output_prefix> after.")
    parser.add_argument("--data_prefix", type=str, default="data/")
    parser.add_argument("--output_prefix", type=str, default=None,
                         help="Defaults to results/per_lambda_unfrozen/<mode_tag>/lambda_<val>/")
    parser.add_argument("--n_folds", type=int, default=5,
                         help="Number of patient-level CV folds; see the package README for the "
                              "fold count used for each reported result")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5,
                         help="Learning rate for the unfrozen encoder parameters; "
                              "lower than --head_lr")
    parser.add_argument("--grad_checkpointing", action="store_true",
                         help="Trade compute for memory by checkpointing encoder activations; "
                              "needed to fit end-to-end fine-tuning on a 24 GB GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                         help="cuda|mps|cpu (default: cuda when available, else cpu)")
    parser.add_argument("--limit_patients", type=int, default=0,
                         help="If >0, use only the first this-many patients. For smoke tests only: "
                              "it changes the patient set and therefore the fold membership.")
    args = parser.parse_args()

    if not args.gcs_bucket and not args.data_dir:
        parser.error("--data_dir is required unless --gcs_bucket is set")
    if not args.gcs_bucket and not args.output_dir:
        parser.error("--output_dir is required unless --gcs_bucket is set")

    set_seed(args.seed)
    mode_tag = args.unfreeze_mode

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | mode={mode_tag} | lambda={args.lambda_val}")

    output_prefix = args.output_prefix or f"results/per_lambda_unfrozen/{mode_tag}/lambda_{args.lambda_val}/"
    if args.gcs_bucket:
        work_dir = tempfile.mkdtemp(prefix=f"per_lambda_unfrozen_{mode_tag}_{args.lambda_val}_")
        data_dir = os.path.join(work_dir, "data")
        output_dir = os.path.join(work_dir, "output")
        download_from_gcs(args.gcs_bucket, args.data_prefix, data_dir)
    else:
        data_dir = args.data_dir
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data_root = Path(data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))
    icbhi_files = sorted(list(data_root.rglob("ICBHI2017/**/*.wav")))
    env_files = sorted(list(data_root.rglob("ESC-50/**/*.wav")) + list(data_root.rglob("UrbanSound8K/**/*.wav")))
    logger.info(f"Found {len(heart_files)} heart, {len(icbhi_files)} lung, {len(env_files)} environmental files")
    if not heart_files:
        raise ValueError(f"No heart audio files found in {data_dir}")

    # Group by patient ("<patient_id>_<valve>.wav") so all of a patient's
    # recordings share a fold.
    patient_map = {}
    for f in heart_files:
        pid = f.name.split("_")[0]
        patient_map.setdefault(pid, []).append(f)
    pids = sorted(list(patient_map.keys()))
    if args.limit_patients > 0:
        pids = pids[:args.limit_patients]
        logger.info(f"[smoke-test] limiting to {len(pids)} patients")
    logger.info(f"Found {len(pids)} unique patients")

    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_mel_bins=128, max_length=1024, sampling_rate=16000, f_min=0, f_max=8000
    )

    # Same fold construction as train_unfreezing_ablation_cv.py
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    convergence_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        logger.info(f"=== FOLD {fold + 1}/{args.n_folds} (lambda={args.lambda_val}) ===")
        tr_hearts = [f for i in train_idx for f in patient_map[pids[i]]]
        te_hearts = [f for i in test_idx for f in patient_map[pids[i]]]

        # Noise corpora are split 80/20 per fold so test noise clips are never
        # used in training.
        rng = random.Random(args.seed + fold)
        tr_icbhi = sorted(list(icbhi_files))
        tr_env = sorted(list(env_files))
        rng.shuffle(tr_icbhi)
        rng.shuffle(tr_env)
        te_icbhi = tr_icbhi[int(len(tr_icbhi) * 0.8):]
        tr_icbhi = tr_icbhi[:int(len(tr_icbhi) * 0.8)]
        te_env = tr_env[int(len(tr_env) * 0.8):]
        tr_env = tr_env[:int(len(tr_env) * 0.8)]

        logger.info(f"  Fold {fold + 1}: {len(tr_hearts)} train / {len(te_hearts)} test hearts")

        train_ds = PerLambdaDataset(tr_hearts, tr_icbhi, tr_env, args.lambda_val, processor)
        test_ds = PerLambdaDataset(te_hearts, te_icbhi, te_env, args.lambda_val, processor, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = train_one_fold(args.unfreeze_mode, train_loader, args.lambda_val, fold + 1, device,
                                args, convergence_rows)
        res = evaluate_fold(model, test_loader, device)
        fold_metrics.append(res["metrics"])
        logger.info(f"  Fold {fold + 1}: AUROC={res['metrics']['auroc']:.4f} F1={res['metrics']['f1']:.4f}")

        preds_dir = os.path.join(output_dir, "raw_predictions", f"lambda_{args.lambda_val}", f"fold_{fold + 1}")
        os.makedirs(preds_dir, exist_ok=True)
        pd.DataFrame(res["predictions"]).to_csv(os.path.join(preds_dir, "predictions.csv"), index=False)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pd.DataFrame(convergence_rows).to_csv(os.path.join(output_dir, "convergence_log.csv"), index=False)
        with open(os.path.join(output_dir, "per_lambda_unfrozen_progress.json"), "w") as f:
            json.dump({"unfreeze_mode": args.unfreeze_mode, "lambda_val": args.lambda_val,
                       "folds_done": fold + 1, "per_fold": fold_metrics}, f, indent=2)

        if args.gcs_bucket:
            # Upload after each fold so an interrupted job keeps finished folds
            upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))

    # Mean over folds and normal-approximation 95% CI half-width
    means = {m: float(np.mean([f[m] for f in fold_metrics])) for m in fold_metrics[0]}
    cis = {m: float(1.96 * np.std([f[m] for f in fold_metrics]) / np.sqrt(args.n_folds)) for m in fold_metrics[0]}
    final_results = {"mean": means, "ci": cis}

    with open(os.path.join(output_dir, "per_lambda_unfrozen_final_results.json"), "w") as f:
        json.dump({"unfreeze_mode": args.unfreeze_mode, "lambda_val": args.lambda_val,
                    "n_folds": args.n_folds, "results": final_results}, f, indent=2)
    logger.info(f"Done! mode={mode_tag} lambda={args.lambda_val} per-lambda-unfrozen CV results saved to {output_dir}")

    if args.gcs_bucket:
        upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))


if __name__ == "__main__":
    main()
