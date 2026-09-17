#!/usr/bin/env python3
"""
Per-lambda matched benchmark with a fully fine-tuned AST backbone.

This is the unfrozen-backbone counterpart of the paper's matched per-lambda
benchmark (src/train_per_lambda_cv.py), in which a fresh model is trained and
tested at the *same* noise intensity, giving the upper bound on achievable
performance at each lambda. The published benchmark trains
ASTHeartQA(freeze_base=True); here the same protocol is run with
ASTHeartQAUnfreeze, by default in end-to-end fine-tuning mode. It is a separate
script rather than a flag on the original so that the code path reproducing the
published table is left untouched.

One lambda per invocation (rather than an internal sweep), so that the
(lambda, fold) grid can be spread over independent jobs instead of being
trained serially in a single process. --lambda_val is therefore required, and
it sets both the training and the test noise intensity -- that is what makes
the benchmark "matched".

--unfreeze_mode selects the backbone adaptation policy: "full" (the default,
used for the reported benchmark) or "frozen". The progressive top-K schedule
is specific to the ablation and is implemented in
train_unfreezing_ablation_cv.py rather than here.

Cross-validation is patient-level: recordings are grouped by the patient id
parsed from the filename (`13918_AV.wav` -> `13918`) and whole patients are
assigned to folds, so no patient appears in both the train and test side of a
fold. The split is recomputed deterministically from
KFold(n_splits=--n_folds, shuffle=True, random_state=--seed) over the sorted
patient ids, reproducing the frozen assignments stored in
revision/fold_assignments/patient_folds_{3,5}fold.csv for the matching fold
count; the CSVs are documentation and audit references, not runtime inputs.

The audio pipeline (load_audio / mix_rms) and PerLambdaDataset are reproduced
unchanged from src/train_per_lambda_cv.py so that all compared methods see
identical inputs; build_optimizer matches
train_unfreezing_ablation_cv.py's two-learning-rate setup.

Outputs (under --output_dir):
  raw_predictions/lambda_<L>/fold_<N>/predictions.csv
  per_lambda_unfrozen_progress.json         per-fold metrics, written each fold
  per_lambda_unfrozen_final_results.json    aggregated mean + CI
  convergence_log.csv                       per-epoch loss, time, peak GPU memory

Fine-tuning the 86M-parameter backbone is intended to run on a GPU; combine
--limit_patients with small --n_folds / --epochs for a local smoke test.

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


# --- Audio pipeline and dataset, reproduced unchanged from
# src/train_per_lambda_cv.py. Every method compared in the revision shares this
# pipeline, so it must stay identical across scripts. ---
def load_audio(path):
    """Load one file as a fixed-length, normalized 16 kHz mono waveform.

    Resamples to TARGET_SR, removes the DC offset, band-passes to the
    20-1000 Hz range where heart-sound energy lives (2nd-order high-pass,
    5th-order low-pass), and peak-normalizes. The result is trimmed or
    loop-padded (np.tile) to exactly MAX_LENGTH samples (10 s).

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

    The noise is rescaled so its RMS equals the heart sound's before being
    weighted by lambda, so lambda is an energy ratio rather than an absolute
    gain: lambda = 1 gives 0 dB SNR, lambda = 10 gives ten times the cardiac
    RMS energy. The mixture is rescaled if it would otherwise clip beyond
    +/-1.0.
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
    indices [n, 2n) are noise-only clips (label 0), giving a 50/50 balance.

    Args:
        is_train: Training sets draw fresh noise from the global RNG on every
            access, so each epoch sees new mixtures. Evaluation sets
            (`is_train=False`) derive the noise from an index-seeded RNG
            instead, so the same evaluation index always receives the same
            interference across lambdas, folds and models.
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
        environmental clip (ESC-50 / UrbanSound8K) and peak-normalized,
        modelling clinical auscultation in which internal physiological
        interference dominates ambient noise. When `idx` is given the two
        source clips are drawn from an RNG seeded by it, making the noise
        reproducible per sample index.
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
# --- end of reproduced audio pipeline and dataset ---


# --- Google Cloud Storage helpers, used only when this script runs as a
# managed cloud training job (--gcs_bucket). They are a no-op for local runs,
# and `google-cloud-storage` is imported lazily so it is not a hard dependency. ---
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
# --- end of cloud-storage helpers ---


def build_optimizer(model, phase, head_lr, backbone_lr):
    """Build the AdamW optimizer for the given training phase.

    With a frozen encoder there is only one parameter group, the QA head at
    `head_lr`. Once any part of the backbone is trainable, the optimizer uses
    two groups: the head stays at `head_lr` while the unfrozen pretrained
    encoder parameters are trained at the lower `backbone_lr`, so fine-tuning
    perturbs the AudioSet representation far less than it adapts the
    randomly-initialized head. Identical to the function of the same name in
    train_unfreezing_ablation_cv.py.
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

    Appends one convergence row per epoch (mean loss, wall-clock duration and
    peak GPU memory, the latter None on CPU) and returns the trained model.
    """
    logger.info(f"  --> [Train] mode={unfreeze_mode} Lambda={lambda_val}, Fold={fold}")
    model = ASTHeartQAUnfreeze(unfreeze_mode=unfreeze_mode, topk_layers=0).to(device)
    if args.grad_checkpointing and unfreeze_mode == "full":
        # use_reentrant=False is required, not a stylistic preference: the
        # default reentrant torch.utils.checkpoint implementation drops the
        # gradients of a checkpointed block's own trainable parameters whenever
        # the activation entering that block does not require grad, which is
        # the case for the first unfrozen block of any partially frozen
        # encoder. The same keyword is used at the corresponding call site in
        # train_unfreezing_ablation_cv.py.
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

    Metrics are the six reported quantities at the fixed 0.5 decision
    threshold; AUROC and AUPRC fall back to 0.5 / 0.0 if the label set is
    degenerate (single class), which can only happen on very small debug runs.
    The returned predictions are written to disk so every metric can be
    recomputed from the stored probabilities.
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
                         help="Learning rate for the unfrozen encoder parameters; deliberately "
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

    # Group recordings by patient: filenames are "<patient_id>_<valve>.wav",
    # so the id is the part before the first underscore. Folds are drawn over
    # patient ids, never over individual files, which is what keeps all four
    # auscultation-point recordings of a patient on the same side of a split.
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

    # Folds are over the sorted patient id list with a fixed seed, so the
    # assignment is identical across lambdas, modes and reruns, and matches
    # the checked-in fold assignment CSVs for the same n_folds and seed. This
    # construction is shared with train_unfreezing_ablation_cv.py.
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    convergence_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        logger.info(f"=== FOLD {fold + 1}/{args.n_folds} (lambda={args.lambda_val}) ===")
        tr_hearts = [f for i in train_idx for f in patient_map[pids[i]]]
        te_hearts = [f for i in test_idx for f in patient_map[pids[i]]]

        # The interference corpora are split 80/20 per fold as well, so the
        # lung and environmental clips heard at test time were never used to
        # build training mixtures. The shuffle is seeded per fold for
        # reproducibility, and the two slices are disjoint by construction.
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
            # Upload after every fold rather than only at the end, so that a
            # job interrupted mid-run still leaves the completed folds'
            # predictions and logs in the bucket.
            upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))

    # Aggregate across folds: mean of the per-fold metrics and a half-width
    # confidence interval from the across-fold spread.
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
