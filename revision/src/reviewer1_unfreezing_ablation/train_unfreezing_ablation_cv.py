#!/usr/bin/env python3
"""
Backbone unfreezing ablation: frozen vs. full fine-tuning vs. top-K unfreezing.

Runs patient-level cross-validation for one backbone mode per invocation and
evaluates the model over the full lambda sweep. One mode per process so each
condition can run as its own GPU job.

Modes:
  --unfreeze_mode frozen              Encoder frozen; only the binary QA head
                                      (~0.1M parameters) trains. This is the
                                      setup used for the paper's main results.
  --unfreeze_mode full                All 12 transformer blocks trained
                                      end-to-end, with a lower learning rate
                                      for the backbone than for the head.
  --unfreeze_mode topk --topk_layers {2,4}
                                      Two phases: --topk_warmup_epochs of
                                      head-only training, then the last K
                                      transformer blocks (plus the final
                                      layernorm) are unfrozen, the optimizer
                                      is rebuilt, and training continues for
                                      the remaining epochs.

Training noise is set with --noise_strategy (the three strategies in the
paper):
  variable_0_10 (default)  lambda ~ Uniform[0, 10] drawn per noisy sample.
  clean                    lambda = 0.0 for every sample (clean-only training).
  fixed_10                 lambda = 10.0 for every noisy sample.

The audio pipeline (load_audio / mix_rms / get_noise) and dataset construction
are copied from src/train_per_lambda_cv.py and src/train_three_strategies_cv.py,
so preprocessing and test-set noise match the other methods in the revision.
Only the model class differs.

Splits are patient-level: recordings are grouped by the patient id in the
filename (`13918_AV.wav` -> `13918`) and whole patients are assigned to folds.
Folds come from KFold(n_splits=--n_folds, shuffle=True, random_state=--seed)
over the sorted patient ids; with the defaults (5 folds, seed 42) this gives
the assignments in revision/fold_assignments/patient_folds_5fold.csv.

Outputs (under --output_dir):
  raw_predictions/lambda_<L>/fold_<N>/predictions.csv   filename, y_true, probs
  unfreezing_ablation_progress.json                     per-fold metrics, written
                                                        after every fold
  unfreezing_ablation_final_results.json                aggregated mean + CI
  unfreezing_ablation_metrics_<tag>.csv                 per-lambda mean metrics
  convergence_log.csv                                   per-epoch loss, wall-clock
                                                        time and peak GPU memory

The full and top-K modes backpropagate into the 86M-parameter backbone and
need a GPU; --grad_checkpointing reduces activation memory.
For a local smoke test, combine --limit_patients with small --n_folds /
--epochs values (and --topk_warmup_epochs 0 for the top-K mode).

Usage:
    python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \\
        --data_dir ../../../dataset/ \\
        --output_dir ../../results/reviewer1_unfreezing_ablation/frozen_5fold/ \\
        --n_folds 5 --epochs 5 --seed 42

    python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 \\
        --data_dir ../../../dataset/ \\
        --output_dir ../../results/reviewer1_unfreezing_ablation/topk4_5fold/ \\
        --n_folds 5 --epochs 5 --topk_warmup_epochs 2 --seed 42
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


# Audio pipeline, copied from src/train_per_lambda_cv.py
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


def get_noise(icbhi_files, env_files, idx=None):
    """Build one composite interference waveform: lung + 0.5 * environmental.

    A respiratory recording (ICBHI 2017) is summed with a half-weighted
    environmental clip (ESC-50 / UrbanSound8K) and peak-normalized.

    Args:
        idx: If given, clips are drawn from random.Random(42 + idx), so test
            sample `idx` gets the same noise across lambdas, folds and modes.
            If None, the global RNG is used (fresh noise every training epoch).
    """
    if idx is not None:
        rng = random.Random(42 + idx)
    else:
        rng = random
    l = load_audio(rng.choice(icbhi_files))
    e = load_audio(rng.choice(env_files))
    if l is not None and e is not None:
        combined = l + 0.5 * e
        peak = np.max(np.abs(combined))
        if peak > 0:
            combined = combined / peak
        return combined
    return np.zeros(MAX_LENGTH)


class StrategyTrainDataset(Dataset):
    """Class-balanced training set for any of the three noise strategies.

    Each heart recording gives two positives (label 1): the clean waveform and
    a noise-mixed copy with lambda set by `noise_strategy`. An equal number of
    noise-only negatives (label 0) is added. Waveforms are converted to AST
    log-mel features by `processor`.

    Matches ThreeStrategyDataset in src/train_three_strategies_cv.py. With
    'clean', the "mixed" sample is mixed at lambda = 0, so each recording
    appears twice and sample counts are the same for all strategies.

    Args:
        noise_strategy: 'variable_0_10' draws lambda ~ Uniform[0, 10] per
            noisy sample; 'fixed_10' uses lambda = 10.0; 'clean' uses
            lambda = 0.0.
    """

    def __init__(self, heart_files, icbhi_files, env_files, processor, noise_strategy="variable_0_10"):
        self.icbhi_files = icbhi_files
        self.env_files = env_files
        self.processor = processor
        self.noise_strategy = noise_strategy
        self.data = []
        for f in heart_files:
            self.data.append((f, 1, "clean"))
            self.data.append((f, 1, "mixed"))
        for _ in range(len(heart_files) * 2):
            self.data.append((None, 0, "noise"))
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, s_type = self.data[idx]
        if s_type == "noise":
            wav = get_noise(self.icbhi_files, self.env_files)
        elif s_type == "clean":
            wav = load_audio(path)
        else:  # mixed
            h = load_audio(path)
            n = get_noise(self.icbhi_files, self.env_files)
            if h is None or n is None:
                wav = None
            else:
                if self.noise_strategy == "variable_0_10":
                    lam = random.uniform(0, 10)
                elif self.noise_strategy == "fixed_10":
                    lam = 10.0
                elif self.noise_strategy == "clean":
                    lam = 0.0
                else:
                    raise ValueError(f"Unknown noise_strategy: {self.noise_strategy!r}")
                wav = mix_rms(h, n, lam)
        if wav is None:
            wav = np.zeros(MAX_LENGTH)
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }


class FixedLambdaEvalDataset(Dataset):
    """Balanced evaluation set at a single fixed test lambda.

    Indices [0, n) are the heart recordings mixed at `lambda_val` (label 1);
    indices [n, 2n) are noise-only clips (label 0). Noise uses the
    index-seeded RNG, so each index gets the same noise for every lambda and
    model.

    Construction matches PerLambdaDataset in src/train_per_lambda_cv.py.
    """

    def __init__(self, heart_files, icbhi_files, env_files, lambda_val, processor):
        self.icbhi_files = icbhi_files
        self.env_files = env_files
        self.lambda_val = lambda_val
        self.processor = processor
        self.data = [(f, 1) for f in heart_files]

    def __len__(self):
        return len(self.data) * 2  # balanced: heart(+noise) and noise-only

    def __getitem__(self, idx):
        if idx >= len(self.data):
            wav = get_noise(self.icbhi_files, self.env_files, idx=idx)
            label = 0
            filename = "noise"
        else:
            path, label = self.data[idx]
            h = load_audio(path)
            n = get_noise(self.icbhi_files, self.env_files, idx=idx)
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


def mode_tag(unfreeze_mode, topk_layers):
    """Short label for a condition, used in log lines and output filenames."""
    return unfreeze_mode if unfreeze_mode != "topk" else f"topk{topk_layers}"


def build_optimizer(model, phase, head_lr, backbone_lr):
    """Build the AdamW optimizer for the given training phase.

    Frozen encoder: one group, the QA head at `head_lr`. Otherwise two groups:
    the head at `head_lr` and the unfrozen encoder parameters at the lower
    `backbone_lr`, so the pretrained AudioSet weights change less than the
    new head.

    Group membership follows the current `requires_grad` flags, so rebuild
    the optimizer after changing the freeze policy (e.g. after top-K warmup).
    """
    if phase == "frozen":
        return torch.optim.AdamW(model.qa_classifier.parameters(), lr=head_lr)
    backbone_params = model.trainable_backbone_parameters()
    return torch.optim.AdamW([
        {"params": model.qa_classifier.parameters(), "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ])


def run_training_epochs(model, loader, optimizer, criterion, device, epochs, phase_label, fold,
                         convergence_rows, mode_label):
    """Train for `epochs` epochs, appending one convergence row per epoch.

    Each row has the mean loss, wall-clock time and peak GPU memory (None on
    CPU). `phase_label` separates the warmup and unfrozen phases of top-K.
    """
    for epoch in range(epochs):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        epoch_loss = 0.0
        t0 = time.time()
        for batch in tqdm(loader, desc=f"{phase_label} F{fold} E{epoch + 1}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)
            _, logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        elapsed = time.time() - t0
        mean_loss = epoch_loss / len(loader)
        peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                       if torch.cuda.is_available() else None)
        logger.info(
            f"  {phase_label} | Fold {fold} | Epoch {epoch + 1}/{epochs} | "
            f"Loss: {mean_loss:.4f} | {elapsed:.1f}s"
            + (f" | peak_mem={peak_mem_mb:.0f}MB" if peak_mem_mb is not None else "")
        )
        convergence_rows.append({
            "mode": mode_label, "fold": fold, "phase": phase_label, "epoch": epoch + 1,
            "loss": mean_loss, "epoch_seconds": elapsed, "peak_gpu_mem_mb": peak_mem_mb,
        })


def train_model(unfreeze_mode, topk_layers, train_loader, fold, device, args, convergence_rows):
    """Train one model for one fold under the requested unfreezing mode.

    The "frozen" and "full" modes are a single training phase. The "topk" mode
    runs two phases: `args.topk_warmup_epochs` of head-only training, then the
    top-K blocks are unfrozen, a new optimizer is built, and the remaining
    `args.epochs - args.topk_warmup_epochs` epochs are run.

    Returns the trained model.
    """
    tag = mode_tag(unfreeze_mode, topk_layers)
    logger.info(f"  --> [Train] mode={tag} fold={fold}")
    criterion = nn.BCEWithLogitsLoss()

    if unfreeze_mode != "topk":
        model = ASTHeartQAUnfreeze(unfreeze_mode=unfreeze_mode, topk_layers=0).to(device)
        if args.grad_checkpointing and unfreeze_mode == "full":
            # use_reentrant=False: reentrant checkpointing drops parameter
            # gradients of a block whose input does not require grad. That
            # can't happen in "full" mode but does in "topk"; the same setting
            # is used in both for consistency.
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info(f"  {tag}: trainable params = {model.num_trainable_parameters():,} "
                    f"/ {model.num_total_parameters():,}")
        optimizer = build_optimizer(model, unfreeze_mode, args.head_lr, args.backbone_lr)
        run_training_epochs(model, train_loader, optimizer, criterion, device,
                             args.epochs, tag, fold, convergence_rows, tag)
        return model

    # Phase 1: head-only warmup
    model = ASTHeartQAUnfreeze(unfreeze_mode="frozen").to(device)
    if args.topk_warmup_epochs > 0:
        optimizer = build_optimizer(model, "frozen", args.head_lr, args.backbone_lr)
        run_training_epochs(model, train_loader, optimizer, criterion, device,
                             args.topk_warmup_epochs, f"{tag}-warmup", fold, convergence_rows, tag)

    # Phase 2: unfreeze the top-K blocks, two-group optimizer
    model.set_unfreeze_mode("topk", topk_layers=topk_layers)
    if args.grad_checkpointing:
        # use_reentrant=False is needed here: the input to the first unfrozen
        # block has requires_grad=False (earlier blocks are frozen), and
        # reentrant checkpointing would then silently drop that block's
        # parameter gradients. See test_unfreeze_ast_qa.py check 9.
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logger.info(f"  {tag}: trainable params after unfreeze = {model.num_trainable_parameters():,} "
                f"/ {model.num_total_parameters():,}")
    remaining_epochs = args.epochs - args.topk_warmup_epochs
    if remaining_epochs <= 0:
        raise ValueError(
            f"--epochs ({args.epochs}) must be greater than --topk_warmup_epochs "
            f"({args.topk_warmup_epochs}) so the top-{topk_layers} layers actually get "
            f"unfrozen-phase training."
        )
    optimizer = build_optimizer(model, "topk", args.head_lr, args.backbone_lr)
    run_training_epochs(model, train_loader, optimizer, criterion, device,
                         remaining_epochs, f"{tag}-unfrozen", fold, convergence_rows, tag)
    return model


def compute_metrics(y_true, probs):
    """Compute the six reported metrics at the fixed 0.5 decision threshold.

    Returns AUROC, AUPRC, F1, accuracy, sensitivity and specificity. AUROC and
    AUPRC fall back to 0.5 / 0.0 if only one class is present (tiny debug runs).
    """
    preds = (probs > 0.5).astype(int)
    try:
        auroc = roc_auc_score(y_true, probs)
    except ValueError:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        auprc = 0.0
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": round(float(acc), 6), "auroc": round(float(auroc), 6),
        "auprc": round(float(auprc), 6), "f1": round(float(f1), 6),
        "sensitivity": round(float(sens), 6), "specificity": round(float(spec), 6),
    }


def evaluate_fold(model, hearts, icbhi, env, processor, device, batch_size, output_dir, fold, lambdas):
    """Evaluate one trained fold model at every lambda in the sweep.

    Writes raw per-clip predictions for each lambda to
    raw_predictions/lambda_<L>/fold_<N>/predictions.csv and returns the
    metrics keyed by lambda.
    """
    per_lambda_metrics = {}
    for lam in lambdas:
        ds = FixedLambdaEvalDataset(hearts, icbhi, env, lam, processor)
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
        metrics = compute_metrics(y_true, y_probs)
        per_lambda_metrics[lam] = metrics
        logger.info(f"  lambda={lam}: AUROC={metrics['auroc']:.4f} F1={metrics['f1']:.4f}")

        preds_dir = os.path.join(output_dir, "raw_predictions", f"lambda_{lam}", f"fold_{fold}")
        os.makedirs(preds_dir, exist_ok=True)
        pd.DataFrame({"filename": filenames, "y_true": y_true, "probs": y_probs}).to_csv(
            os.path.join(preds_dir, "predictions.csv"), index=False
        )
    return per_lambda_metrics


def main():
    parser = argparse.ArgumentParser(description="AST backbone unfreezing ablation, patient-level CV")
    parser.add_argument("--unfreeze_mode", type=str, required=True, choices=["frozen", "full", "topk"])
    parser.add_argument("--topk_layers", type=int, default=0, choices=[0, 2, 4],
                         help="Required (2 or 4) when --unfreeze_mode topk")
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
                         help="Defaults to results/unfreezing_ablation/<mode_tag>/")
    parser.add_argument("--n_folds", type=int, default=5,
                         help="Number of patient-level CV folds; see the package README for the "
                              "fold count used for each reported result")
    parser.add_argument("--epochs", type=int, default=5, help="Total epochs (includes warmup for topk)")
    parser.add_argument("--topk_warmup_epochs", type=int, default=2,
                         help="Head-only warmup epochs before unfreezing top-K layers (topk mode only)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5,
                         help="Learning rate for the unfrozen encoder parameters (full/topk "
                              "modes); lower than --head_lr")
    parser.add_argument("--grad_checkpointing", action="store_true",
                         help="Trade compute for memory by checkpointing encoder activations. "
                              "Only has an effect when part of the encoder is trainable; "
                              "needed to fit --unfreeze_mode full on a 24 GB GPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto-detect cuda>cpu)")
    parser.add_argument("--limit_patients", type=int, default=0,
                         help="If >0, use only the first this-many patients. For smoke tests only: "
                              "it changes the patient set and therefore the fold membership.")
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,1,5,10,25,50,75,100",
                         help="Comma-separated test-time lambda sweep (shorten it for smoke tests)")
    parser.add_argument("--noise_strategy", type=str, default="variable_0_10",
                         choices=["variable_0_10", "clean", "fixed_10"],
                         help="Noise strategy used to build the training set; see module docstring")
    args = parser.parse_args()

    if args.unfreeze_mode == "topk" and args.topk_layers not in (2, 4):
        parser.error("--topk_layers must be 2 or 4 when --unfreeze_mode topk")
    if args.unfreeze_mode == "topk" and args.epochs <= args.topk_warmup_epochs:
        # Fail early instead of after the warmup epochs; train_model() checks
        # again for programmatic calls.
        parser.error(
            f"--epochs ({args.epochs}) must be greater than --topk_warmup_epochs "
            f"({args.topk_warmup_epochs}) so the top-{args.topk_layers} layers actually "
            f"get unfrozen-phase training."
        )
    if not args.gcs_bucket and not args.data_dir:
        parser.error("--data_dir is required unless --gcs_bucket is set")
    if not args.gcs_bucket and not args.output_dir:
        parser.error("--output_dir is required unless --gcs_bucket is set")

    set_seed(args.seed)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    # Strategy is added to the tag only for non-default strategies
    tag = mode_tag(args.unfreeze_mode, args.topk_layers)
    if args.noise_strategy != "variable_0_10":
        tag = f"{tag}_{args.noise_strategy}"

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} | mode={tag}")

    output_prefix = args.output_prefix or f"results/unfreezing_ablation/{tag}/"
    if args.gcs_bucket:
        work_dir = tempfile.mkdtemp(prefix=f"unfreezing_ablation_{tag}_")
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

    # Fixed seed over sorted patient ids: same folds across modes, strategies
    # and reruns (matches patient_folds_5fold.csv at the defaults).
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_fold_metrics = []
    convergence_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        logger.info(f"=== FOLD {fold + 1}/{args.n_folds} (mode={tag}) ===")
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train_hearts = [f for p in train_pids for f in patient_map[p]]
        test_hearts = [f for p in test_pids for f in patient_map[p]]

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

        logger.info(f"  Fold {fold + 1}: {len(train_hearts)} train / {len(test_hearts)} test hearts")

        train_ds = StrategyTrainDataset(train_hearts, tr_icbhi, tr_env, processor,
                                         noise_strategy=args.noise_strategy)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

        model = train_model(args.unfreeze_mode, args.topk_layers, train_loader, fold + 1, device,
                             args, convergence_rows)
        fold_metrics = evaluate_fold(model, test_hearts, te_icbhi, te_env, processor, device,
                                      args.batch_size, output_dir, fold + 1, lambdas)
        all_fold_metrics.append(fold_metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pd.DataFrame(convergence_rows).to_csv(os.path.join(output_dir, "convergence_log.csv"), index=False)
        with open(os.path.join(output_dir, "unfreezing_ablation_progress.json"), "w") as f:
            json.dump({"unfreeze_mode": args.unfreeze_mode, "topk_layers": args.topk_layers,
                       "noise_strategy": args.noise_strategy,
                       "folds_done": fold + 1, "per_fold": all_fold_metrics}, f, indent=2)

        if args.gcs_bucket:
            # Upload after each fold so an interrupted job keeps finished folds
            upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))

    # Mean over folds and normal-approximation 95% CI half-width
    final_results = {}
    for lam in lambdas:
        metrics = [fold[lam] for fold in all_fold_metrics]
        means = {m: float(np.mean([f[m] for f in metrics])) for m in metrics[0]}
        cis = {m: float(1.96 * np.std([f[m] for f in metrics]) / np.sqrt(args.n_folds)) for m in metrics[0]}
        final_results[lam] = {"mean": means, "ci": cis}

    with open(os.path.join(output_dir, "unfreezing_ablation_final_results.json"), "w") as f:
        json.dump({"unfreeze_mode": args.unfreeze_mode, "topk_layers": args.topk_layers,
                    "noise_strategy": args.noise_strategy,
                    "n_folds": args.n_folds, "results": final_results}, f, indent=2)
    df = pd.DataFrame({lam: final_results[lam]["mean"] for lam in lambdas}).T
    df.index.name = "Lambda"
    df.to_csv(os.path.join(output_dir, f"unfreezing_ablation_metrics_{tag}.csv"))
    logger.info(f"Done! mode={tag} unfreezing-ablation CV results saved to {output_dir}")

    if args.gcs_bucket:
        upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))


if __name__ == "__main__":
    main()
