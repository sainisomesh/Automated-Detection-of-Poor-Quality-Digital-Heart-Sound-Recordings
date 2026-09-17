#!/usr/bin/env python3
"""
Frozen vs. Full vs. Top-K Backbone Unfreezing Ablation (Reviewer #1, Comment 2).

Trains ONE unfreeze-mode variant per invocation (mirrors
../../reviewer7_backbone_swap/src/train_backbone_swap_cv.py's one-backbone-
per-invocation design -- lets each mode run as its own parallel Vertex AI
job once GPU quota allows):

  --unfreeze_mode frozen              Mode A: fully frozen encoder (baseline
                                       reproduction -- trains only the 0.11M
                                       qa_classifier head, same as the
                                       published Table 2 / Figure 3 models).
  --unfreeze_mode full                Mode B: all 12 ViT layers unfrozen,
                                       trained end-to-end at a lower backbone
                                       LR than the head.
  --unfreeze_mode topk --topk_layers {2,4}
                                       Mode C: head-only warmup for
                                       --topk_warmup_epochs, then unfreeze
                                       the top K transformer layers (closest
                                       to the classifier) and continue
                                       training.

All variants train on the Variable-Noise (U[0,10]) strategy ONLY -- this is
the reviewer comment's own scope (CLAUDE.md Sec 9.1 Comment 2 explicitly
says "using the primary Variable-Noise strategy", not all three Figure 3
strategies) -- and are evaluated across all 10 lambda test levels.

Audio pipeline (load_audio / mix_rms / get_noise) and dataset construction
(VariableNoiseTrainDataset / FixedLambdaEvalDataset) are copied verbatim
from reproducibility/src/train_per_lambda_cv.py's PerLambdaDataset and
train_three_strategies_cv.py's ThreeStrategyDataset 'noise_0_10' branch --
same pipeline every other method in this revision (AST-QA baseline, Tang,
Giordano, the backbone-swap experiment) is evaluated on. Only the model
class changes (ASTHeartQAUnfreeze instead of ASTHeartQA / BackboneQAHead).

Fold count: 3-fold (KFold(n_splits=3, shuffle=True, random_state=42), same
construction as ../../fold_assignments/patient_folds_3fold.csv) for the
ORIGINAL 4-condition ablation run (2026-09-13/14). This was a SECOND,
explicit deviation on top of the already-documented 5-fold-instead-
of-10-fold policy in CLAUDE.md Sec 9.4: 5-fold is PAPER_REVISIONS' own
default for new experiments, but this specific experiment was deliberately
dropped to 3-fold (decided 2026-09-13, see ../README.md "Why 3-fold, not
5-fold") to bound wall-clock/cost given how much heavier full/topk-mode
backward passes are than the frozen baseline's head-only training. This
must be stated explicitly in any writeup that cites these numbers -- never
presented next to the published 10-fold Table 2/Figure 3 results, or even
next to this revision's OTHER 5-fold PAPER_REVISIONS results, without
flagging the further fold-count drop.

--noise_strategy (added 2026-09-15): after seeing --unfreeze_mode full beat
frozen at every lambda (see ../README.md "Results"), the PI decided to
adopt the unfrozen backbone as the new candidate deployed model, which
means it needs to be trained under the OTHER two Figure-3 strategies too --
clean-only and fixed-noise (lambda_train=10) -- not just variable-noise
U[0,10]. Rather than duplicating the whole CV harness, this flag repurposes
the existing training-set construction (which already knows how to build a
'mixed' clean+heart sample) to fix the sampled lambda instead of drawing it
from U[0,10]:
  --noise_strategy variable_0_10 (default)  lambda ~ Uniform[0, 10] per
                                             mixed sample -- ORIGINAL,
                                             UNCHANGED behavior. Omitting
                                             --noise_strategy entirely
                                             reproduces the existing 3-fold
                                             full/frozen/topk2/topk4 results
                                             byte-for-byte -- nothing about
                                             the default path changed.
  --noise_strategy clean                    lambda = 0.0 for every mixed
                                             sample (equivalent to
                                             reproducibility/train_three_
                                             strategies_cv.py's 'clean').
  --noise_strategy fixed_10                 lambda = 10.0 for every mixed
                                             sample (equivalent to that
                                             script's 'noise_10').
These new strategies are intended to run at 5-fold (PAPER_REVISIONS'
default, per CLAUDE.md Sec 9.4), NOT 3-fold -- the 3-fold drop above was
scoped to the original 4-condition ablation's cost concerns, not to this
new direction. See ../README.md "2026-09-15: PI-directed pivot to unfrozen
backbone" for why the existing 3-fold full/variable_0_10 result is being
rerun at 5-fold rather than reused as-is.

Output layout matches the backbone-swap experiment for direct comparability:
  raw_predictions/lambda_X/fold_Y/predictions.csv  (filename, y_true, probs)
  unfreezing_ablation_progress.json, unfreezing_ablation_metrics_<tag>.csv,
  convergence_log.csv (per-epoch loss, wall-clock time, peak GPU memory --
  CLAUDE.md Sec 9.1 Comment 2 explicitly asks for convergence speed and GPU
  memory footprint to be reported alongside test performance).

GPU-heavy for --unfreeze_mode full/topk -- meant to run as a Vertex AI job.
DO NOT submit to Vertex without explicit confirmation. Use --limit_patients /
--n_folds 2 / --epochs 1 (with --topk_warmup_epochs 0 for topk) for local
smoke-testing.

Usage:
    python train_unfreezing_ablation_cv.py --unfreeze_mode frozen \\
        --data_dir ../../../data_processed/ --output_dir ../results/frozen/ \\
        --n_folds 3 --seed 42

    python train_unfreezing_ablation_cv.py --unfreeze_mode topk --topk_layers 4 \\
        --data_dir ../../../data_processed/ --output_dir ../results/topk4/ \\
        --n_folds 3 --epochs 5 --topk_warmup_epochs 2 --seed 42
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- verbatim block, copied from reproducibility/src/train_per_lambda_cv.py ---
# (same audio pipeline AST-QA, Tang, Giordano, and the backbone-swap
# experiment are all evaluated on -- must stay byte-identical)
def load_audio(path):
    """Load and preprocess a single audio file."""
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
    """Mix heart sound with noise at a given lambda using RMS-based scaling."""
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
    """Structured noise: lung + 0.5*env, peak-normalized -- identical
    formula to train_per_lambda_cv.py's PerLambdaDataset.get_noise."""
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
# --- end verbatim block ---


class StrategyTrainDataset(Dataset):
    """Training set supporting all three Figure-3 noise strategies -- identical
    construction to ThreeStrategyDataset in train_three_strategies_cv.py (and
    to the backbone-swap experiment's dataset of the same name), plus the
    ASTFeatureExtractor step AST needs (unlike the backbone-swap variant,
    which hands raw waveforms to each backbone's own front-end).

    Renamed from VariableNoiseTrainDataset (2026-09-15) when --noise_strategy
    was added -- this class now also serves 'clean' and 'fixed_10', so a
    name implying it only does variable-noise would be misleading. Passing
    noise_strategy='variable_0_10' (the default) reproduces the original
    class's exact behavior."""

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
    """Fixed test-lambda eval set -- identical construction to
    train_per_lambda_cv.py's PerLambdaDataset / the backbone-swap
    experiment's dataset of the same name, plus the ASTFeatureExtractor step."""

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


# --- GCS helpers, copied verbatim from ../../reviewer7_backbone_swap/src/train_backbone_swap_cv.py
# (itself copied from phase6_paper_quality/src/train_quality_vertexai.py --
# established convention for this repo's Vertex AI jobs) ---
def download_from_gcs(bucket_name, prefix, local_dir):
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
# --- end GCS helpers ---


def mode_tag(unfreeze_mode, topk_layers):
    return unfreeze_mode if unfreeze_mode != "topk" else f"topk{topk_layers}"


def build_optimizer(model, phase, head_lr, backbone_lr):
    """phase='frozen' -> head params only. phase in {'full','topk'} -> two
    param groups (head at head_lr, currently-trainable backbone params at
    the lower backbone_lr, per CLAUDE.md Sec 9.1 Comment 2's "lower learning
    rate" instruction for Mode B)."""
    if phase == "frozen":
        return torch.optim.AdamW(model.qa_classifier.parameters(), lr=head_lr)
    backbone_params = model.trainable_backbone_parameters()
    return torch.optim.AdamW([
        {"params": model.qa_classifier.parameters(), "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ])


def run_training_epochs(model, loader, optimizer, criterion, device, epochs, phase_label, fold,
                         convergence_rows, mode_label):
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
    tag = mode_tag(unfreeze_mode, topk_layers)
    logger.info(f"  --> [Train] mode={tag} fold={fold}")
    criterion = nn.BCEWithLogitsLoss()

    if unfreeze_mode != "topk":
        model = ASTHeartQAUnfreeze(unfreeze_mode=unfreeze_mode, topk_layers=0).to(device)
        if args.grad_checkpointing and unfreeze_mode == "full":
            # use_reentrant=False is required, not a style choice -- with the
            # reentrant (default) implementation, if a checkpointed layer's
            # INPUT activation doesn't require grad (irrelevant for 'full',
            # where every input does, but load-bearing for 'topk' below),
            # the checkpoint silently drops gradients for that layer's own
            # trainable parameters. See test_unfreeze_ast_qa.py check 9 and
            # ../README.md "Real bug found during the second audit pass".
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info(f"  {tag}: trainable params = {model.num_trainable_parameters():,} "
                    f"/ {model.num_total_parameters():,}")
        optimizer = build_optimizer(model, unfreeze_mode, args.head_lr, args.backbone_lr)
        run_training_epochs(model, train_loader, optimizer, criterion, device,
                             args.epochs, tag, fold, convergence_rows, tag)
        return model

    # topk: Phase 1 -- head-only warmup, encoder fully frozen.
    model = ASTHeartQAUnfreeze(unfreeze_mode="frozen").to(device)
    if args.topk_warmup_epochs > 0:
        optimizer = build_optimizer(model, "frozen", args.head_lr, args.backbone_lr)
        run_training_epochs(model, train_loader, optimizer, criterion, device,
                             args.topk_warmup_epochs, f"{tag}-warmup", fold, convergence_rows, tag)

    # Phase 2 -- unfreeze the top-K layers, continue training.
    model.set_unfreeze_mode("topk", topk_layers=topk_layers)
    if args.grad_checkpointing:
        # use_reentrant=False is REQUIRED here, not optional: layers 0..(12-K-1)
        # are frozen (including embeddings), so the activation flowing into the
        # first unfrozen layer has requires_grad=False. The default reentrant
        # checkpoint implementation silently drops that first unfrozen layer's
        # own parameter gradients in exactly this situation -- confirmed as a
        # real bug (not just a benign warning) by test_unfreeze_ast_qa.py check 9,
        # which failed with reentrant checkpointing and passes with this fix.
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
    parser = argparse.ArgumentParser(description="Unfreezing ablation (Reviewer #1, Comment 2)")
    parser.add_argument("--unfreeze_mode", type=str, required=True, choices=["frozen", "full", "topk"])
    parser.add_argument("--topk_layers", type=int, default=0, choices=[0, 2, 4],
                         help="Required (2 or 4) when --unfreeze_mode topk")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Local data dir. Required unless --gcs_bucket is set.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Local output dir. Defaults to a temp dir when --gcs_bucket is set.")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                         help="If set, download data from gs://<bucket>/<data_prefix> before "
                              "training and upload --output_dir to gs://<bucket>/<output_prefix> after.")
    parser.add_argument("--data_prefix", type=str, default="data/")
    parser.add_argument("--output_prefix", type=str, default=None,
                         help="Defaults to results/unfreezing_ablation/<mode_tag>/")
    parser.add_argument("--n_folds", type=int, default=3,
                         help="3-fold per this experiment's own documented deviation on top of "
                              "PAPER_REVISIONS' 5-fold default -- see README.md 'Why 3-fold, not 5-fold'")
    parser.add_argument("--epochs", type=int, default=5, help="Total epochs (includes warmup for topk)")
    parser.add_argument("--topk_warmup_epochs", type=int, default=2,
                         help="Head-only warmup epochs before unfreezing top-K layers (topk mode only)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5,
                         help="LR for unfrozen encoder layers (full/topk). CLAUDE.md Sec 9.1 "
                              "Comment 2 suggests 1e-5 or 5e-5 for Mode B.")
    parser.add_argument("--grad_checkpointing", action="store_true",
                         help="Enable gradient checkpointing on the encoder -- recommended for "
                              "--unfreeze_mode full on L4 (24GB), which has less VRAM than the "
                              "A100 the original runs used.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto-detect cuda>cpu)")
    parser.add_argument("--limit_patients", type=int, default=0,
                         help="If >0, only use this many patients (for smoke-testing)")
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,1,5,10,25,50,75,100",
                         help="Comma-separated lambda sweep (override with a short list for smoke-testing)")
    parser.add_argument("--noise_strategy", type=str, default="variable_0_10",
                         choices=["variable_0_10", "clean", "fixed_10"],
                         help="Training-set noise strategy (added 2026-09-15 for the PI-directed "
                              "pivot to the unfrozen backbone -- see module docstring). Default "
                              "'variable_0_10' reproduces the original ablation's exact behavior.")
    args = parser.parse_args()

    if args.unfreeze_mode == "topk" and args.topk_layers not in (2, 4):
        parser.error("--topk_layers must be 2 or 4 when --unfreeze_mode topk")
    if args.unfreeze_mode == "topk" and args.epochs <= args.topk_warmup_epochs:
        # Fail before any training happens, not after wasting a warmup epoch's
        # worth of GPU time on a real job -- train_model() also re-checks this
        # right before Phase 2 as a defense-in-depth safety net.
        parser.error(
            f"--epochs ({args.epochs}) must be greater than --topk_warmup_epochs "
            f"({args.topk_warmup_epochs}) so the top-{args.topk_layers} layers actually "
            f"get unfrozen-phase training."
        )
    if not args.gcs_bucket and not args.data_dir:
        parser.error("--data_dir is required unless --gcs_bucket is set")

    set_seed(args.seed)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    # Only suffix the tag when noise_strategy deviates from the original
    # default -- keeps the existing full/frozen/topk2/topk4 output paths
    # (and the already-published 3-fold results under them) untouched.
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

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_fold_metrics = []
    convergence_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        logger.info(f"=== FOLD {fold + 1}/{args.n_folds} (mode={tag}) ===")
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train_hearts = [f for p in train_pids for f in patient_map[p]]
        test_hearts = [f for p in test_pids for f in patient_map[p]]

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
            # Upload after every fold, not just at the end -- if a preemptible/spot
            # job gets killed mid-run, completed folds' raw predictions and the
            # convergence log are already safe in GCS instead of lost with the container.
            upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))

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
