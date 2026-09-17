#!/usr/bin/env python3
"""
Denoise-then-classify versus noise-aware training (Reviewer #7, Comment 1).

The reviewer asks why a quality classifier should be trained on noisy audio at
all, when one could instead denoise a corrupted recording and pass it to a
clean-trained model. This script builds that comparison:

  1. Train a genuinely clean-only AST-QA model -- heart-only positives,
     noise-only negatives, no mixing at any lambda -- with patient-level
     k-fold cross-validation. A clean-only checkpoint has to be trained here
     because none exists elsewhere: the checkpoint distributed with the
     original paper was trained at lambda = 1.0, not on clean audio.
  2. For each fold and each lambda in the sweep, evaluate that fold's
     clean-only model on the pre-mixed audio of that fold's held-out test
     patients, read from `<mixed_dir>/lambda_<value>/` (produced by the same
     RMS mixing code as the rest of this package), under several conditions:
       - `no_denoise`: the pre-mixed audio fed straight to the model.
       - `denoise_wavelet`: the same files through wavelet_denoise()
         (Candidate 1 -- training-free, deterministic, untuned).
       - `denoise_wavelet_leveldep`: the same files through this work's own
         level-dependent wavelet variant (see wavelet_denoiser.py, including
         its measured signal-degradation limitation).
       - `denoise_lunet`: the same files through lunet_denoise() (Candidate 2
         -- pretrained, with a disclosed ICBHI 2017 training-data overlap with
         this paper's noise source; see lunet_denoiser.py).
     Every condition sees the same folds and the same files, so the
     comparison is paired -- the main evidence this experiment produces.
  3. The noise-aware comparator needs no new training: the `clean` and
     `noise_0_10` (variable noise, lambda ~ U[0, 10]) entries of
     ../../../results/three_strategies_cv/final_results.json are the already
     published numbers the denoiser conditions are compared against.

Two-run workflow. `--n_folds` is required and has no default, because the
script serves two runs that must agree on the patient split:

  1. Clean-only training run: `--conditions no_denoise --save_checkpoints`
     trains and persists one clean-only model per fold.
  2. Denoiser comparison run: `--load_checkpoint_dir` (or, for cloud jobs,
     `--gcs_checkpoint_prefix`) together with the full `--conditions` list
     reloads those per-fold weights and only evaluates, so it needs no
     training time at all. This is valid only if fold i's train/test patient
     split is identical between the two runs, which requires the same
     `--n_folds`, the same `--seed` and the same patient list. The script
     therefore checks that the number of checkpoints found equals `--n_folds`
     and aborts on a mismatch, rather than silently evaluating fold i's model
     against another fold's held-out patients.

Usage (paths relative to this directory):
    # Step 1 -- clean-only training, saving per-fold weights:
    python run_denoiser_benchmark_cv.py \\
        --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \\
        --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only/ \\
        --n_folds 5 --conditions no_denoise --save_checkpoints \\
        --epochs 5 --seed 42

    # Step 2 -- denoiser comparison reusing Step 1's weights (no training):
    python run_denoiser_benchmark_cv.py \\
        --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \\
        --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison/ \\
        --n_folds 5 \\
        --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only/checkpoints/ \\
        --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \\
        --seed 42

The exact commands, fold counts and model variants behind each result set
checked into this package are listed in ../../README.md.
"""

import argparse
import json
import logging
import os
import random
import sys
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
from wavelet_denoiser import wavelet_denoise, wavelet_denoise_level_dependent
from lunet_denoiser import lunet_denoise
# A fourth denoiser candidate (T-BiLSTM) was attempted and abandoned without
# usable weights or reported results, and its module is not included in this
# package. The `denoise_tbilstm` condition therefore imports it lazily, only
# when that condition is explicitly requested, so its absence cannot affect
# the no_denoise / wavelet / LU-Net conditions that the results are based on.

# Locate the package root containing src/models/ast_qa.py. Three layouts have
# to resolve here:
#   * this package, where the model lives at revision/src/models/ast_qa.py
#     (parents[2] == the revision/ directory) -- this is the vendored copy
#     that should always be preferred;
#   * a development checkout, where the experiment sits one level deeper and
#     the model package is at parents[3];
#   * a container image, where only this experiment's src/ is copied in and
#     the model package is mounted at the fixed path /app/repo_src (parents[3]
#     may not exist at all there, hence the length guards below).
# The candidates are tried in that order so that the locally vendored copy
# wins deterministically instead of matching a sibling package by accident:
# from this package's layout, parents[3] also happens to contain a
# src/models/ast_qa.py (the same class), and relying on that would be
# ambiguous.
_CANDIDATE_REPO_ROOTS = [Path("/app/repo_src")]
if len(Path(__file__).resolve().parents) > 2:
    _CANDIDATE_REPO_ROOTS.append(Path(__file__).resolve().parents[2])
if len(Path(__file__).resolve().parents) > 3:
    _CANDIDATE_REPO_ROOTS.append(Path(__file__).resolve().parents[3])
for _candidate in _CANDIDATE_REPO_ROOTS:
    if (_candidate / "src" / "models" / "ast_qa.py").exists():
        REPO_ROOT = _candidate
        break
else:
    raise RuntimeError(
        f"Could not locate src/models/ast_qa.py under any candidate repo root: {_CANDIDATE_REPO_ROOTS}"
    )
sys.path.insert(0, str(REPO_ROOT))
from src.models.ast_qa import ASTHeartQA  # noqa: E402 -- the published, unmodified frozen model

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Local copy of the configurable-freeze model (identical architecture, see
# unfreeze_ast_qa.py); kept per-experiment rather than shared so that each
# experiment's code stays fixed once its results are produced.
from unfreeze_ast_qa import ASTHeartQAUnfreeze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000
DURATION = 10
MAX_LENGTH = TARGET_SR * DURATION
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- audio pipeline, identical to ../../../src/train_per_lambda_cv.py ---
# This is the same preprocessing every other method in the revision is
# evaluated under, and the same code that generated the pre-mixed evaluation
# corpus (see ../../../src/generate_mixed_datasets.py), so applying it to the
# raw clean/noise training files here stays consistent with how the
# evaluation-side files were built. Steps: load at 16 kHz mono, remove DC
# offset, 20 Hz high-pass and 1 kHz low-pass, peak-normalize, then crop or
# loop-pad (np.tile) to exactly 10 s. Returns None for unreadable, too-short
# or numerically silent files so callers can skip them.
def load_audio(path):
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


def get_noise(icbhi_files, env_files, idx=None):
    """Build one composite noise waveform: lung + 0.5 x environmental, peak-normalized.

    Matches Eq. 1 of the paper. `idx` selects a per-sample deterministic RNG
    (seeded 42 + idx) for reproducible pre-mixed generation; passing None uses
    the global `random` stream, which is what the training dataset does so
    that negatives vary across epochs.
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
# --- end shared audio pipeline ---


class CleanOnlyTrainDataset(Dataset):
    """Clean-only training set: unmixed heart recordings as positives,
    composite noise as negatives, balanced 1:1.

    This reproduces the published "Clean-Only Training" strategy (the `clean`
    arm of ../../../src/train_three_strategies_cv.py). The mixing function is
    never called on the positive path: at lambda = 0 it would return the heart
    signal unchanged, so the result is the same either way, but "clean" here
    means literally unmixed audio rather than relying on that equivalence.
    """

    def __init__(self, heart_files, icbhi_files, env_files, processor):
        self.icbhi_files = icbhi_files
        self.env_files = env_files
        self.processor = processor
        self.data = [(f, 1) for f in heart_files] + [(None, 0) for _ in heart_files]
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        wav = load_audio(path) if path is not None else get_noise(self.icbhi_files, self.env_files)
        if wav is None:
            wav = np.zeros(MAX_LENGTH)
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }


class PreMixedEvalDataset(Dataset):
    """Evaluation set for one fold at one lambda, built from the pre-mixed corpus.

    Positives (label 1) are the pre-mixed heart+noise files in
    `<mixed_dir>/lambda_<lam>/` whose source recording belongs to one of this
    fold's held-out `test_pids`; the patient id is the part of the source
    filename before the first underscore. Negatives (label 0) are pure-noise
    files from the same directory, which carry no patient identity, shuffled
    with a per-fold seed and truncated to the number of positives so the set
    is exactly balanced 1:1 and deterministic per fold.

    `condition` selects what happens to each waveform before it reaches the
    feature extractor:
      - 'no_denoise': passed through unmodified.
      - 'denoise_wavelet': wavelet_denoise() -- Candidate 1, the untuned
        skimage BayesShrink/VisuShrink implementation.
      - 'denoise_wavelet_leveldep': wavelet_denoise_level_dependent() -- this
        work's own per-level noise-estimation variant of Candidate 1; see
        wavelet_denoiser.py for its authorship status and the signal energy
        it removes even from clean recordings.
      - 'denoise_lunet': lunet_denoise() -- Candidate 2, pretrained, with the
        ICBHI 2017 training-data overlap documented in lunet_denoiser.py.
      - 'denoise_tbilstm': an abandoned fourth candidate whose module is not
        shipped with this package; selecting it requires supplying that
        module and a trained checkpoint.
    """

    def __init__(self, mixed_dir, lam, test_pids, processor, condition,
                 denoiser_method, denoiser_wavelet, seed, fold, tbilstm_checkpoint=None):
        self.processor = processor
        self.condition = condition
        self.denoiser_method = denoiser_method
        self.denoiser_wavelet = denoiser_wavelet
        self.tbilstm_checkpoint = tbilstm_checkpoint

        lam_dir = Path(mixed_dir) / f"lambda_{lam}"
        manifest = pd.read_csv(lam_dir / "manifest.csv")

        pos = manifest[manifest["label"] == 1].copy()
        pos["patient_id"] = pos["source_heart"].str.split("_").str[0]
        pos = pos[pos["patient_id"].isin(test_pids)]

        neg = manifest[manifest["label"] == 0].copy()
        rng = random.Random(seed + fold)
        neg_files = list(neg["filename"])
        rng.shuffle(neg_files)
        neg_files = neg_files[:len(pos)]  # balanced 1:1, deterministic per fold

        self.lam_dir = lam_dir
        self.samples = [(fn, 1) for fn in pos["filename"]] + [(fn, 0) for fn in neg_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        wav, _ = librosa.load(self.lam_dir / filename, sr=TARGET_SR, mono=True)
        if len(wav) != MAX_LENGTH:
            # Pre-mixed files should already be exactly MAX_LENGTH; crop or
            # loop-pad defensively rather than assume, since the corpus is
            # downloaded separately from this code.
            if len(wav) > MAX_LENGTH:
                wav = wav[:MAX_LENGTH]
            else:
                wav = np.tile(wav, int(np.ceil(MAX_LENGTH / len(wav))))[:MAX_LENGTH]
        if self.condition == "denoise_wavelet":
            wav = wavelet_denoise(wav, method=self.denoiser_method, wavelet=self.denoiser_wavelet)
        elif self.condition == "denoise_wavelet_leveldep":
            wav = wavelet_denoise_level_dependent(wav, method=self.denoiser_method, wavelet=self.denoiser_wavelet)
        elif self.condition == "denoise_lunet":
            wav = lunet_denoise(wav, target_sr=TARGET_SR)
        elif self.condition == "denoise_tbilstm":
            # Imported here, not at module scope, so that the missing module
            # only matters if this condition is actually requested.
            sys.path.insert(0, str(Path(__file__).resolve().parent / "attempted_not_used"))
            from tbilstm_denoiser import tbilstm_denoise
            wav = tbilstm_denoise(wav, weights_path=self.tbilstm_checkpoint, target_sr=TARGET_SR)
        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
            "filename": filename,
        }


def train_clean_only_model(train_loader, fold, device, epochs, lr, unfreeze_mode="frozen",
                            backbone_lr=5e-5, grad_checkpointing=False):
    """Train one fold's clean-only classifier.

    unfreeze_mode='frozen' trains the published frozen-backbone model
    (ASTHeartQA with freeze_base=True) with a head-only optimizer.
    unfreeze_mode='full' trains a fully fine-tuned clean-only model instead,
    using two parameter groups so the encoder gets the lower `backbone_lr`
    while the head keeps `lr`; `grad_checkpointing` trades compute for
    activation memory so backprop through all 12 layers fits on a single GPU.

    The two modes are separate comparators, not replacements for each other:
    the denoiser conditions are evaluated against whichever clean-only model
    a given run trains (or loads).
    """
    criterion = nn.BCEWithLogitsLoss()
    if unfreeze_mode == "frozen":
        model = ASTHeartQA(freeze_base=True).to(device)
        optimizer = torch.optim.AdamW(model.qa_classifier.parameters(), lr=lr)
    elif unfreeze_mode == "full":
        model = ASTHeartQAUnfreeze(unfreeze_mode="full").to(device)
        if grad_checkpointing:
            # use_reentrant=False is required, not stylistic: the reentrant
            # checkpointing path only propagates gradients when the
            # checkpointed block's inputs themselves require grad, which is
            # not the case for this encoder's spectrogram input.
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW([
            {"params": model.qa_classifier.parameters(), "lr": lr},
            {"params": model.trainable_backbone_parameters(), "lr": backbone_lr},
        ])
    else:
        raise ValueError(f"unfreeze_mode must be 'frozen' or 'full', got {unfreeze_mode!r}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"clean-only-{unfreeze_mode} F{fold} E{epoch + 1}"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)
            _, logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"  clean-only-{unfreeze_mode} | Fold {fold} | Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {epoch_loss / len(train_loader):.4f}")
    return model


def load_clean_only_model(checkpoint_path, device, unfreeze_mode="frozen"):
    """Rebuild a clean-only model from a checkpoint saved by a previous run.

    The checkpoint contents depend on the mode, mirroring the save-side
    branch under --save_checkpoints:
      - 'frozen': only the qa_classifier state dict was saved, because the
        frozen AST encoder is byte-identical across every fold; the encoder
        is reloaded from the pretrained checkpoint and the head is restored
        on top of it.
      - 'full': the encoder was itself fine-tuned, and differs per fold, so
        the entire model state dict is saved and loaded.
    """
    if unfreeze_mode == "frozen":
        model = ASTHeartQA(freeze_base=True).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.qa_classifier.load_state_dict(state_dict)
    elif unfreeze_mode == "full":
        model = ASTHeartQAUnfreeze(unfreeze_mode="full").to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"unfreeze_mode must be 'frozen' or 'full', got {unfreeze_mode!r}")
    model.eval()
    return model


def compute_metrics(y_true, probs):
    """Compute the six reported metrics at the fixed 0.5 decision threshold.

    Returns accuracy, AUROC, AUPRC, F1, sensitivity and specificity. AUROC
    and AUPRC fall back to 0.5 and 0.0 respectively when they are undefined
    (a batch containing only one class); sensitivity and specificity fall
    back to 0.0 when their denominator is empty.
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


def evaluate_condition(model, mixed_dir, lambdas, test_pids, processor, device, batch_size,
                        output_dir, fold, condition, denoiser_method, denoiser_wavelet, seed,
                        tbilstm_checkpoint=None):
    """Evaluate one fold's model under one condition, across every lambda.

    For each lambda, builds the fold's balanced evaluation set, runs
    inference, and writes the raw per-recording predictions to
    <output_dir>/raw_predictions/<condition>/lambda_<lam>/fold_<fold>/
    predictions.csv so that every aggregated number can be recomputed from
    the raw outputs. Returns {lambda: metrics dict}.
    """
    per_lambda_metrics = {}
    for lam in lambdas:
        ds = PreMixedEvalDataset(mixed_dir, lam, test_pids, processor, condition,
                                  denoiser_method, denoiser_wavelet, seed, fold, tbilstm_checkpoint)
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
        logger.info(f"  [{condition}] lambda={lam}: AUROC={metrics['auroc']:.4f} F1={metrics['f1']:.4f}")

        preds_dir = os.path.join(output_dir, "raw_predictions", condition, f"lambda_{lam}", f"fold_{fold}")
        os.makedirs(preds_dir, exist_ok=True)
        pd.DataFrame({"filename": filenames, "y_true": y_true, "probs": y_probs}).to_csv(
            os.path.join(preds_dir, "predictions.csv"), index=False
        )
    return per_lambda_metrics


# --- Google Cloud Storage helpers ---
# Used only for cloud (Vertex AI) runs, where the datasets are staged in a
# bucket and results are uploaded after every fold. Local runs never touch
# these, and google-cloud-storage is imported lazily so it is not a hard
# dependency of the local pipeline.
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


def main():
    parser = argparse.ArgumentParser(description="Denoise-then-classify vs. variable-noise (Reviewer #7, Comment 1)")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Root dir containing PhysioNet2022/ICBHI2017/ESC-50/UrbanSound8K. "
                              "Required unless --gcs_bucket is set.")
    parser.add_argument("--mixed_dir", type=str, default=None,
                         help="Root of the pre-mixed evaluation corpus (lambda_X/ subdirs, "
                              "each with a manifest.csv). Required unless --gcs_bucket is set.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Local output dir. Defaults to a temp dir when --gcs_bucket is set.")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                         help="If set, download --data_prefix/--mixed_prefix from this GCS bucket "
                              "before running and upload --output_dir to "
                              "gs://<bucket>/<output_prefix> after every fold, so a preempted "
                              "cloud job does not lose completed folds.")
    parser.add_argument("--data_prefix", type=str, default="data/",
                         help="GCS prefix holding the raw source datasets.")
    parser.add_argument("--mixed_prefix", type=str, default="mixed_dataset/",
                         help="GCS prefix holding the pre-mixed evaluation corpus (~20 GB).")
    parser.add_argument("--output_prefix", type=str, default="results/denoiser_benchmark/")
    parser.add_argument("--n_folds", type=int, required=True,
                         help="Number of patient-level CV folds. No default: when reusing "
                              "checkpoints via --load_checkpoint_dir/--gcs_checkpoint_prefix this "
                              "must equal the fold count of the run that produced them, since each "
                              "fold's model is evaluated on that same fold's held-out patients. See "
                              "../../README.md for the fold count used for each checked-in result set.")
    parser.add_argument("--conditions", type=str, default="no_denoise,denoise_wavelet,denoise_lunet",
                         help="Comma-separated list of evaluation arms: 'no_denoise'; "
                              "'denoise_wavelet' (Candidate 1, untuned skimage BayesShrink/"
                              "VisuShrink); 'denoise_wavelet_leveldep' (this work's own "
                              "level-dependent noise-estimation variant of Candidate 1, not a "
                              "separately published algorithm -- see wavelet_denoiser.py); "
                              "'denoise_lunet' (Candidate 2, pretrained, with a disclosed ICBHI 2017 "
                              "training-data overlap -- see lunet_denoiser.py); 'denoise_tbilstm' "
                              "(an abandoned fourth candidate whose module is not shipped with this "
                              "package; needs that module plus --tbilstm_checkpoint). Use "
                              "'no_denoise' alone for the clean-only training run, then the full "
                              "list for the denoiser comparison.")
    parser.add_argument("--tbilstm_checkpoint", type=str, default=None,
                         help="Local path to trained weights for the abandoned T-BiLSTM candidate. "
                              "Required when --conditions includes denoise_tbilstm, unless "
                              "--gcs_tbilstm_checkpoint is used instead.")
    parser.add_argument("--gcs_tbilstm_checkpoint", type=str, default=None,
                         help="GCS blob path to download before running, when --gcs_bucket is set. "
                              "This is a single file, not a prefix: T-BiLSTM has one weights file "
                              "in total rather than one per fold. Mutually exclusive with "
                              "--tbilstm_checkpoint.")
    parser.add_argument("--save_checkpoints", action="store_true",
                         help="Persist each fold's trained clean-only model under "
                              "<output_dir>/checkpoints/ so a later run can reload the weights and "
                              "evaluate additional conditions without retraining. See "
                              "load_clean_only_model() for what each mode writes.")
    parser.add_argument("--load_checkpoint_dir", type=str, default=None,
                         help="Local directory of per-fold checkpoints from an earlier "
                              "--save_checkpoints run. When set, training is SKIPPED for every fold "
                              "and the run only loads and evaluates, which is what makes the "
                              "denoiser comparison inexpensive. --n_folds and --seed must match the "
                              "run that produced the checkpoints, since reusing fold i's weights is "
                              "only leakage-free if fold i's train/test patient split is identical. "
                              "Mutually exclusive with --gcs_checkpoint_prefix.")
    parser.add_argument("--gcs_checkpoint_prefix", type=str, default=None,
                         help="GCS equivalent of --load_checkpoint_dir for cloud jobs: prefix of an "
                              "earlier run's checkpoints/ directory, downloaded before the folds "
                              "start when --gcs_bucket is set. Same fold-count requirement.")
    parser.add_argument("--unfreeze_mode", type=str, default="frozen", choices=["frozen", "full"],
                         help="Backbone policy for the clean-only model. 'frozen': the frozen-"
                              "backbone comparator, checkpointed as the QA head only. 'full': a "
                              "fully fine-tuned clean-only model, checkpointed as a complete state "
                              "dict. The two are separate comparators; which one a given result set "
                              "used is recorded in the output JSON and in ../../README.md.")
    parser.add_argument("--backbone_lr", type=float, default=5e-5,
                         help="Learning rate for the unfrozen encoder parameters; used only with "
                              "--unfreeze_mode=full. Lower than --lr, which remains the head's rate.")
    parser.add_argument("--grad_checkpointing", action="store_true",
                         help="Used only with --unfreeze_mode=full: enable gradient checkpointing on "
                              "the encoder, trading compute for activation memory so that backprop "
                              "through all 12 layers fits on a single mid-range GPU.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoiser_method", type=str, default="BayesShrink", choices=["BayesShrink", "VisuShrink"],
                         help="Wavelet thresholding rule for both wavelet conditions.")
    parser.add_argument("--denoiser_wavelet", type=str, default="db4", choices=["db4", "sym4"],
                         help="Wavelet family for both wavelet conditions.")
    parser.add_argument("--device", type=str, default=None,
                         help="Torch device; defaults to cuda when available, else cpu.")
    parser.add_argument("--limit_patients", type=int, default=0,
                         help="Use only the first N patients. For smoke tests only -- results from a "
                              "limited run are not comparable to the reported numbers.")
    parser.add_argument("--lambdas", type=str, default=",".join(str(l) for l in LAMBDAS),
                         help="Comma-separated noise levels to evaluate. Each must have a "
                              "corresponding lambda_<value>/ directory under --mixed_dir.")
    args = parser.parse_args()

    if not args.gcs_bucket and (not args.data_dir or not args.mixed_dir):
        parser.error("--data_dir and --mixed_dir are required unless --gcs_bucket is set")
    if not args.gcs_bucket and not args.output_dir:
        parser.error("--output_dir is required unless --gcs_bucket is set")
    if args.load_checkpoint_dir and args.gcs_checkpoint_prefix:
        parser.error("--load_checkpoint_dir and --gcs_checkpoint_prefix are mutually exclusive")

    set_seed(args.seed)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    conditions = args.conditions.split(",")
    valid_conditions = ("no_denoise", "denoise_wavelet", "denoise_wavelet_leveldep", "denoise_lunet", "denoise_tbilstm")
    for c in conditions:
        if c not in valid_conditions:
            parser.error(f"--conditions entries must be one of {valid_conditions}, got {c!r}")
    if args.tbilstm_checkpoint and args.gcs_tbilstm_checkpoint:
        parser.error("--tbilstm_checkpoint and --gcs_tbilstm_checkpoint are mutually exclusive")
    if "denoise_tbilstm" in conditions and not (args.tbilstm_checkpoint or args.gcs_tbilstm_checkpoint):
        parser.error("--tbilstm_checkpoint or --gcs_tbilstm_checkpoint is required when "
                      "--conditions includes denoise_tbilstm")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.gcs_bucket:
        import tempfile
        work_dir = tempfile.mkdtemp(prefix="denoiser_benchmark_")
        data_dir = os.path.join(work_dir, "data")
        mixed_dir = os.path.join(work_dir, "mixed")
        output_dir = args.output_dir or os.path.join(work_dir, "output")
        download_from_gcs(args.gcs_bucket, args.data_prefix, data_dir)
        download_from_gcs(args.gcs_bucket, args.mixed_prefix, mixed_dir)
        if args.gcs_checkpoint_prefix:
            checkpoint_dir = os.path.join(work_dir, "checkpoints")
            download_from_gcs(args.gcs_bucket, args.gcs_checkpoint_prefix, checkpoint_dir)
        else:
            checkpoint_dir = None
        if args.gcs_tbilstm_checkpoint:
            from google.cloud import storage
            local_tbilstm_path = os.path.join(work_dir, "tbilstm_model.weights.h5")
            logger.info(f"Downloading T-BiLSTM checkpoint from gs://{args.gcs_bucket}/{args.gcs_tbilstm_checkpoint}")
            storage.Client().bucket(args.gcs_bucket).blob(args.gcs_tbilstm_checkpoint).download_to_filename(local_tbilstm_path)
            args.tbilstm_checkpoint = local_tbilstm_path
    else:
        data_dir = args.data_dir
        mixed_dir = args.mixed_dir
        output_dir = args.output_dir
        checkpoint_dir = args.load_checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    ckpt_glob = "fold_*_qa_classifier.pth" if args.unfreeze_mode == "frozen" else "fold_*_full.pth"
    if checkpoint_dir:
        found_checkpoints = sorted(Path(checkpoint_dir).glob(ckpt_glob))
        if len(found_checkpoints) != args.n_folds:
            raise ValueError(
                f"--n_folds={args.n_folds} but found {len(found_checkpoints)} checkpoint(s) in "
                f"{checkpoint_dir}. Reusing a saved checkpoint per fold only produces a correct, "
                f"leakage-free evaluation when this run's KFold(n_splits={args.n_folds}, seed="
                f"{args.seed}) split is IDENTICAL to the one that produced these checkpoints -- "
                f"a fold-count mismatch here means fold i's 'held-out' test patients would not "
                f"actually match the patients fold i's checkpoint was trained without. Set "
                f"--n_folds to the checkpoint-producing run's fold count instead of guessing."
            )
        logger.info(f"Loaded {len(found_checkpoints)} checkpoints from {checkpoint_dir} -- "
                    f"training will be SKIPPED for every fold.")

    data_root = Path(data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))
    icbhi_files = sorted(list(data_root.rglob("ICBHI2017/**/*.wav")))
    env_files = sorted(list(data_root.rglob("ESC-50/**/*.wav")) + list(data_root.rglob("UrbanSound8K/**/*.wav")))
    logger.info(f"Found {len(heart_files)} heart, {len(icbhi_files)} lung, {len(env_files)} environmental files")

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
    all_fold_metrics = {c: [] for c in conditions}

    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        logger.info(f"=== FOLD {fold + 1}/{args.n_folds} ===")
        train_pids = [pids[i] for i in train_idx]
        test_pids = set(pids[i] for i in test_idx)
        train_hearts = [f for p in train_pids for f in patient_map[p]]
        logger.info(f"  Fold {fold + 1}: {len(train_hearts)} train hearts / {len(test_pids)} test patients")

        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"fold_{fold + 1}_{'full' if args.unfreeze_mode == 'full' else 'qa_classifier'}.pth")
            logger.info(f"  Loading checkpoint (training skipped): {ckpt_path}")
            model = load_clean_only_model(ckpt_path, device, unfreeze_mode=args.unfreeze_mode)
        else:
            train_ds = CleanOnlyTrainDataset(train_hearts, icbhi_files, env_files, processor)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            model = train_clean_only_model(train_loader, fold + 1, device, args.epochs, args.lr,
                                            unfreeze_mode=args.unfreeze_mode, backbone_lr=args.backbone_lr,
                                            grad_checkpointing=args.grad_checkpointing)

        if args.save_checkpoints:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            if args.unfreeze_mode == "frozen":
                # Frozen mode saves ONLY the qa_classifier state dict. The
                # encoder is never updated by training, so it is identical
                # across every fold (always the same pretrained
                # MIT/ast-finetuned-audioset-10-10-0.4593 weights); storing it
                # per fold would cost ~347 MB per fold for no information
                # gain, while the head that does differ is ~0.4 MB.
                # To reload: construct ASTHeartQA(freeze_base=True), then
                # model.qa_classifier.load_state_dict(torch.load(ckpt_path)) --
                # see load_clean_only_model().
                ckpt_path = os.path.join(ckpt_dir, f"fold_{fold + 1}_qa_classifier.pth")
                torch.save(model.qa_classifier.state_dict(), ckpt_path)
                logger.info(f"  Saved checkpoint (qa_classifier only): {ckpt_path}")
            else:
                # In 'full' mode the encoder is itself fine-tuned and differs
                # per fold, so the entire model state dict must be saved
                # (~347 MB per fold; budget disk accordingly).
                ckpt_path = os.path.join(ckpt_dir, f"fold_{fold + 1}_full.pth")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"  Saved checkpoint (full model): {ckpt_path}")

        for condition in conditions:
            fold_metrics = evaluate_condition(
                model, mixed_dir, lambdas, test_pids, processor, device, args.batch_size,
                output_dir, fold + 1, condition, args.denoiser_method, args.denoiser_wavelet, args.seed,
                tbilstm_checkpoint=args.tbilstm_checkpoint,
            )
            all_fold_metrics[condition].append(fold_metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with open(os.path.join(output_dir, "denoiser_benchmark_progress.json"), "w") as f:
            json.dump({"folds_done": fold + 1, "per_fold": all_fold_metrics}, f, indent=2)

        if args.gcs_bucket:
            # Uploaded after every fold, not only at the end, so that a
            # preempted or crashed cloud job keeps its completed folds.
            upload_to_gcs(args.gcs_bucket, output_dir, args.output_prefix.rstrip("/"))

    # Aggregate across folds: per-metric mean plus a normal-approximation 95%
    # interval, 1.96 * SD / sqrt(n_folds).
    final_results = {}
    for condition in conditions:
        final_results[condition] = {}
        for lam in lambdas:
            metrics = [fold[lam] for fold in all_fold_metrics[condition]]
            means = {m: float(np.mean([f[m] for f in metrics])) for m in metrics[0]}
            cis = {m: float(1.96 * np.std([f[m] for f in metrics]) / np.sqrt(args.n_folds)) for m in metrics[0]}
            final_results[condition][lam] = {"mean": means, "ci": cis}

    with open(os.path.join(output_dir, "denoiser_benchmark_final_results.json"), "w") as f:
        json.dump({
            "n_folds": args.n_folds, "unfreeze_mode": args.unfreeze_mode, "conditions": conditions,
            "denoiser_method": args.denoiser_method,
            "denoiser_wavelet": args.denoiser_wavelet, "results": final_results,
        }, f, indent=2)

    for condition in conditions:
        df = pd.DataFrame({lam: final_results[condition][lam]["mean"] for lam in lambdas}).T
        df.index.name = "Lambda"
        df.to_csv(os.path.join(output_dir, f"denoiser_benchmark_metrics_{condition}.csv"))

    logger.info(f"Done! Results saved to {output_dir}")

    if args.gcs_bucket:
        upload_to_gcs(args.gcs_bucket, output_dir, args.output_prefix.rstrip("/"))


if __name__ == "__main__":
    main()
