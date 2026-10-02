#!/usr/bin/env python3
"""
Denoise-then-classify versus noise-aware training (Reviewer #7, Comment 1).

The reviewer asks why the quality classifier is trained on noisy audio
instead of denoising the recording and using a clean-trained model. This
script runs that comparison:

  1. Train a clean-only AST-QA model (heart-only positives, noise-only
     negatives, no mixing) with patient-level k-fold CV. This has to be
     trained here because the checkpoint released with the paper was
     trained at lambda = 1.0.
  2. For each fold and lambda, evaluate that fold's clean-only model on the
     pre-mixed audio of the fold's test patients
     (`<mixed_dir>/lambda_<value>/`, made with the same RMS mixing code as
     the rest of the package) under these conditions:
       - `no_denoise`: pre-mixed audio as is.
       - `denoise_wavelet`: wavelet_denoise() (candidate 1, no training,
         deterministic, untuned).
       - `denoise_wavelet_leveldep`: our level-dependent wavelet variant
         (see wavelet_denoiser.py, including its limitation on clean audio).
       - `denoise_lunet`: lunet_denoise() (candidate 2, pretrained; its
         training data includes ICBHI 2017, which our noise also uses; see
         lunet_denoiser.py).
     All conditions use the same folds and files, so the comparison is
     paired.
  3. The noise-aware models need no new training: the `clean` and
     `noise_0_10` (lambda ~ U[0, 10]) results in
     ../../../results/three_strategies_cv/final_results.json are the
     published numbers the denoiser conditions are compared with.

Two runs share the patient split, so `--n_folds` has no default:

  1. Training run: `--conditions no_denoise --save_checkpoints` trains and
     saves one clean-only model per fold.
  2. Comparison run: `--load_checkpoint_dir` (or `--gcs_checkpoint_prefix`)
     with the full `--conditions` list loads those models and only
     evaluates. Fold i must have the same train/test patients in both runs,
     so `--n_folds`, `--seed` and the patient list must match; the script
     stops if the number of checkpoints differs from `--n_folds`.

Two configurations are reported: a frozen model with 10-fold CV and a fully
fine-tuned model (`--unfreeze_mode full`) with 5-fold CV. Commands, run from
this directory (the same as in ../../run_all.sh):

    # Frozen, 10-fold: train clean-only models, then compare denoisers
    python run_denoiser_benchmark_cv.py \\
        --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \\
        --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/ \\
        --n_folds 10 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
    python run_denoiser_benchmark_cv.py \\
        --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \\
        --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_10fold/ \\
        --n_folds 10 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \\
        --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_10fold/checkpoints/ \\
        --seed 42

    # Fully fine-tuned, 5-fold
    python run_denoiser_benchmark_cv.py --unfreeze_mode full --backbone_lr 5e-5 --grad_checkpointing \\
        --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \\
        --output_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/ \\
        --n_folds 5 --conditions no_denoise --save_checkpoints --epochs 5 --seed 42
    python run_denoiser_benchmark_cv.py --unfreeze_mode full \\
        --data_dir ../../../dataset/ --mixed_dir ../../../mixed_dataset/ \\
        --output_dir ../../results/reviewer7_denoiser_benchmark/denoiser_comparison_full_5fold/ \\
        --n_folds 5 --conditions no_denoise,denoise_wavelet,denoise_wavelet_leveldep,denoise_lunet \\
        --load_checkpoint_dir ../../results/reviewer7_denoiser_benchmark/clean_only_full_5fold/checkpoints/ \\
        --seed 42
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
# A fourth candidate (T-BiLSTM) was dropped and its module is not included.
# The `denoise_tbilstm` condition imports it only when requested.

# Find the directory containing src/models/ast_qa.py: /app/repo_src in the
# container image, otherwise revision/ (parents[2]) in this package, or
# parents[3] in a deeper checkout. The length checks handle short paths in
# the container.
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
from src.models.ast_qa import ASTHeartQA  # noqa: E402  (published frozen model)

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Local copy of the configurable-freeze model (same architecture, see
# unfreeze_ast_qa.py).
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


# --- audio pipeline, same as ../../../src/train_per_lambda_cv.py ---
# Also the code used to build the pre-mixed corpus
# (../../../src/generate_mixed_datasets.py), so training and test audio are
# processed the same way. Load at 16 kHz mono, remove DC offset, 20 Hz
# high-pass and 1 kHz low-pass, peak-normalize, then crop or loop-pad
# (np.tile) to 10 s. Returns None for unreadable, too short or silent files.
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

    Eq. 1 of the paper. If `idx` is given, clips are chosen with
    random.Random(42 + idx); with None (training) the global `random` stream
    is used, so negatives change between epochs.
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

    Same as the published clean-only strategy (the `clean` arm of
    ../../../src/train_three_strategies_cv.py). Positives are never passed
    through the mixing function.
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
    `<mixed_dir>/lambda_<lam>/` whose source recording is from one of the
    fold's `test_pids` (patient id = source filename before the first
    underscore). Negatives (label 0) are noise-only files from the same
    directory, shuffled with a per-fold seed and cut to the number of
    positives, giving a 1:1 set.

    `condition` sets what is applied to each waveform before feature
    extraction:
      - 'no_denoise': nothing.
      - 'denoise_wavelet': wavelet_denoise() (candidate 1).
      - 'denoise_wavelet_leveldep': wavelet_denoise_level_dependent(), our
        per-level variant (see wavelet_denoiser.py).
      - 'denoise_lunet': lunet_denoise() (candidate 2; see the ICBHI 2017
        note in lunet_denoiser.py).
      - 'denoise_tbilstm': dropped candidate; needs its module and a
        checkpoint, which are not included.
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
            # Pre-mixed files should already be MAX_LENGTH; crop or pad in
            # case they are not.
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
            # Imported here because the module is not included.
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

    'frozen': the published frozen model (ASTHeartQA, freeze_base=True),
    head-only optimizer. 'full': fully fine-tuned model, with the encoder at
    the lower `backbone_lr` and the head at `lr`. `grad_checkpointing` saves
    activation memory so backprop through all 12 layers fits on one GPU.
    """
    criterion = nn.BCEWithLogitsLoss()
    if unfreeze_mode == "frozen":
        model = ASTHeartQA(freeze_base=True).to(device)
        optimizer = torch.optim.AdamW(model.qa_classifier.parameters(), lr=lr)
    elif unfreeze_mode == "full":
        model = ASTHeartQAUnfreeze(unfreeze_mode="full").to(device)
        if grad_checkpointing:
            # use_reentrant=False is needed: reentrant checkpointing only
            # propagates gradients if the block inputs require grad, and the
            # spectrogram input does not.
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

    The checkpoint contents depend on the mode (see --save_checkpoints):
      - 'frozen': only the qa_classifier state dict; the encoder is the
        pretrained one, reloaded from the hub.
      - 'full': the whole model state dict, since the encoder differs per
        fold.
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
    and AUPRC are 0.5 and 0.0 if only one class is present; sensitivity and
    specificity are 0.0 if their denominator is zero.
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

    For each lambda, builds the fold's test set, runs inference and saves
    per-recording predictions to
    <output_dir>/raw_predictions/<condition>/lambda_<lam>/fold_<fold>/
    predictions.csv. Returns {lambda: metrics dict}.
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


# --- Optional Google Cloud Storage helpers, used only with --gcs_bucket ---
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
                              "separately published algorithm: see wavelet_denoiser.py); "
                              "'denoise_lunet' (Candidate 2, pretrained, with a disclosed ICBHI 2017 "
                              "training-data overlap: see lunet_denoiser.py); 'denoise_tbilstm' "
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
                         help="Use only the first N patients. For smoke tests only: results from a "
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
                f"{args.seed}) split is identical to the one that produced these checkpoints: "
                f"a fold-count mismatch here means fold i's 'held-out' test patients would not "
                f"actually match the patients fold i's checkpoint was trained without. Set "
                f"--n_folds to the checkpoint-producing run's fold count instead of guessing."
            )
        logger.info(f"Loaded {len(found_checkpoints)} checkpoints from {checkpoint_dir}: "
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
                # Frozen mode: save only the head (~0.4 MB). The encoder is
                # the unchanged pretrained model (~347 MB). See
                # load_clean_only_model() for reloading.
                ckpt_path = os.path.join(ckpt_dir, f"fold_{fold + 1}_qa_classifier.pth")
                torch.save(model.qa_classifier.state_dict(), ckpt_path)
                logger.info(f"  Saved checkpoint (qa_classifier only): {ckpt_path}")
            else:
                # Full mode: save the whole model (~347 MB per fold).
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
            # Upload after each fold so a preempted job keeps finished folds.
            upload_to_gcs(args.gcs_bucket, output_dir, args.output_prefix.rstrip("/"))

    # Mean across folds and 95% half-width 1.96 * SD / sqrt(n_folds).
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
