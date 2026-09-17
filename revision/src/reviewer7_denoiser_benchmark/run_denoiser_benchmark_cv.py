#!/usr/bin/env python3
"""
Denoise-Then-Classify vs. Variable-Noise Training (Reviewer #7, Comment 1).

CLAUDE.md Sec 9.2 Comment 1 asks: why train a noise-aware classifier at all,
when you could just denoise the corrupted audio and run the existing
clean-trained model on it? This script builds that comparison:

  1. Train a genuinely clean-only AST-QA model (lambda_train=0.0 -- heart-only
     positives, noise-only negatives, NO mixing) via 5-fold patient-level CV.
     NOTE: no such checkpoint already existed anywhere in this repo before
     this script -- phase6_paper_quality/results/best_quality_model.pth,
     which an earlier README draft called "the existing clean-trained
     checkpoint", was actually trained at --train_lambda=1.0 (verified
     directly against phase6_paper_quality/src/train_quality_vertexai.py's
     argparse default and phase6_paper_quality/submit_quality_job.py's
     actual submitted Vertex AI job args -- not clean at all). See
     ../README.md "The stub README's checkpoint claim was wrong" for the
     full paper trail. So Part 1 of this script has to train a real one.
  2. For each fold, for each lambda in the 10-point sweep, evaluate that
     fold's clean-only model on the corresponding test patients' pre-mixed
     audio from ../../../zenodo_mixed/lambda_<lambda>/ (verified
     byte-identical mixing formula to every other script in this repo --
     see ../README.md) under up to three conditions:
       - no_denoise: raw pre-mixed audio, straight into the model.
       - denoise_wavelet: the SAME audio, run through wavelet_denoise() first
         (Candidate 1 -- zero training, zero leakage risk).
       - denoise_lunet: the SAME audio, run through lunet_denoise() first
         (Candidate 2 -- pretrained model, disclosed ICBHI-2017 training-data
         overlap with our own noise source, see ../README.md).
     This gives a PAIRED, same-fold, same-files comparison of "does
     denoising help the clean model" -- the strongest evidence this
     experiment produces.
  3. External context (already published, full 10-fold power, no new
     training needed for either): reproducibility/results/three_strategies_cv/
     final_results.json's 'clean' (no-denoise clean-only, 10-fold) and
     'noise_0_10' (Variable-Noise U[0,10], 10-fold) entries -- the actual
     "does noise-aware training still beat denoise-then-classify" answer.
     Cross-checked against CLAUDE.md's own published Figure 3 prose
     (sensitivity 45.80%/1.01% at lambda=1.0/5.0 for clean-only) before
     trusting it -- matches to 2 decimal places, see ../README.md.

Fold count is a REQUIRED CLI arg (no default) because this script serves two
distinct purposes, per your 2026-09-13/2026-09-14 instructions:
  1. --n_folds 10 --conditions no_denoise --save_checkpoints: the
     "pure clean run" -- train a genuinely clean-only model and save its
     weights, at 10-fold so it "fits back into the paper properly" (same
     KFold(seed=42) construction as the published Table 2/Figure 3 runs --
     this literally reproduces the original fold assignments, not just the
     same fold COUNT). No checkpoint at this fold count existed anywhere in
     this repo before this script -- see ../README.md. COMPLETE as of
     2026-09-14 -- see ../results/clean_only_10fold/.
  2. --n_folds 10 --load_checkpoint_dir/--gcs_checkpoint_prefix (Step 1's
     checkpoints) --conditions no_denoise,denoise_wavelet,denoise_lunet: the
     actual Reviewer #7 Comment 1 denoiser-vs-variable-noise comparison.
     CORRECTED 2026-09-14 from an earlier "--n_folds 5" plan: your later
     instruction ("nothing from denoising comes through until we get the
     final weights of that clean model so the denoiser run itself doesn't
     take long at all") means Step 2 must REUSE Step 1's exact per-fold
     trained weights, not retrain fresh ones -- which only produces a valid,
     leakage-free evaluation if Step 2's KFold split is IDENTICAL to Step
     1's, i.e. same --n_folds (10) and --seed (42), not a different 5-fold
     split. --load_checkpoint_dir/--gcs_checkpoint_prefix enforce this: the
     checkpoint count is checked against --n_folds and the run errors out on
     a mismatch instead of silently evaluating fold i's checkpoint against
     the wrong fold's held-out patients. denoise_lunet carries a disclosed
     ICBHI-2017 training-data overlap caveat -- see ../README.md.

Usage:
    # Step 1 -- pure clean run, 10-fold, save weights, no denoiser yet:
    python run_denoiser_benchmark_cv.py \\
        --data_dir ../../../data_processed/ --mixed_dir ../../../zenodo_mixed/ \\
        --output_dir ../results/clean_only_10fold/ --n_folds 10 \\
        --conditions no_denoise --save_checkpoints --epochs 5 --seed 42

    # Step 2 -- full denoiser comparison, reusing Step 1's weights (fast --
    # no training), same 10-fold split as Step 1:
    python run_denoiser_benchmark_cv.py \\
        --data_dir ../../../data_processed/ --mixed_dir ../../../zenodo_mixed/ \\
        --output_dir ../results/denoiser_comparison_10fold/ --n_folds 10 \\
        --load_checkpoint_dir ../results/clean_only_10fold/checkpoints/ \\
        --conditions no_denoise,denoise_wavelet,denoise_lunet --seed 42
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
# T-BiLSTM (Candidate 3) was canceled with no trained weights or results (see
# ../README.md); its code lives in attempted_not_used/ and is imported lazily,
# only if --conditions actually requests denoise_tbilstm, so its absence never
# breaks the no_denoise/wavelet/lunet conditions this script actually reports.

# Locate the repo root that contains src/models/ast_qa.py. Three different
# layouts need to resolve here: this file vendored into the public
# reproducibility package at <repo_root>/reproducibility/revision/src/... ,
# where reproducibility/revision/src/models/ast_qa.py is the correct local
# copy to use (parents[2] == reproducibility/revision); a PAPER_REVISIONS
# local checkout, where this file sits at
# <repo_root>/PAPER_REVISIONS/reviewer7_denoiser_benchmark/src/... (parents[3]
# == repo root); and the Docker image, where only this experiment's src/ is
# copied in and the real src/ package is placed at the fixed /app/repo_src
# path by ../Dockerfile (parents[3] doesn't exist there -- this is what
# crashed the first Vertex AI submission with IndexError: 3, since the
# container's src/run_denoiser_benchmark_cv.py has only 2 parents).
# Checked in this order so the deterministic, locally-vendored copy always
# wins over an accidental match from a sibling package -- e.g. parents[3]
# from the reproducibility/revision/ location happens to resolve to
# reproducibility/, which also has its own src/models/ast_qa.py; that would
# still work (both files are the same class) but shouldn't be relied on.
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
from src.models.ast_qa import ASTHeartQA  # noqa: E402 -- the published, unmodified baseline model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unfreeze_ast_qa import ASTHeartQAUnfreeze  # noqa: E402 -- local copy, see reviewer1_unfreezing_ablation/src/unfreeze_ast_qa.py; duplicated rather than cross-imported per this repo's convention of never letting a new revision experiment share mutable code with one that already produced trusted results

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


# --- verbatim block, copied from reproducibility/src/train_per_lambda_cv.py ---
# (same audio pipeline every other method in this revision is evaluated on;
# also byte-identical to what generated zenodo_mixed/ in the first place --
# see reproducibility/src/generate_mixed_datasets.py -- so applying it again
# to raw clean/noise training files here stays consistent with how the
# eval-side pre-mixed files were built)
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


class CleanOnlyTrainDataset(Dataset):
    """Clean-only training set: heart-only positives (lambda=0, i.e. literally
    unmixed clean audio), noise-only negatives -- matches Figure 3's
    'Clean-Only Training' strategy exactly (CLAUDE.md Sec 7.2 / the 'clean'
    branch of reproducibility/src/train_three_strategies_cv.py's
    ThreeStrategyDataset), just without the redundant lambda=0.0 'mixed'
    duplicate that scoring showed is byte-identical to 'clean' anyway
    (mix_rms returns the heart signal unchanged when lam==0)."""

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
    """Evaluation set built from zenodo_mixed/lambda_<lambda>/ -- pre-mixed
    heart+noise files (label=1) restricted to this fold's TEST patients, plus
    a matching-sized set of pure-noise files (label=0, no patient identity so
    no leakage concern in picking any subset). condition='denoise_wavelet' runs
    wavelet_denoise() (Candidate 1, off-the-shelf skimage BayesShrink/
    VisuShrink); condition='denoise_wavelet_leveldep' runs
    wavelet_denoise_level_dependent() (an additional, disclosed ablation --
    OUR OWN extension of Candidate 1's method with per-level rather than
    global noise estimation, added 2026-09-14 after diagnosing why Candidate
    1 is a near-no-op on structured noise; see ../README.md); condition=
    'denoise_lunet' runs lunet_denoise() (Candidate 2 -- see ../README.md for
    its disclosed ICBHI-2017 training-data overlap caveat); condition=
    'denoise_tbilstm' runs tbilstm_denoise() (Candidate 3, added 2026-09-14 --
    our own reimplementation of Jakubec et al. 2025's published architecture,
    trained on our own CirCor+ICBHI data since no pretrained weights are
    published for it; see ../README.md "T-BiLSTM candidate" for the
    leakage caveat and data-adaptation disclosures); condition=
    'no_denoise' feeds the raw audio straight through, unmodified."""

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
            # zenodo_mixed files should already be MAX_LENGTH, but pad/crop
            # defensively rather than assume, since this is external data.
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
    """unfreeze_mode='frozen' (default): original behavior, unchanged --
    ASTHeartQA(freeze_base=True), head-only optimizer. unfreeze_mode='full':
    added for the unfrozen-backbone pivot (see ../../UNFROZEN_PIVOT_FOLLOWUP.md
    point 3) -- trains a SEPARATE clean-only full-unfreeze model (does not
    reuse reviewer1_unfreezing_ablation's full_clean_5fold job, which saves
    no checkpoints at all) so Step 2 has real unfrozen weights to evaluate
    denoiser conditions against, alongside the untouched frozen comparator."""
    criterion = nn.BCEWithLogitsLoss()
    if unfreeze_mode == "frozen":
        model = ASTHeartQA(freeze_base=True).to(device)
        optimizer = torch.optim.AdamW(model.qa_classifier.parameters(), lr=lr)
    elif unfreeze_mode == "full":
        model = ASTHeartQAUnfreeze(unfreeze_mode="full").to(device)
        if grad_checkpointing:
            # use_reentrant=False required, not stylistic -- see
            # reviewer1_unfreezing_ablation/src/train_unfreezing_ablation_cv.py's
            # identical comment and test_unfreeze_ast_qa.py check 9.
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
    """unfreeze_mode='frozen' (default, unchanged): reconstructs from a
    saved qa_classifier-only checkpoint, since the frozen AST encoder is
    identical across every fold. unfreeze_mode='full': the encoder itself
    was fine-tuned differently per fold, so the FULL model state dict must
    be saved and loaded, not just the head -- see the --save_checkpoints
    block below for the corresponding save-side branch."""
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


# --- GCS helpers, copied verbatim from ../../reviewer1_unfreezing_ablation/src/train_unfreezing_ablation_cv.py
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


def main():
    parser = argparse.ArgumentParser(description="Denoise-then-classify vs. variable-noise (Reviewer #7, Comment 1)")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Root dir containing PhysioNet2022/ICBHI2017/ESC-50/UrbanSound8K. "
                              "Required unless --gcs_bucket is set.")
    parser.add_argument("--mixed_dir", type=str, default=None,
                         help="zenodo_mixed/ root (lambda_X/ subdirs + manifest.csv). "
                              "Required unless --gcs_bucket is set.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Local output dir. Defaults to a temp dir when --gcs_bucket is set.")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                         help="If set, download data_prefix/mixed_prefix from GCS before running "
                              "and upload --output_dir to gs://<bucket>/<output_prefix> after "
                              "every fold (crash-safe, same convention as every other Vertex job here).")
    parser.add_argument("--data_prefix", type=str, default="data/")
    parser.add_argument("--mixed_prefix", type=str, default="mixed_dataset/",
                         help="GCS prefix for the pre-mixed zenodo_mixed corpus -- already uploaded "
                              "at gs://ast-heart-quality-revisions/mixed_dataset/ (~20GB), per "
                              "PAPER_REVISIONS/README.md's GCP status section.")
    parser.add_argument("--output_prefix", type=str, default="results/denoiser_benchmark/")
    parser.add_argument("--n_folds", type=int, required=True,
                         help="10 for both Step 1 (the paper-aligned clean-only run) and Step 2 (the "
                              "denoiser-vs-variable-noise comparison) -- Step 2 MUST match Step 1's "
                              "fold count exactly when using --load_checkpoint_dir/"
                              "--gcs_checkpoint_prefix, since it's reusing Step 1's exact per-fold "
                              "trained weights against that same fold's held-out patients. No "
                              "default -- pick deliberately, this number changes what the run means.")
    parser.add_argument("--conditions", type=str, default="no_denoise,denoise_wavelet,denoise_lunet",
                         help="Comma list of eval arms to run: 'no_denoise', 'denoise_wavelet' "
                              "(Candidate 1, off-the-shelf skimage BayesShrink/VisuShrink), "
                              "'denoise_wavelet_leveldep' (additional ablation added 2026-09-14 -- "
                              "OUR OWN level-dependent-noise-estimation extension of Candidate 1, "
                              "not a separately-published algorithm, see ../README.md), "
                              "'denoise_lunet' (Candidate 2 -- disclosed ICBHI-2017 "
                              "training overlap, see ../README.md), 'denoise_tbilstm' (Candidate 3, "
                              "added 2026-09-14 -- our own reimplementation of Jakubec et al. 2025's "
                              "T-BiLSTM, trained on our own data since no pretrained weights are "
                              "published; requires --tbilstm_checkpoint, see ../README.md). Use "
                              "'no_denoise' alone for the clean-only weights/validation run; add the "
                              "denoiser conditions once ready for the actual Reviewer #7 Comment 1 "
                              "comparison.")
    parser.add_argument("--tbilstm_checkpoint", type=str, default=None,
                         help="Local path to a trained tbilstm_model.weights.h5 (see "
                              "train_tbilstm.py). Required when --conditions includes "
                              "denoise_tbilstm, unless --gcs_tbilstm_checkpoint is set instead.")
    parser.add_argument("--gcs_tbilstm_checkpoint", type=str, default=None,
                         help="GCS blob path (e.g. results/denoiser_benchmark/tbilstm_training/"
                              "tbilstm_model.weights.h5) to download before running, when "
                              "--gcs_bucket is set. Single-file download, NOT a prefix -- "
                              "T-BiLSTM has one weights file total, not one per fold, unlike "
                              "--gcs_checkpoint_prefix's per-fold AST checkpoints. Mutually "
                              "exclusive with --tbilstm_checkpoint.")
    parser.add_argument("--save_checkpoints", action="store_true",
                         help="Save each fold's trained clean-only model to "
                              "<output_dir>/checkpoints/fold_<i>.pth -- previously NO script in this "
                              "repo persisted a clean-only checkpoint at all (see ../README.md); this "
                              "is what makes the weights reusable afterward instead of training-and-"
                              "discarding like every other CV script here.")
    parser.add_argument("--load_checkpoint_dir", type=str, default=None,
                         help="Local dir of fold_<i>_qa_classifier.pth files from a prior "
                              "--save_checkpoints run (e.g. Step 1's clean_only_10fold/checkpoints/). "
                              "When set, SKIPS training entirely for every fold and just loads+evaluates "
                              "-- this is what actually makes Step 2 fast, per your instruction that "
                              "'nothing from denoising comes through until we get the final weights of "
                              "that clean model so the denoiser run itself doesn't take long at all.' "
                              "Requires --n_folds to exactly match Step 1's fold count (10) -- reusing "
                              "fold i's checkpoint only means anything if fold i's train/test patient "
                              "split is IDENTICAL between the two runs, which only holds when n_folds, "
                              "seed, and the patient list are all the same. Mutually exclusive with "
                              "--gcs_checkpoint_prefix (pick local-dir OR GCS, not both).")
    parser.add_argument("--gcs_checkpoint_prefix", type=str, default=None,
                         help="GCS prefix of a prior --save_checkpoints run's checkpoints/ dir (e.g. "
                              "results/denoiser_benchmark/clean_only_10fold/checkpoints/), downloaded "
                              "before training starts when --gcs_bucket is set. Same fold-count "
                              "requirement and purpose as --load_checkpoint_dir above -- this is its "
                              "GCS equivalent for Vertex AI jobs.")
    parser.add_argument("--unfreeze_mode", type=str, default="frozen", choices=["frozen", "full"],
                         help="'frozen' (default, unchanged behavior): the original published-"
                              "comparator clean-only model, qa_classifier-only checkpoints. "
                              "'full': added for the unfrozen-backbone pivot -- trains/loads a "
                              "SEPARATE full-unfreeze clean-only model (full state dict "
                              "checkpoints) so the denoiser conditions can also be evaluated "
                              "against the new candidate deployed model, alongside the "
                              "untouched frozen results (not a replacement for them). See "
                              "../../UNFROZEN_PIVOT_FOLLOWUP.md point 3.")
    parser.add_argument("--backbone_lr", type=float, default=5e-5,
                         help="Only used when --unfreeze_mode=full -- learning rate for the "
                              "unfrozen encoder params (lower than --lr, which stays the head's "
                              "rate). Matches reviewer1_unfreezing_ablation's default.")
    parser.add_argument("--grad_checkpointing", action="store_true",
                         help="Only used when --unfreeze_mode=full -- enable gradient "
                              "checkpointing on the encoder to fit backprop through all 12 "
                              "layers into the L4's memory. Matches reviewer1_unfreezing_ablation.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoiser_method", type=str, default="BayesShrink", choices=["BayesShrink", "VisuShrink"])
    parser.add_argument("--denoiser_wavelet", type=str, default="db4", choices=["db4", "sym4"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit_patients", type=int, default=0, help="Smoke-test only")
    parser.add_argument("--lambdas", type=str, default=",".join(str(l) for l in LAMBDAS))
    args = parser.parse_args()

    if not args.gcs_bucket and (not args.data_dir or not args.mixed_dir):
        parser.error("--data_dir and --mixed_dir are required unless --gcs_bucket is set")
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
                # Save ONLY qa_classifier's state dict, not the whole model.
                # The encoder is frozen and IDENTICAL across every fold (always
                # the same pretrained MIT/ast-finetuned-audioset-10-10-0.4593
                # weights, never updated by training) -- saving it per fold
                # would be 347MB x n_folds for zero information gain (10 folds
                # = ~3.5GB, which would have blown this machine's ~3.1GB free
                # disk budget, caught before running this at real scale, not
                # after). qa_classifier is the only thing that actually differs
                # fold-to-fold (~0.4MB each) and is the only thing worth keeping.
                # To reload: ASTHeartQA(freeze_base=True) then
                # model.qa_classifier.load_state_dict(torch.load(ckpt_path)).
                ckpt_path = os.path.join(ckpt_dir, f"fold_{fold + 1}_qa_classifier.pth")
                torch.save(model.qa_classifier.state_dict(), ckpt_path)
                logger.info(f"  Saved checkpoint (qa_classifier only): {ckpt_path}")
            else:
                # unfreeze_mode='full': the encoder is fine-tuned differently
                # per fold, so the full model must be saved (~347MB/fold,
                # 5 folds = ~1.7GB -- budgeted into this job's boot disk).
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
            # Upload after every fold, not just at the end -- same
            # crash-safety convention as every other Vertex job here.
            upload_to_gcs(args.gcs_bucket, output_dir, args.output_prefix.rstrip("/"))

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
