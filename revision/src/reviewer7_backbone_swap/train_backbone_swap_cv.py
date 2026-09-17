#!/usr/bin/env python3
"""
Backbone-swap experiment (Reviewer #7, Comment 2): PANNs CNN14 / YAMNet /
HuBERT, each with the identical trainable binary QA head from
backbone_qa_model.py, trained with the Variable-Noise (lambda ~ U[0, 10])
strategy and evaluated across the full lambda sweep. Only the variable-noise
strategy is run per backbone; the clean-only and fixed-noise strategies are
covered for AST in the published Figure 3 and are not re-run here.

The pipeline is assembled from the two training scripts of the original
package so that the audio handling is identical to the published results:
  - the 'noise_0_10' branch of ../../../src/train_three_strategies_cv.py for
    the TRAIN split (per heart recording: one clean and one mixed sample at a
    lambda drawn uniformly from [0, 10], balanced with noise-only negatives);
  - the fixed-test-lambda PerLambdaDataset of
    ../../../src/train_per_lambda_cv.py for the EVAL split, evaluated once
    per lambda in the sweep.
load_audio / mix_rms / get_noise are copied verbatim from
train_per_lambda_cv.py, so every method compared in this package sees exactly
the same waveforms. The single substitution is the encoder: instead of an
ASTFeatureExtractor mel-spectrogram feeding the AST transformer, the
preprocessed waveform goes directly into the backbone selected by --backbone,
which applies its own front-end internally (see backbones.py).

Patient-level grouping: heart recordings are named {patient_id}_{valve}.wav,
so the patient id is the filename prefix before the first underscore. Folds
are formed over unique patient ids, never over individual files, so every
recording from a patient falls entirely within either the training or the
test side of a fold.

Output layout matches the Tang/Giordano baselines for direct comparability:
  raw_predictions/lambda_<L>/fold_<K>/predictions.csv  (filename, y_true, probs)
  backbone_swap_progress.json      (incremental, rewritten after each fold)
  backbone_swap_final_results.json (per-lambda mean and 95% CI across folds)
  backbone_swap_metrics_<backbone>[_<mode>].csv

Training is GPU-heavy; --unfreeze_mode full in particular backpropagates
through the whole backbone. --limit_patients, a small --n_folds and --epochs 1
give a fast CPU smoke test. The --gcs_bucket / --data_prefix / --output_prefix
flags drive the cloud pipeline the reported results were produced on and
default to off for local reproduction.

See ../../README.md for the fold counts and other settings used for the
results checked into this package.

Usage:
    python train_backbone_swap_cv.py --backbone panns --unfreeze_mode frozen \
        --data_dir ../../../dataset/ \
        --output_dir ../../results/reviewer7_backbone_swap/panns/ \
        --n_folds 3 --epochs 5 --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys
import tempfile
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_qa_model import BackboneQAHead

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


# --- Verbatim block, copied from ../../../src/train_per_lambda_cv.py. ---
# This is the audio pipeline AST-QA, Tang, and Giordano are all evaluated on;
# it must stay byte-identical across scripts so that the only difference
# between methods is the model, not the waveforms.
def load_audio(path):
    """Load one recording and apply the shared preprocessing chain.

    Resamples to 16 kHz mono, removes the DC offset, applies a 20 Hz
    second-order high-pass and a 1 kHz fifth-order low-pass (Butterworth,
    SOS form), peak-normalizes to [-1, 1], then crops or loop-pads with
    np.tile to exactly 10 s (160000 samples).

    Returns None for unreadable, near-empty, or effectively silent files.
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
    """Mix a heart sound with noise at intensity lambda, RMS energy-matched.

    Implements Eq. 2 of the paper: the noise is rescaled so its RMS matches
    the heart sound's before being added at weight lambda, so lambda = 1
    corresponds to 0 dB SNR regardless of the absolute level of either
    recording. The result is rescaled if it would otherwise clip beyond 1.0.
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
# --- end verbatim block ---


def get_noise(icbhi_files, env_files, idx=None):
    """Build one composite structured-noise waveform (Eq. 1 of the paper):
    an ICBHI lung recording plus 0.5x an ESC-50/UrbanSound8K environmental
    clip, peak-normalized. Same formula as
    PerLambdaDataset.get_noise in train_per_lambda_cv.py.

    idx: when given, the two source clips are chosen from a per-item seeded
    RNG (random.Random(42 + idx)), which makes the evaluation sets
    deterministic and identical across backbones and across runs. When None
    (used for training samples) the global RNG is used, so the noise drawn
    varies from epoch to epoch as intended for augmentation.

    Returns silence if either source clip fails to load.
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


class VariableNoiseTrainDataset(Dataset):
    """Variable-noise (lambda ~ U[0, 10]) training set.

    Same construction as the 'noise_0_10' branch of ThreeStrategyDataset in
    train_three_strategies_cv.py: each heart recording contributes one clean
    and one noise-mixed positive sample, and an equal number of noise-only
    negatives is added, giving a balanced 50/50 label distribution. The
    mixing lambda is redrawn from U[0, 10] on every __getitem__ call, so a
    recording is seen at many noise levels over the course of training.

    The only difference from the AST version is that this returns the raw
    waveform instead of AST feature-extractor output, leaving each backbone
    to apply its own front-end.
    """

    def __init__(self, heart_files, icbhi_files, env_files):
        self.icbhi_files = icbhi_files
        self.env_files = env_files
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
                lam = random.uniform(0, 10)
                wav = mix_rms(h, n, lam)
        if wav is None:
            wav = np.zeros(MAX_LENGTH)
        return {
            "wav": torch.tensor(wav, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.float),
        }


class FixedLambdaEvalDataset(Dataset):
    """Evaluation set at a single fixed test lambda.

    Same construction as PerLambdaDataset in train_per_lambda_cv.py, minus
    the AST feature-extractor step. Length is 2x the number of heart
    recordings: indices below len(self.data) are heart recordings mixed at
    `lambda_val` (label 1), and indices at or above it are noise-only
    negatives (label 0), keeping the test set balanced. Noise selection is
    seeded per index, so the same waveforms are used for every backbone.

    The emitted `filename` is the source path for positives and the literal
    string "noise" for negatives; the significance scripts rely on this
    column to align predictions row-for-row across methods.
    """

    def __init__(self, heart_files, icbhi_files, env_files, lambda_val):
        self.heart_files = heart_files
        self.icbhi_files = icbhi_files
        self.env_files = env_files
        self.lambda_val = lambda_val
        self.data = [(f, 1) for f in heart_files]

    def __len__(self):
        return len(self.data) * 2  # balanced: heart+noise and noise-only

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
        return {
            "wav": torch.tensor(wav, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.float),
            "filename": filename,
        }


# --- Cloud-storage helpers, used only when --gcs_bucket is given.
# The cloud training jobs stage data with explicit downloads/uploads through
# the google-cloud-storage client rather than a FUSE mount, which keeps I/O
# predictable on preemptible instances. Local reproduction does not use these.
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
# --- end cloud-storage helpers ---


def build_optimizer(model, unfreeze_mode, head_lr, backbone_lr):
    """Build the AdamW optimizer for the requested unfreeze mode.

    In 'frozen' mode only the head's parameters are passed to the optimizer.
    In 'full' mode two parameter groups are used, with the head at head_lr
    and the backbone at the lower backbone_lr; a reduced backbone learning
    rate is the standard precaution against catastrophic forgetting when
    fine-tuning a large pretrained encoder on a small dataset. This mirrors
    build_optimizer in
    ../reviewer1_unfreezing_ablation/train_unfreezing_ablation_cv.py so that
    the AST and backbone-swap fine-tuning runs use the same schedule.
    """
    if unfreeze_mode == "frozen":
        return torch.optim.AdamW(model.qa_classifier.parameters(), lr=head_lr)
    return torch.optim.AdamW([
        {"params": model.qa_classifier.parameters(), "lr": head_lr},
        {"params": model.backbone.parameters(), "lr": backbone_lr},
    ])


def train_model(backbone_name, unfreeze_mode, train_loader, fold, device, epochs=5,
                 head_lr=1e-4, backbone_lr=5e-5):
    """Train one fold from scratch and return the fitted model.

    A fresh BackboneQAHead is built per fold so that no information leaks
    between folds. Loss is BCEWithLogits on the balanced usable/noise labels.
    """
    logger.info(f"  --> [Train] backbone={backbone_name} mode={unfreeze_mode}, fold={fold}")
    model = BackboneQAHead(backbone_name, freeze_backbone=(unfreeze_mode == "frozen")).to(device)
    optimizer = build_optimizer(model, unfreeze_mode, head_lr, backbone_lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{backbone_name} F{fold} E{epoch + 1}"):
            wav = batch["wav"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)
            logits = model(wav)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"  {backbone_name} | Fold {fold} | Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f}")
    return model


def compute_metrics(y_true, probs):
    """Compute the six metrics reported throughout the paper.

    Threshold-free: AUROC and AUPRC. At the fixed 0.5 decision threshold:
    accuracy, F1, sensitivity (recall for usable recordings) and specificity
    (recall for noise). AUROC/AUPRC fall back to chance-level values if a
    subset happens to contain a single class, which sklearn treats as
    undefined.
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


def evaluate_fold(model, hearts, icbhi, env, device, batch_size, output_dir, fold, lambdas):
    """Evaluate one trained fold model across the whole lambda sweep.

    One fixed-lambda test set is built per lambda from the same held-out
    patients, so a single model is scored at every noise level. Per-recording
    predictions are written to
    raw_predictions/lambda_<L>/fold_<K>/predictions.csv before aggregation,
    so that every reported number can be recomputed from raw output.
    Returns a dict mapping lambda to its metric dict.
    """
    per_lambda_metrics = {}
    for lam in lambdas:
        ds = FixedLambdaEvalDataset(hearts, icbhi, env, lam)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        model.eval()
        y_true, y_probs, filenames = [], [], []
        with torch.no_grad():
            for batch in loader:
                wav = batch["wav"].to(device)
                logits = model(wav)
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
    parser = argparse.ArgumentParser(description="Backbone-swap experiment (Reviewer #7, Comment 2)")
    parser.add_argument("--backbone", type=str, required=True, choices=["panns", "yamnet", "hubert"])
    parser.add_argument("--unfreeze_mode", type=str, default="frozen", choices=["frozen", "full"],
                         help="frozen: train only the QA head on fixed backbone features. "
                              "full: fine-tune the backbone end to end as well, at --backbone_lr.")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Local data dir. Required unless --gcs_bucket is set.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Local output dir. Defaults to a temp dir when --gcs_bucket is set.")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                         help="If set, download data from gs://<bucket>/<data_prefix> before "
                              "training and upload --output_dir to gs://<bucket>/<output_prefix> after.")
    parser.add_argument("--data_prefix", type=str, default="data/")
    parser.add_argument("--output_prefix", type=str, default=None,
                         help="Defaults to results/backbone_swap/<backbone>/")
    parser.add_argument("--n_folds", type=int, default=5,
                         help="Number of patient-level CV folds. See ../../README.md for the "
                              "fold count used for the results checked into this package.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5,
                         help="LR for the backbone params when --unfreeze_mode full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (default: auto-detect cuda>cpu)")
    parser.add_argument("--limit_patients", type=int, default=0,
                         help="If >0, only use this many patients (for smoke-testing)")
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,1,5,10,25,50,75,100",
                         help="Comma-separated lambda sweep (override with a short list for smoke-testing)")
    args = parser.parse_args()

    if not args.gcs_bucket and not args.data_dir:
        parser.error("--data_dir is required unless --gcs_bucket is set")

    set_seed(args.seed)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    mode_suffix = "" if args.unfreeze_mode == "frozen" else f"_{args.unfreeze_mode}"
    output_prefix = args.output_prefix or f"results/backbone_swap/{args.backbone}{mode_suffix}/"
    if args.gcs_bucket:
        work_dir = tempfile.mkdtemp(prefix=f"backbone_swap_{args.backbone}_")
        data_dir = os.path.join(work_dir, "data")
        output_dir = os.path.join(work_dir, "output")
        download_from_gcs(args.gcs_bucket, args.data_prefix, data_dir)
    else:
        data_dir = args.data_dir
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # File lists are sorted so that the fold split and the per-index noise
    # seeding are reproducible independently of filesystem traversal order.
    data_root = Path(data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))
    icbhi_files = sorted(list(data_root.rglob("ICBHI2017/**/*.wav")))
    env_files = sorted(list(data_root.rglob("ESC-50/**/*.wav")) + list(data_root.rglob("UrbanSound8K/**/*.wav")))
    logger.info(f"Found {len(heart_files)} heart, {len(icbhi_files)} lung, {len(env_files)} environmental files")
    if not heart_files:
        raise ValueError(f"No heart audio files found in {data_dir}")

    # Group recordings by patient id -- the filename prefix before the first
    # underscore, e.g. 13918_AV.wav -> 13918. Cross-validation is performed
    # over these ids, never over individual files, which is what prevents
    # recordings from the same patient appearing in both train and test.
    patient_map = {}
    for f in heart_files:
        pid = f.name.split("_")[0]
        patient_map.setdefault(pid, []).append(f)
    pids = sorted(list(patient_map.keys()))

    if args.limit_patients > 0:
        pids = pids[:args.limit_patients]
        logger.info(f"[smoke-test] limiting to {len(pids)} patients")
    logger.info(f"Found {len(pids)} unique patients")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    all_fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        logger.info(f"=== FOLD {fold + 1}/{args.n_folds} (backbone={args.backbone}) ===")
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train_hearts = [f for p in train_pids for f in patient_map[p]]
        test_hearts = [f for p in test_pids for f in patient_map[p]]

        # The noise corpora are split per fold as well (80% train / 20% test),
        # so the interfering lung and environmental recordings used to build
        # the test set are never the ones seen during training. The shuffle is
        # seeded per fold for reproducibility, and matches the construction
        # used by the AST unfreezing ablation so the two experiments' test
        # sets line up recording for recording.
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

        train_ds = VariableNoiseTrainDataset(train_hearts, tr_icbhi, tr_env)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

        model = train_model(args.backbone, args.unfreeze_mode, train_loader, fold + 1, device,
                             epochs=args.epochs, head_lr=args.head_lr, backbone_lr=args.backbone_lr)
        fold_metrics = evaluate_fold(model, test_hearts, te_icbhi, te_env, device, args.batch_size,
                                     output_dir, fold + 1, lambdas)
        all_fold_metrics.append(fold_metrics)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with open(os.path.join(output_dir, "backbone_swap_progress.json"), "w") as f:
            json.dump({"backbone": args.backbone, "unfreeze_mode": args.unfreeze_mode,
                       "folds_done": fold + 1, "per_fold": all_fold_metrics}, f, indent=2)

        if args.gcs_bucket:
            # Upload after every fold rather than only at the end: on
            # preemptible instances a job killed mid-run would otherwise lose
            # the raw predictions of the folds it had already finished.
            upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))

    # Aggregate across folds: per-lambda mean of each metric with a
    # normal-approximation 95% half-width (1.96 * SD / sqrt(n_folds)).
    final_results = {}
    for lam in lambdas:
        metrics = [fold[lam] for fold in all_fold_metrics]
        means = {m: float(np.mean([f[m] for f in metrics])) for m in metrics[0]}
        cis = {m: float(1.96 * np.std([f[m] for f in metrics]) / np.sqrt(args.n_folds)) for m in metrics[0]}
        final_results[lam] = {"mean": means, "ci": cis}

    with open(os.path.join(output_dir, "backbone_swap_final_results.json"), "w") as f:
        json.dump({"backbone": args.backbone, "unfreeze_mode": args.unfreeze_mode,
                   "n_folds": args.n_folds, "results": final_results}, f, indent=2)
    df = pd.DataFrame({lam: final_results[lam]["mean"] for lam in lambdas}).T
    df.index.name = "Lambda"
    df.to_csv(os.path.join(output_dir, f"backbone_swap_metrics_{args.backbone}{mode_suffix}.csv"))
    logger.info(f"Done! {args.backbone} ({args.unfreeze_mode}) backbone-swap CV results saved to {output_dir}")

    if args.gcs_bucket:
        upload_to_gcs(args.gcs_bucket, output_dir, output_prefix.rstrip("/"))


if __name__ == "__main__":
    main()
