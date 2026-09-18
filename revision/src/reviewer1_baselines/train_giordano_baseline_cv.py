#!/usr/bin/env python3
"""
Giordano, Rosati & Knaflitz (2021) SNR-based PCG quality baseline, evaluated
under the same patient-level cross-validation and noise-mixing protocol as
AST-QA and the Tang et al. baseline. The shared audio pipeline is documented
in train_tang_baseline_cv.py and cloned here.

Unlike Tang et al. and AST-QA, the original method is not a trained
classifier: it is a single SNR formula compared against a fixed threshold.
It is fitted into the same train/test discipline as follows:

- The per-recording SNR score (`giordano_snr.compute_snr_db`) is computed
  identically for training and test recordings.
- The decision threshold used for accuracy, F1, sensitivity, and specificity
  is selected on the training fold only, by maximizing F1, and then applied
  unchanged to the held-out fold. Test labels never influence the threshold.
- AUROC and AUPRC require no threshold and are computed directly from the raw
  SNR score, which is the most faithful evaluation of a pure scoring rule.

See giordano_snr.py for the documented deviation in cardiac-cycle
segmentation: the original method uses R-peaks from a synchronized ECG, which
these PCG-only datasets do not provide.

Usage:
    python train_giordano_baseline_cv.py --data_dir ../../../dataset/ \\
        --output_dir ../../results/reviewer1_baselines/giordano_full/ \\
        --n_folds 5 --seed 42
"""

import argparse
import os
import sys
import random
import json
import logging
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from giordano_snr import compute_snr_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SR = 16000
MAX_LENGTH = TARGET_SR * 10
GIORDANO_FS = 1000.0  # sampling rate the SNR method is defined at


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ── Shared audio pipeline ───────────────────────────────────────────────
# Mirrors train_tang_baseline_cv.py and ../../../src/train_per_lambda_cv.py so
# that all three methods see exactly the same audio; see
# train_tang_baseline_cv.py for the per-function documentation. Any change
# here must be mirrored in those scripts, and vice versa.
def load_audio(path):
    try:
        wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    except Exception:
        return None
    if len(wav) < 100 or np.max(np.abs(wav)) < 1e-6:
        return None
    wav = wav - np.mean(wav)
    from scipy.signal import butter, sosfilt
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


def get_noise(icbhi_files, env_files, rng):
    for _ in range(10):
        lung = load_audio(rng.choice(icbhi_files))
        env = load_audio(rng.choice(env_files))
        if lung is not None and env is not None:
            combined = lung + 0.5 * env
            peak = np.max(np.abs(combined))
            if peak > 0:
                combined = combined / peak
            return combined
    return np.zeros(MAX_LENGTH)
# ── End shared audio pipeline ───────────────────────────────────────────


def _score_one(args):
    """Build one sample and return (snr_db, label, filename).

    Counterpart of train_tang_baseline_cv.py's `_extract_one`, with the SNR
    score in place of the feature vector: "mixed" samples are a heart
    recording plus composite noise at intensity `lam` (label 1) and "noise"
    samples are noise only (label 0). Evaluation samples use the per-sample
    seed `random.Random(42 + seed_idx)` for a reproducible noise draw;
    training samples draw unseeded. Returns None if the recording cannot be
    loaded or scored.
    """
    kind, heart_path, icbhi_files, env_files, lam, seed_idx, is_train, train_noise_seed = args
    if kind == "mixed":
        h = load_audio(heart_path)
        if h is None:
            return None
        rng = random.Random(train_noise_seed + seed_idx) if is_train else random.Random(42 + seed_idx)
        n = get_noise(icbhi_files, env_files, rng)
        wav = mix_rms(h, n, lam)
        label = 1
        filename = str(heart_path)
    else:
        rng = random.Random(train_noise_seed + seed_idx) if is_train else random.Random(42 + seed_idx)
        wav = get_noise(icbhi_files, env_files, rng)
        label = 0
        filename = "noise"

    if wav is None:
        wav = np.zeros(MAX_LENGTH)
    try:
        wav_1k = librosa.resample(wav, orig_sr=TARGET_SR, target_sr=GIORDANO_FS)
        score = compute_snr_db(wav_1k, GIORDANO_FS)
    except Exception as e:
        logger.warning(f"SNR scoring failed for {filename}: {e}")
        return None
    return score, label, filename


def build_score_set(heart_files, icbhi_files, env_files, lam, is_train, n_jobs, train_noise_seed=0):
    """Score one fold's samples: one noise-mixed positive per heart recording
    plus an equal number of noise-only negatives. `seed_idx` runs 0..N-1 over
    the positives and N..2N-1 over the negatives so no two evaluation samples
    share a noise draw. Returns (scores, labels, filenames).
    """
    tasks = []
    for i, hf in enumerate(heart_files):
        tasks.append(("mixed", hf, icbhi_files, env_files, lam, i, is_train, train_noise_seed))
    for i in range(len(heart_files)):
        tasks.append(("noise", None, icbhi_files, env_files, lam, len(heart_files) + i, is_train,
                       train_noise_seed))

    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = pool.map(_score_one, tasks)
    else:
        results = [_score_one(t) for t in tasks]

    results = [r for r in results if r is not None]
    scores = np.array([r[0] for r in results])
    labels = np.array([r[1] for r in results])
    filenames = [r[2] for r in results]
    return scores, labels, filenames


def best_threshold_for_f1(scores, labels, max_candidates=200):
    """Select the SNR decision threshold that maximizes F1 on the training set.

    Scores above the threshold are predicted acceptable (label 1), following
    the method's premise that added noise lowers SNR.

    Parameters
    ----------
    max_candidates : int
        Upper bound on thresholds evaluated. If the training scores have more
        than this many distinct values, candidates are taken as evenly spaced
        percentiles of the score distribution instead of every unique value,
        which keeps the search linear in the number of training samples.

    Non-finite scores are excluded from the candidate set; 0.0 is returned if
    no score is finite.
    """
    finite = np.isfinite(scores)
    if not np.any(finite):
        return 0.0
    finite_scores = scores[finite]
    n_unique = len(np.unique(finite_scores))
    if n_unique <= max_candidates:
        candidates = np.unique(finite_scores)
    else:
        candidates = np.percentile(finite_scores, np.linspace(0, 100, max_candidates))

    best_f1, best_t = -1.0, candidates[0] if len(candidates) else 0.0
    for t in candidates:
        preds = (scores > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def scores_to_pseudo_prob(scores, threshold, scale=5.0):
    """Map SNR values in dB onto [0, 1] for output-format parity with the
    other methods' `probs` column: a logistic centered on the train-derived
    threshold, with `scale` setting its width in dB. Monotone in the raw
    score, so it preserves ranking. The reported AUROC/AUPRC are computed
    from the raw SNR values, not from this transform.
    """
    return 1.0 / (1.0 + np.exp(-(scores - threshold) / scale))


def compute_metrics(y_true, raw_scores, threshold):
    """Evaluate one fold from raw SNR scores and the train-derived threshold.

    AUROC and AUPRC are ranking metrics computed on the raw scores; accuracy,
    F1, sensitivity, specificity, and the confusion-matrix counts use the
    thresholded predictions. AUROC falls back to 0.5 (AUPRC to 0.0) if only
    one class is present.
    """
    preds = (raw_scores > threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true, raw_scores)
    except ValueError:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, raw_scores)
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main():
    parser = argparse.ArgumentParser(description="Giordano SNR baseline, AST-QA-identical CV protocol")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5,
                         help="Number of patient-level CV folds (see ../../README.md)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,1,5,10,25,50,75,100",
                         help="Comma-separated noise intensities to sweep")
    parser.add_argument("--n_jobs", type=int, default=1,
                         help="Parallel worker processes for SNR scoring")
    parser.add_argument("--limit_patients", type=int, default=0,
                         help="If >0, only use this many patients (for smoke-testing)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    data_root = Path(args.data_dir)
    heart_files = sorted(list(data_root.rglob("PhysioNet2022/**/*.wav")))
    icbhi_files = sorted(list(data_root.rglob("ICBHI2017/**/*.wav")))
    env_files = sorted(list(data_root.rglob("ESC-50/**/*.wav")) + list(data_root.rglob("UrbanSound8K/**/*.wav")))

    logger.info(f"Found {len(heart_files)} heart, {len(icbhi_files)} lung, {len(env_files)} environmental files")
    if not heart_files:
        raise ValueError(f"No heart audio files found in {args.data_dir}")

    # Group recordings by patient so folds can be split at the patient level:
    # filenames are {patient_id}_{valve}.wav, e.g. 13918_AV.wav -> 13918.
    patient_map = {}
    for f in heart_files:
        pid = f.name.split('_')[0]
        patient_map.setdefault(pid, []).append(f)
    pids = sorted(list(patient_map.keys()))

    if args.limit_patients > 0:
        pids = pids[:args.limit_patients]
        logger.info(f"[smoke-test] limiting to {len(pids)} patients")

    logger.info(f"Found {len(pids)} unique patients")

    final_results = {}

    for l_val in lambdas:
        logger.info(f"=== Starting {args.n_folds}-Fold CV for Lambda={l_val} (Giordano SNR baseline) ===")
        # Folds are formed over patient IDs, not files.
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        lambda_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
            tr_hearts = [f for i in train_idx for f in patient_map[pids[i]]]
            te_hearts = [f for i in test_idx for f in patient_map[pids[i]]]

            # The noise corpora are also split 80/20 per fold, so evaluation
            # noise comes from clips never used during training.
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

            train_scores, train_labels, _ = build_score_set(tr_hearts, tr_icbhi, tr_env, l_val, is_train=True, n_jobs=args.n_jobs,
                                            train_noise_seed=1_000_000 + args.seed * 100_000 + fold * 10_000)
            test_scores, test_labels, test_filenames = build_score_set(te_hearts, te_icbhi, te_env, l_val, is_train=False, n_jobs=args.n_jobs)

            # Threshold fitted on the training fold only, then frozen.
            threshold = best_threshold_for_f1(train_scores, train_labels)
            metrics = compute_metrics(test_labels, test_scores, threshold)
            metrics["learned_threshold_db"] = threshold
            lambda_metrics.append(metrics)
            logger.info(f"  Fold {fold + 1}: AUROC={metrics['auroc']:.4f} F1={metrics['f1']:.4f} thresh={threshold:.2f}dB")

            # Raw per-sample held-out predictions, saved alongside the
            # aggregated metrics. Both the raw SNR in dB and its bounded
            # transform are written; downstream comparisons use `probs`.
            preds_dir = os.path.join(args.output_dir, "raw_predictions", f"lambda_{l_val}", f"fold_{fold + 1}")
            os.makedirs(preds_dir, exist_ok=True)
            pseudo_prob = scores_to_pseudo_prob(test_scores, threshold)
            pd.DataFrame({
                "filename": test_filenames, "y_true": test_labels,
                "snr_db": test_scores, "probs": pseudo_prob,
            }).to_csv(os.path.join(preds_dir, "predictions.csv"), index=False)

        # Fold-level aggregation. The reported half-widths are normal-
        # approximation 95% intervals (1.96 * SD / sqrt(n_folds)) over folds.
        avg_auroc = float(np.mean([m['auroc'] for m in lambda_metrics]))
        ci_auroc = float(1.96 * np.std([m['auroc'] for m in lambda_metrics]) / np.sqrt(args.n_folds))
        avg_f1 = float(np.mean([m['f1'] for m in lambda_metrics]))
        ci_f1 = float(1.96 * np.std([m['f1'] for m in lambda_metrics]) / np.sqrt(args.n_folds))
        avg_auprc = float(np.mean([m['auprc'] for m in lambda_metrics]))
        avg_acc = float(np.mean([m['accuracy'] for m in lambda_metrics]))
        avg_sens = float(np.mean([m['sensitivity'] for m in lambda_metrics]))
        avg_spec = float(np.mean([m['specificity'] for m in lambda_metrics]))

        final_results[l_val] = {
            "mean": {"auroc": avg_auroc, "f1": avg_f1, "auprc": avg_auprc,
                     "accuracy": avg_acc, "sensitivity": avg_sens, "specificity": avg_spec},
            "ci": {"auroc": ci_auroc, "f1": ci_f1},
            "per_fold": lambda_metrics,
        }

        with open(os.path.join(args.output_dir, "giordano_baseline_progress.json"), 'w') as f:
            json.dump(final_results, f, indent=2)

    df = pd.DataFrame.from_dict(final_results, orient='index')
    df.index.name = "Lambda"
    df.to_csv(os.path.join(args.output_dir, "giordano_baseline_metrics.csv"))
    logger.info("Done! Giordano SNR baseline CV results saved.")


if __name__ == "__main__":
    main()
