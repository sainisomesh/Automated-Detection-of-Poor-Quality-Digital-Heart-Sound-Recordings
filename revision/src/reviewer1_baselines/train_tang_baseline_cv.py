#!/usr/bin/env python3
"""
Tang et al. (2021) signal-quality baseline, evaluated with the same
patient-level cross-validation and noise-mixing protocol as AST-QA.

The audio pipeline follows `../../../src/train_per_lambda_cv.py`: same dataset
discovery, patient-level folds, composite noise (lung + 0.5 * environmental,
peak-normalized), and RMS-matched mixing

    mixed = heart + lambda * (noise * RMS(heart) / RMS(noise))

Instead of log-Mel spectrograms and an AST backbone, each mixed or noise-only
waveform is reduced to the 10 Tang et al. features (`tang_features.py`) and
classified with an SVM.

Classifier configuration from the authors' MATLAB code,
    fitcsvm(..., 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto')
whose scikit-learn equivalent is
    Pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', probability=True))
('Standardize', true -> StandardScaler; 'KernelScale', 'auto' -> gamma='scale').

Labels: heart recordings mixed with noise at intensity lambda are the positive
class (1, acceptable quality); noise-only signals are the negative class (0).
One negative is synthesized per positive, so the classes are balanced.

Cross-validation uses 5 patient-level folds by default (see ../../README.md).

Usage:
    python train_tang_baseline_cv.py --data_dir ../../../dataset/ \
        --output_dir ../../results/reviewer1_baselines/tang_full/ \
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tang_features import pre_processing as tang_pre_processing, extract_features as tang_extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SR = 16000
DURATION = 10
MAX_LENGTH = TARGET_SR * DURATION
TANG_FS = 1000.0  # sampling rate the Tang et al. features are defined at


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Audio pipeline: the three functions below match ../../../src/train_per_lambda_cv.py
# so this baseline sees the same preprocessing and mixing as AST-QA.
def load_audio(path):
    """Load one recording and apply the shared conditioning pipeline.

    Resamples to 16 kHz mono, removes the DC offset, applies a 20-1000 Hz
    bandpass (2nd-order high-pass, 5th-order low-pass), peak-normalizes, and
    forces a uniform 10 s length by truncation or loop padding (`np.tile`).

    Returns None for unreadable, too-short, or effectively silent files.
    """
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
    """Mix `noise` into `heart` at RMS-matched noise intensity `lam`.

    The noise is first rescaled to the heart sound's RMS energy, so lam = 1
    corresponds to 0 dB SNR and lam = 10 to ten times the cardiac RMS energy.
    The mixture is rescaled if it would otherwise clip beyond +/-1.0.
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


def get_noise(icbhi_files, env_files, rng):
    """Draw one composite noise signal: lung + 0.5 * environmental.

    A respiratory recording (ICBHI 2017) is summed with a half-weighted
    environmental clip (ESC-50 / UrbanSound8K) and peak-normalized. Retries up
    to 10 times if either draw fails to load, then returns silence.
    """
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


def wav_to_tang_features(wav_16k):
    """16 kHz waveform -> 10 Tang et al. features: resample to 1 kHz, apply
    the method's preprocessing, then extract."""
    wav_1k = librosa.resample(wav_16k, orig_sr=TARGET_SR, target_sr=TANG_FS)
    processed = tang_pre_processing(wav_1k, TANG_FS)
    return tang_extract_features(processed, TANG_FS)


def _extract_one(args):
    """Build one training/test sample and return (features, label, filename).

    Module-level so it can be pickled for multiprocessing.Pool. `args` is
    (kind, heart_path, icbhi_files, env_files, lam, seed_idx, is_train,
    train_noise_seed):

    - kind == "mixed": load `heart_path`, mix in composite noise at `lam`,
      label 1.
    - kind == "noise": composite noise only, label 0, filename "noise".

    Noise draws are seeded per sample: test samples use
    `random.Random(42 + seed_idx)`, training samples use
    `random.Random(train_noise_seed + seed_idx)`. `train_noise_seed` is set
    per fold in main(), so training noise is reproducible, differs across
    folds, and does not reuse the test seeds. Returns None if the recording
    cannot be loaded or feature extraction fails; the sample is then dropped.
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
    else:  # noise-only
        rng = random.Random(train_noise_seed + seed_idx) if is_train else random.Random(42 + seed_idx)
        wav = get_noise(icbhi_files, env_files, rng)
        label = 0
        filename = "noise"

    if wav is None:
        wav = np.zeros(MAX_LENGTH)
    try:
        feats = wav_to_tang_features(wav)
    except Exception as e:
        logger.warning(f"Feature extraction failed for {filename}: {e}")
        return None
    return feats, label, filename


def build_feature_set(heart_files, icbhi_files, env_files, lam, is_train, n_jobs, train_noise_seed=0):
    """Build the balanced feature matrix for one fold and noise level.

    Each heart recording gives one noise-mixed positive, and an equal number
    of noise-only negatives is generated, as in the AST-QA dataset class.
    `seed_idx` runs 0..N-1 for positives and N..2N-1 for negatives so no two
    samples share a noise draw.

    Returns
    (X, y, filenames)
        Feature matrix, labels, and the source path per row ("noise" for
        negatives).
    """
    tasks = []
    for i, hf in enumerate(heart_files):
        tasks.append(("mixed", hf, icbhi_files, env_files, lam, i, is_train, train_noise_seed))
    n_noise = len(heart_files)
    for i in range(n_noise):
        tasks.append(("noise", None, icbhi_files, env_files, lam, len(heart_files) + i, is_train,
                       train_noise_seed))

    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = pool.map(_extract_one, tasks)
    else:
        results = [_extract_one(t) for t in tasks]

    results = [r for r in results if r is not None]
    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    filenames = [r[2] for r in results]
    return X, y, filenames


def compute_metrics(y_true, probs, threshold=0.5):
    """Per-fold AUROC, AUPRC, accuracy, F1, sensitivity, specificity and
    confusion-matrix counts. Thresholded metrics use `threshold`. AUROC falls
    back to 0.5 (AUPRC to 0.0) if only one class is present.
    """
    preds = (probs > threshold).astype(int)
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main():
    parser = argparse.ArgumentParser(description="Tang et al. baseline, AST-QA-identical CV protocol")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5,
                         help="Number of patient-level CV folds (see ../../README.md)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,1,5,10,25,50,75,100",
                         help="Comma-separated noise intensities to sweep")
    parser.add_argument("--n_jobs", type=int, default=1,
                         help="Parallel worker processes for feature extraction")
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

    # Group by patient (filenames are {patient_id}_{valve}.wav, e.g.
    # 13918_AV.wav -> 13918) so all of a patient's recordings share a fold.
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
        logger.info(f"=== Starting {args.n_folds}-Fold CV for Lambda={l_val} (Tang et al. baseline) ===")
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        lambda_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
            tr_hearts = [f for i in train_idx for f in patient_map[pids[i]]]
            te_hearts = [f for i in test_idx for f in patient_map[pids[i]]]

            # Noise corpora are split 80/20 per fold so test noise clips are
            # never seen in training.
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

            X_train, y_train, _ = build_feature_set(tr_hearts, tr_icbhi, tr_env, l_val, is_train=True, n_jobs=args.n_jobs,
                                            train_noise_seed=1_000_000 + args.seed * 100_000 + fold * 10_000)
            X_test, y_test, filenames_test = build_feature_set(te_hearts, te_icbhi, te_env, l_val, is_train=False, n_jobs=args.n_jobs)

            # Equivalent of the authors' MATLAB call:
            # fitcsvm(..., 'Standardize', true, 'KernelFunction', 'RBF',
            #         'KernelScale', 'auto')
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", gamma="scale", probability=True, random_state=args.seed)),
            ])
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, probs)
            lambda_metrics.append(metrics)
            logger.info(f"  Fold {fold + 1}: AUROC={metrics['auroc']:.4f} F1={metrics['f1']:.4f}")

            # Raw per-sample predictions, used to recompute metrics and by the
            # paired significance script.
            preds_dir = os.path.join(args.output_dir, "raw_predictions", f"lambda_{l_val}", f"fold_{fold + 1}")
            os.makedirs(preds_dir, exist_ok=True)
            pd.DataFrame({
                "filename": filenames_test, "y_true": y_test, "probs": probs,
            }).to_csv(os.path.join(preds_dir, "predictions.csv"), index=False)

        # Normal-approximation 95% CI half-widths over folds (1.96 * SD / sqrt(n_folds))
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

        with open(os.path.join(args.output_dir, "tang_baseline_progress.json"), 'w') as f:
            json.dump(final_results, f, indent=2)

    df = pd.DataFrame.from_dict(final_results, orient='index')
    df.index.name = "Lambda"
    df.to_csv(os.path.join(args.output_dir, "tang_baseline_metrics.csv"))
    logger.info("Done! Tang et al. baseline CV results saved.")


if __name__ == "__main__":
    main()
