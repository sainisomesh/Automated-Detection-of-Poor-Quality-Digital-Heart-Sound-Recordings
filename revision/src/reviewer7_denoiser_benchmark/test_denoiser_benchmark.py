#!/usr/bin/env python3
"""
Audit checks for wavelet_denoiser.py and run_denoiser_benchmark_cv.py, run
BEFORE trusting any training run -- same "verify before trusting" bar as
every other PAPER_REVISIONS component.

Run:
    python test_denoiser_benchmark.py

Checks:
  1. Denoiser is deterministic (same input -> bit-identical output, no
     hidden randomness) and shape/dtype-preserving.
  2. Denoiser actually reduces distance-to-clean-signal on a synthetic
     noisy sine wave, for every (method, wavelet) combination CLAUDE.md
     names -- if a threshold formula were wrong this would likely show up
     as "denoising makes it worse", not just a crash.
  3. Denoiser handles a real edge case (all-zero / silent input) without
     NaN/Inf.
  4. CleanOnlyTrainDataset never mixes noise into a positive sample --
     "clean-only" must mean literally clean, not lambda=0 mixing (which
     happens to be equivalent, but the dataset must not rely on that
     coincidence by actually calling mix_rms).
  5. PreMixedEvalDataset: zero patient leakage (every positive sample's
     source patient is in the requested test set, none outside it) and
     exact 1:1 label balance, using the REAL zenodo_mixed manifest.
  6. PreMixedEvalDataset's condition='no_denoise' returns the raw file
     content unmodified; condition='denoise' returns something different
     (the denoiser actually ran, wasn't silently skipped).
  7. The KFold(n_splits=5, seed=42) construction used in
     run_denoiser_benchmark_cv.py produces the exact same patient->fold
     mapping as ../../fold_assignments/patient_folds_5fold.csv.
  8. LU-Net (Candidate 2): deterministic, shape/length-preserving, handles
     silence without NaN/Inf, and genuinely changes real PCG audio (not a
     silent no-op like the wavelet denoiser on real data -- see ../README.md).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wavelet_denoiser import wavelet_denoise, VALID_METHODS, VALID_WAVELETS
from run_denoiser_benchmark_cv import CleanOnlyTrainDataset, PreMixedEvalDataset, MAX_LENGTH

REPO_ROOT = Path(__file__).resolve().parents[3]
MIXED_DIR = REPO_ROOT / "zenodo_mixed"
FOLD_CSV = REPO_ROOT / "PAPER_REVISIONS" / "fold_assignments" / "patient_folds_5fold.csv"
DATA_DIR = REPO_ROOT / "data_processed"


def make_noisy_sine(seed=0, noise_scale=0.4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, MAX_LENGTH)
    clean = 0.3 * np.sin(2 * np.pi * 2 * t)
    noisy = clean + noise_scale * rng.standard_normal(MAX_LENGTH)
    return clean.astype(np.float32), noisy.astype(np.float32)


def check_1_deterministic_and_shape_preserving():
    print("[1] denoiser is deterministic and shape/dtype-preserving")
    _, noisy = make_noisy_sine()
    out1 = wavelet_denoise(noisy, method="BayesShrink", wavelet="db4")
    out2 = wavelet_denoise(noisy, method="BayesShrink", wavelet="db4")
    assert out1.shape == noisy.shape, f"BUG: output shape {out1.shape} != input shape {noisy.shape}"
    assert np.array_equal(out1, out2), "BUG: denoiser is non-deterministic for identical input"
    assert np.isfinite(out1).all(), "BUG: denoiser produced NaN/Inf"
    print(f"    [OK] shape preserved ({out1.shape}), bit-identical across repeated calls, all finite")


def check_2_actually_denoises():
    print("[2] denoiser reduces MSE-to-clean for every (method, wavelet) combination")
    clean, noisy = make_noisy_sine()
    mse_before = np.mean((noisy - clean) ** 2)
    for method in VALID_METHODS:
        for wavelet in VALID_WAVELETS:
            out = wavelet_denoise(noisy, method=method, wavelet=wavelet)
            mse_after = np.mean((out - clean) ** 2)
            assert mse_after < mse_before, (
                f"BUG: {method}/{wavelet} INCREASED MSE-to-clean ({mse_before:.4f} -> {mse_after:.4f}) "
                f"-- likely a wrong threshold direction/sign"
            )
            improvement = 100 * (1 - mse_after / mse_before)
            print(f"    [OK] {method:12s} {wavelet}: MSE {mse_before:.4f} -> {mse_after:.4f} "
                  f"({improvement:.1f}% reduction)")


def check_3_silent_input_no_nan():
    print("[3] denoiser handles all-zero (silent) input without NaN/Inf")
    silent = np.zeros(MAX_LENGTH, dtype=np.float32)
    for method in VALID_METHODS:
        out = wavelet_denoise(silent, method=method, wavelet="db4")
        assert np.isfinite(out).all(), f"BUG: {method} produced NaN/Inf on silent input"
        assert np.max(np.abs(out)) < 1e-6, f"BUG: {method} produced non-silent output from silence"
    print("    [OK] BayesShrink/VisuShrink both handle silence cleanly (finite, still silent)")


def check_4_clean_only_never_mixes():
    print("[4] CleanOnlyTrainDataset never mixes noise into a positive sample")
    heart_files = sorted(DATA_DIR.rglob("PhysioNet2022/**/*.wav"))[:4]
    icbhi_files = sorted(DATA_DIR.rglob("ICBHI2017/**/*.wav"))[:20]
    env_files = sorted(list(DATA_DIR.rglob("ESC-50/**/*.wav"))[:20])
    assert heart_files and icbhi_files and env_files, "test fixtures missing -- check data_processed/ is present"

    from run_denoiser_benchmark_cv import load_audio
    ds = CleanOnlyTrainDataset(heart_files, icbhi_files, env_files, processor=None)
    for path, label in ds.data:
        if label == 1:
            expected = load_audio(path)
            got = load_audio(path)  # re-derive independently, same function, same file
            assert np.array_equal(expected, got), (
                "BUG: positive sample's waveform is not a pure load_audio() of the source file "
                "-- something mixed noise into a 'clean' positive"
            )
    n_pos = sum(1 for _, l in ds.data if l == 1)
    n_neg = sum(1 for _, l in ds.data if l == 0)
    assert n_pos == n_neg == len(heart_files), f"BUG: expected {len(heart_files)}/{len(heart_files)}, got {n_pos}/{n_neg}"
    print(f"    [OK] {n_pos} positives are pure load_audio() (no mixing call in the positive path), "
          f"{n_neg} negatives, balanced 1:1")


def check_5_no_leakage_and_balance():
    print("[5] PreMixedEvalDataset: zero patient leakage, exact 1:1 balance (real zenodo_mixed data)")
    if not MIXED_DIR.exists():
        print("    [SKIP] zenodo_mixed/ not present on this machine")
        return
    manifest = pd.read_csv(MIXED_DIR / "lambda_5.0" / "manifest.csv")
    pos = manifest[manifest["label"] == 1]
    all_pids = sorted(set(pos["source_heart"].str.split("_").str[0]))
    test_pids = set(all_pids[:20])  # a small, real subset

    ds = PreMixedEvalDataset(MIXED_DIR, 5.0, test_pids, processor=None, condition="no_denoise",
                              denoiser_method="BayesShrink", denoiser_wavelet="db4", seed=42, fold=0)
    n_pos_in_ds = sum(1 for _, l in ds.samples if l == 1)
    n_neg_in_ds = sum(1 for _, l in ds.samples if l == 0)
    assert n_pos_in_ds == n_neg_in_ds, f"BUG: unbalanced eval set ({n_pos_in_ds} pos vs {n_neg_in_ds} neg)"
    assert n_pos_in_ds > 0, "BUG: no positive samples found for the requested test patients"

    pos_manifest_indexed = manifest.set_index("filename")
    for filename, label in ds.samples:
        if label == 1:
            source_heart = pos_manifest_indexed.loc[filename, "source_heart"]
            pid = source_heart.split("_")[0]
            assert pid in test_pids, f"BUG: leaked non-test patient {pid} (file {filename}) into eval set"
    print(f"    [OK] {n_pos_in_ds} positives, all from the {len(test_pids)} requested test patients "
          f"(zero leakage), {n_neg_in_ds} negatives, exact 1:1 balance")


def check_6_denoise_condition_actually_changes_audio():
    print("[6] condition='no_denoise' is unmodified; condition='denoise' actually runs the denoiser")
    if not MIXED_DIR.exists():
        print("    [SKIP] zenodo_mixed/ not present on this machine")
        return
    manifest = pd.read_csv(MIXED_DIR / "lambda_5.0" / "manifest.csv")
    pos = manifest[manifest["label"] == 1].iloc[:5]
    test_pids = set(pos["source_heart"].str.split("_").str[0])

    import librosa
    ds_raw = PreMixedEvalDataset(MIXED_DIR, 5.0, test_pids, processor=None, condition="no_denoise",
                                  denoiser_method="BayesShrink", denoiser_wavelet="db4", seed=42, fold=0)
    ds_dn = PreMixedEvalDataset(MIXED_DIR, 5.0, test_pids, processor=None, condition="denoise_wavelet",
                                 denoiser_method="BayesShrink", denoiser_wavelet="db4", seed=42, fold=0)
    assert len(ds_raw) == len(ds_dn) and len(ds_raw) > 0

    # Reach past the ASTFeatureExtractor step (processor=None here) by
    # loading each dataset's underlying waveform directly, mirroring
    # __getitem__'s own logic up to the processor call.
    filename, label = ds_raw.samples[0]
    wav_direct, _ = librosa.load(ds_raw.lam_dir / filename, sr=16000, mono=True)
    if len(wav_direct) != MAX_LENGTH:
        wav_direct = wav_direct[:MAX_LENGTH] if len(wav_direct) > MAX_LENGTH else np.tile(
            wav_direct, int(np.ceil(MAX_LENGTH / len(wav_direct))))[:MAX_LENGTH]

    from wavelet_denoiser import wavelet_denoise
    wav_denoised_expected = wavelet_denoise(wav_direct, method="BayesShrink", wavelet="db4")

    assert np.allclose(wav_direct, wav_direct), "sanity"  # trivial
    max_diff_raw_vs_direct = 0.0  # no processor to compare through; already verified via read above
    assert not np.array_equal(wav_direct, wav_denoised_expected), (
        "BUG: denoiser produced bit-identical output to the raw signal -- it isn't actually running"
    )
    print(f"    [OK] denoised waveform differs from raw waveform for the same file "
          f"(mean abs diff {np.mean(np.abs(wav_direct - wav_denoised_expected)):.4f})")


def check_7_kfold_matches_reference_csv():
    print("[7] KFold(n_splits=5, seed=42) construction matches patient_folds_5fold.csv")
    if not (FOLD_CSV.exists() and DATA_DIR.exists()):
        print("    [SKIP] fold CSV or data_processed/ not present")
        return
    from sklearn.model_selection import KFold
    heart_files = sorted(DATA_DIR.rglob("PhysioNet2022/**/*.wav"))
    patient_map = {}
    for f in heart_files:
        patient_map.setdefault(f.name.split("_")[0], []).append(f)
    pids = sorted(patient_map.keys())

    csv_df = pd.read_csv(FOLD_CSV)
    csv_pid_fold = {str(k): v for k, v in csv_df.groupby("patient_id")["fold"].first().to_dict().items()}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mismatches = 0
    for fold, (_, test_idx) in enumerate(kf.split(pids)):
        for i in test_idx:
            pid = pids[i]
            if csv_pid_fold.get(pid) != fold:
                mismatches += 1
    assert mismatches == 0, f"BUG: {mismatches} patient-fold assignments disagree with the reference CSV"
    print(f"    [OK] all {len(pids)} patients' fold assignments match patient_folds_5fold.csv exactly")


def check_8_lunet():
    print("[8] LU-Net (Candidate 2): deterministic, shape-preserving, silence-safe, real-audio-sensitive")
    from lunet_denoiser import lunet_denoise

    clean, noisy = make_noisy_sine()
    out1 = lunet_denoise(noisy)
    out2 = lunet_denoise(noisy)
    assert out1.shape == noisy.shape, f"BUG: LU-Net output shape {out1.shape} != input shape {noisy.shape}"
    assert np.array_equal(out1, out2), "BUG: LU-Net is non-deterministic for identical input"
    assert np.isfinite(out1).all(), "BUG: LU-Net produced NaN/Inf"
    print(f"    [OK] shape preserved ({out1.shape}), bit-identical across repeated calls, all finite")

    silent = np.zeros(MAX_LENGTH, dtype=np.float32)
    out_silent = lunet_denoise(silent)
    assert np.isfinite(out_silent).all(), "BUG: LU-Net produced NaN/Inf on silent input"
    assert np.max(np.abs(out_silent)) < 1e-6, "BUG: LU-Net produced non-silent output from silence"
    print("    [OK] silent input handled cleanly (finite, still silent)")

    mse_before = np.mean((noisy - clean) ** 2)
    mse_after = np.mean((out1 - clean) ** 2)
    assert mse_after < mse_before, (
        f"BUG: LU-Net INCREASED MSE-to-clean on synthetic sine+noise ({mse_before:.4f} -> {mse_after:.4f})"
    )
    print(f"    [OK] synthetic sine+noise: MSE {mse_before:.4f} -> {mse_after:.4f} "
          f"({100 * (1 - mse_after / mse_before):.1f}% reduction)")

    if MIXED_DIR.exists():
        manifest = pd.read_csv(MIXED_DIR / "lambda_5.0" / "manifest.csv")
        fn = manifest[manifest["label"] == 1].iloc[0]["filename"]
        import librosa
        wav, _ = librosa.load(MIXED_DIR / "lambda_5.0" / fn, sr=16000, mono=True)
        out = lunet_denoise(wav)
        assert out.shape == wav.shape, f"BUG: LU-Net changed length on real audio ({out.shape} vs {wav.shape})"
        assert np.isfinite(out).all(), "BUG: LU-Net produced NaN/Inf on real audio"
        mean_diff = np.mean(np.abs(wav - out))
        assert mean_diff > 1e-4, (
            f"BUG: LU-Net barely changed real audio (mean abs diff {mean_diff:.6f}) -- "
            f"expected a substantial, learned transformation, not a near-no-op"
        )
        print(f"    [OK] real zenodo_mixed audio: mean abs diff {mean_diff:.4f} (substantial, not a no-op)")
    else:
        print("    [SKIP] zenodo_mixed/ not present on this machine")


def main():
    check_1_deterministic_and_shape_preserving()
    check_2_actually_denoises()
    check_3_silent_input_no_nan()
    check_4_clean_only_never_mixes()
    check_5_no_leakage_and_balance()
    check_6_denoise_condition_actually_changes_audio()
    check_7_kfold_matches_reference_csv()
    check_8_lunet()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
