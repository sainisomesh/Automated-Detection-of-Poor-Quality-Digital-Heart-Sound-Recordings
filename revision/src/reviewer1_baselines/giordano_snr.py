"""
SNR-based PCG quality scoring, adapted from:

  Giordano N, Rosati S, Knaflitz M. "Automated Assessment of the Quality of
  Phonocardographic Recordings through Signal-to-Noise Ratio for Home
  Monitoring Applications." Sensors, 2021, 21(21):7246.
  DOI: 10.3390/s21217246 (fully open access, MDPI).
  Full text verified via PMC8588421 (2026-09-12).

Core formula, quoted directly from the paper:
    SNR = 20 * log10(AS / (4 * sigma_n))
where AS is the peak-to-peak heart-sound amplitude and 4*sigma_n is the 95%
confidence band of the noise amplitude, measured in a quiet window within
70-85% of the cardiac cycle where no heart sound occurs.

ADAPTATION NOTE (must stay visible, not silently done): the original paper
identifies cardiac cycle boundaries using a SIMULTANEOUS ECG channel's
R-peaks. Our datasets are PCG-only -- there is no synchronized ECG. Cycle
boundaries here are instead estimated from the PCG envelope's own
autocorrelation peak (same technique Tang et al.'s released code uses for
its "max autocorrelation coefficient" feature -- see tang_features.py). The
SNR formula itself is exact and unmodified; only the cycle-segmentation
input to it is a necessary substitution given data availability. This must
be stated explicitly in any writeup that reports numbers from this baseline.
"""

import numpy as np

from tang_features import get_envelope_from_stft


def _estimate_cycle_duration_samples(phs: np.ndarray, fs: float,
                                      min_cf: float = 0.3, max_cf: float = 2.5) -> int:
    """Cycle duration via envelope autocorrelation peak lag, in samples.
    Same technique as Tang et al.'s getCycleDur.m / getMaxAxcorCoef.m, reused
    here (not duplicated) via tang_features.get_envelope_from_stft."""
    enve = get_envelope_from_stft(phs, fs)
    enve = enve - np.mean(enve)
    n = len(enve)
    full_corr = np.correlate(enve, enve, mode="full")
    norm = full_corr[n - 1] if full_corr[n - 1] != 0 else 1.0
    single_side = full_corr[n - 1:] / norm

    start = int(round((1 / max_cf) * fs))  # shortest plausible cycle: 1/max_cf seconds
    end = int(round((1 / min_cf) * fs))    # longest plausible cycle: 1/min_cf seconds
    end = min(end, len(single_side) - 1)
    if start >= end or start < 1:
        return int(round(fs))  # fallback: assume a 1-second cycle
    peak_lag = start + int(np.argmax(single_side[start:end]))
    return max(peak_lag, 2)


def compute_snr_db(wav: np.ndarray, fs: float) -> float:
    """Segment `wav` into estimated cardiac cycles, compute per-cycle SNR
    via the exact Giordano et al. formula, return the median across cycles
    (median chosen for robustness -- a single corrupted cycle shouldn't
    dominate the whole-recording score, and the paper itself reports
    recording-level SNR without specifying its own cross-cycle aggregator)."""
    wav = np.asarray(wav, dtype=np.float64).flatten()
    cycle_len = _estimate_cycle_duration_samples(wav, fs)

    n_cycles = len(wav) // cycle_len
    if n_cycles < 1:
        cycle_len = len(wav)
        n_cycles = 1

    quiet_start_frac, quiet_end_frac = 0.70, 0.85
    snrs = []
    for c in range(n_cycles):
        seg = wav[c * cycle_len:(c + 1) * cycle_len]
        if len(seg) < 10:
            continue
        AS = float(np.max(seg) - np.min(seg))  # peak-to-peak heart sound amplitude

        q_start = int(round(quiet_start_frac * len(seg)))
        q_end = int(round(quiet_end_frac * len(seg)))
        quiet = seg[q_start:q_end]
        sigma_n = float(np.std(quiet)) if len(quiet) > 1 else 1e-9
        sigma_n = max(sigma_n, 1e-9)

        snr_db = 20.0 * np.log10(AS / (4.0 * sigma_n) + 1e-12)
        snrs.append(snr_db)

    if not snrs:
        return -np.inf
    return float(np.median(snrs))
