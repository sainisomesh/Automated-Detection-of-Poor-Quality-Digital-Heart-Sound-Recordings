"""
SNR-based phonocardiogram quality scoring, implementing:

  Giordano N, Rosati S, Knaflitz M. "Automated Assessment of the Quality of
  Phonocardographic Recordings through Signal-to-Noise Ratio for Home
  Monitoring Applications." Sensors, 2021, 21(21):7246.
  DOI: 10.3390/s21217246 (open access; also PMC8588421)

The quality score is the signal-to-noise ratio

    SNR = 20 * log10(AS / (4 * sigma_n))

where AS is the peak-to-peak heart-sound amplitude within a cardiac cycle and
4 * sigma_n is the 95% confidence band of the noise amplitude, sigma_n being
the standard deviation of a quiet window located at 70-85% of the cardiac
cycle, where no heart sound is expected to occur.

Deviation from the original method: cardiac-cycle segmentation
--------------------------------------------------------------
The original paper delimits cardiac cycles using R-peaks from an ECG channel
recorded simultaneously with the PCG. The datasets used here are PCG-only and
contain no synchronized ECG, so cycle duration is instead estimated from the
peak of the PCG envelope's own autocorrelation -- the same envelope
autocorrelation technique used by the Tang et al. feature set (see
`tang_features.get_max_axcor_coef`). The SNR formula itself is unmodified;
only the cycle-boundary input to it differs. Results obtained from this
baseline should be reported with that substitution stated.
"""

import numpy as np

from tang_features import get_envelope_from_stft


def _estimate_cycle_duration_samples(phs: np.ndarray, fs: float,
                                      min_cf: float = 0.3, max_cf: float = 2.5) -> int:
    """Estimate the cardiac cycle duration in samples from the PCG itself.

    Computes the STFT amplitude envelope (reusing
    `tang_features.get_envelope_from_stft` rather than duplicating it), takes
    its normalized autocorrelation, and returns the lag of the largest peak
    within the plausible heart-rate range.

    Parameters
    ----------
    min_cf, max_cf : float
        Heart-rate bounds in Hz, which map to the longest (1/min_cf s) and
        shortest (1/max_cf s) cycle durations searched. The 0.3-2.5 Hz default
        covers roughly 18-150 bpm.

    Returns
    -------
    int
        Estimated cycle length in samples; falls back to one second's worth of
        samples if the search window is degenerate.
    """
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
    """Recording-level quality score in dB for a PCG waveform.

    Splits `wav` into consecutive cycles of the estimated cycle length,
    computes SNR = 20*log10(AS / (4*sigma_n)) per cycle, and aggregates across
    cycles with the median. The median is used so that a single badly
    corrupted cycle cannot dominate the recording-level score; the original
    paper reports a recording-level SNR without specifying its cross-cycle
    aggregator.

    Returns -inf if no cycle yields a usable measurement.
    """
    wav = np.asarray(wav, dtype=np.float64).flatten()
    cycle_len = _estimate_cycle_duration_samples(wav, fs)

    n_cycles = len(wav) // cycle_len
    if n_cycles < 1:
        cycle_len = len(wav)
        n_cycles = 1

    # Noise is measured in the paper's quiet window at 70-85% of the cardiac
    # cycle, i.e. late diastole, after S2 and before the next S1.
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
