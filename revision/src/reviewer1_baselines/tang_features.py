"""
Python port of the reference MATLAB implementation of the Tang et al. heart
sound signal-quality feature set:

  Tang H, Wang M, Hu Y, Guo B, Li T. "Automated Signal Quality Assessment for
  Heart Sound Signal by Novel Features and Evaluation in Open Public Datasets."
  BioMed Research International, 2021. DOI: 10.1155/2021/7565398

Ported from the authors' released source:
  https://github.com/tanghongdlut/signal-quality-assessment-of-heart-sound-signal
Each function below names the specific .m file it corresponds to. All numeric
constants (window lengths, filter cutoffs, sample-entropy m/r, cycle-frequency
band) are taken from that published code rather than re-derived, and should not
be altered without breaking correspondence with the original method.

`extract_features` returns the 10 published features in their original order
(see FEATURE_NAMES at the bottom of this module).

Sampling rate: the original preprocessing operates at a fixed rate chosen at
feature-extraction time, and the published features were computed at
fs = 1000 Hz ("down sampled to 1000 Hz" in the paper). Callers are therefore
expected to resample to 1000 Hz before calling `pre_processing` /
`extract_features`.

Two documented departures from bit-exact equivalence with the MATLAB code are
described inline: the Welch segmentation in `get_energy_ratio` and the
resampling method in `_resample_poly_like`.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch, stft, czt


# ── remove_spike.m ──────────────────────────────────────────────────────
def remove_spike(x: np.ndarray) -> np.ndarray:
    """Spike removal (remove_spike.m).

    Defines a reference amplitude TH as the mean of the largest 10% of |x|,
    then clips any sample exceeding R*TH (R = 3) to sign(x)*R*TH. At most 1%
    of samples are clipped; if more samples exceed the bound, only the 1%
    with the largest amplitude are replaced.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    R = 3.0
    abs_x = np.abs(x)
    n = len(x)
    sort_abs = np.sort(abs_x)[::-1]
    top10 = max(1, int(np.floor(n * 0.1)))
    TH = np.mean(sort_abs[:top10])

    ind_spike = np.where(abs_x > R * TH)[0]
    if len(ind_spike) > 0:
        one_pct = round(n * 0.01)
        if len(ind_spike) > one_pct:
            # only replace the `one_pct` largest-amplitude spikes
            order = np.argsort(abs_x[ind_spike])[::-1]
            chosen = ind_spike[order[:one_pct]]
            x[chosen] = np.sign(x[chosen]) * R * TH
        else:
            x[ind_spike] = np.sign(x[ind_spike]) * R * TH
    return x


# ── pre_processing.m ────────────────────────────────────────────────────
def pre_processing(input_signal: np.ndarray, fs: float) -> np.ndarray:
    """Signal conditioning applied before feature extraction (pre_processing.m).

    Pipeline: divide by the standard deviation -> remove spikes -> 3rd-order
    Butterworth high-pass at 2 Hz applied with zero-phase `filtfilt` -> divide
    by the standard deviation again.
    """
    x = np.asarray(input_signal, dtype=np.float64)
    x = x / (np.std(x) + 1e-12)
    x = remove_spike(x)

    fc = 2.0
    b, a = butter(3, 2 * fc / fs, btype="high")
    x = filtfilt(b, a, x)

    x = x / (np.std(x) + 1e-12)
    return x


# ── getEnvelopeFromSTFT.m ───────────────────────────────────────────────
def get_envelope_from_stft(phs: np.ndarray, fs: float) -> np.ndarray:
    """Amplitude envelope from the short-time Fourier transform
    (getEnvelopeFromSTFT.m).

    Uses a rectangular (boxcar) window of 0.03*fs samples advanced one sample
    at a time (noverlap = win_len - 1) with nfft = fs, so the envelope has
    roughly sample-rate resolution. Per frame, the magnitudes of all frequency
    bins are summed and divided by nfft. The resulting envelope is then
    low-pass filtered (3rd-order Butterworth, 20 Hz, zero-phase) and passed
    through `remove_spike`.
    """
    win_len = max(1, int(round(0.03 * fs)))
    nfft = int(round(fs))
    noverlap = win_len - 1

    _, _, Zxx = stft(
        phs, fs=fs, window="boxcar", nperseg=win_len,
        noverlap=noverlap, nfft=nfft, boundary=None, padded=False,
    )
    ins_fre_raw = np.sum(np.abs(Zxx), axis=0) / nfft

    fc = 20.0
    b, a = butter(3, 2 * fc / fs)
    ins_fre = filtfilt(b, a, ins_fre_raw)

    return remove_spike(ins_fre)


# ── getkurtosis.m ───────────────────────────────────────────────────────
def get_kurtosis(x: np.ndarray) -> float:
    """Kurtosis as defined in getkurtosis.m.

    Non-excess kurtosis of the mean-removed signal: mean(x^4) / mean(x^2)^2
    (a Gaussian signal therefore gives ~3). `eps` guards a zero denominator.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    k1 = np.mean(x ** 4)
    k2 = np.mean(x ** 2)
    eps = np.finfo(np.float64).eps
    return float(k1 / (k2 ** 2 + eps))


# ── getEnergyRatio.m ────────────────────────────────────────────────────
def get_energy_ratio(x: np.ndarray, fre: tuple, fs: float) -> float:
    """Fraction of spectral energy inside a frequency band (getEnergyRatio.m).

    Estimates the power spectral density with Welch's method at
    nfft = round(fs) -- i.e. 1 Hz per bin, which is what makes the method's
    fixed Hz band edges directly addressable -- and returns the PSD mass in
    [fre[0], fre[1]] Hz divided by the total PSD mass.

    Parameters
    ----------
    fre : tuple
        (low, high) band edges in Hz, inclusive.

    Segmentation deviation
    ----------------------
    MATLAB's `pwelch` with default window/overlap splits the signal into 8
    Hamming-windowed segments at 50% overlap, i.e. nperseg = N/4.5 (the
    formula retained below). For the fixed 10 s @ 1000 Hz inputs used here
    (N = 10000) that gives nperseg = 2222, which exceeds nfft = 1000; scipy's
    `welch` does not accept nperseg > nfft. nperseg is therefore capped at
    nfft, so the effective segmentation for these inputs is nperseg = 1000 /
    noverlap = 500, i.e. roughly 19 segments at 50% overlap rather than 8.

    The Hamming window, the 50% overlap fraction, and nfft = fs are unchanged.
    The cap is applied identically to every recording irrespective of label,
    noise level, or fold, and averaging over more segments at the same overlap
    fraction yields an equally or more stable PSD estimate.
    """
    x = np.asarray(x, dtype=np.float64)
    nfft = int(round(fs))
    n = len(x)
    nperseg = max(8, int(np.floor(n / 4.5)))
    nperseg = min(nperseg, n, nfft)
    noverlap = nperseg // 2

    w, px = welch(x, fs=fs, window="hamming", nperseg=nperseg,
                  noverlap=noverlap, nfft=nfft)
    ind = np.where((w >= fre[0]) & (w <= fre[1]))[0]
    eps = np.finfo(np.float64).eps
    return float(np.sum(px[ind]) / (np.sum(px) + eps))


# ── getMaxAxcorCoef.m ───────────────────────────────────────────────────
def get_max_axcor_coef(x_single_side: np.ndarray, fs: float) -> float:
    """Peak autocorrelation coefficient in the cardiac-cycle lag range
    (getMaxAxcorCoef.m).

    Returns the maximum of the single-sided (lag >= 0) autocorrelation over
    lags of 0.3 s to 2 s, the plausible range for one cardiac cycle. Expects
    an already normalized autocorrelation, so the result is a coefficient in
    [-1, 1]. Falls back to max(|x|) if the signal is shorter than the lag
    window.
    """
    start = int(round(0.3 * fs))
    end = int(round(2 * fs))
    end = min(end, len(x_single_side))
    if start >= end:
        return float(np.max(np.abs(x_single_side))) if len(x_single_side) else 0.0
    return float(np.max(x_single_side[start:end]))


# ── getSampEn_fast.m ────────────────────────────────────────────────────
def get_sampen_fast(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy (getSampEn_fast.m; Richman & Moorman, 2000).

    Returns -log(A/B), where B is the fraction of embedded-vector pairs of
    length `m` within Chebyshev distance `r` of each other and A is the same
    fraction at length m+1. Lower values indicate a more regular, repetitive
    signal.

    Parameters
    ----------
    m : int
        Embedding dimension (template length).
    r : float
        Similarity tolerance, in units of the z-scored signal's standard
        deviation.

    The input is z-scored internally, mirroring the MATLAB implementation; any
    normalization already applied by the caller is therefore redundant but
    harmless. Degenerate cases (constant signal, signal shorter than m+1, no
    matching pairs) return 0.0.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    std = np.std(x)
    if std < 1e-12:
        return 0.0
    x = (x - np.mean(x)) / std

    N = len(x)
    if N <= m + 1:
        return 0.0

    def _phi(mm):
        n_vec = N - mm
        if n_vec <= 1:
            return 0.0
        templates = np.array([x[i:i + mm] for i in range(n_vec)])
        # Chebyshev (max-abs) distance matrix, matching pdist(...,'chebychev')
        diff = np.max(
            np.abs(templates[:, None, :] - templates[None, :, :]), axis=2
        )
        iu = np.triu_indices(n_vec, k=1)
        d = diff[iu]
        if len(d) == 0:
            return 0.0
        count = np.sum(d <= r) * 2.0 / (n_vec * (n_vec - 1))
        return count

    cm = _phi(m)
    ca = _phi(m + 1)
    if cm <= 0 or ca <= 0:
        return 0.0
    return float(-np.log(ca / cm))


# ── fast_cfs.m / getDegree_cycle.m ──────────────────────────────────────
def get_degree_cycle(rx: np.ndarray, min_cf: float, max_cf: float, fs: float,
                      M: int = 200) -> float:
    """Degree of periodicity from cyclostationary analysis
    (getDegree_cycle.m + fast_cfs.m).

    Takes the instantaneous amplitude of the signal (|Hilbert transform|,
    mean-removed), evaluates its cyclic spectrum on `M` points spanning the
    cycle-frequency band [min_cf, max_cf] Hz via a chirp-Z transform, and
    returns max(|cfs|) / median(|cfs|). A strongly periodic signal
    concentrates cyclic-spectrum energy at its cycle frequency and so scores
    high; broadband noise scores near 1.

    Parameters
    ----------
    min_cf, max_cf : float
        Cycle-frequency search band in Hz (0.3-2.5 Hz for heart rate,
        i.e. ~18-150 bpm).
    M : int
        Number of cycle-frequency points evaluated across that band.

    The hand-rolled fast convolution in fast_cfs.m is the Rabiner chirp-Z
    transform; scipy.signal.czt is used here with the same w and a
    parameterization as the MATLAB source.
    """
    from scipy.signal import hilbert

    x = np.abs(hilbert(np.asarray(rx, dtype=np.float64)))
    x = x - np.mean(x)

    w = np.exp(-1j * 2 * np.pi * (max_cf - min_cf) / (M * fs))
    a = np.exp(1j * 2 * np.pi * min_cf / fs)

    g = czt(x, m=M, w=w, a=a)
    cfs = np.abs(g)

    med = np.median(cfs)
    if med <= 0:
        return 0.0
    return float(np.max(cfs) / med)


# ── get_features_Tang.m ─────────────────────────────────────────────────
def extract_features(phs: np.ndarray, fs: float) -> np.ndarray:
    """Compute the 10 Tang et al. quality features (get_features_Tang.m).

    Parameters
    ----------
    phs : np.ndarray
        Phonocardiogram already conditioned by `pre_processing()`.
    fs : float
        Sampling rate of `phs` in Hz (1000 Hz in the original work).

    Returns
    -------
    np.ndarray
        The 10 features in published order; see FEATURE_NAMES.
    """
    phs = np.asarray(phs, dtype=np.float64).flatten()
    enve = get_envelope_from_stft(phs, fs)

    # 1: kurtosis of the heart sound signal
    kur_hs = get_kurtosis(phs)

    # 2-4: energy ratio in low/high/middle frequency bands
    fre_low = (24, 144)
    fre_high = (200, fs / 2)
    fre_midd = (144, 200)
    energy_ratio_low = get_energy_ratio(phs, fre_low, fs)
    energy_ratio_high = get_energy_ratio(phs, fre_high, fs)
    energy_ratio_midd = get_energy_ratio(phs, fre_midd, fs)

    # 5: std of the envelope, scaled by 1/1000 (matches Std_enve=std(enve)/1000)
    std_enve = np.std(enve) / 1000.0

    # autocorrelation of the mean-removed envelope (double- and single-sided)
    enve_dm = enve - np.mean(enve)
    n = len(enve_dm)
    full_corr = np.correlate(enve_dm, enve_dm, mode="full")
    norm = full_corr[n - 1] if full_corr[n - 1] != 0 else 1.0
    axcor_double = full_corr / norm            # matches xcorr(...,'coeff')
    axcor_single = axcor_double[n - 1:]         # single-sided, lag >= 0

    # 6: kurtosis of the (double-sided) autocorrelation of the envelope
    kur_corr = get_kurtosis(axcor_double)

    # 7: max autocorrelation coefficient in lag range [0.3*fs, 2*fs]
    max_correlation_coef = get_max_axcor_coef(axcor_single, fs)

    # 8: sample entropy of the envelope, downsampled to 30 Hz
    m, r, fsd = 2, 0.2, 30
    n_down = max(2, int(round(len(enve_dm) * fsd / fs)))
    down_enve = _resample_poly_like(enve_dm, fsd, fs, n_down)
    std_down = np.std(down_enve)
    samp_en = get_sampen_fast(down_enve / (std_down + 1e-12), m, r)

    # 9: sample entropy of the envelope autocorrelation, downsampled to 30 Hz
    n_down_ax = max(2, int(round(len(axcor_single) * fsd / fs)))
    down_axcor = _resample_poly_like(axcor_single, fsd, fs, n_down_ax)
    std_down_ax = np.std(down_axcor)
    samp_en_axcor = get_sampen_fast(down_axcor / (std_down_ax + 1e-12), m, r)

    # 10: degree of periodicity (cyclostationary analysis)
    min_cf, max_cf = 0.3, 2.5
    d_cfs = get_degree_cycle(phs, min_cf, max_cf, fs)

    return np.array([
        kur_hs, energy_ratio_low, energy_ratio_high, energy_ratio_midd,
        std_enve, kur_corr, max_correlation_coef,
        samp_en, samp_en_axcor, d_cfs,
    ], dtype=np.float64)


def _resample_poly_like(x: np.ndarray, fs_new: float, fs_old: float, n_out: int) -> np.ndarray:
    """Downsample `x` from `fs_old` to `fs_new` Hz, producing `n_out` samples.

    Approximation rather than an exact equivalent of MATLAB's `resample`:
    MATLAB applies a polyphase FIR anti-aliasing filter suited to
    non-periodic signals, whereas scipy.signal.resample is FFT-based and
    implicitly treats the input as periodic, which can introduce Gibbs-type
    edge artifacts.

    Only the two sample-entropy features (8 and 9) go through this path, and
    they are z-scored pattern-matching statistics rather than
    amplitude-sensitive ones, which limits but does not remove sensitivity to
    the difference. The same resampling is applied to every recording
    irrespective of label or noise level, so it affects fidelity to the
    original MATLAB feature values rather than the balance of any comparison.
    """
    from scipy.signal import resample
    return resample(x, n_out)


FEATURE_NAMES = [
    "kurtosis_hs", "energy_ratio_low", "energy_ratio_high", "energy_ratio_midd",
    "std_envelope", "kurtosis_autocorr", "max_autocorr_coef",
    "sampen_envelope", "sampen_autocorr", "degree_periodicity",
]
