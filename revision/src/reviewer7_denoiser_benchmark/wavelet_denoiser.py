"""
Wavelet (DWT) denoising, candidate 1 of the denoise-then-classify comparison
(Reviewer #7, Comment 1).

Two denoisers:

  * ``wavelet_denoise``: soft-threshold DWT denoising using
    ``skimage.restoration.denoise_wavelet``, with one of two standard
    threshold rules:
      - BayesShrink (Chang, Yu & Vetterli, IEEE TIP 2000, "Adaptive Wavelet
        Thresholding for Image Denoising and Compression"): per-subband
        threshold T = sigma^2 / sigma_signal, with sigma estimated from the
        finest-level detail coefficients by the median absolute deviation,
        sigma_hat = median(|detail_coeffs|) / 0.6745.
      - VisuShrink (Donoho & Johnstone, 1994): universal threshold
        tau = sigma * sqrt(2 * log(N)) for all detail coefficients.

    Default library parameters are used. The wavelet, depth, threshold mode
    and noise estimator were not tuned on our data, so this baseline needs
    no training and is deterministic. It is applied to 1-D waveforms.

  * ``wavelet_denoise_level_dependent``: our own variant that estimates the
    noise level separately at each decomposition level (see the comment
    above that function). It is not a published method.
"""

import numpy as np
import pywt
from scipy import stats
from skimage.restoration import denoise_wavelet

VALID_METHODS = ("BayesShrink", "VisuShrink")
VALID_WAVELETS = ("db4", "sym4")


def wavelet_denoise(wav: np.ndarray, method: str = "BayesShrink", wavelet: str = "db4",
                     wavelet_levels: int = None) -> np.ndarray:
    """Denoise a single 1-D waveform with soft-threshold DWT.

    The input is expected to have already been through this benchmark's
    ``load_audio()``: 16 kHz mono, DC-removed, bandpass-filtered and
    peak-normalized.

    Args:
        wav: 1-D float waveform, any length.
        method: "BayesShrink" (per-subband, default; smooths less) or
            "VisuShrink" (single universal threshold, more aggressive).
        wavelet: "db4" (Daubechies) or "sym4" (Symlet), both common for PCG.
        wavelet_levels: decomposition depth. ``None`` uses skimage's default,
            which depends on signal length.

    Returns:
        Denoised float32 waveform of the same length, rescaled to unit peak
        if the reconstruction exceeds 1.0 (DWT can overshoot slightly at
        sharp transients).

    Near-silent input (peak < 1e-8) is returned unchanged.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if wavelet not in VALID_WAVELETS:
        raise ValueError(f"wavelet must be one of {VALID_WAVELETS}, got {wavelet!r}")

    wav = np.asarray(wav, dtype=np.float64)

    # For silent input all coefficients are zero and the BayesShrink
    # threshold is 0/0 (NaN). Failed audio loads fall back to zeros, so
    # return silence unchanged.
    if np.max(np.abs(wav)) < 1e-8:
        return wav.astype(np.float32)

    denoised = denoise_wavelet(
        wav, method=method, mode="soft", wavelet=wavelet,
        wavelet_levels=wavelet_levels, rescale_sigma=True,
    )
    peak = np.max(np.abs(denoised))
    if peak > 1.0:
        denoised = denoised / peak
    return denoised.astype(np.float32)


# ---------------------------------------------------------------------------
# Level-dependent noise estimation (our own variant, used as an extra
# condition next to wavelet_denoise()). It is not a published method.
#
# skimage (skimage.restoration._denoise._wavelet_threshold) estimates sigma
# once, from the finest-level detail coefficients, and uses it at every level
# for both BayesShrink and VisuShrink. That suits white noise, but our
# lung + environmental noise also has energy at coarser scales, where it
# overlaps the heart sound, so those levels are barely thresholded. This is
# why wavelet_denoise() changes real PCG audio very little (see
# ../../README.md).
#
# Here sigma is estimated separately at each level with the same MAD
# estimator (Donoho & Johnstone 1994, Biometrika 81(3):425-455, sec. 4.2), and
# the BayesShrink/VisuShrink thresholds use that level's sigma. skimage's
# public API has no per-level option, so the estimator is reimplemented.
# Level-dependent thresholds for correlated noise are a known idea (e.g.
# Johnstone & Silverman); this particular combination is ours. Parameters
# were not tuned on our data; the variant was only checked on synthetic
# signals (test_denoiser_benchmark.py).
#
# Limitation: on clean heart recordings this variant removes about 2-13% of
# the signal energy (versus about 0% for wavelet_denoise()) and lowers AUROC
# at lambda = 0.0. S1/S2 are broadband transients with energy at many scales,
# so per-level estimates treat part of the heart sound as noise. Results for
# this condition should be reported with this caveat.
# ---------------------------------------------------------------------------

def _sigma_est_level(detail_coeffs: np.ndarray) -> float:
    """MAD-based Gaussian noise sigma estimate for one decomposition level.

    Same formula as skimage.restoration._denoise._sigma_est_dwt, applied to
    one level. Zero coefficients are excluded, as in skimage; an all-zero
    level gives sigma = 0 (no thresholding).
    """
    coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
    if coeffs.size == 0:
        return 0.0
    return float(np.median(np.abs(coeffs)) / stats.norm.ppf(0.75))


def _bayes_thresh_level(level_coeffs: np.ndarray, var: float) -> float:
    """BayesShrink threshold for one decomposition level.

    Same formula as skimage.restoration._denoise._bayes_thresh,
    threshold = var / sqrt(max(signal_var - var, eps)) (T = sigma^2 /
    sigma_signal), with this level's noise variance `var`.
    """
    dvar = np.mean(level_coeffs.astype(np.float64) ** 2)
    eps = np.finfo(np.float64).eps
    return float(var / np.sqrt(max(dvar - var, eps)))


def wavelet_denoise_level_dependent(wav: np.ndarray, method: str = "BayesShrink",
                                     wavelet: str = "db4", wavelet_levels: int = None) -> np.ndarray:
    """Level-dependent BayesShrink/VisuShrink (our variant, see above).

    Same arguments, silence handling and peak rescaling as wavelet_denoise().
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if wavelet not in VALID_WAVELETS:
        raise ValueError(f"wavelet must be one of {VALID_WAVELETS}, got {wavelet!r}")

    wav = np.asarray(wav, dtype=np.float64)
    if np.max(np.abs(wav)) < 1e-8:
        return wav.astype(np.float32)

    if wavelet_levels is None:
        # skimage's default depth (max level - 3), so the only difference from
        # wavelet_denoise() is per-level vs global sigma.
        wavelet_levels = max(pywt.dwt_max_level(len(wav), wavelet) - 3, 1)

    coeffs = pywt.wavedec(wav, wavelet=wavelet, level=wavelet_levels)
    # pywt.wavedec returns [approximation, detail_coarsest, ..., detail_finest].
    approx, *details = coeffs

    denoised_details = []
    for level_coeffs in details:
        sigma = _sigma_est_level(level_coeffs)
        var = sigma ** 2
        if method == "BayesShrink":
            thresh = _bayes_thresh_level(level_coeffs, var)
        else:
            # VisuShrink with this level's sigma. N is the signal length, as
            # in skimage's _universal_thresh().
            thresh = sigma * np.sqrt(2 * np.log(len(wav)))
        denoised_details.append(pywt.threshold(level_coeffs, value=thresh, mode="soft"))

    denoised = pywt.waverec([approx] + denoised_details, wavelet=wavelet)
    # waverec can return one extra sample for odd-length input.
    denoised = denoised[:len(wav)]
    peak = np.max(np.abs(denoised))
    if peak > 1.0:
        denoised = denoised / peak
    return denoised.astype(np.float32)


if __name__ == "__main__":
    # Quick self-check: each (method, wavelet) pair should lower the MSE on a
    # synthetic sine + white noise.
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 160000)
    clean = 0.3 * np.sin(2 * np.pi * 2 * t)
    noisy = clean + 0.4 * rng.standard_normal(160000)
    for method in VALID_METHODS:
        for wavelet in VALID_WAVELETS:
            out = wavelet_denoise(noisy, method=method, wavelet=wavelet)
            mse_before = np.mean((noisy - clean) ** 2)
            mse_after = np.mean((out - clean) ** 2)
            print(f"{method:12s} {wavelet}: MSE {mse_before:.4f} -> {mse_after:.4f} "
                  f"({'improved' if mse_after < mse_before else 'WORSE'})")
