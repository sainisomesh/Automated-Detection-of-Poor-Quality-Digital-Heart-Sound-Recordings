"""
Classical wavelet (DWT) denoising -- Candidate 1 of the denoise-then-classify
comparison (Reviewer #7, Comment 1).

This module exposes two denoisers:

  * ``wavelet_denoise`` -- soft-threshold DWT denoising delegated to
    ``skimage.restoration.denoise_wavelet``, i.e. a published library
    implementation of two standard, independently published thresholding
    rules:
      - BayesShrink (Chang, Yu & Vetterli, IEEE TIP 2000, "Adaptive Wavelet
        Thresholding for Image Denoising and Compression"): a per-subband
        threshold T = sigma^2 / sigma_signal, where sigma is estimated from
        the finest-level detail coefficients via the median absolute
        deviation, sigma_hat = median(|detail_coeffs|) / 0.6745.
      - VisuShrink (Donoho & Johnstone, 1994): a single universal threshold
        tau = sigma * sqrt(2 * log(N)) applied to every detail coefficient.

    Both rules are used with the standard literature default parameters.
    Nothing here -- wavelet family, decomposition depth, thresholding mode or
    noise-estimation rule -- was tuned against this paper's own heart-sound
    data. That is a deliberate methodological commitment: this candidate is
    meant to be a training-free, fully deterministic, untuned reference
    point, with no possibility of information leaking from the evaluation
    corpora into the denoiser. Calling the library implementation rather than
    re-deriving the threshold arithmetic keeps the numbers identical to the
    widely cited reference implementation.

    ``denoise_wavelet`` is dimension-agnostic (it decomposes whatever array
    shape it is given via ``pywt.wavedecn``); it is used here on plain 1-D
    waveforms, matching the 1-D DWT convention used for PCG/ECG signals in
    the denoising literature.

  * ``wavelet_denoise_level_dependent`` -- a level-dependent noise-estimation
    variant developed for this work rather than a separately published
    algorithm. See the section comment above that function for its
    motivation, authorship status and a measured limitation.
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
        method: "BayesShrink" (per-subband adaptive, the default; reported in
            the literature as less over-smoothing than VisuShrink) or
            "VisuShrink" (a single universal threshold, more aggressive).
        wavelet: "db4" (Daubechies) or "sym4" (Symlet), the two families used
            for PCG wavelet denoising in the cited literature.
        wavelet_levels: decomposition depth. ``None`` keeps skimage's own
            signal-length-dependent auto-selection, which is a documented and
            tested library behaviour; the PCG literature does not converge on
            a single universal level count, so no hand-picked value is
            imposed here.

    Returns:
        Denoised 1-D waveform (float32) of the same length as the input,
        rescaled to unit peak if the reconstruction exceeds 1.0. DWT
        reconstruction can overshoot slightly at sharp transients; the
        rescaling is the same clip-avoidance convention used by the RMS
        mixing code elsewhere in this package.

    Near-silent input (peak < 1e-8) is returned unchanged: the MAD-based
    noise-sigma estimate is undefined for an all-zero signal (every wavelet
    coefficient is zero, so BayesShrink's sigma^2 / sigma_signal becomes 0/0),
    and there is nothing to denoise in silence anyway.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if wavelet not in VALID_WAVELETS:
        raise ValueError(f"wavelet must be one of {VALID_WAVELETS}, got {wavelet!r}")

    wav = np.asarray(wav, dtype=np.float64)

    # Degenerate-input guard. For an all-zero (or numerically silent) input
    # every wavelet coefficient is zero, so BayesShrink's
    # T = sigma^2 / sigma_signal is a genuine 0/0 and yields NaN. The
    # surrounding pipeline can produce such a buffer (a failed audio load
    # falls back to zeros), and a single NaN probability would corrupt the
    # AUROC/F1 of the whole evaluation batch, so silence is returned as-is.
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
# Level-dependent noise estimation -- an extension developed for this work.
# It is an additional evaluation arm alongside wavelet_denoise() above, not a
# replacement for it.
#
# AUTHORSHIP. Unlike wavelet_denoise(), which is a direct call into a
# peer-reviewed library's implementation of two independently published
# thresholding rules, the function below is this work's own variant. It
# carries none of the external validation that BayesShrink, VisuShrink or
# LU-Net do, and must be reported as an extension developed here rather than
# as a published baseline method.
#
# MOTIVATION. skimage's thresholding routine
# (skimage.restoration._denoise._wavelet_threshold) estimates the noise sigma
# once, globally, from only the finest wavelet decomposition level's detail
# coefficients, and then reuses that single value for every coarser level's
# threshold -- for both BayesShrink and VisuShrink. That is the correct
# convention for i.i.d. white Gaussian noise, whose energy is roughly uniform
# across scales. The composite interference used in this paper (lung sounds +
# 0.5x environmental sounds) is not concentrated at the finest scale: it has
# real structure across coarse and mid-frequency subbands, which is also
# where it overlaps the heart signal. A finest-scale-only estimate therefore
# systematically underestimates the noise level at coarser scales, which end
# up barely thresholded at all. This is the mechanism behind the observation
# that wavelet_denoise() is close to a no-op on real PCG audio (see
# ../../README.md).
#
# APPROACH. Estimate sigma separately at each decomposition level, using the
# same MAD-based Gaussian estimator skimage uses internally (Donoho &
# Johnstone 1994, Biometrika 81(3):425-455, sec. 4.2), then apply the
# BayesShrink/VisuShrink threshold formulas per level with that level's own
# local sigma. The estimator is reproduced here only because skimage's public
# denoise_wavelet() API exposes no per-level sigma control. Scale-adaptive
# noise estimation for correlated or non-stationary noise is a known general
# technique in the wavelet denoising literature (e.g. Johnstone & Silverman
# on threshold estimators for correlated noise); the particular combination
# used here is ours. As with wavelet_denoise(), no parameter was selected by
# examining performance on this paper's PCG, lung-sound or environmental
# audio: the variant was developed and checked against synthetic, non-PCG
# test signals only (see test_denoiser_benchmark.py).
#
# MEASURED LIMITATION. On genuinely clean, noise-free heart-sound recordings
# this variant still removes a measurable fraction of the signal's own energy
# -- roughly 2-13% across sampled CirCor files, versus about 0% for the
# global-sigma version -- and correspondingly lowers classification AUROC at
# lambda = 0.0, where there is no noise to remove at all. Mechanism: per-level
# estimation implicitly assumes most of the energy at every scale is noise,
# which holds for sparse or simple signals but not for heart sounds. S1/S2 are
# broadband transients that populate detail coefficients across many scales,
# including the coarse ones where the heart sound's own low-frequency
# structure lives, so the per-level estimate cannot separate "coarse-scale
# energy because of noise" from "coarse-scale energy because of a real S1/S2
# transient" and attenuates both. Any result from this condition should be
# reported together with this signal-degradation effect.
# ---------------------------------------------------------------------------

def _sigma_est_level(detail_coeffs: np.ndarray) -> float:
    """MAD-based Gaussian noise sigma estimate for one decomposition level.

    Same formula as skimage.restoration._denoise._sigma_est_dwt, but called
    once per decomposition level instead of once globally on the finest
    level only. Zero coefficients are excluded from the median, as skimage
    does; an all-zero level yields sigma = 0 (no thresholding).
    """
    coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
    if coeffs.size == 0:
        return 0.0
    return float(np.median(np.abs(coeffs)) / stats.norm.ppf(0.75))


def _bayes_thresh_level(level_coeffs: np.ndarray, var: float) -> float:
    """BayesShrink threshold for one decomposition level.

    Same formula as skimage.restoration._denoise._bayes_thresh --
    threshold = var / sqrt(max(signal_var - var, eps)), i.e. Chang, Yu &
    Vetterli's T = sigma^2 / sigma_signal -- but evaluated with this level's
    own noise variance `var` instead of a shared global one.
    """
    dvar = np.mean(level_coeffs.astype(np.float64) ** 2)
    eps = np.finfo(np.float64).eps
    return float(var / np.sqrt(max(dvar - var, eps)))


def wavelet_denoise_level_dependent(wav: np.ndarray, method: str = "BayesShrink",
                                     wavelet: str = "db4", wavelet_levels: int = None) -> np.ndarray:
    """Level-dependent BayesShrink/VisuShrink (this work's own variant).

    See the section comment above for the motivation, authorship status and
    measured limitation. Arguments, near-silence guard and peak-rescaling
    behaviour are identical to wavelet_denoise(), so the two denoisers are
    interchangeable as `--conditions` arms in run_denoiser_benchmark_cv.py
    with no other code changes.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if wavelet not in VALID_WAVELETS:
        raise ValueError(f"wavelet must be one of {VALID_WAVELETS}, got {wavelet!r}")

    wav = np.asarray(wav, dtype=np.float64)
    if np.max(np.abs(wav)) < 1e-8:
        return wav.astype(np.float32)

    if wavelet_levels is None:
        # Reproduces skimage's default depth ("maximum level minus 3"), so
        # that this variant and wavelet_denoise() differ only in the one
        # mechanism under test -- per-level versus global noise estimation --
        # and not also in decomposition depth.
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
            # VisuShrink: universal threshold evaluated with this level's own
            # sigma. N is the whole signal length, not the level's
            # coefficient count, matching skimage's _universal_thresh().
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
    # Quick self-check: every (method, wavelet) pair should reduce the
    # distance to a known clean signal on synthetic sine + white noise.
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
