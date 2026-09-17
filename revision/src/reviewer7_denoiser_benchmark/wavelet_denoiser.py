"""
Wavelet (DWT) denoiser for Reviewer #7 Comment 1 -- Candidate 1 from
CLAUDE.md Sec 9.2/9.3: "Automatic Wavelet Denoising (DWT): classical discrete
wavelet thresholding via PyWavelets, Symlet/Daubechies wavelets with soft
BayesShrink/VisuShrink thresholding. Zero training, 100% deterministic, zero
risk of data leakage between ICBHI/CirCor splits."

Implementation choice: wraps `skimage.restoration.denoise_wavelet` rather
than hand-rolling the BayesShrink/VisuShrink threshold math from scratch.
Both methods verified against the literature before writing this (see
../README.md "Wavelet denoiser -- literature check"):
  - BayesShrink (Chang, Yu & Vetterli, IEEE TIP 2000, "Adaptive Wavelet
    Thresholding for Image Denoising and Compression"): per-subband
    threshold T = sigma^2 / sigma_signal, with sigma estimated from the
    finest-level detail coefficients via the median absolute deviation:
    sigma_hat = median(|detail_coeffs|) / 0.6745.
  - VisuShrink (Donoho & Johnstone, 1994): a single universal threshold
    tau = sigma * sqrt(2 * log(N)) applied to every detail coefficient.
`denoise_wavelet` implements exactly this (confirmed by reading its
docstring/behavior, not assumed from the name) and is dimension-agnostic --
it operates on whatever array shape it's given via `pywt.wavedecn`, and was
verified empirically here to accept plain 1D arrays (the literature's own
convention for PCG/ECG signals is 1D DWT via `pywt.wavedec`, not a 2D/image
transform -- confirmed this matches, not just assumed compatible).
Reusing a well-tested, widely-cited library implementation instead of a
from-scratch threshold calculation removes an entire class of "I got the
formula subtly wrong" bugs for a Candidate whose whole selling point is
"zero risk", per the module docstring's own framing.
"""

import numpy as np
import pywt
from scipy import stats
from skimage.restoration import denoise_wavelet

VALID_METHODS = ("BayesShrink", "VisuShrink")
VALID_WAVELETS = ("db4", "sym4")


def wavelet_denoise(wav: np.ndarray, method: str = "BayesShrink", wavelet: str = "db4",
                     wavelet_levels: int = None) -> np.ndarray:
    """Denoise a single 1D waveform (already 16kHz, DC-removed, bandpass-filtered,
    peak-normalized -- i.e. already through load_audio()) via soft-threshold DWT.

    Args:
        wav: 1D float waveform, any length.
        method: "BayesShrink" (per-subband adaptive, default -- literature
            finds it less over-smoothing than VisuShrink) or "VisuShrink"
            (single universal threshold, more aggressive/conservative).
        wavelet: "db4" or "sym4" -- the two families CLAUDE.md Sec 9.3 names.
        wavelet_levels: decomposition depth. None = skimage's own auto-select
            (based on signal length); left as the library default rather than
            an arbitrary hand-picked number, since PCG-specific literature
            doesn't converge on one universal level count and skimage's
            auto-selection is itself a documented, tested behavior.

    Returns:
        Denoised 1D waveform, same length and dtype family as input,
        re-peak-normalized if the denoised signal's peak exceeds 1.0 (the
        DWT reconstruction can occasionally overshoot slightly at
        transients -- matches the same "clip-safe" convention mix_rms()
        uses elsewhere in this codebase, not a new behavior).
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if wavelet not in VALID_WAVELETS:
        raise ValueError(f"wavelet must be one of {VALID_WAVELETS}, got {wavelet!r}")

    wav = np.asarray(wav, dtype=np.float64)

    # Degenerate-input guard: found via test_denoiser_benchmark.py check 3.
    # skimage's BayesShrink estimates a noise sigma from the finest-level
    # detail coefficients' MAD and divides by an estimated per-subband
    # signal sigma (Chang/Yu/Vetterli's T = sigma^2 / sigma_signal). For an
    # all-zero (or numerically-silent) input, every wavelet coefficient is
    # zero, so that division is a real 0/0 -> NaN, not a library bug -- but
    # it's a case our own pipeline can hit (load_audio() falls back to
    # np.zeros(MAX_LENGTH) on a failed file load elsewhere in this repo's
    # scripts), and a NaN probability silently poisons AUROC/F1 for that
    # entire eval batch. There is nothing to denoise in silence anyway, so
    # return it unchanged rather than call into the division.
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
# Level-dependent variant (added 2026-09-14) -- an additional, disclosed
# ablation alongside wavelet_denoise() above, NOT a replacement for it.
#
# Diagnosis (from reading skimage's actual installed source,
# skimage.restoration._denoise._wavelet_threshold -- not assumed, not from a
# secondary summary): both BayesShrink and VisuShrink there estimate the
# noise sigma ONCE, GLOBALLY, from ONLY the finest wavelet decomposition
# level's detail coefficients, then reuse that single number for every
# coarser level's threshold too. That's the right convention for i.i.d.
# white Gaussian noise, whose energy is roughly uniform across scales. Our
# composite noise (lung sounds + 0.5x environmental, per the paper's own
# mixing formula) is NOT concentrated at the finest scale the way white
# noise is -- it has real structure spread across coarser/mid-frequency
# subbands, which is exactly where it overlaps with the heart signal. A
# finest-scale-only noise estimate systematically UNDERESTIMATES the true
# noise level everywhere except the finest scale, so coarser subbands barely
# get thresholded at all -- a precise mechanism for wavelet_denoise()'s
# near-no-op finding on real PCG audio (see ../README.md), not just "doesn't
# fit the model" as a vague explanation.
#
# Fix: estimate sigma SEPARATELY AT EACH decomposition level (using the
# exact same MAD-based Gaussian estimator skimage already uses -- Donoho &
# Johnstone 1994, Biometrika 81(3):425-455 sec 4.2 -- verified against
# skimage.restoration._denoise._sigma_est_dwt's actual source, reproduced
# here rather than reused directly since skimage's public denoise_wavelet()
# API doesn't expose per-level sigma control), then apply BayesShrink/
# VisuShrink's already-verified threshold formulas per level with that
# level's own local sigma. "Level-dependent"/"scale-adaptive" noise
# estimation for correlated/non-stationary noise is itself a documented
# general technique in the wavelet denoising literature (e.g. Johnstone &
# Silverman's work on wavelet threshold estimators for correlated noise) --
# this is not a new algorithm invented for this dataset, and no parameter
# here was chosen by looking at how it performs on our actual PCG/lung/
# environmental audio (validated first on synthetic non-PCG test signals in
# test_denoiser_benchmark.py, same as every other candidate here).
#
# IMPORTANT DISCLOSURE: unlike wavelet_denoise() above (a direct call into a
# peer-reviewed library's implementation of two independently-published
# methods), this is OUR OWN implementation, not a separately-benchmarked
# published algorithm -- report it as such, not as if it carried the same
# external validation as BayesShrink/VisuShrink or LU-Net.
# ---------------------------------------------------------------------------

def _sigma_est_level(detail_coeffs: np.ndarray) -> float:
    """Per-level MAD-based noise sigma estimate -- identical formula to
    skimage.restoration._denoise._sigma_est_dwt (verified against its actual
    source 2026-09-14), called once per decomposition level here instead of
    once globally on only the finest level."""
    coeffs = detail_coeffs[np.nonzero(detail_coeffs)]
    if coeffs.size == 0:
        return 0.0
    return float(np.median(np.abs(coeffs)) / stats.norm.ppf(0.75))


def _bayes_thresh_level(level_coeffs: np.ndarray, var: float) -> float:
    """Identical formula to skimage.restoration._denoise._bayes_thresh
    (verified against its actual source 2026-09-14): threshold = var /
    sqrt(max(signal_var - var, eps)), i.e. Chang/Yu/Vetterli's T = sigma^2 /
    sigma_signal, just invoked with a per-level `var` instead of a shared
    global one."""
    dvar = np.mean(level_coeffs.astype(np.float64) ** 2)
    eps = np.finfo(np.float64).eps
    return float(var / np.sqrt(max(dvar - var, eps)))


def wavelet_denoise_level_dependent(wav: np.ndarray, method: str = "BayesShrink",
                                     wavelet: str = "db4", wavelet_levels: int = None) -> np.ndarray:
    """Level-dependent BayesShrink/VisuShrink -- see the module-level comment
    block above for the full diagnosis and literature basis. Same call
    signature and same guard/clipping conventions as wavelet_denoise() above,
    so the two can be swapped in run_denoiser_benchmark_cv.py's --conditions
    without any other code changes.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if wavelet not in VALID_WAVELETS:
        raise ValueError(f"wavelet must be one of {VALID_WAVELETS}, got {wavelet!r}")

    wav = np.asarray(wav, dtype=np.float64)
    if np.max(np.abs(wav)) < 1e-8:
        return wav.astype(np.float32)

    if wavelet_levels is None:
        # Same "skip the coarsest 3 scales" convention skimage's
        # denoise_wavelet uses by default (see its _wavelet_threshold source)
        # -- not a new choice, matched deliberately so the two candidates
        # only differ in the one mechanism being tested (per-level vs.
        # global noise estimation), not also in decomposition depth.
        wavelet_levels = max(pywt.dwt_max_level(len(wav), wavelet) - 3, 1)

    coeffs = pywt.wavedec(wav, wavelet=wavelet, level=wavelet_levels)
    approx, *details = coeffs  # details: coarsest -> finest, matches pywt.wavedec's documented ordering

    denoised_details = []
    for level_coeffs in details:
        sigma = _sigma_est_level(level_coeffs)
        var = sigma ** 2
        if method == "BayesShrink":
            thresh = _bayes_thresh_level(level_coeffs, var)
        else:  # VisuShrink -- per-level universal threshold using this level's own sigma,
            # N = len(wav) matches skimage's _universal_thresh(img, sigma) using the
            # WHOLE signal's size, not just this level's coefficient count -- verified
            # against skimage's actual call site, not assumed.
            thresh = sigma * np.sqrt(2 * np.log(len(wav)))
        denoised_details.append(pywt.threshold(level_coeffs, value=thresh, mode="soft"))

    denoised = pywt.waverec([approx] + denoised_details, wavelet=wavelet)
    denoised = denoised[:len(wav)]  # pywt.waverec can be 1 sample longer for odd-length input
    peak = np.max(np.abs(denoised))
    if peak > 1.0:
        denoised = denoised / peak
    return denoised.astype(np.float32)


if __name__ == "__main__":
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
