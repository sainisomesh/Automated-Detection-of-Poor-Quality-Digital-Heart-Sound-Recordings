"""
LU-Net denoiser for Reviewer #7 Comment 1 -- Candidate 2 from CLAUDE.md
Sec 9.3: Ali, Shuvo, Al-Manzo, Hasan & Hasan, "An end-to-end deep learning
framework for real-time denoising of heart sounds for cardiac disease
detection in unseen noise," IEEE Access, 2023.
    github.com/ShamsNafisaAli/LU-Net-Heart-Sound-Denoising-

**LEAKAGE CAVEAT -- included deliberately, not hidden (your explicit call,
2026-09-13): "let's still build in the LU-Net thing as some sort of model
implementation, even if it's on similar data training-wise, we can still
note that in the paper."** LU-Net was trained on PhysioNet heart sounds
mixed with ICBHI 2017 lung noise -- our own test set ALSO uses ICBHI 2017 as
its structured-noise source. Any result from this candidate must be reported
with this domain-overlap explicitly stated, not presented as a clean,
independent comparison. This is a disclosed limitation of the comparison,
not a reason to skip the candidate.

Corrections to CLAUDE.md Sec 9.3's own description, found by reading the
actual repository rather than trusting the table:
  - It says "PyTorch U-Net architecture" -- it's actually Keras/TensorFlow
    (confirmed via Codes/model.py: `from tensorflow.keras...`). No PyTorch
    port is attempted here -- the pretrained .h5 is loaded via
    `keras.models.load_model()` directly (architecture + weights together,
    the safest possible integration: zero risk of an architecture-porting
    bug, since nothing is reimplemented).
  - "Weights provided in repo GDrive" -- also not quite right. The actual
    pretrained weights are committed directly in the GitHub repo at
    `Models/LU-Net.h5` (16.1MB, confirmed present via the GitHub API), not
    behind a separate Google Drive link. Fetched here from
    raw.githubusercontent.com, not vendored into this repo's git history.
  - License: no repo-level LICENSE file (GitHub's own API reports
    `license: null`), but `Codes/model.py`'s own docstring explicitly states
    "This code is licensed under the terms of the MIT-license." -- treated
    as authoritative permission, with attribution preserved here per MIT's
    own requirement.

Preprocessing/inference recipe below is reverse-engineered directly from the
authors' own `Codes/config.py`, `Codes/processing_initial.py`, and
`Codes/utils.py` (their real inference pipeline in `result_making.py`), not
guessed:
  - Model operates at 1000 Hz (`sampling_rate_new` in config.py), not 16kHz.
  - Fixed 0.8s (800-sample) non-overlapping segments (`window_size`,
    `input_shape`/`output_shape` in config.py); `get_files_and_resample()`
    takes `k = len(signal) // duration_samples` FULL segments and drops any
    remainder -- no padding of a partial trailing segment.
  - Each segment is peak-normalized before being used (their `real_signal =
    x_files / max(abs(x_files))` convention, applied throughout their
    pipeline) -- inference-time analogue: normalize each input segment, run
    the model, then re-scale the output back to the input segment's
    original peak (their training/eval code doesn't need this step since it
    always works with already-normalized data; we do, since we're feeding
    it real, differently-scaled evaluation audio it never saw a
    denormalization step for).

Adaptation needed because our pipeline is 16kHz/10s and theirs is 1kHz/0.8s
(disclosed, not hidden): resample down to 1kHz, denoise in 800-sample
segments, resample the denoised concatenation back to 16kHz, and because
`k * 800` samples at 1kHz never evenly divides our 10s clip (10000/800 =
12.5), the trailing ~0.4s that LU-Net's own convention would simply drop is
instead filled back in from the ORIGINAL (non-denoised) audio at that
position, so the output stays the expected 160,000 samples without
inventing silence or looping.
"""

import os

import numpy as np
import librosa
# NOTE: no longer setting os.environ["TF_USE_LEGACY_KERAS"]=1 here (removed
# 2026-09-14). It was unnecessary -- `import tf_keras` below is already a
# standalone package with its own Keras-2-compatible API surface, and loads
# LU-Net's legacy-format .h5 fine without the env var (verified directly).
# The env var's real effect is much broader than "make tf_keras available":
# it redirects `tensorflow.keras` itself to the legacy implementation for
# the ENTIRE process. Since this module gets imported into the same process
# as tbilstm_denoiser.py (both are denoiser candidates in
# run_denoiser_benchmark_cv.py), that process-wide side effect broke loading
# of T-BiLSTM's own model (saved with plain modern Keras 3) with a real,
# reproduced crash -- caught by test_tbilstm.py check 7. Removing the
# unnecessary env var fixes the root cause instead of working around it.

LUNET_SR = 1000
LUNET_SEGMENT_SECONDS = 0.8
LUNET_SEGMENT_SAMPLES = int(LUNET_SR * LUNET_SEGMENT_SECONDS)  # 800
LUNET_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/ShamsNafisaAli/"
    "LU-Net-Heart-Sound-Denoising-/main/Models/LU-Net.h5"
)
DEFAULT_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "ast-heart-quality", "LU-Net.h5"
)

_model_cache = {}  # keyed by cache_path, not a single global slot -- found and fixed as a
# real (if previously latent) bug 2026-09-14 while building tbilstm_denoiser.py's analogous
# cache: a single-slot "if _model is None" cache silently ignores the requested path on every
# call after the first, always returning whichever model loaded first. Harmless here today
# since this module is only ever called with DEFAULT_CACHE_PATH in practice, but the same
# pattern produced an observable bug in tbilstm_denoiser.py (see its comment) as soon as a
# second, different weights_path was used -- fixed here too before it could bite the same way.


def _download_weights(cache_path):
    # requests (bundles its own certifi CA bundle) rather than urllib --
    # macOS's python.framework build doesn't wire urllib to the system trust
    # store by default, which fails with CERTIFICATE_VERIFY_FAILED on a
    # plain urllib.request.urlretrieve() call (same gotcha already
    # documented in ../../reviewer7_backbone_swap/README.md for
    # torch.hub's downloader). requests sidesteps it entirely.
    import requests
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        resp = requests.get(LUNET_WEIGHTS_URL, timeout=60)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(resp.content)
    return cache_path


def _get_model(cache_path=DEFAULT_CACHE_PATH):
    if cache_path not in _model_cache:
        import tf_keras
        weights_path = _download_weights(cache_path)
        _model_cache[cache_path] = tf_keras.models.load_model(weights_path, compile=False)
    return _model_cache[cache_path]


def lunet_denoise(wav_16k: np.ndarray, target_sr: int = 16000, cache_path: str = DEFAULT_CACHE_PATH) -> np.ndarray:
    """Denoise a 1D waveform (as produced by this repo's load_audio(): 16kHz,
    DC-removed, bandpass-filtered, peak-normalized) using the pretrained
    LU-Net model, following the authors' own preprocessing convention.

    Returns a waveform at the SAME length and sample rate as the input.
    """
    wav = np.asarray(wav_16k, dtype=np.float64)
    n_samples_in = len(wav)

    if np.max(np.abs(wav)) < 1e-8:
        return wav.astype(np.float32)  # nothing to denoise in silence -- same guard as wavelet_denoiser.py

    wav_1k = librosa.resample(wav, orig_sr=target_sr, target_sr=LUNET_SR)
    n_segments = len(wav_1k) // LUNET_SEGMENT_SAMPLES
    used_samples_1k = n_segments * LUNET_SEGMENT_SAMPLES

    model = _get_model(cache_path)

    if n_segments > 0:
        segments = wav_1k[:used_samples_1k].reshape(n_segments, LUNET_SEGMENT_SAMPLES)
        peaks = np.maximum(np.max(np.abs(segments), axis=1, keepdims=True), 1e-8)
        segments_norm = segments / peaks
        model_input = segments_norm[..., np.newaxis].astype(np.float32)  # (n_segments, 800, 1)
        denoised_norm = model.predict(model_input, verbose=0)[..., 0]  # (n_segments, 800)
        denoised_segments = denoised_norm * peaks  # undo the per-segment normalization
        denoised_1k = denoised_segments.reshape(-1)
    else:
        denoised_1k = np.zeros(0, dtype=np.float64)

    denoised_16k = librosa.resample(denoised_1k, orig_sr=LUNET_SR, target_sr=target_sr) if n_segments > 0 else np.zeros(0)

    # Fill in whatever length is missing (the trailing <0.8s LU-Net's own
    # convention drops, plus any resampling round-off) with the ORIGINAL,
    # non-denoised audio at that position -- disclosed adaptation, not a
    # silent gap or a loop.
    out = np.array(wav, dtype=np.float64, copy=True)
    n_denoised = min(len(denoised_16k), n_samples_in)
    out[:n_denoised] = denoised_16k[:n_denoised]

    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 160000)
    clean = 0.3 * np.sin(2 * np.pi * 2 * t)
    noisy = (clean + 0.4 * rng.standard_normal(160000)).astype(np.float32)
    peak = np.max(np.abs(noisy))
    if peak > 0:
        noisy = noisy / peak
    out = lunet_denoise(noisy)
    mse_before = np.mean((noisy - clean) ** 2)
    mse_after = np.mean((out - clean) ** 2)
    print(f"LU-Net: MSE {mse_before:.4f} -> {mse_after:.4f} "
          f"({'improved' if mse_after < mse_before else 'WORSE'}), output shape {out.shape}")
