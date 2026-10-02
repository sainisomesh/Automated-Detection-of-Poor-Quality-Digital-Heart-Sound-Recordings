"""
LU-Net heart-sound denoiser, candidate 2 of the denoise-then-classify
comparison (Reviewer #7, Comment 1).

Reference: Ali, Shuvo, Al-Manzo, Hasan & Hasan, "An end-to-end deep learning
framework for real-time denoising of heart sounds for cardiac disease
detection in unseen noise," IEEE Access, 2023.
    github.com/ShamsNafisaAli/LU-Net-Heart-Sound-Denoising-

Training-data overlap: LU-Net was trained on PhysioNet 2016 heart sounds
mixed with ICBHI 2017 lung sounds. Our noise also uses ICBHI 2017, so LU-Net
is not fully independent of our test noise. It is included because it is the
only published pretrained PCG denoiser we found, and its results should be
reported with this caveat.

Framework and weights:
  - LU-Net is Keras/TensorFlow. The released `Models/LU-Net.h5` (~16 MB, in
    the authors' repository) contains architecture and weights and is loaded
    as is. It is in the pre-Keras-3 HDF5 format, so it is loaded with the
    `tf_keras` package.
  - The weights are downloaded on first use from raw.githubusercontent.com
    and cached under ~/.cache/ast-heart-quality/.
  - The upstream repository has no LICENSE file, but `Codes/model.py` states
    it is MIT-licensed.

Inference follows the authors' `Codes/config.py`, `Codes/processing_initial.py`
and `Codes/utils.py`:
  - The model runs at 1000 Hz.
  - Input is split into non-overlapping 0.8 s (800-sample) segments; the
    authors keep `len(signal) // 800` full segments and drop the remainder.
  - Each segment is peak-normalized. We normalize each input segment, run
    the model, and scale the output back to the segment's original peak
    (their code skips the last step because it only uses normalized data).

For our 16 kHz, 10 s clips: resample to 1 kHz, denoise in 800-sample
segments, resample back to 16 kHz. 10 s at 1 kHz is 12.5 segments, so the
last ~0.4 s is not denoised and is copied from the input. The output always
has 160,000 samples.
"""

import os

import numpy as np
import librosa

# The .h5 is loaded with `tf_keras` (see _get_model). TF_USE_LEGACY_KERAS is
# not set because it would change `tensorflow.keras` for the whole process.

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

# Loaded models, keyed by weights path.
_model_cache = {}


def _download_weights(cache_path):
    """Fetch LU-Net.h5 into `cache_path` on first use; reuse it afterwards."""
    # Use requests (bundled certifi CAs): on some macOS Python builds urllib
    # fails here with CERTIFICATE_VERIFY_FAILED.
    import requests
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        resp = requests.get(LUNET_WEIGHTS_URL, timeout=60)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(resp.content)
    return cache_path


def _get_model(cache_path=DEFAULT_CACHE_PATH):
    """Load (and memoize) the pretrained LU-Net model.

    `tf_keras` is imported here so TensorFlow is only loaded when LU-Net is
    used. compile=False skips the training optimizer state.
    """
    if cache_path not in _model_cache:
        import tf_keras
        weights_path = _download_weights(cache_path)
        _model_cache[cache_path] = tf_keras.models.load_model(weights_path, compile=False)
    return _model_cache[cache_path]


def lunet_denoise(wav_16k: np.ndarray, target_sr: int = 16000, cache_path: str = DEFAULT_CACHE_PATH) -> np.ndarray:
    """Denoise a 1-D waveform with the pretrained LU-Net model.

    Input is 16 kHz mono, DC-removed, bandpass-filtered and peak-normalized
    (the output of ``load_audio()``).

    Resample to 1 kHz, split into 800-sample segments, peak-normalize each
    segment, run the model, restore each segment's peak, and resample back
    to `target_sr` (see the module docstring).

    Args:
        wav_16k: 1-D float waveform at `target_sr`.
        target_sr: sample rate of the input and of the returned waveform.
        cache_path: local path for the downloaded LU-Net weights.

    Returns:
        Denoised float32 waveform with the same rate and length as the
        input, rescaled to unit peak if it exceeds 1.0. Trailing samples not
        covered by a full 800-sample segment keep the original audio.
        Near-silent input (peak < 1e-8) is returned unchanged, as in
        wavelet_denoiser.py.
    """
    wav = np.asarray(wav_16k, dtype=np.float64)
    n_samples_in = len(wav)

    if np.max(np.abs(wav)) < 1e-8:
        return wav.astype(np.float32)  # nothing to denoise in silence

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

    # Overwrite the denoised prefix of a copy of the input; the trailing
    # <0.8 s (and any resampling round-off) keeps the original audio.
    out = np.array(wav, dtype=np.float64, copy=True)
    n_denoised = min(len(denoised_16k), n_samples_in)
    out[:n_denoised] = denoised_16k[:n_denoised]

    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


if __name__ == "__main__":
    # Quick self-check on a synthetic sine + white noise (downloads weights
    # if needed).
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
