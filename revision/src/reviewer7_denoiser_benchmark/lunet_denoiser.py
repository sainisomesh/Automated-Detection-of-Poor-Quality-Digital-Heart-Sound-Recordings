"""
LU-Net heart-sound denoiser -- Candidate 2 of the denoise-then-classify
comparison (Reviewer #7, Comment 1).

Reference: Ali, Shuvo, Al-Manzo, Hasan & Hasan, "An end-to-end deep learning
framework for real-time denoising of heart sounds for cardiac disease
detection in unseen noise," IEEE Access, 2023.
    github.com/ShamsNafisaAli/LU-Net-Heart-Sound-Denoising-

TRAINING-DATA OVERLAP CAVEAT (important for interpreting any result from this
candidate). LU-Net was trained on PhysioNet 2016 heart sounds mixed with
ICBHI 2017 lung sounds. The noise construction evaluated in this paper also
draws its biological interference from ICBHI 2017, so LU-Net's performance
here is not fully independent of its own training distribution. The candidate
is included deliberately -- it is the only published, pretrained PCG denoiser
available for this comparison -- but its numbers must always be reported
together with this overlap, never as a clean independent comparison.

Framework and weights:
  - LU-Net is Keras/TensorFlow, not PyTorch. The released `Models/LU-Net.h5`
    (~16 MB, committed in the authors' GitHub repository) is loaded whole --
    architecture and weights together -- so nothing about the architecture is
    reimplemented here and there is no porting risk. Because the file is in
    the legacy pre-Keras-3 HDF5 format, it is loaded through the standalone
    `tf_keras` compatibility package rather than modern `keras`.
  - The weights are fetched on first use from raw.githubusercontent.com and
    cached under ~/.cache/ast-heart-quality/; they are not vendored into this
    repository.
  - The upstream repository has no top-level LICENSE file, but
    `Codes/model.py` states "This code is licensed under the terms of the
    MIT-license"; attribution is preserved here accordingly.

Inference recipe, following the authors' own `Codes/config.py`,
`Codes/processing_initial.py` and `Codes/utils.py`:
  - The model operates at 1000 Hz, not 16 kHz.
  - It consumes fixed 0.8 s (800-sample) non-overlapping segments; the
    authors' loader takes `k = len(signal) // 800` full segments and drops
    any remainder rather than padding a partial trailing segment.
  - Every segment is peak-normalized before use in their pipeline. At
    inference time that means normalizing each input segment, running the
    model, then rescaling the output back to that segment's original peak.
    Their own training/evaluation code omits the rescaling step because it
    only ever handles already-normalized data; it is needed here because the
    evaluation audio has its own amplitude scale.

Adaptation to this paper's 16 kHz / 10 s clips: resample to 1 kHz, denoise in
800-sample segments, then resample the denoised result back to 16 kHz. A 10 s
clip at 1 kHz is 10,000 samples, which is 12.5 segments, so the trailing
partial segment (~0.4 s) that LU-Net's own convention would discard is filled
back in from the ORIGINAL, non-denoised audio at that position. The output is
therefore always the expected 160,000 samples, without inserting silence or
looping content -- but note that a short tail of every clip passes through
un-denoised.
"""

import os

import numpy as np
import librosa

# The legacy-format .h5 is loaded via the standalone `tf_keras` package (see
# _get_model below), which provides its own Keras-2-compatible API surface.
# Note that the TF_USE_LEGACY_KERAS environment variable is deliberately NOT
# set here: it would redirect `tensorflow.keras` to the legacy implementation
# process-wide, affecting any other Keras model loaded in the same process,
# and it is unnecessary when importing `tf_keras` directly.

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

# Loaded models are cached per weights path rather than in a single global
# slot, so that a call with a different cache_path loads the model it asked
# for instead of returning whichever model was loaded first.
_model_cache = {}


def _download_weights(cache_path):
    """Fetch LU-Net.h5 into `cache_path` on first use; reuse it afterwards."""
    # `requests` is used rather than urllib because it bundles its own certifi
    # CA bundle. Some macOS Python builds do not wire urllib to the system
    # trust store, which makes urllib.request.urlretrieve() fail with
    # CERTIFICATE_VERIFY_FAILED on this URL.
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

    `tf_keras` is imported lazily so that importing this module does not pull
    in TensorFlow unless LU-Net is actually used. The released .h5 predates
    Keras 3, so `tf_keras.models.load_model` is required to read it;
    compile=False skips reconstructing the training-time optimizer state,
    which inference does not need.
    """
    if cache_path not in _model_cache:
        import tf_keras
        weights_path = _download_weights(cache_path)
        _model_cache[cache_path] = tf_keras.models.load_model(weights_path, compile=False)
    return _model_cache[cache_path]


def lunet_denoise(wav_16k: np.ndarray, target_sr: int = 16000, cache_path: str = DEFAULT_CACHE_PATH) -> np.ndarray:
    """Denoise a 1-D waveform with the pretrained LU-Net model.

    The input is expected to have already been through this benchmark's
    ``load_audio()``: 16 kHz mono, DC-removed, bandpass-filtered and
    peak-normalized.

    Processing follows the authors' convention (see the module docstring):
    resample to 1 kHz, split into 800-sample segments, peak-normalize each
    segment, run the model, restore each segment's original peak, then
    resample back to `target_sr`.

    Args:
        wav_16k: 1-D float waveform at `target_sr`.
        target_sr: sample rate of the input and of the returned waveform.
        cache_path: local path for the downloaded LU-Net weights.

    Returns:
        Denoised waveform (float32) at the same sample rate and exactly the
        same length as the input, rescaled to unit peak if it exceeds 1.0.
        Any trailing samples not covered by a whole 800-sample segment (plus
        resampling round-off) retain the ORIGINAL, non-denoised audio.
        Near-silent input (peak < 1e-8) is returned unchanged, matching
        wavelet_denoiser.py's behaviour.
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

    # Start from a copy of the original waveform and overwrite the denoised
    # prefix, so whatever length is missing -- the trailing <0.8 s that
    # LU-Net's own segmenting convention drops, plus any resampling round-off
    # -- keeps the original, non-denoised audio instead of a gap or a loop.
    out = np.array(wav, dtype=np.float64, copy=True)
    n_denoised = min(len(denoised_16k), n_samples_in)
    out[:n_denoised] = denoised_16k[:n_denoised]

    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


if __name__ == "__main__":
    # Quick self-check: downloads the weights if needed and confirms the
    # denoiser runs end to end, preserves length, and reduces the distance to
    # a known clean signal on synthetic sine + white noise.
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
