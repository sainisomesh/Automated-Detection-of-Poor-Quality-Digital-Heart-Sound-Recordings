"""
AudioSet-pretrained embedding backbones for the backbone-swap experiment
(Reviewer #7, Comment 2): PANNs CNN14, YAMNet and HuBERT, compared with AST.

Each wrapper has the same interface:
    backbone.embedding_dim        -> int
    backbone(wav) -> (B, embedding_dim) float tensor

`wav` is a (B, num_samples) float32 tensor at 16 kHz that has gone through
the same preprocessing as every other method in this package (`load_audio`
in ../../../src/train_per_lambda_cv.py, copied into train_backbone_swap_cv.py):
DC offset removed, 20-1000 Hz bandpass, peak-normalized to [-1, 1], and
looped/cropped to 10 s (160000 samples). Each backbone then applies its own
front-end (mel-spectrogram for PANNs/YAMNet, convolutional waveform encoder
for HuBERT). Noise mixing and cross-validation are the same for all backbones.

Frozen backbones are kept in eval mode
All three backbones have layers whose behaviour depends on
`nn.Module.training` regardless of `requires_grad`:

  - PANNs CNN14: BatchNorm2d (bn0 and one per ConvBlock), five
    `F.dropout(..., training=self.training)` calls, and SpecAugment.
  - YAMNet: BatchNorm in every convolutional block.
  - HuBERT (ALM/hubert-base-audioset): the config sets
    `apply_spec_augment=True`, so time/feature masking is applied in train
    mode, plus several p=0.1 dropout layers.

The training loop calls `model.train()` every epoch so the head's dropout is
active, and `train()` recurses into submodules. Without an override, a frozen
backbone would have its BatchNorm running statistics updated and HuBERT would
mask its hidden states, so the embeddings would change during training. Each
wrapper therefore overrides `train()` to stay in eval mode when frozen.
"""

import contextlib
import os
import tempfile
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

TARGET_SR = 16000  # sample rate of the waveform this module receives


class _TogglableFreezeBackbone(nn.Module):
    """Base class for the three wrappers; the mode is fixed at construction.

    freeze=True (default): the module stays in eval mode whatever mode the
    parent model is set to, and forward() runs the backbone under
    torch.no_grad() (see the module docstring).

    freeze=False: a normal nn.Module. train()/eval() propagate as usual and
    forward() does not force no_grad, so the backbone is fine-tuned.
    """

    def __init__(self, freeze: bool):
        super().__init__()
        self._freeze = freeze

    def train(self, mode=True):
        if self._freeze:
            return super().train(False)
        return super().train(mode)


# PANNs CNN14 (Kong et al., IEEE/ACM TASLP 2020)
# github.com/qiuqiangkong/audioset_tagging_cnn. Checkpoint "Cnn14_mAP=0.431.pth"
# from Zenodo record 3987831. The architecture comes from the `panns_inference`
# package, which includes the authors' model.py.
PANNS_SAMPLE_RATE = 32000  # the released Cnn14 checkpoint was trained at 32 kHz
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_CHECKPOINT_PATH = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
# Fallback source for the same checkpoint when Zenodo is unreachable.
# 'nicofarr/panns_Cnn14' on the Hugging Face Hub re-hosts the official weights.
# It was saved with PyTorchModelHubMixin, which wraps Cnn14 as `self.backbone`,
# so every key has a "backbone." prefix that _load_state_dict strips. Its keys
# and shapes match Cnn14 (fc_audioset is (527, 2048)).
PANNS_HF_MIRROR_REPO = "nicofarr/panns_Cnn14"


PANNS_LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


def _ensure_panns_labels_csv():
    """Fetch the AudioSet label list that panns_inference reads at import.

    panns_inference downloads it with a shell call to wget, which is missing on
    stock macOS and Windows, so fetch it here first if it is absent or empty.
    """
    path = Path.home() / "panns_data" / "class_labels_indices.csv"
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PANNS_LABELS_URL, path)


class PANNsBackbone(_TogglableFreezeBackbone):
    """PANNs CNN14 embedding extractor, embedding_dim=2048.

    The released checkpoint was trained at 32 kHz and Cnn14's torchlibrosa
    front-end does not resample, so the 16 kHz input is resampled to 32 kHz
    first. Feeding 16 kHz audio directly would shift every mel bin to the
    wrong frequency without raising an error.
    """

    embedding_dim = 2048

    def __init__(self, checkpoint_path: str = None, download_if_missing: bool = True,
                 source: str = "auto", freeze: bool = True):
        """checkpoint_path: local Cnn14 checkpoint; defaults to
        PANNS_CHECKPOINT_PATH.
        source: 'zenodo', 'hf_mirror' (see PANNS_HF_MIRROR_REPO), or 'auto'
        (Zenodo first, then the mirror).
        """
        super().__init__(freeze=freeze)
        _ensure_panns_labels_csv()
        from panns_inference.models import Cnn14

        self.net = Cnn14(
            sample_rate=PANNS_SAMPLE_RATE, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        )

        state_dict = self._load_state_dict(checkpoint_path, download_if_missing, source)
        self.net.load_state_dict(state_dict)

        for p in self.net.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.eval()
        else:
            # Keep the STFT/mel front-end frozen. torchlibrosa stores these
            # fixed DSP matrices as nn.Parameter (so .to(device) moves them),
            # and the loop above would make them trainable. Their gradients
            # are orders of magnitude larger than those of the conv/BN layers
            # (test_backbones.py check 8 guards against this).
            # fc_audioset (the 527-class AudioSet head) is unused, since
            # forward() only reads out["embedding"], so it is frozen too to
            # keep the trainable parameter count accurate.
            for p in self.net.spectrogram_extractor.parameters():
                p.requires_grad = False
            for p in self.net.logmel_extractor.parameters():
                p.requires_grad = False
            for p in self.net.fc_audioset.parameters():
                p.requires_grad = False

    @staticmethod
    def _load_state_dict(checkpoint_path, download_if_missing, source):
        ckpt_path = Path(checkpoint_path) if checkpoint_path else PANNS_CHECKPOINT_PATH
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            return checkpoint["model"]

        if not download_if_missing:
            raise FileNotFoundError(f"PANNs checkpoint not found at {ckpt_path}")

        if source in ("zenodo", "auto"):
            try:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(PANNS_CHECKPOINT_URL, str(ckpt_path), progress=True)
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                return checkpoint["model"]
            except Exception as e:
                if source == "zenodo":
                    raise
                print(f"[PANNsBackbone] Zenodo download failed ({e}); falling back to HF mirror "
                      f"'{PANNS_HF_MIRROR_REPO}'.")

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        hf_path = hf_hub_download(PANNS_HF_MIRROR_REPO, "model.safetensors")
        raw_sd = load_file(hf_path)
        # The mixin prefixes every key with "backbone."; strip it so the
        # state dict matches a bare Cnn14.
        return {k[len("backbone."):]: v for k, v in raw_sd.items()}

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav_32k = torchaudio.functional.resample(wav, TARGET_SR, PANNS_SAMPLE_RATE)
        if self._freeze:
            with torch.no_grad():
                out = self.net(wav_32k)
        else:
            out = self.net(wav_32k)
        return out["embedding"]


# YAMNet (Google AudioSet), via the `torch-vggish-yamnet` PyTorch port
# (github.com/w-hc/torch_audioset).
#
# Caveat: this is a community port of the weights, not an official Google
# release, and we did not re-check its AudioSet mAP. test_backbones.py only
# checks that it loads and gives finite, input-dependent embeddings.
YAMNET_SAMPLE_RATE = 16000  # YAMNet is trained at 16 kHz; no resampling needed


class YAMNetBackbone(_TogglableFreezeBackbone):
    """MobileNet-v1-style YAMNet embedding extractor, embedding_dim=1024.

    Clip-level pooling: YAMNet works on ~0.96 s log-mel patches
    (WaveformToInput returns (num_chunks, 1, 96, 64)) and has no clip-level
    embedding, so the pre-classifier embeddings are averaged over the clip's
    patches to get one vector per 10 s recording.

    WaveformToInput takes one unbatched waveform per call, so forward() loops
    over the batch. This is slower than the other two backbones.
    """

    embedding_dim = 1024

    def __init__(self, freeze: bool = True):
        super().__init__(freeze=freeze)
        from torch_vggish_yamnet.yamnet.model import yamnet
        from torch_vggish_yamnet.input_proc import WaveformToInput

        self.net = yamnet(pretrained=True)
        self.input_proc = WaveformToInput()

        for p in self.net.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.eval()
        else:
            # `net.classifier` is the unused 521-class AudioSet head (as with
            # PANNs' fc_audioset). The mel front-end is in self.input_proc,
            # which has no parameters, so nothing else needs freezing.
            for p in self.net.classifier.parameters():
                p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        embeddings = []
        # When not frozen, use the caller's grad mode so that evaluation under
        # torch.no_grad() does not build autograd graphs.
        ctx = torch.no_grad() if self._freeze else contextlib.nullcontext()
        with ctx:
            for i in range(wav.shape[0]):
                patches = self.input_proc(wav[i:i + 1], YAMNET_SAMPLE_RATE)
                emb, _ = self.net(patches)              # (num_chunks, 1024, 1, 1)
                emb = emb.reshape(emb.shape[0], -1).mean(dim=0)  # (1024,)
                embeddings.append(emb)
        return torch.stack(embeddings, dim=0)


# HuBERT (AudioSet): transformers.HubertModel with the
# "ALM/hubert-base-audioset" weights, a 12-layer HuBERT-base (hidden_size=768)
# with apply_spec_augment=True in its config.
#
# The Hub repo only has a legacy pytorch_model.bin, which transformers 4.57.3
# will only load with torch >= 2.6. Our training container used torch 2.5.1
# for all three backbones, so we loaded from a copy converted to safetensors
# (same weights). The bucket holding that copy is optional; without it the
# code loads the public Hub model, which works with the torch version in
# ../../requirements.txt.
HUBERT_CHECKPOINT = "ALM/hubert-base-audioset"
HUBERT_SAFETENSORS_GCS_BUCKET = "ast-heart-quality-revisions"
HUBERT_SAFETENSORS_GCS_PREFIX = "checkpoints/hubert-base-audioset-safetensors/"
HUBERT_SAFETENSORS_LOCAL_DIR = Path(tempfile.gettempdir()) / "hubert-base-audioset-safetensors"
HUBERT_SAMPLE_RATE = 16000  # HuBERT operates at 16 kHz; no resampling needed


def _ensure_hubert_safetensors_checkpoint() -> str:
    """Return a path or model id for HubertModel.from_pretrained.

    Tries, in order: a local safetensors copy (downloaded once per machine),
    the optional cloud bucket copy, then the public Hub id.
    """
    marker = HUBERT_SAFETENSORS_LOCAL_DIR / "model.safetensors"
    if marker.exists():
        return str(HUBERT_SAFETENSORS_LOCAL_DIR)

    try:
        from google.cloud import storage

        HUBERT_SAFETENSORS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(HUBERT_SAFETENSORS_GCS_BUCKET)
        blobs = [b for b in bucket.list_blobs(prefix=HUBERT_SAFETENSORS_GCS_PREFIX)
                 if not b.name.endswith("/")]
        if not blobs:
            raise FileNotFoundError(
                f"no objects under gs://{HUBERT_SAFETENSORS_GCS_BUCKET}/{HUBERT_SAFETENSORS_GCS_PREFIX}"
            )
        for blob in blobs:
            rel = blob.name[len(HUBERT_SAFETENSORS_GCS_PREFIX):].lstrip("/")
            local_path = HUBERT_SAFETENSORS_LOCAL_DIR / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
        return str(HUBERT_SAFETENSORS_LOCAL_DIR)
    except Exception as exc:
        # Bucket not reachable: use the public checkpoint.
        print(
            f"[HubertBackbone] pre-converted checkpoint unavailable ({type(exc).__name__}); "
            f"loading '{HUBERT_CHECKPOINT}' from the Hugging Face Hub instead."
        )
        return HUBERT_CHECKPOINT


class HubertBackbone(_TogglableFreezeBackbone):
    """HuBERT-base embedding extractor, embedding_dim=768.

    Input normalization: HuBERT is usually given zero-mean, unit-variance
    audio (Wav2Vec2FeatureExtractor). Here it gets the same peak-normalized
    waveform as the other methods, so preprocessing is identical across
    backbones. HuBERT may do somewhat worse than with its usual
    normalization, so its numbers are not an upper bound for HuBERT.

    HuBERT has no [CLS] token, so the clip embedding is the time-average of
    last_hidden_state.
    """

    embedding_dim = 768

    def __init__(self, freeze: bool = True):
        super().__init__(freeze=freeze)
        from transformers import HubertModel

        checkpoint_dir = _ensure_hubert_safetensors_checkpoint()
        self.net = HubertModel.from_pretrained(checkpoint_dir)

        for p in self.net.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.eval()
        else:
            # Keep the CNN feature encoder frozen, as is standard when
            # fine-tuning Wav2Vec2/HuBERT. The task-head classes do this via
            # freeze_feature_encoder(); bare HubertModel lacks that helper,
            # so call the underlying method directly.
            self.net.feature_extractor._freeze_parameters()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self._freeze:
            with torch.no_grad():
                out = self.net(wav)
        else:
            out = self.net(wav)
        return out.last_hidden_state.mean(dim=1)


BACKBONES = {
    "panns": PANNsBackbone,
    "yamnet": YAMNetBackbone,
    "hubert": HubertBackbone,
}


def build_backbone(name: str, freeze: bool = True) -> nn.Module:
    """Build a backbone from BACKBONES by name.

    freeze=True gives a frozen feature extractor; freeze=False gives a
    trainable backbone (fixed front-ends stay frozen).
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}', choose from {list(BACKBONES)}")
    return BACKBONES[name](freeze=freeze)
