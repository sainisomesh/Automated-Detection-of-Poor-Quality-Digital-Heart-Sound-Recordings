"""
Frozen audio-embedding backbones for the Reviewer #7 Comment 2 backbone-swap
experiment (PANNs CNN14 / YAMNet / HuBERT vs. AST).

Each wrapper exposes one minimal, identical interface:
    backbone.embedding_dim        -> int
    backbone(wav) -> (B, embedding_dim) float tensor

`wav` is a (B, num_samples) float32 tensor at 16 kHz, already run through the
EXACT SAME preprocessing as the rest of this revision's pipeline (DC-removed,
20-1000Hz bandpassed, peak-normalized to [-1,1], looped/cropped to 10s =
160000 samples -- reproducibility/src/train_per_lambda_cv.py's load_audio,
copied verbatim in train_backbone_swap_cv.py). Each backbone then does
whatever its own published front-end expects internally (mel-spectrogram for
PANNs/YAMNet, a raw-waveform conv encoder for HuBERT) -- nothing about our
own noise-mixing/CV pipeline changes, only what happens between "get a
preprocessed waveform" and "get an embedding", matching how the Tang/Giordano
baselines only swapped what happens after load_audio/mix_rms.

CRITICAL BUG CAUGHT BEFORE ANY TRAINING CODE WAS WRITTEN (2026-09-13): all
three backbones have submodules whose behavior depends on nn.Module.training
even though every one of their *parameters* has requires_grad=False:
  - PANNs Cnn14: nn.BatchNorm2d(bn0, and one per ConvBlock) -- in train mode
    BatchNorm uses/updates running statistics from THIS batch; in eval mode
    it uses the frozen pretrained running stats. Also has 5 internal
    F.dropout(..., training=self.training) calls and a SpecAugment layer
    gated on self.training.
  - YAMNet: every conv block's nn.BatchNorm2d has the same issue.
  - HuBERT (ALM/hubert-base-audioset): apply_spec_augment=True in its config
    means HubertModel applies SpecAugment-style time/feature masking to
    hidden states whenever self.training is True, plus several p=0.1
    dropout layers.
If the composite model's .train() is called at the start of each training
epoch (as train_backbone_swap_cv.py does, to put the trainable qa_classifier
head's own Dropout(0.1) into train mode), nn.Module.train() recursively
cascades to every submodule INCLUDING these "frozen" backbones. Without an
explicit guard, that would silently drift PANNs'/YAMNet's BatchNorm running
stats away from their pretrained values over 5 epochs, and inject random
masking noise into HuBERT's embeddings during training -- while giving
numerically different, non-reproducible embeddings for the exact same input
at eval time depending on incidental call order. This would NOT crash or
look wrong; it would just quietly make the backbone not-actually-frozen.
Fix: every backbone class below overrides train() to force eval() no matter
what mode is requested on the parent.
"""

import contextlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

TARGET_SR = 16000  # sample rate of the waveform this module receives


class _TogglableFreezeBackbone(nn.Module):
    """Base class supporting two modes, set once at construction:

    freeze=True (default, Reviewer #7 Comment 2's original ask): pins the
    module in eval() mode permanently regardless of what .train(mode) the
    parent composite model requests, and every forward() call below wraps
    its backbone call in torch.no_grad(). See module docstring for exactly
    why the eval-lock matters (BatchNorm running stats / SpecAugment).

    freeze=False (added 2026-09-14, for the "does AST only win because of
    frozen features?" follow-up ablation -- see ../README.md "Full-unfreeze
    extension"): behaves like a normal nn.Module -- .train()/.eval() cascade
    through it as usual (so BatchNorm updates its running stats and
    HuBERT's SpecAugment/dropout are actually active during training, which
    is the correct behavior for genuine fine-tuning, not a bug), and
    forward() runs without no_grad so gradients reach every backbone param.
    """

    def __init__(self, freeze: bool):
        super().__init__()
        self._freeze = freeze

    def train(self, mode=True):
        if self._freeze:
            return super().train(False)
        return super().train(mode)


# ---------------------------------------------------------------------------
# PANNs CNN14 (Kong et al., IEEE/ACM TASLP 2020)
# github.com/qiuqiangkong/audioset_tagging_cnn, checkpoint hosted on
# Zenodo record 3987831 ("Cnn14_mAP=0.431.pth"). Installed here via the
# official `panns_inference` PyPI package, which vendors the author's exact
# model.py (verified 2026-09-13: Cnn14.__init__ signature, forward() control
# flow, and the Zenodo checkpoint URL were read directly from the installed
# package source, not reconstructed from the paper).
# ---------------------------------------------------------------------------
PANNS_SAMPLE_RATE = 32000  # Cnn14 checkpoint was trained at 32kHz, NOT 16kHz
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_CHECKPOINT_PATH = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
# Zenodo (the canonical host CLAUDE.md Sec 9.3 points at) was unreachable for
# >30 minutes straight (connection timeouts, not just slow -- verified GitHub/HF
# were fine at the same time, so this is a Zenodo-side outage) on 2026-09-13.
# 'nicofarr/panns_Cnn14' on the HF Hub is a direct re-hosting of the exact same
# official checkpoint (explicitly linked to github.com/qiuqiangkong/audioset_tagging_cnn
# in its model card, pushed via PyTorchModelHubMixin) -- verified by inspecting
# its state_dict keys/shapes (fc_audioset: (527, 2048), matching classes_num=527
# and embedding_dim=2048 exactly) before trusting it, not just the repo name.
# Its keys are prefixed "backbone." (the mixin wraps Cnn14 as self.backbone),
# stripped below before loading into our own bare Cnn14 instance.
PANNS_HF_MIRROR_REPO = "nicofarr/panns_Cnn14"


class PANNsBackbone(_TogglableFreezeBackbone):
    """CNN14, embedding_dim=2048.

    ADAPTATION (disclosed): Cnn14's internal Spectrogram/LogmelFilterBank
    (torchlibrosa) do NOT resample -- they assume the input is already at
    the sample_rate passed to the constructor (32000, matching the released
    checkpoint). Our pipeline produces 16kHz waveforms, so this wrapper
    resamples 16kHz -> 32kHz on the fly (torchaudio.functional.resample)
    before calling the backbone. Feeding 16kHz audio into a model configured
    for 32kHz without resampling would silently halve every frequency bin's
    true location in the mel filterbank -- a real bug we are avoiding here,
    not a hypothetical one.
    """

    embedding_dim = 2048

    def __init__(self, checkpoint_path: str = None, download_if_missing: bool = True,
                 source: str = "auto", freeze: bool = True):
        """source: 'zenodo' (canonical, CLAUDE.md Sec 9.3), 'hf_mirror' (fallback,
        see PANNS_HF_MIRROR_REPO above), or 'auto' (try zenodo first, fall back
        to the HF mirror if it's unreachable -- e.g. the 2026-09-13 outage)."""
        super().__init__(freeze=freeze)
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
            # Cnn14 itself constructs spectrogram_extractor/logmel_extractor with
            # freeze_parameters=True (verified directly in the installed
            # panns_inference source: torchlibrosa's Spectrogram/LogmelFilterBank
            # register their STFT/mel-filterbank basis matrices as nn.Parameter
            # purely so .to(device) moves them, NOT so they can be learned -- these
            # are deterministic DSP transforms). The blanket "unfreeze every net
            # param" above would otherwise silently start gradient-updating that
            # fixed mel-filterbank matrix -- caught in testing via an astronomically
            # large gradient (~5e11) on logmel_extractor.melW, versus O(1-100) on
            # every genuinely learnable conv/BN param. Re-lock the DSP front-end
            # explicitly. Also freeze fc_audioset (the original 527-class AudioSet
            # head): forward() below only reads out["embedding"], never
            # out["clipwise_output"], so fc_audioset never receives a gradient
            # regardless -- explicitly freezing it keeps "trainable backbone
            # params" an accurate count of what full-unfreeze actually fine-tunes.
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
        # mixin prefixes every key with "backbone." -- strip it to match our bare Cnn14
        return {k[len("backbone."):]: v for k, v in raw_sd.items()}

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav_32k = torchaudio.functional.resample(wav, TARGET_SR, PANNS_SAMPLE_RATE)
        if self._freeze:
            with torch.no_grad():
                out = self.net(wav_32k)
        else:
            out = self.net(wav_32k)
        return out["embedding"]


# ---------------------------------------------------------------------------
# YAMNet (Google AudioSet) -- community PyTorch port `torch-vggish-yamnet`
# (github.com/w-hc/torch_audioset). NOT an official Google release -- see
# test_backbones.py's sanity check and PAPER_REVISIONS/README.md open
# question #4 before treating its embeddings as an uncontroversial stand-in
# for the official TF YAMNet.
# ---------------------------------------------------------------------------
YAMNET_SAMPLE_RATE = 16000  # trained at 16kHz -- matches our pipeline exactly


class YAMNetBackbone(_TogglableFreezeBackbone):
    """MobileNet-v1-style YAMNet, embedding_dim=1024.

    YAMNet's own input pipeline (torch_vggish_yamnet.input_proc.WaveformToInput)
    slices a clip into non-overlapping 0.96s log-mel patches and expects ONE
    unbatched (channels, time) waveform per call, returning (num_chunks, 1,
    96, 64). There is no single-vector "clip embedding" in the reference
    implementation -- it classifies each 0.96s patch independently. We take
    the mean embedding across a clip's patches as its single feature vector
    (a disclosed adaptation, analogous to AST's own single [CLS] embedding
    standing in for the whole 10s clip).

    PERFORMANCE NOTE (not a correctness issue): this loops over the batch
    dimension in Python because WaveformToInput doesn't support batched
    input. Fine for the CPU smoke test in test_backbones.py; worth batching
    properly before submitting a real Vertex AI training job if this backbone
    turns out to be a throughput bottleneck. When freeze=False this loop
    still builds one autograd graph per sample rather than a single batched
    graph -- functionally correct (gradients still reach every conv param
    through the loop) but slower per step than the other two backbones.
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
            # Same pattern as PANNs' fc_audioset above: `net.classifier` is
            # YAMNet's original 521-class AudioSet head. forward() below only
            # reads the pre-classifier embedding (mean-pooled conv features),
            # never net.classifier's output, so it never receives a gradient
            # regardless of requires_grad -- explicitly freezing it keeps
            # "trainable backbone params" an accurate count of what
            # full-unfreeze actually fine-tunes (the 14 MobileNet conv
            # layers), not dead optimizer state. No fixed DSP-transform
            # parameters live inside net itself (unlike PANNs' torchlibrosa
            # front-end) -- the mel computation happens in self.input_proc,
            # which is not an nn.Module with parameters at all.
            for p in self.net.classifier.parameters():
                p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        embeddings = []
        # freeze=True: force no_grad (matches PANNs/HubertBackbone). freeze=False:
        # inherit whatever ambient grad context the caller is in -- nullcontext(),
        # NOT torch.enable_grad(). An earlier version of this used
        # torch.enable_grad() here, which would have force-overridden
        # evaluate_fold()'s `with torch.no_grad():` wrapper during evaluation,
        # needlessly building an autograd graph over this per-sample loop with
        # no backward() ever called on it -- wasted memory/compute, not a
        # correctness bug (eval never backprops), but inconsistent with how
        # PANNsBackbone/HubertBackbone correctly just inherit ambient context
        # instead of forcing grad on unconditionally.
        ctx = torch.no_grad() if self._freeze else contextlib.nullcontext()
        with ctx:
            for i in range(wav.shape[0]):
                patches = self.input_proc(wav[i:i + 1], YAMNET_SAMPLE_RATE)
                emb, _ = self.net(patches)              # (num_chunks, 1024, 1, 1)
                emb = emb.reshape(emb.shape[0], -1).mean(dim=0)  # (1024,)
                embeddings.append(emb)
        return torch.stack(embeddings, dim=0)


# ---------------------------------------------------------------------------
# HuBERT (AudioSet) -- transformers.HubertModel.from_pretrained(
# "ALM/hubert-base-audioset"). Config verified 2026-09-13 (hidden_size=768,
# standard 12-layer HuBERT-base, apply_spec_augment=True -- see module
# docstring for why that matters).
#
# The Hub repo only ships a legacy "pytorch_model.bin" (no .safetensors --
# verified via the HF Hub API's /files listing, not assumed). transformers
# 4.57.3's from_pretrained() refuses to torch.load() any non-safetensors
# checkpoint unless torch>=2.6 (CVE-2025-32434 guard), and this project's
# Docker image is pinned to torch 2.5.1 (base image
# pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime) to stay identical across all
# three backbone-swap jobs -- bumping torch just for HuBERT would make the
# ablation run on two different torch versions, an inconsistency CLAUDE.md
# Sec 10.7 says must never happen silently. So instead of touching the image,
# the checkpoint was converted to safetensors OFFLINE, on 2026-09-13, using a
# local venv with torch 2.9.1 (which is not subject to the guard): loaded via
# HubertModel.from_pretrained("ALM/hubert-base-audioset"), then
# model.save_pretrained(..., safe_serialization=True). Verified bit-identical
# (max abs weight diff == 0.0 against the original .bin) before uploading to
# gs://ast-heart-quality-revisions/checkpoints/hubert-base-audioset-safetensors/.
# Loading a local safetensors file never calls torch.load, so the guard never
# triggers, regardless of the container's torch version.
# ---------------------------------------------------------------------------
HUBERT_CHECKPOINT = "ALM/hubert-base-audioset"
HUBERT_SAFETENSORS_GCS_BUCKET = "ast-heart-quality-revisions"
HUBERT_SAFETENSORS_GCS_PREFIX = "checkpoints/hubert-base-audioset-safetensors/"
HUBERT_SAFETENSORS_LOCAL_DIR = Path("/tmp/hubert-base-audioset-safetensors")
HUBERT_SAMPLE_RATE = 16000  # matches our pipeline exactly, no resampling needed


def _ensure_hubert_safetensors_checkpoint() -> str:
    """Download the pre-converted safetensors HuBERT checkpoint from GCS to
    a local dir (once per container, cached across folds) and return its
    path. See the comment above HUBERT_CHECKPOINT for why this exists."""
    marker = HUBERT_SAFETENSORS_LOCAL_DIR / "model.safetensors"
    if marker.exists():
        return str(HUBERT_SAFETENSORS_LOCAL_DIR)

    from google.cloud import storage

    HUBERT_SAFETENSORS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(HUBERT_SAFETENSORS_GCS_BUCKET)
    for blob in bucket.list_blobs(prefix=HUBERT_SAFETENSORS_GCS_PREFIX):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(HUBERT_SAFETENSORS_GCS_PREFIX):].lstrip("/")
        local_path = HUBERT_SAFETENSORS_LOCAL_DIR / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
    return str(HUBERT_SAFETENSORS_LOCAL_DIR)


class HubertBackbone(_TogglableFreezeBackbone):
    """HuBERT-base, embedding_dim=768.

    ADAPTATION (disclosed, deliberate): HuBERT is normally fed audio
    normalized by Wav2Vec2FeatureExtractor (zero-mean, unit-variance),
    not peak-normalized to [-1,1]. We deliberately feed it the exact same
    peak-normalized waveform every other method in this revision receives
    (from load_audio/mix_rms) rather than adding a HuBERT-specific
    normalization step -- Reviewer #7 Comment 2 asks for an apples-to-apples
    backbone comparison under identical preprocessing, and adding a
    backbone-specific input transform would reopen exactly the kind of
    "is this actually a fair comparison" question already resolved for the
    noise-mixing pipeline. This may cost HuBERT some absolute performance
    relative to its own typical usage convention; that tradeoff is the
    price of comparability and must be stated in the writeup, not hidden.

    HuBERT has no [CLS] token (unlike AST's ViT-style encoder), so the
    per-clip embedding is the mean of last_hidden_state over the time axis.
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
            # Official HF convention for HuBERT/Wav2Vec2-family fine-tuning
            # (verified 2026-09-14 directly in transformers.models.hubert.
            # modeling_hubert -- HubertForCTC/HubertForSequenceClassification's
            # freeze_feature_encoder(), which literally calls
            # `self.hubert.feature_extractor._freeze_parameters()`; not exposed
            # on the bare HubertModel class we use, so replicated here) is to
            # keep the low-level CNN feature_extractor frozen even during full
            # fine-tuning -- only feature_projection + the transformer encoder
            # get fine-tuned. This mirrors the original Wav2Vec2 paper's
            # fine-tuning recipe (the raw-waveform conv front-end is treated
            # like a fixed feature extractor, analogous to why PANNs' own
            # authors freeze their torchlibrosa DSP front-end -- see
            # PANNsBackbone above for that parallel finding).
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
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}', choose from {list(BACKBONES)}")
    return BACKBONES[name](freeze=freeze)
