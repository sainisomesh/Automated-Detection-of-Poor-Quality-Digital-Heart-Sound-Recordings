"""
AudioSet-pretrained audio-embedding backbones for the Reviewer #7 Comment 2
backbone-swap experiment (PANNs CNN14 / YAMNet / HuBERT vs. AST).

Each wrapper exposes one minimal, identical interface:
    backbone.embedding_dim        -> int
    backbone(wav) -> (B, embedding_dim) float tensor

`wav` is a (B, num_samples) float32 tensor at 16 kHz that has already been
through the same preprocessing as every other method in this package: DC
offset removed, 20-1000 Hz bandpass, peak-normalized to [-1, 1], and
looped/cropped to 10 s (160000 samples). That pipeline is `load_audio` in
../../../src/train_per_lambda_cv.py, copied verbatim into
train_backbone_swap_cv.py. Each backbone then applies whatever front-end its
own published implementation expects (a mel-spectrogram for PANNs/YAMNet, a
raw-waveform convolutional encoder for HuBERT). Nothing about the
noise-mixing or cross-validation pipeline changes between backbones: the only
substitution is what happens between "preprocessed waveform in" and
"embedding out", mirroring how the Tang/Giordano classical baselines swap only
what happens after load_audio/mix_rms.

Why every frozen backbone is also eval()-locked
-----------------------------------------------
All three backbones contain submodules whose forward behaviour depends on
`nn.Module.training`, independently of whether their parameters have
`requires_grad=False`:

  - PANNs CNN14: `nn.BatchNorm2d` (bn0, plus one per ConvBlock). In train mode
    BatchNorm normalizes using the current batch's statistics and updates its
    running estimates; in eval mode it uses the pretrained running statistics.
    Cnn14 additionally contains five `F.dropout(..., training=self.training)`
    calls and a SpecAugment layer gated on `self.training`.
  - YAMNet: the same BatchNorm behaviour in every convolutional block.
  - HuBERT (ALM/hubert-base-audioset): its config sets
    `apply_spec_augment=True`, so HubertModel applies SpecAugment-style
    time/feature masking to hidden states whenever `self.training` is True,
    on top of several p=0.1 dropout layers.

The training loop calls `model.train()` at the start of every epoch so that
the trainable head's own `Dropout(0.1)` is active. `nn.Module.train()`
recurses into every submodule, so it would also put a nominally frozen
backbone into train mode. That would let PANNs'/YAMNet's BatchNorm running
statistics drift away from their pretrained values across epochs and inject
random masking into HuBERT's embeddings, and would make the embedding for a
given input depend on which mode the module happened to be in -- all without
raising an error. To keep the "frozen backbone" premise strictly true, each
wrapper below overrides `train()` so that a frozen backbone stays in eval
mode regardless of the mode requested by the parent module.
"""

import contextlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

TARGET_SR = 16000  # sample rate of the waveform this module receives


class _TogglableFreezeBackbone(nn.Module):
    """Common base for the three wrappers, supporting two modes fixed at
    construction time.

    freeze=True (default): the module is pinned in eval() mode regardless of
    the mode requested on the parent composite model, and each subclass's
    forward() wraps the backbone call in torch.no_grad(). See the module
    docstring for why the eval lock is required for the frozen comparison to
    mean what it says (BatchNorm running statistics, SpecAugment, dropout).

    freeze=False: the module behaves like an ordinary nn.Module -- train()
    and eval() cascade through it normally, so BatchNorm updates its running
    statistics and HuBERT's SpecAugment/dropout are active during training,
    which is the intended behaviour for genuine fine-tuning -- and forward()
    does not force no_grad, so gradients reach every unfrozen backbone
    parameter.
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
# github.com/qiuqiangkong/audioset_tagging_cnn; the released checkpoint
# ("Cnn14_mAP=0.431.pth") is hosted on Zenodo record 3987831. The architecture
# comes from the `panns_inference` PyPI package, which vendors the authors'
# own model.py unmodified.
# ---------------------------------------------------------------------------
PANNS_SAMPLE_RATE = 32000  # the released Cnn14 checkpoint was trained at 32 kHz
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_CHECKPOINT_PATH = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
# Secondary source for the same checkpoint, used when Zenodo is unreachable.
# 'nicofarr/panns_Cnn14' on the Hugging Face Hub re-hosts the official
# checkpoint (its model card links back to
# github.com/qiuqiangkong/audioset_tagging_cnn) and was pushed via
# PyTorchModelHubMixin, so every key is prefixed "backbone." because the mixin
# wraps Cnn14 as `self.backbone`; the prefix is stripped in _load_state_dict
# before loading into a bare Cnn14. Equivalence was confirmed by checking the
# state_dict keys and shapes (fc_audioset is (527, 2048), matching
# classes_num=527 and embedding_dim=2048) rather than by trusting the repo name.
PANNS_HF_MIRROR_REPO = "nicofarr/panns_Cnn14"


class PANNsBackbone(_TogglableFreezeBackbone):
    """PANNs CNN14 embedding extractor, embedding_dim=2048.

    Sample-rate adaptation (disclosed): the released CNN14 checkpoint was
    trained at 32 kHz, and Cnn14's internal torchlibrosa
    Spectrogram/LogmelFilterBank modules do not resample -- they assume the
    input is already at the `sample_rate` passed to the constructor. This
    pipeline produces 16 kHz waveforms, so the wrapper resamples 16 kHz ->
    32 kHz with torchaudio.functional.resample before calling the backbone.
    Passing 16 kHz audio to a front-end configured for 32 kHz would halve the
    true frequency of every mel bin without any error being raised, so the
    resampling step is what makes the pretrained filterbank meaningful here.
    """

    embedding_dim = 2048

    def __init__(self, checkpoint_path: str = None, download_if_missing: bool = True,
                 source: str = "auto", freeze: bool = True):
        """checkpoint_path: explicit path to a local Cnn14 checkpoint; when
        omitted, PANNS_CHECKPOINT_PATH is used as the cache location.
        source: 'zenodo' (the authors' canonical host), 'hf_mirror' (the
        Hugging Face re-hosting, see PANNS_HF_MIRROR_REPO), or 'auto' (try
        Zenodo first and fall back to the mirror if it is unreachable).
        """
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
            # Even when fine-tuning end to end, the STFT/mel front-end stays
            # frozen. Cnn14 constructs spectrogram_extractor/logmel_extractor
            # with freeze_parameters=True: torchlibrosa registers their
            # STFT and mel-filterbank basis matrices as nn.Parameter only so
            # that .to(device) moves them, not because they are learnable --
            # they are deterministic DSP transforms. The blanket
            # "requires_grad = not freeze" loop above would otherwise start
            # gradient-updating the fixed filterbank, which produces gradient
            # magnitudes many orders of magnitude larger than any genuinely
            # learnable conv/BN parameter (test_backbones.py check 8 asserts a
            # sane bound for exactly this reason). fc_audioset, the original
            # 527-class AudioSet head, is frozen for a different reason:
            # forward() reads only out["embedding"] and never
            # out["clipwise_output"], so fc_audioset can never receive a
            # gradient anyway, and freezing it keeps the reported count of
            # trainable backbone parameters equal to what is actually
            # fine-tuned.
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


# ---------------------------------------------------------------------------
# YAMNet (Google AudioSet), via the `torch-vggish-yamnet` PyTorch port
# (github.com/w-hc/torch_audioset).
#
# CAVEAT TO CARRY INTO ANY INTERPRETATION OF THE YAMNET RESULTS: this is a
# community port of the weights, not an official Google release, and its
# AudioSet mAP was not independently revalidated in this work. Its embeddings
# are therefore treated as a close but unverified stand-in for the official
# TensorFlow YAMNet. test_backbones.py checks that the port loads and produces
# input-dependent, finite embeddings, which is a sanity check, not a
# replication of the original AudioSet evaluation.
# ---------------------------------------------------------------------------
YAMNET_SAMPLE_RATE = 16000  # YAMNet is trained at 16 kHz; no resampling needed


class YAMNetBackbone(_TogglableFreezeBackbone):
    """MobileNet-v1-style YAMNet embedding extractor, embedding_dim=1024.

    Clip-level pooling (disclosed adaptation): YAMNet has no single
    clip-level embedding. Its input pipeline
    (torch_vggish_yamnet.input_proc.WaveformToInput) slices a clip into
    non-overlapping ~0.96 s log-mel patches, returning (num_chunks, 1, 96,
    64), and the model classifies each patch independently. To obtain one
    feature vector per 10 s recording, this wrapper averages the
    pre-classifier embeddings across a clip's patches -- the analogue of
    AST's single [CLS] embedding standing in for the whole clip.

    Performance note: WaveformToInput accepts only one unbatched
    (channels, time) waveform per call, so forward() loops over the batch
    dimension in Python. This is correct but slower per step than the other
    two backbones, and with freeze=False it builds one autograd graph per
    sample rather than a single batched graph.
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
            # Same rationale as PANNs' fc_audioset: `net.classifier` is
            # YAMNet's original 521-class AudioSet head, and forward() reads
            # only the pre-classifier embedding, so the head can never
            # receive a gradient. Freezing it explicitly keeps the count of
            # trainable backbone parameters equal to what is actually
            # fine-tuned (the MobileNet convolutional stack) instead of
            # carrying dead optimizer state. Unlike PANNs there is no fixed
            # DSP transform to re-lock inside `net`: the mel computation lives
            # in self.input_proc, which holds no nn.Parameters.
            for p in self.net.classifier.parameters():
                p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        embeddings = []
        # freeze=True: force no_grad, matching PANNsBackbone/HubertBackbone.
        # freeze=False: use nullcontext() so the loop inherits the caller's
        # ambient autograd context rather than forcing gradients on. This
        # matters because evaluate_fold() wraps inference in
        # `with torch.no_grad():`; forcing torch.enable_grad() here would
        # override that and build an autograd graph per sample that is never
        # backpropagated, wasting memory and time during evaluation.
        ctx = torch.no_grad() if self._freeze else contextlib.nullcontext()
        with ctx:
            for i in range(wav.shape[0]):
                patches = self.input_proc(wav[i:i + 1], YAMNET_SAMPLE_RATE)
                emb, _ = self.net(patches)              # (num_chunks, 1024, 1, 1)
                emb = emb.reshape(emb.shape[0], -1).mean(dim=0)  # (1024,)
                embeddings.append(emb)
        return torch.stack(embeddings, dim=0)


# ---------------------------------------------------------------------------
# HuBERT (AudioSet): transformers.HubertModel with the
# "ALM/hubert-base-audioset" weights -- a standard 12-layer HuBERT-base
# (hidden_size=768) whose config sets apply_spec_augment=True (see the module
# docstring for why that matters when the backbone is frozen).
#
# Checkpoint format note: the Hub repo ships only a legacy "pytorch_model.bin"
# with no safetensors variant, and transformers 4.57.3 (the version pinned in
# ../../requirements.txt) refuses to torch.load a non-safetensors checkpoint
# unless torch >= 2.6. The training container is pinned to torch 2.5.1
# (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime) so that all three
# backbone-swap runs execute on an identical torch version; raising torch for
# HuBERT alone would have made one arm of the comparison run on a different
# numerical stack. The weights were therefore converted to safetensors
# out-of-band -- loaded with a torch version not subject to the guard and
# re-saved via save_pretrained(..., safe_serialization=True), checked to be
# bit-identical to the original .bin -- and the converted directory is staged
# in the project's cloud bucket. Loading a local safetensors directory never
# calls torch.load, so the guard is not triggered by any torch version.
#
# External reproduction: the bucket below is private, so
# _ensure_hubert_safetensors_checkpoint() falls back to loading
# "ALM/hubert-base-audioset" straight from the Hub. That path requires
# torch >= 2.6, which a fresh install of ../../requirements.txt satisfies, and
# is the route an outside reproduction takes. Alternatively, perform the same
# conversion locally and point HUBERT_SAFETENSORS_LOCAL_DIR at the result.
# ---------------------------------------------------------------------------
HUBERT_CHECKPOINT = "ALM/hubert-base-audioset"
HUBERT_SAFETENSORS_GCS_BUCKET = "ast-heart-quality-revisions"
HUBERT_SAFETENSORS_GCS_PREFIX = "checkpoints/hubert-base-audioset-safetensors/"
HUBERT_SAFETENSORS_LOCAL_DIR = Path("/tmp/hubert-base-audioset-safetensors")
HUBERT_SAMPLE_RATE = 16000  # HuBERT operates at 16 kHz; no resampling needed


def _ensure_hubert_safetensors_checkpoint() -> str:
    """Return a path or model id that HubertModel.from_pretrained can load.

    Resolution order:

    1. A pre-converted safetensors directory already present locally, reused
       so the fetch happens at most once per machine rather than once per fold.
    2. The same directory staged in the project's cloud bucket, used when
       running inside the pinned torch 2.5.1 training container.
    3. The public Hub id, which is what an external reproduction uses. On
       torch >= 2.6 -- what a fresh install of ../../requirements.txt provides
       -- transformers can load the Hub repo's legacy .bin directly, so no
       conversion is needed. See the comment above HUBERT_CHECKPOINT for why
       the converted copy exists at all.
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
        # The staged copy is only reachable from the project's own cloud
        # environment; fall back to the public checkpoint.
        print(
            f"[HubertBackbone] pre-converted checkpoint unavailable ({type(exc).__name__}); "
            f"loading '{HUBERT_CHECKPOINT}' from the Hugging Face Hub instead."
        )
        return HUBERT_CHECKPOINT


class HubertBackbone(_TogglableFreezeBackbone):
    """HuBERT-base embedding extractor, embedding_dim=768.

    Input-normalization adaptation (deliberate, and important when reading
    the HuBERT numbers): HuBERT is normally fed audio normalized by
    Wav2Vec2FeatureExtractor to zero mean and unit variance, not
    peak-normalized to [-1, 1]. This wrapper instead receives exactly the same
    peak-normalized waveform as every other method in this package (the
    output of load_audio/mix_rms), with no HuBERT-specific normalization
    step. The comparison being made is between backbone architectures under
    identical preprocessing, so introducing a per-backbone input transform
    would confound it. The tradeoff is that HuBERT may underperform here
    relative to its conventional usage, and its reported numbers should be
    interpreted with that in mind rather than as a ceiling for HuBERT.

    Clip-level pooling: HuBERT has no [CLS] token (unlike AST's ViT-style
    encoder), so the per-clip embedding is the mean of last_hidden_state over
    the time axis.
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
            # The low-level CNN feature encoder stays frozen even during full
            # fine-tuning, following the standard Wav2Vec2/HuBERT fine-tuning
            # convention: the raw-waveform convolutional front-end is treated
            # as a fixed feature extractor and only feature_projection plus
            # the transformer encoder are updated. In transformers this is
            # exposed as freeze_feature_encoder() on the task heads
            # (HubertForCTC, HubertForSequenceClassification), which calls
            # `self.hubert.feature_extractor._freeze_parameters()`; the bare
            # HubertModel used here has no such helper, so the same call is
            # made directly. This parallels keeping PANNs' fixed DSP front-end
            # frozen above, for the same reason: it is not a learnable part of
            # the model.
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
    """Instantiate one of the wrappers in BACKBONES by name.

    freeze=True returns an eval-locked, no-grad backbone (feature-extractor
    use); freeze=False returns a trainable backbone for end-to-end
    fine-tuning, with each wrapper's fixed front-end still held frozen.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}', choose from {list(BACKBONES)}")
    return BACKBONES[name](freeze=freeze)
