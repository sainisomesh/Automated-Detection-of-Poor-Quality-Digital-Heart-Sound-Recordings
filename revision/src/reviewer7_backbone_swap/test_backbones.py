#!/usr/bin/env python3
"""
Audit checks for backbones.py and backbone_qa_model.py, intended to be run
before any training run is trusted. They verify the properties the
backbone-swap comparison depends on, using synthetic waveforms only (no
dataset required), and they are assertions rather than a report: the script
exits non-zero on the first violation.

Run one backbone at a time, since each loads a real pretrained checkpoint:
    python test_backbones.py --backbone panns
    python test_backbones.py --backbone yamnet
    python test_backbones.py --backbone hubert

Frozen mode:
  1. The backbone loads and produces embeddings of the declared
     embedding_dim, with no NaN or Inf.
  2. No backbone parameter is trainable.
  3. Calling composite_model.train(), as the training loop does at the start
     of every epoch, leaves the backbone in eval mode and leaves its output
     for a fixed input unchanged. This is the property that makes the frozen
     comparison meaningful: BatchNorm running statistics (PANNs, YAMNet) and
     SpecAugment/dropout (HuBERT) are gated on nn.Module.training, not on
     requires_grad, so without the eval lock in backbones.py a "frozen"
     backbone would still drift and inject masking noise.
  4. Acoustically different inputs (silence versus broadband noise) give
     different embeddings, i.e. the wrapper is not returning a constant.
  5. Forward and backward work end to end and gradients reach only the
     qa_classifier head.

Full fine-tuning mode:
  6. Backbone parameters are trainable. Not all of them: each wrapper keeps
     its fixed, non-learnable front-end frozen by design, so the check is
     that a non-zero subset is trainable.
  7. train() and eval() now cascade into the backbone -- the inverse of
     check 3, confirming the eval lock is not applied when it should not be.
  8. A backward pass reaches backbone parameters, and their gradient sum is
     of a plausible magnitude. An implausibly large value indicates that a
     fixed DSP transform (such as PANNs' mel filterbank) was unfrozen by
     mistake, which produces gradients many orders of magnitude larger than
     any genuinely learnable layer.
  9. The unfrozen backbone still honours an ambient torch.no_grad() context
     instead of forcing gradients on, so evaluation does not build unused
     autograd graphs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_qa_model import BackboneQAHead

MAX_LENGTH = 16000 * 10


def make_wav(kind, seed=0):
    """Build a deterministic 10 s, 16 kHz synthetic test waveform.

    'silence' is all zeros, 'noise' is uniform broadband noise, and 'tone' is
    an 80 Hz sinusoid, which sits in the frequency range the pipeline's
    20-1000 Hz bandpass preserves for heart sounds.
    """
    rng = np.random.default_rng(seed)
    if kind == "silence":
        wav = np.zeros(MAX_LENGTH, dtype=np.float32)
    elif kind == "noise":
        wav = rng.uniform(-1, 1, MAX_LENGTH).astype(np.float32)
    elif kind == "tone":
        t = np.arange(MAX_LENGTH) / 16000.0
        wav = 0.5 * np.sin(2 * np.pi * 80 * t).astype(np.float32)  # ~heart-rate-ish low tone
    else:
        raise ValueError(kind)
    return torch.tensor(wav, dtype=torch.float32)


def run_checks(backbone_name):
    """Run the frozen-mode checks (1-5) for one backbone."""
    print(f"=== Testing backbone: {backbone_name} ===")
    model = BackboneQAHead(backbone_name)
    model.eval()

    # --- Check 1: embedding dim + finite output ---
    batch = torch.stack([make_wav("silence"), make_wav("noise", seed=1), make_wav("tone")])
    with torch.no_grad():
        emb = model.backbone(batch)
    assert emb.shape == (3, model.backbone.embedding_dim), f"unexpected embedding shape {emb.shape}"
    assert torch.isfinite(emb).all(), "backbone produced NaN/Inf"
    print(f"  [OK] embedding shape {emb.shape}, all finite")

    # --- Check 2: backbone frozen ---
    n_trainable_backbone = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert n_trainable_backbone == 0, f"{n_trainable_backbone} backbone params are NOT frozen"
    n_trainable_head = sum(p.numel() for p in model.qa_classifier.parameters() if p.requires_grad)
    print(f"  [OK] backbone frozen (0 trainable params), head has {n_trainable_head} trainable params")

    # --- Check 3: the frozen backbone stays eval-locked under model.train() ---
    with torch.no_grad():
        emb_before = model.backbone(batch).clone()
    model.train()  # what the training loop calls at the start of every epoch
    assert model.backbone.training is False, (
        "backbone.training is True after model.train() -- the eval-lock override "
        "in backbones.py is not working, so BatchNorm/SpecAugment would alter "
        "supposedly frozen embeddings"
    )
    with torch.no_grad():
        emb_after = model.backbone(batch).clone()
    model.eval()
    max_diff = (emb_before - emb_after).abs().max().item()
    assert max_diff < 1e-5, (
        f"backbone output changed after model.train() (max diff {max_diff}) -- "
        f"the backbone is not actually frozen/eval-locked"
    )
    print(f"  [OK] backbone.training stays False and output is identical after model.train() "
          f"(max diff {max_diff:.2e})")

    # --- Check 4: different inputs give different embeddings ---
    pairwise_diff = (emb[0] - emb[1]).abs().mean().item()
    assert pairwise_diff > 1e-4, "silence and noise produced near-identical embeddings"
    print(f"  [OK] silence vs. noise embeddings differ (mean abs diff {pairwise_diff:.4f})")

    # --- Check 5: gradient only flows into qa_classifier ---
    model.train()
    wav = torch.stack([make_wav("noise", seed=2), make_wav("tone", seed=3)])
    labels = torch.tensor([[1.0], [0.0]])
    logits = model(wav)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    for name, p in model.backbone.named_parameters():
        assert p.grad is None, f"gradient flowed into frozen backbone param {name}"
    head_grad_norm = sum(p.grad.abs().sum().item() for p in model.qa_classifier.parameters() if p.grad is not None)
    assert head_grad_norm > 0, "no gradient reached the trainable head"
    print(f"  [OK] no gradient in backbone params, head gradient norm={head_grad_norm:.4f}")

    print(f"=== {backbone_name}: FROZEN-MODE CHECKS PASSED ===\n")


def run_unfrozen_checks(backbone_name):
    """Run the full-fine-tuning-mode checks (6-9) for one backbone."""
    print(f"=== Testing backbone (full-unfreeze mode): {backbone_name} ===")
    model = BackboneQAHead(backbone_name, freeze_backbone=False)

    # --- Check 6: backbone actually trainable ---
    # Deliberately not asserting that ALL parameters are trainable: each
    # wrapper keeps its fixed front-end frozen even in full mode (PANNs' fixed
    # torchlibrosa STFT/mel matrices and its unused AudioSet head, YAMNet's
    # unused AudioSet head, HuBERT's CNN feature encoder -- see backbones.py).
    # The invariant is that the genuinely learnable layers are now trainable.
    n_total = sum(p.numel() for p in model.backbone.parameters())
    n_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert n_trainable > 0, "freeze_backbone=False left zero backbone params trainable"
    print(f"  [OK] {n_trainable}/{n_total} backbone params trainable")

    # --- Check 7: train() actually cascades into the backbone now ---
    model.train()
    assert model.backbone.training is True, (
        "backbone.training is False after model.train() with freeze_backbone=False -- "
        "the eval lock is still engaged when it should not be"
    )
    model.eval()
    assert model.backbone.training is False, "backbone did not respond to model.eval() either"
    print("  [OK] backbone.training correctly tracks model.train()/model.eval() now")

    # --- Check 8: gradient reaches backbone params, not just the head ---
    model.train()
    batch = torch.stack([make_wav("noise", seed=2), make_wav("tone", seed=3)])
    labels = torch.tensor([[1.0], [0.0]])
    logits = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    backbone_grad_norm = sum(
        p.grad.abs().sum().item() for p in model.backbone.parameters() if p.grad is not None
    )
    n_with_grad = sum(1 for p in model.backbone.parameters() if p.grad is not None)
    assert n_with_grad > 0, "no backbone parameter received a gradient at all"
    assert backbone_grad_norm > 0, "backbone gradients are all exactly zero"
    # Upper bound as well as "non-zero". Unfreezing a fixed DSP transform such
    # as PANNs' mel-filterbank matrix produces a gradient sum on the order of
    # 1e11, against O(10-1000) for genuinely learnable layers, so 1e6 sits far
    # above any plausible value for this two-sample synthetic batch while
    # remaining far below what that class of mistake yields.
    assert backbone_grad_norm < 1e6, (
        f"backbone gradient sum {backbone_grad_norm:.3e} is implausibly large -- "
        f"a fixed, non-learnable transform was most likely left unfrozen"
    )
    head_grad_norm = sum(p.grad.abs().sum().item() for p in model.qa_classifier.parameters() if p.grad is not None)
    assert head_grad_norm > 0, "no gradient reached the trainable head either"
    print(f"  [OK] gradient reached {n_with_grad}/{sum(1 for _ in model.backbone.parameters())} "
          f"backbone params (grad norm={backbone_grad_norm:.4f}, sane magnitude), "
          f"head grad norm={head_grad_norm:.4f}")

    # --- Check 9: an unfrozen backbone still inherits the caller's autograd
    # context rather than forcing gradients on. evaluate_fold() runs inference
    # inside `with torch.no_grad():`, so a forward() that wrapped itself in
    # torch.enable_grad() would build an autograd graph that is never
    # backpropagated, wasting memory and time on every evaluation pass. ---
    model.train()
    with torch.no_grad():
        out = model(batch)
    assert not out.requires_grad, (
        "model output still requires_grad under an ambient torch.no_grad() context -- "
        "a backbone forward() is overriding the caller's autograd context instead of "
        "inheriting it"
    )
    print("  [OK] unfrozen backbone still respects an ambient torch.no_grad() context")

    print(f"=== {backbone_name}: FULL-UNFREEZE CHECKS PASSED ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["panns", "yamnet", "hubert"])
    args = parser.parse_args()
    run_checks(args.backbone)
    run_unfrozen_checks(args.backbone)
