#!/usr/bin/env python3
"""
Checks for backbones.py and backbone_qa_model.py, run before trusting a
training run. Uses synthetic waveforms only (no dataset needed) and exits
non-zero on the first failed assertion.

Each backbone loads a real pretrained checkpoint, so run one at a time:
    python test_backbones.py --backbone panns
    python test_backbones.py --backbone yamnet
    python test_backbones.py --backbone hubert

Frozen mode:
  1. Embeddings have the declared embedding_dim and no NaN/Inf.
  2. No backbone parameter is trainable.
  3. After composite_model.train() (called by the training loop every
     epoch) the backbone is still in eval mode and its output for a fixed
     input is unchanged. BatchNorm (PANNs, YAMNet) and SpecAugment/dropout
     (HuBERT) depend on nn.Module.training, not requires_grad.
  4. Silence and broadband noise give different embeddings.
  5. Forward/backward works and gradients reach only qa_classifier.

Full fine-tuning mode:
  6. Some backbone parameters are trainable. Not all: each wrapper keeps its
     fixed front-end frozen.
  7. train() and eval() propagate into the backbone (inverse of check 3).
  8. Gradients reach the backbone and their sum is of a plausible size. A
     very large value means a fixed DSP transform (e.g. PANNs' mel
     filterbank) was unfrozen by mistake.
  9. The unfrozen backbone respects an outer torch.no_grad(), so evaluation
     does not build autograd graphs.
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
    """Deterministic 10 s, 16 kHz synthetic waveform.

    'silence' is zeros, 'noise' is uniform broadband noise, 'tone' is an
    80 Hz sine (inside the pipeline's 20-1000 Hz band).
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
        "backbone.training is True after model.train(): the eval-lock override "
        "in backbones.py is not working, so BatchNorm/SpecAugment would alter "
        "supposedly frozen embeddings"
    )
    with torch.no_grad():
        emb_after = model.backbone(batch).clone()
    model.eval()
    max_diff = (emb_before - emb_after).abs().max().item()
    assert max_diff < 1e-5, (
        f"backbone output changed after model.train() (max diff {max_diff}): "
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
    # Not all parameters are trainable in full mode: PANNs' STFT/mel matrices
    # and AudioSet head, YAMNet's AudioSet head and HuBERT's CNN feature
    # encoder stay frozen (see backbones.py).
    n_total = sum(p.numel() for p in model.backbone.parameters())
    n_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert n_trainable > 0, "freeze_backbone=False left zero backbone params trainable"
    print(f"  [OK] {n_trainable}/{n_total} backbone params trainable")

    # --- Check 7: train() actually cascades into the backbone now ---
    model.train()
    assert model.backbone.training is True, (
        "backbone.training is False after model.train() with freeze_backbone=False: "
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
    # Upper bound: an unfrozen mel-filterbank matrix gives a gradient sum
    # around 1e11, versus roughly 10-1000 for learnable layers on this batch.
    assert backbone_grad_norm < 1e6, (
        f"backbone gradient sum {backbone_grad_norm:.3e} is implausibly large: "
        f"a fixed, non-learnable transform was most likely left unfrozen"
    )
    head_grad_norm = sum(p.grad.abs().sum().item() for p in model.qa_classifier.parameters() if p.grad is not None)
    assert head_grad_norm > 0, "no gradient reached the trainable head either"
    print(f"  [OK] gradient reached {n_with_grad}/{sum(1 for _ in model.backbone.parameters())} "
          f"backbone params (grad norm={backbone_grad_norm:.4f}, sane magnitude), "
          f"head grad norm={head_grad_norm:.4f}")

    # --- Check 9: unfrozen backbone respects an outer torch.no_grad(), which
    # evaluate_fold() uses for inference ---
    model.train()
    with torch.no_grad():
        out = model(batch)
    assert not out.requires_grad, (
        "model output still requires_grad under an ambient torch.no_grad() context: "
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
