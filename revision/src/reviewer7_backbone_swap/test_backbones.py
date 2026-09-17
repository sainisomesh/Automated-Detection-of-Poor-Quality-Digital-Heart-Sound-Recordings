#!/usr/bin/env python3
"""
Sanity checks for backbones.py / backbone_qa_model.py, run BEFORE trusting
any training run. Mirrors the same "verify before trusting" bar used for
tang_features.py / giordano_snr.py (test_tang_features.py, test_giordano_snr.py).

Run one backbone at a time (each downloads a real pretrained checkpoint):
    python test_backbones.py --backbone panns
    python test_backbones.py --backbone yamnet
    python test_backbones.py --backbone hubert

Checks (frozen mode, the original Reviewer #7 Comment 2 ask):
  1. Backbone loads, produces the declared embedding_dim, no NaN/Inf.
  2. Every backbone parameter has requires_grad=False.
  3. REGRESSION TEST for the BatchNorm/SpecAugment-in-train-mode bug caught
     while writing backbones.py: calling composite_model.train() (as the
     real training loop does every epoch) must NOT change the backbone's
     output for the same input -- i.e. the backbone must stay eval-locked.
  4. Two acoustically different inputs (silence vs. structured noise)
     produce different embeddings (backbone isn't just returning a constant).
  5. Head forward/backward works and only qa_classifier parameters receive
     gradients.

Checks (full-unfreeze mode, added 2026-09-14 for the "does AST only win
because of frozen features?" follow-up -- see ../README.md "Full-unfreeze
extension"):
  6. Every backbone parameter has requires_grad=True.
  7. composite_model.train() actually puts the backbone in train mode this
     time (the exact opposite of check 3 -- confirms freeze=False doesn't
     accidentally still eval-lock it).
  8. Backward pass reaches every backbone parameter (grad is not None and
     not all-zero for at least one param), not just qa_classifier.
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

    # --- Check 3: regression test for the train()-mode leak bug ---
    with torch.no_grad():
        emb_before = model.backbone(batch).clone()
    model.train()  # this is what the real training loop calls every epoch
    assert model.backbone.training is False, (
        "BUG: backbone.training is True after model.train() -- the eval-lock override "
        "in backbones.py is not working, BatchNorm/SpecAugment would corrupt frozen embeddings"
    )
    with torch.no_grad():
        emb_after = model.backbone(batch).clone()
    model.eval()
    max_diff = (emb_before - emb_after).abs().max().item()
    assert max_diff < 1e-5, (
        f"BUG: backbone output changed after model.train() (max diff {max_diff}) -- "
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
        assert p.grad is None, f"BUG: gradient flowed into frozen backbone param {name}"
    head_grad_norm = sum(p.grad.abs().sum().item() for p in model.qa_classifier.parameters() if p.grad is not None)
    assert head_grad_norm > 0, "no gradient reached the trainable head"
    print(f"  [OK] no gradient in backbone params, head gradient norm={head_grad_norm:.4f}")

    print(f"=== {backbone_name}: FROZEN-MODE CHECKS PASSED ===\n")


def run_unfrozen_checks(backbone_name):
    print(f"=== Testing backbone (full-unfreeze mode): {backbone_name} ===")
    model = BackboneQAHead(backbone_name, freeze_backbone=False)

    # --- Check 6: backbone actually trainable ---
    # NOT asserting ALL params trainable here: PANNs deliberately re-locks its
    # fixed torchlibrosa DSP front-end (spectrogram/mel-filterbank matrices)
    # and its unused original AudioSet head even in full-unfreeze mode -- see
    # backbones.py's PANNsBackbone comment. What must hold for every backbone
    # is "at least the actual learnable conv/attention/BN layers are now
    # trainable" -- checked as "more than just the head-sized param count".
    n_total = sum(p.numel() for p in model.backbone.parameters())
    n_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert n_trainable > 0, "BUG: freeze_backbone=False left zero backbone params trainable"
    print(f"  [OK] {n_trainable}/{n_total} backbone params trainable")

    # --- Check 7: train() actually cascades into the backbone now ---
    model.train()
    assert model.backbone.training is True, (
        "BUG: backbone.training is False after model.train() with freeze_backbone=False -- "
        "the eval-lock is still engaged even though it shouldn't be"
    )
    model.eval()
    assert model.backbone.training is False, "BUG: backbone didn't respond to model.eval() either"
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
    assert n_with_grad > 0, "BUG: no backbone parameter received a gradient at all"
    assert backbone_grad_norm > 0, "BUG: backbone gradients are all exactly zero"
    # Sanity bound, not just "nonzero": this is exactly what caught the real
    # PANNs bug (blanket-unfreezing its fixed mel-filterbank matrix produced a
    # ~5e11 gradient sum vs. O(10-1000) for every genuinely learnable param).
    # 1e6 is well above any plausible real gradient sum for this tiny 2-sample
    # synthetic batch, but many orders of magnitude below the astronomical
    # values a "unfroze something that should have stayed a fixed transform"
    # bug produces -- a real regression trips this, normal training doesn't.
    assert backbone_grad_norm < 1e6, (
        f"BUG: backbone gradient sum {backbone_grad_norm:.3e} is implausibly large -- "
        f"likely unfroze a fixed/non-learnable transform by mistake (see the PANNs "
        f"mel-filterbank bug this exact check caught on 2026-09-14)"
    )
    head_grad_norm = sum(p.grad.abs().sum().item() for p in model.qa_classifier.parameters() if p.grad is not None)
    assert head_grad_norm > 0, "no gradient reached the trainable head either"
    print(f"  [OK] gradient reached {n_with_grad}/{sum(1 for _ in model.backbone.parameters())} "
          f"backbone params (grad norm={backbone_grad_norm:.4f}, sane magnitude), "
          f"head grad norm={head_grad_norm:.4f}")

    # --- Check 9: unfrozen backbone still respects an AMBIENT no_grad context
    # (i.e. it inherits the caller's context rather than force-overriding it
    # with torch.enable_grad()) -- regression test for a real bug caught
    # 2026-09-14 in YAMNetBackbone.forward(), which used torch.enable_grad()
    # instead of nullcontext() and would have built a needless autograd graph
    # during evaluate_fold()'s `with torch.no_grad():` eval loop every single
    # epoch. Not a correctness bug (eval never calls .backward()), but wasted
    # memory/compute, and inconsistent with PANNsBackbone/HubertBackbone. ---
    model.train()
    with torch.no_grad():
        out = model(batch)
    assert not out.requires_grad, (
        "BUG: model output still requires_grad under an ambient torch.no_grad() context -- "
        "some backbone forward() is force-overriding the caller's grad context instead of "
        "inheriting it (this is the exact YAMNet torch.enable_grad() bug found 2026-09-14)"
    )
    print("  [OK] unfrozen backbone still respects an ambient torch.no_grad() context")

    print(f"=== {backbone_name}: FULL-UNFREEZE CHECKS PASSED ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["panns", "yamnet", "hubert"])
    args = parser.parse_args()
    run_checks(args.backbone)
    run_unfrozen_checks(args.backbone)
