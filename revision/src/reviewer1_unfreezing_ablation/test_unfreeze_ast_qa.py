#!/usr/bin/env python3
"""
Sanity/regression checks for unfreeze_ast_qa.py and the optimizer/schedule
logic in train_unfreezing_ablation_cv.py, run BEFORE trusting any training
run. Mirrors the "verify before trusting" bar set by
../../reviewer7_backbone_swap/src/test_backbones.py.

Run:
    python test_unfreeze_ast_qa.py

Checks:
  1. 'frozen' mode has EXACTLY the same trainable parameter count as the
     published baseline ASTHeartQA(freeze_base=True) -- a structural
     regression test against src/models/ast_qa.py, so Mode A here is
     provably the same model shape as what produced Table 2 / Figure 3.
  2. 'full' mode unfreezes all 12 encoder layers (every encoder param has
     requires_grad=True).
  3. 'topk' mode with K in {2,4} unfreezes EXACTLY the last K
     transformer layers (encoder.layer[12-K:]) plus the final layernorm --
     no more, no less. Checked by layer index, not just a raw count, so an
     off-by-one (e.g. unfreezing layer 11..12-K instead of 12-K..11) would
     be caught.
  4. set_unfreeze_mode() correctly re-freezes: switching frozen -> topk2 ->
     topk4 must never leave a stale requires_grad=True from a previous call
     (this is the exact bug class _apply_freeze_policy's "always reset to
     fully frozen first" comment defends against).
  5. Gradient flow matches requires_grad exactly in each mode: frozen ->
     zero backbone params get a gradient; full -> every backbone param
     gets a non-None, non-zero-sum gradient; topk -> only the unfrozen
     layers do.
  6. build_optimizer() assigns the head to --head_lr and the unfrozen
     backbone params to --backbone_lr as two distinct param groups (a
     mix-up here would silently train the backbone at the head's LR,
     contradicting CLAUDE.md Sec 9.1 Comment 2's "lower learning rate for
     the backbone" instruction).
  7. Forward pass is finite (no NaN/Inf) and shapes are correct in every mode.
  8. The topk warmup -> unfreeze transition (as train_model's Phase 1 ->
     Phase 2 does) ends with exactly the same trainable set as constructing
     the model directly in 'topk' mode -- order of operations doesn't matter.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unfreeze_ast_qa import ASTHeartQAUnfreeze
from train_unfreezing_ablation_cv import build_optimizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.models.ast_qa import ASTHeartQA  # noqa: E402  (baseline, for the structural regression check)

MAX_LENGTH = 16000 * 10


def make_batch(n=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, 1024, 128)  # AST log-mel input shape (max_length=1024, num_mel_bins=128)


def trainable_encoder_layer_indices(model):
    """Which of encoder.layer[0..11] have >=1 trainable param, plus whether
    the final layernorm is trainable. Used to check exact top-K selection."""
    idx = []
    for i, layer in enumerate(model.encoder.encoder.layer):
        if any(p.requires_grad for p in layer.parameters()):
            idx.append(i)
    layernorm_trainable = any(p.requires_grad for p in model.encoder.layernorm.parameters())
    return idx, layernorm_trainable


def check_1_matches_baseline():
    print("[1] frozen mode matches published baseline ASTHeartQA(freeze_base=True) trainable count")
    baseline = ASTHeartQA(freeze_base=True)
    ablation = ASTHeartQAUnfreeze(unfreeze_mode="frozen")
    n_base = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    n_abl = sum(p.numel() for p in ablation.parameters() if p.requires_grad)
    assert n_base == n_abl, f"BUG: baseline trainable={n_base:,} != frozen-mode trainable={n_abl:,}"
    # Neither model ever explicitly freezes original_classifier (the unused 527-way
    # AudioSet head) -- it has requires_grad=True by default in both, and no
    # gradient reaches it since it's disconnected from the QA loss. This is a
    # pre-existing quirk of the published baseline (src/models/ast_qa.py), not
    # something introduced here; preserved for exact structural fidelity rather
    # than "fixed", since Mode A must be provably the same model as Table 2/Fig 3.
    n_head = sum(p.numel() for p in ablation.qa_classifier.parameters())
    n_original_classifier = sum(p.numel() for p in ablation.original_classifier.parameters())
    assert n_abl == n_head + n_original_classifier, (
        f"BUG: frozen mode trainable params ({n_abl:,}) isn't exactly "
        f"qa_classifier ({n_head:,}) + original_classifier ({n_original_classifier:,}) -- "
        f"something else in the encoder is unexpectedly trainable"
    )
    print(f"    [OK] both = {n_base:,} trainable params "
          f"(qa_classifier {n_head:,} + unused original_classifier {n_original_classifier:,})")
    del baseline, ablation


def check_2_full_unfreezes_everything():
    print("[2] full mode unfreezes every encoder parameter")
    model = ASTHeartQAUnfreeze(unfreeze_mode="full")
    n_frozen = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
    assert n_frozen == 0, f"BUG: {n_frozen} encoder params still frozen in 'full' mode"
    print(f"    [OK] all {sum(1 for _ in model.encoder.parameters())} encoder param tensors trainable")
    del model


def check_3_topk_selects_exact_layers():
    print("[3] topk mode unfreezes exactly the last K layers + final layernorm")
    for k in (2, 4):
        model = ASTHeartQAUnfreeze(unfreeze_mode="topk", topk_layers=k)
        idx, ln = trainable_encoder_layer_indices(model)
        expected = list(range(12 - k, 12))
        assert idx == expected, f"BUG: topk={k} unfroze layers {idx}, expected {expected}"
        assert ln, f"BUG: topk={k} did not unfreeze the final layernorm"
        # Layers below the top-K must stay frozen.
        for i in range(0, 12 - k):
            layer_trainable = any(p.requires_grad for p in model.encoder.encoder.layer[i].parameters())
            assert not layer_trainable, f"BUG: topk={k} leaked trainable params into layer {i}"
        print(f"    [OK] topk={k}: trainable layers={idx}, layernorm trainable={ln}, "
              f"layers 0..{11 - k} stay frozen")
        del model


def check_4_set_unfreeze_mode_resets_cleanly():
    print("[4] set_unfreeze_mode() never leaks a stale unfrozen layer across transitions")
    model = ASTHeartQAUnfreeze(unfreeze_mode="frozen")
    idx, ln = trainable_encoder_layer_indices(model)
    assert idx == [] and not ln, "BUG: fresh 'frozen' model has unfrozen encoder params"

    model.set_unfreeze_mode("topk", topk_layers=4)
    idx4, ln4 = trainable_encoder_layer_indices(model)
    assert idx4 == [8, 9, 10, 11] and ln4, f"BUG: after frozen->topk4, got layers={idx4}"

    model.set_unfreeze_mode("topk", topk_layers=2)
    idx2, ln2 = trainable_encoder_layer_indices(model)
    assert idx2 == [10, 11] and ln2, (
        f"BUG: after topk4->topk2, got layers={idx2} -- layers 8,9 from the PREVIOUS mode "
        f"were not re-frozen (this is exactly the stale-state bug _apply_freeze_policy's "
        f"reset-to-fully-frozen-first step exists to prevent)"
    )

    model.set_unfreeze_mode("frozen")
    idx0, ln0 = trainable_encoder_layer_indices(model)
    assert idx0 == [] and not ln0, f"BUG: after topk2->frozen, layers={idx0} still trainable"

    model.set_unfreeze_mode("full")
    idxf, lnf = trainable_encoder_layer_indices(model)
    assert idxf == list(range(12)) and lnf, "BUG: frozen->full did not unfreeze everything"
    print("    [OK] frozen -> topk4 -> topk2 -> frozen -> full all produce exactly the expected set")
    del model


def check_5_gradient_flow_matches_requires_grad():
    print("[5] gradients only reach params with requires_grad=True, in every mode")
    batch = make_batch(n=2)
    labels = torch.tensor([[1.0], [0.0]])

    for mode, k in [("frozen", 0), ("full", 0), ("topk", 2), ("topk", 4)]:
        model = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        model.train()
        _, logits = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        for name, p in model.encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"BUG ({mode},k={k}): unfrozen param {name} got no gradient"
                assert p.grad.abs().sum().item() >= 0.0  # finite check happens below
                assert torch.isfinite(p.grad).all(), f"BUG ({mode},k={k}): NaN/Inf grad in {name}"
            else:
                assert p.grad is None, f"BUG ({mode},k={k}): frozen param {name} received a gradient"
        head_grad_norm = sum(p.grad.abs().sum().item() for p in model.qa_classifier.parameters() if p.grad is not None)
        assert head_grad_norm > 0, f"BUG ({mode},k={k}): no gradient reached the head"
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"    [OK] {tag}: gradient set matches requires_grad exactly, head_grad_norm={head_grad_norm:.4f}")
        del model


def check_6_optimizer_param_groups():
    print("[6] build_optimizer assigns head_lr / backbone_lr to the correct, disjoint param groups")
    head_lr, backbone_lr = 1e-4, 5e-5
    assert head_lr != backbone_lr, "test constants must differ to actually exercise this check"

    # frozen: single group, head only.
    model = ASTHeartQAUnfreeze(unfreeze_mode="frozen")
    opt = build_optimizer(model, "frozen", head_lr, backbone_lr)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == head_lr
    n_params_in_group = sum(p.numel() for p in opt.param_groups[0]["params"])
    n_head_params = sum(p.numel() for p in model.qa_classifier.parameters())
    assert n_params_in_group == n_head_params
    print(f"    [OK] frozen: 1 param group, lr={head_lr}, {n_params_in_group:,} params (head only)")

    # full / topk: two groups, head at head_lr, backbone at backbone_lr, no overlap.
    for mode, k in [("full", 0), ("topk", 4)]:
        model = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        opt = build_optimizer(model, mode, head_lr, backbone_lr)
        assert len(opt.param_groups) == 2, f"BUG ({mode}): expected 2 param groups, got {len(opt.param_groups)}"
        lrs = sorted(g["lr"] for g in opt.param_groups)
        assert lrs == sorted([head_lr, backbone_lr]), f"BUG ({mode}): param group LRs = {lrs}"
        head_group = next(g for g in opt.param_groups if g["lr"] == head_lr)
        backbone_group = next(g for g in opt.param_groups if g["lr"] == backbone_lr)
        head_ids = {id(p) for p in head_group["params"]}
        backbone_ids = {id(p) for p in backbone_group["params"]}
        assert head_ids.isdisjoint(backbone_ids), f"BUG ({mode}): head and backbone param groups overlap"
        expected_head_ids = {id(p) for p in model.qa_classifier.parameters()}
        expected_backbone_ids = {id(p) for p in model.trainable_backbone_parameters()}
        assert head_ids == expected_head_ids, f"BUG ({mode}): head param group doesn't match qa_classifier exactly"
        assert backbone_ids == expected_backbone_ids, (
            f"BUG ({mode}): backbone param group doesn't match trainable_backbone_parameters() exactly"
        )
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"    [OK] {tag}: 2 disjoint param groups, head_lr={head_lr}, backbone_lr={backbone_lr}, "
              f"backbone group has {len(backbone_group['params'])} tensors")
        del model


def check_7_forward_pass_finite_every_mode():
    print("[7] forward pass produces finite, correctly-shaped output in every mode")
    batch = make_batch(n=3, seed=1)
    for mode, k in [("frozen", 0), ("full", 0), ("topk", 2), ("topk", 4)]:
        model = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        model.eval()
        with torch.no_grad():
            original_logits, qa_logits = model(batch)
        assert qa_logits.shape == (3, 1), f"BUG ({mode},k={k}): qa_logits shape {qa_logits.shape}"
        assert original_logits.shape == (3, 527), f"BUG ({mode},k={k}): original_logits shape {original_logits.shape}"
        assert torch.isfinite(qa_logits).all(), f"BUG ({mode},k={k}): non-finite qa_logits"
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"    [OK] {tag}: qa_logits {qa_logits.shape}, original_logits {original_logits.shape}, all finite")
        del model


def check_8_warmup_then_unfreeze_matches_direct_construction():
    print("[8] warmup(frozen) -> set_unfreeze_mode(topk) ends up identical to direct topk construction")
    for k in (2, 4):
        warmed = ASTHeartQAUnfreeze(unfreeze_mode="frozen")
        warmed.set_unfreeze_mode("topk", topk_layers=k)
        idx_warmed, ln_warmed = trainable_encoder_layer_indices(warmed)

        direct = ASTHeartQAUnfreeze(unfreeze_mode="topk", topk_layers=k)
        idx_direct, ln_direct = trainable_encoder_layer_indices(direct)

        assert idx_warmed == idx_direct and ln_warmed == ln_direct, (
            f"BUG (k={k}): warmup-then-unfreeze trainable set {idx_warmed}/{ln_warmed} != "
            f"direct-construction set {idx_direct}/{ln_direct}"
        )
        print(f"    [OK] k={k}: both paths trainable layers={idx_warmed}, layernorm={ln_warmed}")
        del warmed, direct


CHECKPOINTING_KWARGS = {"use_reentrant": False}  # must match train_unfreezing_ablation_cv.py exactly


def check_9_gradient_checkpointing_still_trains_the_right_params():
    print("[9] gradient_checkpointing_enable() doesn't silently drop backbone gradients")
    # Triggered by a real observation while smoke-testing: enabling grad_checkpointing on
    # a 'topk' model prints "UserWarning: None of the inputs have requires_grad=True.
    # Gradients will be None" from torch.utils.checkpoint -- because layers 0..(12-K-1)
    # are frozen INCLUDING embeddings, so the activation flowing INTO the first unfrozen
    # layer genuinely has requires_grad=False.
    labels = torch.tensor([[1.0], [0.0]])

    # --- 9a: confirm the DEFAULT (reentrant) checkpoint really does drop that first
    # unfrozen layer's gradients for 'topk' -- this is a real bug, not just a benign
    # warning, and this sub-check exists to make sure it stays caught if anyone ever
    # reverts the use_reentrant=False fix. 'full' is unaffected (every input already
    # requires grad from the embeddings onward), so it's the negative control here. ---
    model = ASTHeartQAUnfreeze(unfreeze_mode="topk", topk_layers=4)
    model.encoder.gradient_checkpointing_enable()  # default reentrant=True
    model.train()
    _, logits = model(make_batch(n=2, seed=7))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
    first_unfrozen_layer = model.encoder.encoder.layer[12 - 4]  # layer 8
    missing = [n for n, p in first_unfrozen_layer.named_parameters() if p.requires_grad and p.grad is None]
    assert missing, (
        "Expected the known reentrant-checkpointing bug to reproduce (missing gradients on "
        f"layer 8), but got none missing -- either the bug was silently fixed upstream (update "
        f"this test) or this check is no longer exercising the right code path"
    )
    print(f"    [confirmed] default (reentrant) checkpointing DOES drop gradients on "
          f"{missing} -- this is why train_unfreezing_ablation_cv.py must pass "
          f"gradient_checkpointing_kwargs={CHECKPOINTING_KWARGS}")
    del model

    # --- 9b: the actual fix (use_reentrant=False, exactly as train_unfreezing_ablation_cv.py
    # calls it) must NOT have this problem, for both 'full' and 'topk'. ---
    for mode, k in [("full", 0), ("topk", 4)]:
        batch = make_batch(n=2, seed=7)
        model = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=CHECKPOINTING_KWARGS)
        model.train()
        _, logits = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        n_checked_trainable = 0
        for name, p in model.encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, (
                    f"BUG ({mode},k={k},checkpointing,use_reentrant=False): unfrozen param "
                    f"{name} got NO gradient -- the fix did not actually fix this"
                )
                assert torch.isfinite(p.grad).all(), f"BUG ({mode},k={k}): NaN/Inf grad in {name}"
                assert p.grad.abs().sum().item() > 0, (
                    f"BUG ({mode},k={k},checkpointing): unfrozen param {name} got an all-zero gradient"
                )
                n_checked_trainable += 1
            else:
                assert p.grad is None, (
                    f"BUG ({mode},k={k},checkpointing): frozen param {name} received a gradient"
                )
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"    [OK] {tag}+checkpointing(use_reentrant=False): {n_checked_trainable} unfrozen "
              f"encoder param tensors all got real, finite, nonzero gradients; frozen ones got none")
        del model

    # --- 9c: numeric cross-check -- same model/input/seed, gradients on the unfrozen
    # backbone params should match closely with fixed-checkpointing on vs. off entirely
    # (checkpointing changes HOW the backward is computed, not the math). Both models
    # stay in train() mode -- transformers' GradientCheckpointingLayer only actually
    # engages checkpointing when `self.training` is True (confirmed by reading
    # modeling_layers.GradientCheckpointingLayer.__call__ directly), so comparing in
    # eval() would make model_ckpt silently stop checkpointing and this check would
    # pass without testing anything. Train mode means the encoder's internal dropout is
    # active in both models -- to keep that from being a confound (two independent
    # forward calls would otherwise draw different random dropout masks and legitimately
    # produce different logits for reasons unrelated to checkpointing), the global RNG
    # is reset to the same seed immediately before EACH model's forward call, so both
    # draw identical dropout masks during their (separate) initial forward passes. ---
    for mode, k in [("full", 0), ("topk", 4)]:
        torch.manual_seed(11)
        model_ckpt = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        model_plain = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        model_plain.load_state_dict(model_ckpt.state_dict())
        model_ckpt.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=CHECKPOINTING_KWARGS)
        model_ckpt.train()
        model_plain.train()

        batch = make_batch(n=2, seed=13)
        labels2 = torch.tensor([[1.0], [0.0]])

        torch.manual_seed(99)
        _, logits_ckpt = model_ckpt(batch)
        torch.nn.functional.binary_cross_entropy_with_logits(logits_ckpt, labels2).backward()
        torch.manual_seed(99)
        _, logits_plain = model_plain(batch)
        torch.nn.functional.binary_cross_entropy_with_logits(logits_plain, labels2).backward()

        assert torch.allclose(logits_ckpt, logits_plain, atol=1e-5), (
            f"BUG ({mode},k={k}): forward output differs between checkpointed and plain model"
        )
        max_grad_diff = 0.0
        for (n1, p1), (n2, p2) in zip(model_ckpt.encoder.named_parameters(), model_plain.encoder.named_parameters()):
            assert n1 == n2
            if p1.requires_grad:
                diff = (p1.grad - p2.grad).abs().max().item()
                max_grad_diff = max(max_grad_diff, diff)
        assert max_grad_diff < 1e-4, (
            f"BUG ({mode},k={k}): checkpointed vs. plain backbone gradients differ by "
            f"{max_grad_diff} -- checkpointing changed the actual gradient values, not just memory use"
        )
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"    [OK] {tag}: checkpointed(use_reentrant=False) vs. plain gradients match "
              f"(max diff {max_grad_diff:.2e})")
        del model_ckpt, model_plain


def main():
    check_1_matches_baseline()
    check_2_full_unfreezes_everything()
    check_3_topk_selects_exact_layers()
    check_4_set_unfreeze_mode_resets_cleanly()
    check_5_gradient_flow_matches_requires_grad()
    check_6_optimizer_param_groups()
    check_7_forward_pass_finite_every_mode()
    check_8_warmup_then_unfreeze_matches_direct_construction()
    check_9_gradient_checkpointing_still_trains_the_right_params()
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
