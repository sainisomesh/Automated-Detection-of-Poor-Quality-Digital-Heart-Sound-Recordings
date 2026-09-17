#!/usr/bin/env python3
"""
Audit checks for unfreeze_ast_qa.py and the optimizer construction in
train_unfreezing_ablation_cv.py.

These run in seconds on CPU and are intended to be run before any training
job: they verify that each freeze policy trains exactly the parameters it
claims to, which is the kind of error that otherwise shows up only as
inexplicably poor results.

Run:
    python test_unfreeze_ast_qa.py

Checks:
  1. "frozen" mode has exactly the same trainable parameter count as the
     paper's ASTHeartQA(freeze_base=True), so the ablation's reference
     condition is structurally identical to the published model.
  2. "full" mode leaves no encoder parameter frozen.
  3. "topk" mode with K in {2,4} unfreezes exactly the LAST K transformer
     blocks (encoder.layer[12-K:]) plus the final layernorm, and nothing
     else. Verified by block index rather than by parameter count, so
     unfreezing the first K blocks instead of the last K would be caught.
  4. set_unfreeze_mode() re-freezes cleanly: no parameter left trainable by
     a previous policy survives a switch (frozen -> topk4 -> topk2 -> frozen
     -> full).
  5. Gradient flow matches requires_grad exactly in every mode: trainable
     parameters receive finite gradients, frozen ones receive none, and the
     head always receives a non-zero gradient.
  6. build_optimizer() places the head and the unfrozen backbone in two
     disjoint parameter groups at head_lr and backbone_lr respectively. A
     mix-up here would silently fine-tune the pretrained backbone at the
     head's higher learning rate.
  7. Forward pass output is finite and correctly shaped in every mode.
  8. Running the progressive schedule (construct frozen, then
     set_unfreeze_mode("topk", K)) yields the same trainable set as
     constructing the model in "topk" mode directly.
  9. Gradient checkpointing with use_reentrant=False preserves both the set
     and the values of the backbone gradients; see that check for why the
     keyword is mandatory.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unfreeze_ast_qa import ASTHeartQAUnfreeze
from train_unfreezing_ablation_cv import build_optimizer

# Package root, so the paper's model can be imported for the check 1 comparison.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.models.ast_qa import ASTHeartQA  # noqa: E402

MAX_LENGTH = 16000 * 10


def make_batch(n=2, seed=0):
    """Random batch shaped like ASTFeatureExtractor output: (n, 1024, 128)."""
    torch.manual_seed(seed)
    return torch.randn(n, 1024, 128)


def trainable_encoder_layer_indices(model):
    """Return (indices, layernorm_trainable) describing the trainable encoder.

    `indices` lists which of encoder.layer[0..11] contain at least one
    trainable parameter; the flag reports whether the final layernorm is
    trainable. Together these pin down the exact top-K selection.
    """
    idx = []
    for i, layer in enumerate(model.encoder.encoder.layer):
        if any(p.requires_grad for p in layer.parameters()):
            idx.append(i)
    layernorm_trainable = any(p.requires_grad for p in model.encoder.layernorm.parameters())
    return idx, layernorm_trainable


def check_1_matches_baseline():
    """Frozen mode must be structurally identical to the paper's model."""
    print("[1] frozen mode matches published baseline ASTHeartQA(freeze_base=True) trainable count")
    baseline = ASTHeartQA(freeze_base=True)
    ablation = ASTHeartQAUnfreeze(unfreeze_mode="frozen")
    n_base = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    n_abl = sum(p.numel() for p in ablation.parameters() if p.requires_grad)
    assert n_base == n_abl, f"BUG: baseline trainable={n_base:,} != frozen-mode trainable={n_abl:,}"
    # Neither model explicitly freezes original_classifier, the pretrained
    # 527-way AudioSet head, so it counts as trainable in both. It is harmless:
    # it is disconnected from the QA loss, receives no gradient, and is not
    # included in the optimizer's parameter groups. The behaviour is inherited
    # unchanged from src/models/ast_qa.py and is kept so that the frozen mode
    # remains parameter-for-parameter identical to the published model.
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
    """End-to-end mode must leave nothing frozen."""
    print("[2] full mode unfreezes every encoder parameter")
    model = ASTHeartQAUnfreeze(unfreeze_mode="full")
    n_frozen = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
    assert n_frozen == 0, f"BUG: {n_frozen} encoder params still frozen in 'full' mode"
    print(f"    [OK] all {sum(1 for _ in model.encoder.parameters())} encoder param tensors trainable")
    del model


def check_3_topk_selects_exact_layers():
    """Top-K must mean the LAST K blocks, i.e. indices 12-K..11, plus the
    final layernorm. Comparing index lists (not counts) catches a selection
    that unfroze the first K blocks instead."""
    print("[3] topk mode unfreezes exactly the last K layers + final layernorm")
    for k in (2, 4):
        model = ASTHeartQAUnfreeze(unfreeze_mode="topk", topk_layers=k)
        idx, ln = trainable_encoder_layer_indices(model)
        expected = list(range(12 - k, 12))
        assert idx == expected, f"BUG: topk={k} unfroze layers {idx}, expected {expected}"
        assert ln, f"BUG: topk={k} did not unfreeze the final layernorm"
        # Everything below the top-K must stay frozen.
        for i in range(0, 12 - k):
            layer_trainable = any(p.requires_grad for p in model.encoder.encoder.layer[i].parameters())
            assert not layer_trainable, f"BUG: topk={k} leaked trainable params into layer {i}"
        print(f"    [OK] topk={k}: trainable layers={idx}, layernorm trainable={ln}, "
              f"layers 0..{11 - k} stay frozen")
        del model


def check_4_set_unfreeze_mode_resets_cleanly():
    """Switching policies must not carry trainable layers over from the previous one."""
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
        f"BUG: after topk4->topk2, got layers={idx2} -- layers 8,9 from the previous policy "
        f"were not re-frozen, which is what _apply_freeze_policy's reset-to-frozen step "
        f"is there to prevent"
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
    """requires_grad is only a declaration; this confirms the backward pass agrees."""
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
    """The head and the backbone must land in separate groups at their own LRs."""
    print("[6] build_optimizer assigns head_lr / backbone_lr to the correct, disjoint param groups")
    head_lr, backbone_lr = 1e-4, 5e-5
    assert head_lr != backbone_lr, "test constants must differ to actually exercise this check"

    # Frozen: a single group containing the head only.
    model = ASTHeartQAUnfreeze(unfreeze_mode="frozen")
    opt = build_optimizer(model, "frozen", head_lr, backbone_lr)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == head_lr
    n_params_in_group = sum(p.numel() for p in opt.param_groups[0]["params"])
    n_head_params = sum(p.numel() for p in model.qa_classifier.parameters())
    assert n_params_in_group == n_head_params
    print(f"    [OK] frozen: 1 param group, lr={head_lr}, {n_params_in_group:,} params (head only)")

    # Full / topk: two non-overlapping groups, head at head_lr and the
    # trainable backbone at the lower backbone_lr.
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
    """The progressive schedule must converge on the same trainable set as
    direct top-K construction, so the warmup phase has no lasting side effect
    on which parameters train."""
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


CHECKPOINTING_KWARGS = {"use_reentrant": False}  # must match the training scripts exactly


def check_9_gradient_checkpointing_still_trains_the_right_params():
    """Gradient checkpointing must not change which parameters get gradients.

    Requirement being verified: gradient checkpointing on a partially frozen
    encoder must be enabled with use_reentrant=False. With a top-K policy,
    blocks 0..(12-K-1) and the patch embeddings are frozen, so the activation
    entering the first unfrozen block has requires_grad=False. In that
    situation the default reentrant torch.utils.checkpoint implementation
    returns no gradients for the checkpointed block's own trainable
    parameters (it also emits "None of the inputs have requires_grad=True.
    Gradients will be None"), so that block would never be updated while the
    loss curve still looked normal.

    Sub-checks: 9a demonstrates the reentrant behaviour so that reverting the
    keyword cannot pass silently; 9b verifies the non-reentrant path gives
    every trainable parameter a real gradient; 9c verifies checkpointing
    changes only how the backward pass is computed, not its values.
    """
    print("[9] gradient_checkpointing_enable() doesn't silently drop backbone gradients")
    labels = torch.tensor([[1.0], [0.0]])

    # --- 9a: with the default (reentrant) implementation, the first unfrozen
    # block of a top-K model loses its parameter gradients. "full" mode is
    # unaffected, since the trainable embeddings make every activation require
    # grad, so it serves as the negative control in 9b/9c below. ---
    model = ASTHeartQAUnfreeze(unfreeze_mode="topk", topk_layers=4)
    model.encoder.gradient_checkpointing_enable()  # default reentrant=True
    model.train()
    _, logits = model(make_batch(n=2, seed=7))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
    first_unfrozen_layer = model.encoder.encoder.layer[12 - 4]  # layer 8
    missing = [n for n, p in first_unfrozen_layer.named_parameters() if p.requires_grad and p.grad is None]
    assert missing, (
        "Expected reentrant checkpointing to drop gradients on layer 8, but all of them "
        "were present -- either the upstream behaviour changed (in which case update this "
        "check) or this check is no longer exercising the intended code path"
    )
    print(f"    [confirmed] default (reentrant) checkpointing DOES drop gradients on "
          f"{missing} -- this is why the training scripts must pass "
          f"gradient_checkpointing_kwargs={CHECKPOINTING_KWARGS}")
    del model

    # --- 9b: with use_reentrant=False, exactly as the training scripts call
    # it, every trainable encoder parameter must receive a real gradient and
    # every frozen one none, in both "full" and "topk" modes. ---
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
                    f"{name} received no gradient"
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

    # --- 9c: numeric cross-check. Checkpointing changes how the backward pass
    # is computed, not the mathematics, so two identically initialized models
    # fed the same batch must produce matching logits and matching gradients
    # with checkpointing on and off.
    #
    # Two details make this comparison valid rather than vacuous:
    #   * Both models stay in train() mode. transformers' checkpointing layer
    #     only engages when self.training is True, so comparing in eval() would
    #     silently disable checkpointing in the "checkpointed" model.
    #   * Train mode leaves the encoder's dropout active, so the global RNG is
    #     reset to the same seed immediately before each forward call; otherwise
    #     the two models would draw different dropout masks and differ for
    #     reasons unrelated to checkpointing. ---
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
            f"{max_grad_diff} -- checkpointing altered the gradient values, not just memory use"
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
