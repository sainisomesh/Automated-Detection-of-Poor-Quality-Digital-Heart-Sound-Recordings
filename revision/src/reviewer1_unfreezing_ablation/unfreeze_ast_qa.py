"""
AST-QA model variant for Reviewer #1 Comment 2 (frozen vs. full vs. top-K
backbone unfreezing ablation).

Structurally identical to src/models/ast_qa.py's ASTHeartQA -- same AST
backbone, same qa_classifier head (Linear(768,128) -> ReLU -> Dropout(0.1)
-> Linear(128,1)) -- but built as a SEPARATE file here rather than editing
the shared src/models/ast_qa.py / reproducibility/src/models/ast_qa.py.
Same convention PAPER_REVISIONS/reviewer7_backbone_swap/src/backbone_qa_model.py
already established: shared code that produced the published Table 2 /
Figure 3 numbers stays untouched, so a bug in a new revision experiment can
never retroactively change a published result.

Encoder structure (verified directly against the installed `transformers`
version, not assumed from docs -- see audit notes in ../README.md):
self.encoder is an ASTModel with
    embeddings                     (patch + positional embeddings)
    encoder.layer[0..11]            (12 ASTLayer transformer blocks)
    layernorm                       (final LayerNorm, feeds the CLS token
                                      that both classifiers consume)

"Top-K" unfreezing = the LAST K layers (encoder.layer[12-K:]) plus the
final layernorm -- the layers closest to the classifier head. This is the
conventional choice for progressive unfreezing: keep the early, general
AudioSet features frozen the longest, adapt only the late, task-specific
layers. CLAUDE.md Sec 9.1 Comment 2 specifies K in {2, 4}.
"""

import torch.nn as nn
from transformers import ASTConfig, ASTForAudioClassification

UNFREEZE_MODES = ("frozen", "full", "topk")
VALID_TOPK = (2, 4)


class ASTHeartQAUnfreeze(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 unfreeze_mode="frozen", topk_layers=0):
        super().__init__()
        if unfreeze_mode not in UNFREEZE_MODES:
            raise ValueError(f"unfreeze_mode must be one of {UNFREEZE_MODES}, got {unfreeze_mode!r}")
        if unfreeze_mode == "topk" and topk_layers not in VALID_TOPK:
            raise ValueError(
                f"topk_layers must be one of {VALID_TOPK} per CLAUDE.md Sec 9.1 Comment 2, "
                f"got {topk_layers!r}"
            )

        self.config = ASTConfig.from_pretrained(model_name)
        self.original_model = ASTForAudioClassification.from_pretrained(model_name)
        self.encoder = self.original_model.audio_spectrogram_transformer
        self.original_classifier = self.original_model.classifier

        self.qa_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        self.n_encoder_layers = len(self.encoder.encoder.layer)
        self.unfreeze_mode = unfreeze_mode
        self.topk_layers = topk_layers
        self._apply_freeze_policy()

    def _apply_freeze_policy(self):
        # Always reset to fully frozen first, then selectively re-enable --
        # this is what makes set_unfreeze_mode() safe to call more than
        # once (e.g. the topk warmup -> unfreeze transition) without ever
        # leaving a stale requires_grad=True behind from a prior mode.
        for p in self.encoder.parameters():
            p.requires_grad = False

        if self.unfreeze_mode == "frozen":
            return

        if self.unfreeze_mode == "full":
            for p in self.encoder.parameters():
                p.requires_grad = True
            return

        # topk
        k = self.topk_layers
        if not (0 < k <= self.n_encoder_layers):
            raise ValueError(f"topk_layers={k} out of range for {self.n_encoder_layers} encoder layers")
        for layer in self.encoder.encoder.layer[self.n_encoder_layers - k:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.encoder.layernorm.parameters():
            p.requires_grad = True

    def set_unfreeze_mode(self, unfreeze_mode, topk_layers=0):
        """Switch freeze policy mid-training. Used for Mode C's progressive
        schedule: head-only warmup (mode='frozen'), then unfreeze the top-K
        layers (mode='topk') and keep training."""
        if unfreeze_mode not in UNFREEZE_MODES:
            raise ValueError(f"unfreeze_mode must be one of {UNFREEZE_MODES}, got {unfreeze_mode!r}")
        if unfreeze_mode == "topk" and topk_layers not in VALID_TOPK:
            raise ValueError(f"topk_layers must be one of {VALID_TOPK}, got {topk_layers!r}")
        self.unfreeze_mode = unfreeze_mode
        self.topk_layers = topk_layers
        self._apply_freeze_policy()

    def trainable_backbone_parameters(self):
        """Encoder params currently marked trainable (empty list in frozen mode)."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_values):
        outputs = self.encoder(input_values)
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        # Original AudioSet logits (527 classes) -- unused by the QA loss,
        # kept only for structural parity with src/models/ast_qa.py's
        # ASTHeartQA (same forward signature, same two return values).
        original_logits = self.original_classifier(cls_token_state)
        qa_logits = self.qa_classifier(cls_token_state)
        return original_logits, qa_logits


if __name__ == "__main__":
    for mode, k in [("frozen", 0), ("full", 0), ("topk", 2), ("topk", 4)]:
        m = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"{tag:10s} trainable={m.num_trainable_parameters():>10,d} "
              f"total={m.num_total_parameters():>10,d}")
