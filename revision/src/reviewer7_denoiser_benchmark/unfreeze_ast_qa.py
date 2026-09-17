"""
AST-QA model with a configurable backbone freeze policy: frozen, fully
fine-tuned, or top-K layer unfreezing.

The architecture is identical to the published model in
`../models/ast_qa.py` (ASTHeartQA) -- same AST backbone, same QA head
Linear(768, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 1) -- and is kept as
a separate class rather than added to that file, so that the code path which
reproduces the originally published results is never modified by a revision
experiment.

Encoder structure (as exposed by `transformers`' ASTModel):
    embeddings              patch + positional embeddings
    encoder.layer[0..11]    12 ASTLayer transformer blocks
    layernorm               final LayerNorm, producing the [CLS] token
                            representation that both classifiers consume

"Top-K" unfreezing means the LAST K transformer blocks
(`encoder.layer[12-K:]`) plus the final layernorm, i.e. the layers closest to
the classifier head. This is the conventional progressive-unfreezing choice:
the early, general AudioSet features stay frozen while only the late,
task-specific layers adapt. K is restricted to {2, 4}, the two settings
reported in the ablation.
"""

import torch.nn as nn
from transformers import ASTConfig, ASTForAudioClassification

UNFREEZE_MODES = ("frozen", "full", "topk")
VALID_TOPK = (2, 4)


class ASTHeartQAUnfreeze(nn.Module):
    """AST binary quality classifier with a selectable backbone freeze policy.

    Args:
        model_name: pretrained AST checkpoint to load.
        unfreeze_mode: "frozen" (train the QA head only), "full" (fine-tune
            the whole encoder) or "topk" (fine-tune the last `topk_layers`
            transformer blocks plus the final layernorm).
        topk_layers: number of trailing encoder layers to unfreeze; used only
            when `unfreeze_mode="topk"`, and restricted to {2, 4}.
    """

    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 unfreeze_mode="frozen", topk_layers=0):
        super().__init__()
        if unfreeze_mode not in UNFREEZE_MODES:
            raise ValueError(f"unfreeze_mode must be one of {UNFREEZE_MODES}, got {unfreeze_mode!r}")
        if unfreeze_mode == "topk" and topk_layers not in VALID_TOPK:
            raise ValueError(
                f"topk_layers must be one of {VALID_TOPK} (the settings reported in the "
                f"unfreezing ablation), got {topk_layers!r}"
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
        """Apply the current freeze policy to the encoder parameters.

        The encoder is always reset to fully frozen first and then
        selectively re-enabled, which makes set_unfreeze_mode() safe to call
        repeatedly (e.g. a head-only warmup followed by top-K unfreezing)
        without leaving a stale requires_grad=True from an earlier mode.
        """
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
        """Switch the freeze policy mid-training.

        Used for the progressive schedule: a head-only warmup with
        `unfreeze_mode="frozen"`, then `unfreeze_mode="topk"` to unfreeze the
        last K layers and continue training the same model instance.
        """
        if unfreeze_mode not in UNFREEZE_MODES:
            raise ValueError(f"unfreeze_mode must be one of {UNFREEZE_MODES}, got {unfreeze_mode!r}")
        if unfreeze_mode == "topk" and topk_layers not in VALID_TOPK:
            raise ValueError(f"topk_layers must be one of {VALID_TOPK}, got {topk_layers!r}")
        self.unfreeze_mode = unfreeze_mode
        self.topk_layers = topk_layers
        self._apply_freeze_policy()

    def trainable_backbone_parameters(self):
        """Encoder parameters currently trainable (empty in frozen mode).

        Returned separately from the head so that the optimizer can give the
        backbone its own, lower learning rate.
        """
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_values):
        """Return (original AudioSet logits, QA logit) for a batch of spectrograms."""
        outputs = self.encoder(input_values)
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        # The original AudioSet head (527 classes) is not used by the QA loss;
        # it is returned only so this class has the same forward signature and
        # return arity as the published ASTHeartQA, letting the two be used
        # interchangeably by the training/eval loops.
        original_logits = self.original_classifier(cls_token_state)
        qa_logits = self.qa_classifier(cls_token_state)
        return original_logits, qa_logits


if __name__ == "__main__":
    # Print the trainable/total parameter counts for each freeze policy.
    for mode, k in [("frozen", 0), ("full", 0), ("topk", 2), ("topk", 4)]:
        m = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"{tag:10s} trainable={m.num_trainable_parameters():>10,d} "
              f"total={m.num_total_parameters():>10,d}")
