"""
AST-QA model with a configurable backbone freezing policy.

Used for the unfreezing ablation: fully frozen encoder, end-to-end
fine-tuning, and top-K layer unfreezing.

The architecture is the same as the paper's `ASTHeartQA`
(`src/models/ast_qa.py`): an AudioSet-pretrained AST encoder whose
768-dimensional [CLS] embedding feeds a binary QA head
(Linear(768,128) -> ReLU -> Dropout(0.1) -> Linear(128,1)). It is a separate
module so the code that reproduces the published Table 2 / Figure 3 results
stays unchanged.

Encoder layout (`self.encoder` is a `transformers` ASTModel):
    embeddings              patch + positional embeddings
    encoder.layer[0..11]    12 ASTLayer transformer blocks
    layernorm               final LayerNorm, applied to the hidden states
                            from which the [CLS] token consumed by both
                            classifier heads is taken

Freezing policies:
    "frozen"  encoder entirely frozen; only the QA head trains.
    "full"    every encoder parameter trains (end-to-end fine-tuning).
    "topk"    the last K transformer blocks (encoder.layer[12-K:]) and the
              final layernorm train; earlier blocks and the patch embeddings
              stay frozen. Early blocks hold generic AudioSet features, so only
              the blocks nearest the head adapt. The ablation uses K in {2, 4}.
"""

import torch.nn as nn
from transformers import ASTConfig, ASTForAudioClassification

UNFREEZE_MODES = ("frozen", "full", "topk")
VALID_TOPK = (2, 4)


class ASTHeartQAUnfreeze(nn.Module):
    """AST encoder + binary QA head, with a selectable backbone freeze policy.

    Args:
        model_name: Hugging Face id of the pretrained AST checkpoint.
        unfreeze_mode: One of ``UNFREEZE_MODES`` ("frozen", "full", "topk");
            see the module docstring for what each policy trains.
        topk_layers: Number of trailing transformer blocks to unfreeze.
            Required (and restricted to ``VALID_TOPK``) when
            ``unfreeze_mode="topk"``; ignored otherwise.
    """

    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 unfreeze_mode="frozen", topk_layers=0):
        super().__init__()
        if unfreeze_mode not in UNFREEZE_MODES:
            raise ValueError(f"unfreeze_mode must be one of {UNFREEZE_MODES}, got {unfreeze_mode!r}")
        if unfreeze_mode == "topk" and topk_layers not in VALID_TOPK:
            raise ValueError(
                f"topk_layers must be one of {VALID_TOPK}, got {topk_layers!r}"
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
        """Set ``requires_grad`` on the encoder according to the current mode.

        The encoder is first reset to fully frozen, so calling this again on a
        partially unfrozen model (as the progressive top-K schedule does after
        warmup) leaves no stale trainable parameters.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False

        if self.unfreeze_mode == "frozen":
            return

        if self.unfreeze_mode == "full":
            for p in self.encoder.parameters():
                p.requires_grad = True
            return

        # topk: unfreeze the trailing K blocks (encoder.layer[12-K:]) and the
        # final layernorm that produces the [CLS] state used by the QA head.
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

        Used by the progressive top-K schedule: train with
        ``unfreeze_mode="frozen"`` for the head-only warmup epochs, then call
        this with ``("topk", K)`` and continue training. The optimizer must be
        rebuilt afterwards, since the set of trainable parameters changes.
        """
        if unfreeze_mode not in UNFREEZE_MODES:
            raise ValueError(f"unfreeze_mode must be one of {UNFREEZE_MODES}, got {unfreeze_mode!r}")
        if unfreeze_mode == "topk" and topk_layers not in VALID_TOPK:
            raise ValueError(f"topk_layers must be one of {VALID_TOPK}, got {topk_layers!r}")
        self.unfreeze_mode = unfreeze_mode
        self.topk_layers = topk_layers
        self._apply_freeze_policy()

    def trainable_backbone_parameters(self):
        """Encoder parameters currently marked trainable.

        Returns an empty list in "frozen" mode. Used to build the lower
        learning-rate optimizer parameter group for the backbone.
        """
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def num_trainable_parameters(self):
        """Total number of scalar parameters with ``requires_grad=True``."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self):
        """Total number of scalar parameters, trainable or not."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_values):
        """Run the encoder and both heads on a batch of log-mel spectrograms.

        Args:
            input_values: (batch, 1024 frames, 128 mel bins) tensor as
                produced by ``ASTFeatureExtractor``.

        Returns:
            (original_logits, qa_logits): the pretrained 527-class AudioSet
            logits and the binary quality logit per clip. The AudioSet logits
            are not used in the loss; they are returned so the signature
            matches ``ASTHeartQA`` and the two models are interchangeable.
        """
        outputs = self.encoder(input_values)
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        original_logits = self.original_classifier(cls_token_state)
        qa_logits = self.qa_classifier(cls_token_state)
        return original_logits, qa_logits


if __name__ == "__main__":
    # Print trainable/total parameter counts for each ablation condition
    for mode, k in [("frozen", 0), ("full", 0), ("topk", 2), ("topk", 4)]:
        m = ASTHeartQAUnfreeze(unfreeze_mode=mode, topk_layers=k)
        tag = mode if mode != "topk" else f"topk{k}"
        print(f"{tag:10s} trainable={m.num_trainable_parameters():>10,d} "
              f"total={m.num_total_parameters():>10,d}")
