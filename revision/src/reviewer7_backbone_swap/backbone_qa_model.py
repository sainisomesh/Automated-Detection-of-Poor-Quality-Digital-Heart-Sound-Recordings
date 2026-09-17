"""
Generic audio backbone plus the paper's binary quality-assurance head
(Reviewer #7, Comment 2).

The head is architecturally identical to ASTHeartQA's in ../models/ast_qa.py
-- Linear(embedding_dim, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 1) -- so
that the only difference between this model and the published AST-QA model is
which encoder produces the embedding. Holding the head fixed is what makes the
backbone the sole independent variable in the comparison.

Two modes, matching the frozen/full split used by the unfreezing ablation in
../reviewer1_unfreezing_ablation/:
  - freeze_backbone=True (default): the backbone is a fixed feature extractor
    and only the head is trained, isolating the effect of the pretrained
    representation and architectural inductive bias (self-attention versus
    convolutional) from fine-tuning capacity.
  - freeze_backbone=False: the backbone is fine-tuned end to end alongside the
    head, which tests whether any AST advantage observed in the frozen setting
    is specific to frozen-feature transfer rather than to the architecture.
    Each wrapper still keeps its fixed, non-learnable front-end frozen; see
    backbones.py.
"""

import torch.nn as nn

from backbones import build_backbone


class BackboneQAHead(nn.Module):
    """Binary usable-vs-noise classifier: backbone embedding -> QA head logit.

    Args:
        backbone_name: one of the keys in backbones.BACKBONES
            ("panns", "yamnet", "hubert").
        freeze_backbone: if True the backbone is eval-locked and run under
            no_grad, so only qa_classifier is trained.

    forward() takes a (B, 160000) float32 16 kHz waveform batch and returns
    raw (B, 1) logits, for use with BCEWithLogitsLoss; apply a sigmoid to
    obtain the probability that a recording is of usable quality.
    """

    def __init__(self, backbone_name: str, freeze_backbone: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.backbone = build_backbone(backbone_name, freeze=freeze_backbone)

        self.qa_classifier = nn.Sequential(
            nn.Linear(self.backbone.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, wav):
        embedding = self.backbone(wav)
        logits = self.qa_classifier(embedding)
        return logits
