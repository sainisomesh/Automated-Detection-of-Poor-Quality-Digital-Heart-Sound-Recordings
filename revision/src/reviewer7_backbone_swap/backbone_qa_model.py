"""
Audio backbone plus the binary quality head (Reviewer #7, Comment 2).

The head is the same as ASTHeartQA's in ../models/ast_qa.py,
Linear(embedding_dim, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 1), so the
encoder is the only thing that changes between this model and AST-QA.

Two modes, matching the frozen/full split of the unfreezing ablation in
../reviewer1_unfreezing_ablation/:
  - freeze_backbone=True (default): the backbone is a fixed feature extractor
    and only the head is trained.
  - freeze_backbone=False: the backbone is fine-tuned end to end with the
    head. Each wrapper still keeps its fixed, non-learnable front-end frozen
    (see backbones.py).
"""

import torch.nn as nn

from backbones import build_backbone


class BackboneQAHead(nn.Module):
    """Binary usable-vs-noise classifier: backbone embedding -> QA head logit.

    Args:
        backbone_name: one of the keys in backbones.BACKBONES
            ("panns", "yamnet", "hubert").
        freeze_backbone: if True the backbone is kept in eval mode and run
            under no_grad, so only qa_classifier is trained.

    forward() takes a (B, 160000) float32 16 kHz waveform batch and returns
    raw (B, 1) logits for BCEWithLogitsLoss. Sigmoid of the logit is the
    probability that the recording is usable.
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
