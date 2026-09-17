"""
Generic backbone + binary QA head, for Reviewer #7 Comment 2.

Structurally identical to src/models/ast_qa.py's ASTHeartQA -- same head
architecture (Linear(hidden_dim, 128) -> ReLU -> Dropout(0.1) -> Linear(128,
1)) -- just with the backbone swapped for PANNs/YAMNet/HuBERT instead of AST.

Two modes, mirroring ../../reviewer1_unfreezing_ablation's frozen/full split:
  - freeze_backbone=True (default): "Freeze each backbone encoder; attach an
    identical classification head," per CLAUDE.md Sec 9.2 Comment 2's
    original ask -- isolates architecture (attention vs. conv/MobileNet
    inductive bias) from fine-tuning capacity.
  - freeze_backbone=False (added 2026-09-14): unfreezes the whole backbone
    too, per your follow-up question of whether AST's win in the frozen
    comparison is partly just a frozen-feature-transfer artifact rather than
    an architectural one -- see ../README.md "Full-unfreeze extension".
"""

import torch.nn as nn

from backbones import build_backbone


class BackboneQAHead(nn.Module):
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
