"""Small siamese CNN that classifies a pair of 40x40 tile crops as same-type or
different-type -- the learned replacement for the NCC classifier in issue #3.

Architecture: a shared lightweight CNN backbone maps each crop to a 64-d
embedding. The pair head consumes |e_a - e_b| and e_a * e_b (the "absolute +
product" siamese features, which separate same/different better than either
alone) and emits a same-probability logit. The backbone is small because the
inputs are 40x40; it trains in seconds/epoch on CPU.

The embedding is also exposed (``embed``) so the model can build a gallery /
compute pairwise similarity at runtime, dropping in for ``gallery.color_ncc``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CANON = 40
EMBED_DIM = 64


class Backbone(nn.Module):
    """3 conv blocks -> (CANON/8)^2 * 64 features -> EMBED_DIM embedding."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 20
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 10
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 5
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 5 * 5, embed_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.head(self.net(x))


class PairNet(nn.Module):
    """Siamese same/different classifier."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = Backbone(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def _features(self, ea, eb):
        d = torch.abs(ea - eb)
        p = ea * eb
        return torch.cat([d, p], dim=1)

    def forward(self, a, b):
        ea, eb = self.backbone(a), self.backbone(b)
        return self.classifier(self._features(ea, eb)).squeeze(-1)

    @torch.no_grad()
    def embed(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def pair_proba(self, a, b):
        return torch.sigmoid(self.forward(a, b))
