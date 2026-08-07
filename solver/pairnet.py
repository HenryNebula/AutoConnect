"""Small siamese CNN that classifies a pair of 40x40 tile crops as same-type or
different-type -- the learned replacement for the NCC classifier in issue #3.

Architecture: a shared lightweight CNN backbone maps each crop to an embedding.
The pair head consumes |e_a - e_b| and e_a * e_b (the "absolute + product"
siamese features, which separate same/different better than either alone) and
emits a same-probability logit. The backbone is small because the inputs are
40x40; it trains in seconds/epoch on GPU.

The net is configurable: the default ``widths=(16,32,64)`` maximises accuracy;
``widths=(12,24,36), embed=40`` (the "tiny" variant) trades a little accuracy
for ~2x lower CPU latency so single-pair ONNX inference stays within the 3x-NCC
serving budget. The embedding is also exposed (``embed``) so the model can build
a gallery / compute pairwise similarity at runtime, dropping in for
``gallery.color_ncc``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CANON = 40
EMBED_DIM = 64


def _conv_stack(widths):
    layers = []
    inch = 3
    for w in widths:
        layers += [nn.Conv2d(inch, w, 3, padding=1), nn.BatchNorm2d(w),
                   nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        inch = w
    return nn.Sequential(*layers)


class Backbone(nn.Module):
    """N conv/pool blocks -> flatten -> embedding. Handles variable widths."""

    def __init__(self, widths=(16, 32, 64), embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = _conv_stack(widths)
        side = CANON >> len(widths)            # CANON / 2**#blocks (each block pools)
        flat = widths[-1] * side * side
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, embed_dim),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.head(self.net(x))


class PairNet(nn.Module):
    """Siamese same/different classifier."""

    def __init__(self, widths=(16, 32, 64), embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = Backbone(widths, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )
        self.widths = widths
        self.embed_dim = embed_dim

    def _features(self, ea, eb):
        return torch.cat([torch.abs(ea - eb), ea * eb], dim=1)

    def forward(self, a, b):
        ea, eb = self.backbone(a), self.backbone(b)
        return self.classifier(self._features(ea, eb)).squeeze(-1)

    @torch.no_grad()
    def embed(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def pair_proba(self, a, b):
        return torch.sigmoid(self.forward(a, b))


def load_pairnet(model_path, map_location="cpu"):
    """Reconstruct a PairNet from a checkpoint, reading widths/embed_dim (with
    defaults for checkpoints saved before the model was made configurable)."""
    import os
    ckpt = torch.load(model_path, map_location=map_location)
    widths = tuple(ckpt.get("widths", (16, 32, 64)))
    embed_dim = int(ckpt.get("embed_dim", EMBED_DIM))
    pn = PairNet(widths=widths, embed_dim=embed_dim)
    pn.load_state_dict(ckpt["state_dict"])
    pn.eval()
    return pn, dict(widths=widths, embed_dim=embed_dim, canon=ckpt.get("canon", CANON))


# preset configurations
PRESETS = {
    "default": dict(widths=(16, 32, 64), embed_dim=EMBED_DIM),
    "tiny": dict(widths=(12, 24, 36), embed_dim=40),      # ~2x fewer channels
    "micro": dict(widths=(16, 32), embed_dim=48),         # 2 blocks: fewest ops
}
