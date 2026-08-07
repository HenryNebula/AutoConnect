"""Learned (NN) similarity for runtime tile classification -- the drop-in
replacement for ``gallery.color_ncc`` that issue #3 asks for.

Two uses:

1. Runtime pair similarity: ``nn_sim(a, b)`` returns the trained PairNet's
   same-type probability in [0, 1]. ``bot._pick_move`` can call it exactly where
   it currently calls ``color_ncc`` (just raise the decision threshold, e.g.
   0.55 -> 0.8, since a probability is better calibrated than NCC).

2. Gallery / type-count evaluation: cluster a set of crops by NN similarity
   (union-find over pairs above ``link_thr``) and report the number of types.
   The NCC gallery over-segments level 13 (46-50 types vs the true 42); a good
   NN should recover ~42. This is the headline metric from issue #3.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch

import dsio
from pairnet import PairNet, CANON, load_pairnet
import gallery as galmod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NNClassifier:
    """Wraps a trained PairNet as a pairwise same-type scorer."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path is None:
            model_path = _latest_model()
        if model_path and os.path.exists(model_path):
            self.model, _cfg = load_pairnet(model_path, map_location=DEVICE)
            self.model.to(DEVICE).eval()
            self.path = model_path
        else:
            self.path = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _batch_embed(self, crops, batch=512):
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), batch):
                x = torch.from_numpy(crops[i:i + batch].astype(np.float32) / 255.0)
                x = x.permute(0, 3, 1, 2).to(DEVICE)
                out.append(self.model.embed(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Same-type probability for two CANON crops (drop-in for color_ncc)."""
        if not self.available:
            return galmod.color_ncc(a, b)
        with torch.no_grad():
            ta = torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)
            tb = torch.from_numpy(b.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)
            return float(torch.sigmoid(self.model(ta, tb)).item())

    def sims(self, crops_a, crops_b, batch=512):
        """Vectorised same-type probabilities for aligned crop arrays."""
        if not self.available:
            return np.array([galmod.color_ncc(crops_a[i], crops_b[i])
                             for i in range(len(crops_a))], dtype=np.float32)
        ea = self._batch_embed(crops_a)
        eb = self._batch_embed(crops_b)
        ea_t = torch.from_numpy(ea).to(DEVICE)
        eb_t = torch.from_numpy(eb).to(DEVICE)
        with torch.no_grad():
            d = torch.abs(ea_t - eb_t)
            p = ea_t * eb_t
            logits = self.model.classifier(torch.cat([d, p], dim=1)).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()


def _latest_model() -> str | None:
    if not os.path.isdir(dsio.MODELS_DIR):
        return None
    cands = sorted(f for f in os.listdir(dsio.MODELS_DIR)
                   if f.startswith("pairnet_") and f.endswith(".pt"))
    return os.path.join(dsio.MODELS_DIR, cands[-1]) if cands else None


def cluster_types(crops, sim_fn, link_thr):
    """Union-find clustering: link two crops when sim_fn >= link_thr.
    Returns (n_types, labels)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    n = len(crops)
    if n == 0:
        return 0, []
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            v = sim_fn(crops[i], crops[j])
            if v >= link_thr:
                rows.append(i); cols.append(j); vals.append(1.0)
    if rows:
        sm = coo_matrix((vals, (rows, cols)), shape=(n, n))
        sm = sm.maximum(sm.T)
        n_comp, lab = connected_components(sm, directed=False)
    else:
        n_comp, lab = n, list(range(n))
    return n_comp, lab


def eval_clustering(level=13, link_thr_nn=0.82, link_thr_ncc=0.90):
    """Compare NN vs NCC type-clustering on harvested crops of one level.

    Reports the number of types each produces. NCC over-segments (~46-50 at L13);
    a good NN recovers ~42."""
    print(f"[eval-cluster] level {level} crops")
    crops = []
    for _, sh in dsio.iter_harvest_shards():
        if int(sh["level"]) != level:
            continue
        pairs = sh["pairs"]
        crops.append(pairs.reshape(pairs.shape[0] * 2, CANON, CANON, 3))
    if not crops:
        print(f"  no harvested crops for level {level}")
        return
    crops = np.concatenate(crops)
    print(f"  {len(crops)} crops")

    def ncc_sim(a, b):
        return galmod.color_ncc(a, b)

    n_ncc, _ = cluster_types(crops, ncc_sim, link_thr_ncc)
    print(f"  NCC link_thr={link_thr_ncc}: {n_ncc} types")

    nn = NNClassifier()
    if nn.available:
        # vectorised NN similarity matrix
        ea = nn._batch_embed(crops)
        import torch as _t
        ea_t = _t.from_numpy(ea).to(DEVICE)
        with _t.no_grad():
            d = _t.abs(ea_t[None] - ea_t[:, None])
            p = ea_t[None] * ea_t[:, None]
            import torch.nn.functional as _F
            sim = _t.sigmoid(nn.model.classifier(
                _t.cat([d, p], dim=-1)).squeeze(-1)).cpu().numpy()
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import coo_matrix
        iu, ju = np.triu_indices(len(crops), 1)
        keep = sim[iu, ju] >= link_thr_nn
        sm = coo_matrix((np.ones(keep.sum()), (iu[keep], ju[keep])), shape=(len(crops), len(crops)))
        sm = sm.maximum(sm.T)
        n_nn, _ = connected_components(sm, directed=False)
        print(f"  NN  link_thr={link_thr_nn}: {n_nn} types  (model={os.path.basename(nn.path)})")
        print(f"  -> NCC over-segments by {n_ncc - n_nn:+d} vs NN")
    else:
        print("  (no trained model found; train_classifier.py first)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=13)
    ap.add_argument("--nn-thr", type=float, default=0.82)
    ap.add_argument("--ncc-thr", type=float, default=0.90)
    a = ap.parse_args()
    eval_clustering(a.level, a.nn_thr, a.ncc_thr)
