"""Reference-gallery tile classifier (a trained CV model).

Built OFFLINE from same-type crop pairs harvested by observing the oracle's
legal moves (collect.py). Pairs are must-links; very-high colour-NCC between
crops adds further must-links. Connected components of that graph -> stable
type clusters -> a gallery of representative crops per type.

At RUNTIME a tile is classified by max translation-tolerant colour-NCC to the
gallery's templates. No game state is read at runtime; the gallery is just a
bundle of labelled images.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import cv2
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

CANON = 40


def _gray(t):
    return cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32)


def color_ncc(a: np.ndarray, b: np.ndarray, r_half: int = 16, t_half: int = 13) -> float:
    """Max translation-tolerant colour NCC between two CANON-size crops.

    Slides b's central template over a's central region (3-channel
    TM_CCOEFF_NORMED) to absorb residual sub-pixel jitter.
    """
    ga = a.astype(np.float32)
    gb = b.astype(np.float32)
    H = CANON
    ra0, ra1 = H // 2 - r_half, H // 2 + r_half
    tb0, tb1 = H // 2 - t_half, H // 2 + t_half
    region = ga[ra0:ra1, ra0:ra1]
    templ = gb[tb0:tb1, tb0:tb1]
    if region.shape[0] < templ.shape[0] or region.shape[1] < templ.shape[1]:
        return float(cv2.matchTemplate(ga, templ, cv2.TM_CCOEFF_NORMED).max())
    return float(cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED).max())


def build_gallery(pairs_path: str, out_path: str, link_thr: float = 0.90,
                  max_per_type: int = 12) -> dict:
    """Build a gallery .npz from harvested same-type pairs.

    gallery = {'crops': (T, max_per_type, CANON, CANON, 3) with zeros for
    unused slots, 'counts': (T,) usable reps per type}.
    """
    d = np.load(pairs_path, allow_pickle=True)
    pairs = d["pairs"]      # (N,2,CANON,CANON,3)
    n = len(pairs)
    print(f"[gallery] {n} pairs loaded")
    # node = each crop; build must-link graph
    reps = pairs.reshape(-1, CANON, CANON, 3)        # (2n, ...)
    N = reps.shape[0]
    # pair must-links: crop 2k and 2k+1 are same type
    edges = set()
    for k in range(n):
        a, b = 2 * k, 2 * k + 1
        edges.add((a, b))
    # high-NCC links between pair representatives (first crop of each pair)
    # to merge the same type across boards without chaining different types.
    head = reps[0::2]   # one rep per pair
    H = len(head)
    # pairwise NCC among heads (H can be a few hundred -> H^2 fine)
    print(f"[gallery] computing {H}x{H} head NCC ...")
    for i in range(H):
        ai = reps[2 * i]
        for j in range(i + 1, H):
            if color_ncc(ai, reps[2 * j]) >= link_thr:
                edges.add((2 * i, 2 * j))
                edges.add((2 * i + 1, 2 * j))
                edges.add((2 * i, 2 * j + 1))
                edges.add((2 * i + 1, 2 * j + 1))
    if edges:
        r, c = zip(*edges)
        sm = coo_matrix((np.ones(len(r)), (np.array(r), np.array(c))), shape=(N, N))
        sm = sm.maximum(sm.T)
        n_comp, comp = connected_components(sm, directed=False)
    else:
        n_comp, comp = 0, np.zeros(N, dtype=int)
    # group crops by component
    types = []
    for ci in range(n_comp):
        crops = reps[comp == ci]
        if len(crops) < 2:
            continue
        # pick a stable ordering; keep up to max_per_type
        crops = crops[:max_per_type]
        types.append(crops)
    T = len(types)
    print(f"[gallery] {T} distinct types, sizes:", sorted(len(t) for t in types), "...")
    counts = np.array([len(t) for t in types], dtype=np.int32)
    arr = np.zeros((T, max_per_type, CANON, CANON, 3), dtype=np.uint8)
    for i, t in enumerate(types):
        arr[i, :len(t)] = np.stack(t)
    np.savez_compressed(out_path, crops=arr, counts=counts)
    print(f"[gallery] saved {T} types -> {out_path}")
    return dict(crops=arr, counts=counts, T=T)


class GalleryClassifier:
    """Nearest-gallery-type classifier."""

    def __init__(self, gallery_path: str, min_match: float = 0.72, max_templates: int = 4):
        d = np.load(gallery_path, allow_pickle=True)
        self.crops = d["crops"]          # (T, K, CANON, CANON, 3)
        self.counts = d["counts"]        # (T,)
        self.T = self.crops.shape[0]
        self.min_match = min_match
        # pre-extract up to max_templates central templates per type
        self._templates = []
        H = CANON
        tb0, tb1 = H // 2 - 13, H // 2 + 13
        for ti in range(self.T):
            reps = []
            for k in range(min(int(self.counts[ti]), max_templates)):
                reps.append(self.crops[ti, k, tb0:tb1, tb0:tb1].astype(np.float32))
            self._templates.append(reps)
        self._r_half = 16
        self._ra0 = H // 2 - self._r_half
        self._ra1 = H // 2 + self._r_half

    def classify_crop(self, crop: np.ndarray) -> tuple[int, float]:
        """Return (type_id, best_ncc). type_id=-1 if below min_match."""
        region = crop.astype(np.float32)[self._ra0:self._ra1, self._ra0:self._ra1]
        best_t, best_v = -1, self.min_match
        for ti, reps in enumerate(self._templates):
            for templ in reps:
                if region.shape[0] < templ.shape[0]:
                    continue
                v = float(cv2.matchTemplate(region, templ, cv2.TM_CCOEFF_NORMED).max())
                if v > best_v:
                    best_v, best_t = v, ti
        return best_t, best_v


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--link-thr", type=float, default=0.90)
    a = ap.parse_args()
    build_gallery(a.pairs, a.out, link_thr=a.link_thr)
