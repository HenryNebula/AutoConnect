"""Robustness + hard-negative evaluation -- the metrics that actually justify the
learned classifier for issue #3.

On clean harvested crops NCC is already AUC ~1.0 (translation-tolerant colour
NCC + clean harvest), so plain accuracy flatters the baseline. Two metrics
separate the models:

1. Jitter robustness: apply INDEPENDENT per-crop jitter (translation/scale/
   rotation/brightness/blur, mimicking live perception) to each pair member and
   measure same/different AUC. NCC collapses (1.0 -> ~0.75); an augmentation-
   trained NN holds.
2. Hard-negative accuracy: on the mined confusable cross-type pairs, the fraction
   correctly rejected.

Auto-discovers every model in $AC_DATA_DIR/models plus the NCC baseline, and
prints one table. Handles both siamese PairNet checkpoints and SupCon
(ContrastiveModel) checkpoints.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F
import cv2

import dsio
from pairnet import load_pairnet, CANON
from train_classifier import _auc
from gallery import color_ncc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _jitter(img, lvl, rng):
    if lvl == 0:
        return img
    H = CANON
    t = lvl * 2
    M = np.float32([[1, 0, rng.uniform(-t, t)], [0, 1, rng.uniform(-t, t)]])
    img = cv2.warpAffine(img, M, (H, H), borderMode=cv2.BORDER_REPLICATE)
    s = 1 + lvl * 0.06 * rng.uniform(-1, 1)
    c = H / 2
    M = np.float32([[s, 0, c * (1 - s)], [0, s, c * (1 - s)]])
    img = cv2.warpAffine(img, M, (H, H), borderMode=cv2.BORDER_REPLICATE)
    a = rng.uniform(-lvl * 0.08, lvl * 0.08)
    M = cv2.getRotationMatrix2D((c, c), np.degrees(a), 1)
    img = cv2.warpAffine(img, M, (H, H), borderMode=cv2.BORDER_REPLICATE)
    img = np.clip(img * (1 + rng.uniform(-lvl * 0.12, lvl * 0.12)), 0, 255).astype(np.uint8)
    if lvl >= 2:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def _jitter_pairs(ca, cb, lvl, seed):
    rng = np.random.default_rng(seed)
    ja = np.stack([_jitter(c, lvl, rng) for c in ca])
    jb = np.stack([_jitter(c, lvl, rng) for c in cb])
    return ja, jb


def _is_supcon(ckpt):
    return "proj.0.weight" in ckpt["state_dict"]


def _load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    if _is_supcon(ckpt):
        import contrastive as C
        m = C.ContrastiveModel(ckpt["widths"], ckpt["embed_dim"]).to(DEVICE)
        m.load_state_dict(ckpt["state_dict"]); m.eval()
        return m, "supcon"
    pn, _ = load_pairnet(path, map_location=DEVICE)
    pn.to(DEVICE).eval()
    return pn, "siamese"


@torch.no_grad()
def _score(model, kind, ca, cb, batch=1024):
    if kind == "ncc":
        return np.array([color_ncc(ca[i], cb[i]) for i in range(len(ca))], dtype=np.float32)
    if kind == "siamese":
        out = []
        for i in range(0, len(ca), batch):
            a = torch.from_numpy(ca[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
            b = torch.from_numpy(cb[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
            out.append(torch.sigmoid(model(a, b)).cpu().numpy())
        return np.concatenate(out)
    # supcon: cosine of embeddings
    ea = _embed(model, ca, batch); eb = _embed(model, cb, batch)
    return (ea * eb).sum(1)


@torch.no_grad()
def _embed(model, crops, batch=1024):
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(crops[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
        out.append(F.normalize(model.embed(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def _best_thr(scores, ys):
    order = np.argsort(-scores); s, y = scores[order], ys[order]
    P = y.sum(); N = len(y) - P
    j = np.cumsum(y == 1) / max(P, 1) - np.cumsum(y == 0) / max(N, 1)
    return float(s[int(j.argmax())])


def run(n=3000, levels=(0, 1, 2, 3), seed=0):
    # test pair set
    ca_all, cb_all, lab_all, kind_all = [], [], [], []
    for fn in sorted(os.listdir(dsio.DATASET_DIR)):
        if fn.startswith("shard_test_") and fn.endswith(".npz"):
            d = np.load(os.path.join(dsio.DATASET_DIR, fn))
            ca_all.append(d["ca"]); cb_all.append(d["cb"])
            lab_all.append(d["label"]); kind_all.append(d["kind"])
    ca = np.concatenate(ca_all)[:n]; cb = np.concatenate(cb_all)[:n]
    lab = np.concatenate(lab_all)[:n].astype(int); kind = np.concatenate(kind_all)[:n].astype(int)
    print(f"[robust] {len(ca)} test pairs; jitter levels {levels} (independent per-crop jitter)\n")

    # pre-jitter once per level (same crops for all models -> fair)
    jittered = {lvl: _jitter_pairs(ca, cb, lvl, seed) for lvl in levels}

    # gather scorers: NCC baseline + every .pt
    scorers = [("NCC", None, "ncc")]
    for fn in sorted(os.listdir(dsio.MODELS_DIR)):
        if fn.startswith("pairnet_") and fn.endswith(".pt"):
            scorers.append((fn.replace("pairnet_", "").replace(".pt", ""),
                            os.path.join(dsio.MODELS_DIR, fn), None))

    header = ["model"] + [f"j{l}" for l in levels] + ["hard-neg acc"]
    rows = []
    for name, path, forced in scorers:
        kind_m = forced
        model = None
        if path is not None:
            model, kind_m = _load_model(path)
        aucs = []
        for lvl in levels:
            ja, jb = jittered[lvl]
            sc = _score(model, kind_m, ja, jb)
            aucs.append(_auc(sc, lab))
        # hard-neg acc on clean (fraction of confusable cross-type pairs rejected)
        sc_clean = _score(model, kind_m, ca, cb)
        thr = _best_thr(sc_clean, lab)
        h = kind == 1
        hard_acc = float((sc_clean[h] < thr).mean()) if h.any() else float("nan")
        rows.append([name] + [f"{a:.4f}" for a in aucs] + [f"{hard_acc:.4f}"])

    w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
    print("\n(j# = same/different AUC at jitter level #; hard-neg acc = confusable-pair rejection on clean)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.n, tuple(a.levels), a.seed)
