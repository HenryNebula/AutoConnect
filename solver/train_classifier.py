"""Train the siamese tile-pair classifier and evaluate it head-to-head against
the NCC baseline (issue #3).

The headline question: do same-type and different-type pairs separate cleanly?
NCC does not -- within-pair NCC bottoms out at ~0.71 while cross-type reaches
~0.75 (an overlap with no clean threshold). We train the NN on harvested,
exactly-labelled pairs (positives + easy & hard negatives) and report the
separation of both classifiers on a held-out, board-disjoint test set.

Augmentation (small translations/scale/brightness) mirrors the translation
tolerance of the NCC it replaces, so the NN is robust to sub-pixel jitter.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dsio
from pairnet import PairNet, CANON, PRESETS
from gallery import color_ncc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _load_split(split):
    ca, cb, lab, lvl, kind = [], [], [], [], []
    for f in sorted(os.listdir(dsio.DATASET_DIR)):
        if not (f.startswith(f"shard_{split}_") and f.endswith(".npz")):
            continue
        d = np.load(os.path.join(dsio.DATASET_DIR, f))
        ca.append(d["ca"]); cb.append(d["cb"]); lab.append(d["label"])
        lvl.append(d["level"]); kind.append(d["kind"])
    if not ca:
        return None
    return (np.concatenate(ca), np.concatenate(cb), np.concatenate(lab),
            np.concatenate(lvl), np.concatenate(kind))


class PairDataset(Dataset):
    def __init__(self, ca, cb, label, augment=False, seed=0):
        self.ca = ca.astype(np.float32)
        self.cb = cb.astype(np.float32)
        self.label = label.astype(np.float32)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.label)

    def _aug(self, img):
        # translation +-2px, scale 0.95-1.05, brightness +-10%
        t = self.rng.randint(-2, 3)
        M = np.float32([[1, 0, t], [0, 1, self.rng.randint(-2, 3)]])
        img = cv2.warpAffine(img, M, (CANON, CANON), borderMode=cv2.BORDER_REPLICATE)
        s = 0.95 + 0.10 * self.rng.rand()
        M2 = np.float32([[s, 0, CANON / 2 * (1 - s)], [0, s, CANON / 2 * (1 - s)]])
        img = cv2.warpAffine(img, M2, (CANON, CANON), borderMode=cv2.BORDER_REPLICATE)
        img *= (0.90 + 0.20 * self.rng.rand())
        return np.clip(img, 0, 255)

    def __getitem__(self, i):
        a, b = self.ca[i], self.cb[i]
        if self.augment:
            a, b = self._aug(a), self._aug(b)
        a = torch.from_numpy(a / 255.0).permute(2, 0, 1)
        b = torch.from_numpy(b / 255.0).permute(2, 0, 1)
        return a, b, self.label[i]


# ---------------------------------------------------------------------------
# train + eval
# ---------------------------------------------------------------------------
def _auc(scores, labels):
    """Rank-based AUC (1.0 = perfect: all positives score above all negatives).
    Uses ascending average ranks so ties are handled correctly."""
    from scipy.stats import rankdata
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)                # ascending, average ranks for ties
    s = ranks[labels == 1].sum()
    return float((s - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def _nn_scores(model, ca, cb, batch=2048):
    model.eval()
    out = []
    for i in range(0, len(ca), batch):
        a = torch.from_numpy(ca[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
        b = torch.from_numpy(cb[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
        out.append(torch.sigmoid(model(a, b)).cpu().numpy())
    return np.concatenate(out)


def _ncc_scores(ca, cb):
    return np.array([color_ncc(ca[i], cb[i]) for i in range(len(ca))], dtype=np.float32)


def _report(name, scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    same = scores[labels == 1]
    diff = scores[labels == 0]
    auc = _auc(scores, labels)
    acc = float(((scores >= 0.5).astype(int) == labels).mean())
    if len(same) and len(diff):
        sep = float(same.min() - diff.max())
        same_min, same_med = float(same.min()), float(np.median(same))
        diff_max, diff_med = float(diff.max()), float(np.median(diff))
    else:
        sep = float("nan")
        same_min = same_med = float("nan")
        diff_max = diff_med = float("nan")
    print(f"  {name:5s} AUC={auc:.4f} acc@0.5={acc:.4f} | "
          f"same[min={same_min:.3f} med={same_med:.3f}] "
          f"diff[max={diff_max:.3f} med={diff_med:.3f}] "
          f"gap(min_same-max_diff)={sep:+.3f}  (n_same={len(same)} n_diff={len(diff)})")
    return dict(auc=auc, acc=acc, same_min=same_min, same_med=same_med,
                diff_max=diff_max, diff_med=diff_med, gap=sep,
                n_same=int(len(same)), n_diff=int(len(diff)))


def train(epochs=25, batch=256, lr=1e-3, seed=0, tag=None, preset="default"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dsio.ensure_dirs(dsio.MODELS_DIR)
    cfg = PRESETS[preset]
    print(f"[train] preset={preset} widths={cfg['widths']} embed={cfg['embed_dim']}")

    tr = _load_split("train")
    va = _load_split("val")
    te = _load_split("test")
    if tr is None:
        raise SystemExit(f"no dataset in {dsio.DATASET_DIR}; run build_dataset.py first")
    print(f"[train] train={len(tr[2])} val={len(va[2]) if va else 0} "
          f"test={len(te[2]) if te else 0} device={DEVICE}")

    tr_ds = PairDataset(*tr[:3], augment=True, seed=seed)
    loader = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=True)
    model = PairNet(**cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    best_val_auc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for a, b, y in loader:
            a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logit = model(a, b)
            loss = bce(logit, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        val_auc = float("nan")
        if va is not None:
            sc = _nn_scores(model, va[0], va[1])
            val_auc = _auc(sc, va[2])
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[train] ep{ep+1:02d}/{epochs} loss={tot/n:.4f} val_auc={val_auc:.4f} "
              f"({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    tag = tag or f"{preset}_auc{best_val_auc:.3f}"
    out_path = os.path.join(dsio.MODELS_DIR, f"pairnet_{tag}.pt")
    torch.save({"state_dict": model.state_dict(), "widths": list(cfg["widths"]),
                "embed_dim": cfg["embed_dim"], "canon": CANON}, out_path)
    print(f"[train] saved best (val_auc={best_val_auc:.4f}) -> {out_path}")

    # ---- head-to-head evaluation on the board-disjoint test set ----
    print("\n[eval] held-out TEST set separation (higher gap = better):")
    if te is not None:
        ca, cb, lab, lvl, kind = te
        nn_rep = _report("NN", _nn_scores(model, ca, cb), lab)
        ncc_rep = _report("NCC", _ncc_scores(ca, cb), lab)
        # hard-negative subset (the cases NCC fails on)
        hard = kind == 1
        if hard.any():
            print("[eval] HARD-negative subset (cross-type, NCC-confusable):")
            _report("NN", _nn_scores(model, ca[hard], cb[hard]), lab[hard])
            _report("NCC", _ncc_scores(ca[hard], cb[hard]), lab[hard])
        dsio.write_json_manifest(
            os.path.join(dsio.MODELS_DIR, f"eval_{tag}.json"),
            {"test_n": int(len(lab)), "nn": nn_rep,
             "ncc": ncc_rep, "model": os.path.basename(out_path)})
    return model, out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", choices=list(PRESETS), default="default",
                    help="default=max accuracy; tiny=~2x faster CPU inference")
    a = ap.parse_args()
    train(a.epochs, a.batch, a.lr, a.seed, preset=a.preset)
