"""Supervised contrastive (SupCon) variant of the tile classifier.

Instead of a siamese pair classifier with explicit same/different pairs, we train
an EMBEDDING backbone with the supervised contrastive loss (Khosla et al. 2020):
within each mini-batch, all crops of the same type are pulled together and all
others pushed apart, using every same/different pair in the batch. This scales
with the crop count (linear), not the pair count (quadratic).

Labels are board-local type clusters from the oracle (build_crops.py) -- exactly
the within-board identity the runtime bot needs. After training, same/different is
embedding cosine similarity, and the embedding also drives type clustering (the
gallery). Reported head-to-head vs NCC and vs the siamese PairNet.

Eval:
  * same/different AUC by cosine on test-board crops (same label vs different).
  * level-13 within-board type-count (target 42; NCC over-segments).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dsio
from pairnet import Backbone, CANON, PRESETS
from train_classifier import _auc  # reuse the (fixed) rank-based AUC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJ_DIM = 128


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, feats, labels):
        # feats: (B,D) L2-normalised; labels: (B,)
        B = feats.shape[0]
        sim = feats @ feats.t() / self.t
        mask = torch.eye(B, dtype=torch.bool, device=feats.device)
        sim = sim.masked_fill(mask, -1e9)
        logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (labels[:, None] == labels[None, :]) & ~mask
        has_pos = pos.any(1)
        loss = -(logprob * pos.float()).sum(1) / pos.float().sum(1).clamp(min=1)
        loss = loss[has_pos]
        return loss.mean() if loss.numel() else feats.sum() * 0


class CropDataset(Dataset):
    def __init__(self, crops, labels, augment=False, seed=0):
        import cv2
        self._cv2 = cv2
        self.crops = crops.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.labels)

    def _aug(self, img):
        cv2 = self._cv2
        M = np.float32([[1, 0, self.rng.randint(-2, 3)], [0, 1, self.rng.randint(-2, 3)]])
        img = cv2.warpAffine(img, M, (CANON, CANON), borderMode=cv2.BORDER_REPLICATE)
        s = 0.95 + 0.10 * self.rng.rand()
        M2 = np.float32([[s, 0, CANON / 2 * (1 - s)], [0, s, CANON / 2 * (1 - s)]])
        img = cv2.warpAffine(img, M2, (CANON, CANON), borderMode=cv2.BORDER_REPLICATE)
        return np.clip(img * (0.90 + 0.20 * self.rng.rand()), 0, 255)

    def __getitem__(self, i):
        x = self.crops[i]
        if self.augment:
            x = self._aug(x)
        x = torch.from_numpy(x / 255.0).permute(2, 0, 1)
        return x, self.labels[i]


class ContrastiveModel(nn.Module):
    def __init__(self, widths=(16, 32, 64), embed_dim=64, proj_dim=PROJ_DIM):
        super().__init__()
        self.backbone = Backbone(widths, embed_dim)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
                                  nn.Linear(embed_dim, proj_dim))

    def features(self, x):
        return self.proj(self.backbone(x))

    def embed(self, x):
        return self.backbone(x)


def _load(split):
    f = os.path.join(os.path.join(dsio.DATA_DIR, "crops"), f"crops_{split}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    return d["crops"], d["label"], d["level"], d["board"]


@torch.no_grad()
def _embed_all(model, crops, batch=1024):
    model.eval()
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(crops[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
        e = model.embed(x)
        e = F.normalize(e, dim=1)
        out.append(e.cpu().numpy())
    return np.concatenate(out)


def _eval_same_diff(emb, labels, boards, max_per_board=4000):
    """same/different AUC by cosine, per board then pooled."""
    scores, ys = [], []
    for b in np.unique(boards):
        idx = np.where(boards == b)[0]
        if len(idx) < 4:
            continue
        e = emb[idx]
        lab = labels[idx]
        sim = e @ e.T
        n = len(idx)
        iu, ju = np.triu_indices(n, 1)
        s = sim[iu, ju]
        y = (lab[iu] == lab[ju]).astype(int)
        # subsample per board
        if len(s) > max_per_board:
            sel = np.random.RandomState(0).choice(len(s), max_per_board, replace=False)
            s, y = s[sel], y[sel]
        scores.append(s); ys.append(y)
    scores = np.concatenate(scores); ys = np.concatenate(ys)
    auc = _auc(scores, ys)
    same = scores[ys == 1]; diff = scores[ys == 0]
    # cosine similarity lives in ~[0.7, 1.0], so 0.5 is not a useful threshold;
    # report accuracy at the optimal (Youden's-J) operating point.
    thr = _best_threshold(scores, ys)
    acc = float(((scores >= thr).astype(int) == ys).mean())
    return auc, acc, same, diff, thr


def _best_threshold(scores, ys):
    """Youden's J: the threshold maximising TPR - FPR."""
    order = np.argsort(-scores)
    s, y = scores[order], ys[order]
    P = y.sum(); N = len(y) - P
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    tpr = tp / max(P, 1); fpr = fp / max(N, 1)
    j = tpr - fpr
    return float(s[int(j.argmax())])


def _l13_type_count(emb, labels, boards, levels, thr=0.86):
    """Within-board type clustering on level-13 test boards -> median type count."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    counts = []
    for b in np.unique(boards):
        idx = np.where((boards == b) & (levels == 13))[0]
        if len(idx) < 20:
            continue
        e = emb[idx]
        sim = e @ e.T
        iu, ju = np.triu_indices(len(idx), 1)
        keep = sim[iu, ju] >= thr
        sm = coo_matrix((np.ones(keep.sum()), (iu[keep], ju[keep])), shape=(len(idx), len(idx)))
        sm = sm.maximum(sm.T)
        nc, _ = connected_components(sm, directed=False)
        counts.append(nc)
    return float(np.median(counts)) if counts else float("nan"), counts


def train(epochs=30, batch=512, lr=1e-3, temperature=0.07, seed=0, preset="default",
          tag=None):
    torch.manual_seed(seed); np.random.seed(seed)
    dsio.ensure_dirs(dsio.MODELS_DIR)
    cfg = PRESETS[preset]
    print(f"[supcon] preset={preset} widths={cfg['widths']} embed={cfg['embed_dim']} "
          f"proj={PROJ_DIM} temp={temperature} device={DEVICE}")

    tr = _load("train"); va = _load("val"); te = _load("test")
    if tr is None:
        raise SystemExit("no crops; run build_crops.py first")
    print(f"[supcon] crops: train={len(tr[0])} val={len(va[0]) if va else 0} "
          f"test={len(te[0]) if te else 0}")

    ds = CropDataset(tr[0], tr[1], augment=True, seed=seed)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    model = ContrastiveModel(cfg["widths"], cfg["embed_dim"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = SupConLoss(temperature)

    best_val, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0 = time.time(); tot = n = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            feats = F.normalize(model.features(x), dim=1)
            loss = loss_fn(feats, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        val_auc = float("nan")
        if va is not None:
            ev = _embed_all(model, va[0])
            val_auc, _, _, _, _ = _eval_same_diff(ev, va[1], va[3])
            if val_auc > best_val:
                best_val = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[supcon] ep{ep+1:02d}/{epochs} loss={tot/n:.4f} val_auc={val_auc:.4f} "
              f"({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    tag = tag or f"supcon_{preset}_auc{best_val:.3f}"
    out = os.path.join(dsio.MODELS_DIR, f"pairnet_{tag}.pt")
    torch.save({"state_dict": model.state_dict(), "widths": list(cfg["widths"]),
                "embed_dim": cfg["embed_dim"], "proj_dim": PROJ_DIM, "canon": CANON,
                "kind": "supcon"}, out)
    print(f"[supcon] saved best (val_auc={best_val:.4f}) -> {out}")

    if te is not None:
        emb = _embed_all(model, te[0])
        auc, acc, same, diff, thr = _eval_same_diff(emb, te[1], te[3])
        print(f"\n[supcon] TEST same/different by cosine:")
        print(f"  AUC={auc:.4f} acc@{thr:.3f}={acc:.4f} | "
              f"same[min={same.min():.3f} med={np.median(same):.3f}] "
              f"diff[max={diff.max():.3f} med={np.median(diff):.3f}]")
        med, counts = _l13_type_count(emb, te[1], te[3], te[2])
        print(f"[supcon] L13 within-board type-count (cos thr=0.86): "
              f"median={med} per-board={counts} (true 42)")
        dsio.write_json_manifest(
            os.path.join(dsio.MODELS_DIR, f"eval_{tag}.json"),
            {"test_auc": auc, "test_acc": acc, "l13_type_count_median": med,
             "l13_type_counts": counts, "model": os.path.basename(out)})
    return model, out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--preset", choices=list(PRESETS), default="default")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.epochs, a.batch, a.lr, a.temperature, a.seed, a.preset)
