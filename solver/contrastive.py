"""Supervised contrastive (SupCon) tile classifier, with optional hard-negative
pair term.

Plain SupCon trains an embedding with the contrastive loss on randomly-sampled
batches of type-labelled crops; same/different is embedding cosine at runtime.

The optional **hard-negative pair term** (`--hard-neg`) adds a margin contrastive
loss on the mined hard-negative pairs (build_dataset's NCC>=0.28 cross-type
pairs) + matching same-type positives, sharing the same backbone. NOTE: on the
10x test set it reduces the *bulk* hard-negative cosine (median 0.74 -> 0.55) but
does NOT change hard-negative accuracy (~98.6% either way -- plain SupCon does
NOT collapse on hard negatives). It is an optional margin tightener, not a
required fix. (An earlier version of this doc claimed a "1.4% collapse"; that was
a metric bug -- the misclassification rate was mislabeled as accuracy.)

Eval reports: same/different by embedding cosine (crop test), hard-negative
accuracy on the pair test set, and the within-board L13 type-count. The metric
that actually distinguishes these models from NCC is jitter robustness -- see
`solver/eval_robustness.py`.
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
from train_classifier import _auc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJ_DIM = 128


# ---------------------------------------------------------------------------
# data + augmentation
# ---------------------------------------------------------------------------
def _augment(img, rng):
    import cv2
    M = np.float32([[1, 0, rng.randint(-2, 3)], [0, 1, rng.randint(-2, 3)]])
    img = cv2.warpAffine(img, M, (CANON, CANON), borderMode=cv2.BORDER_REPLICATE)
    s = 0.95 + 0.10 * rng.rand()
    M2 = np.float32([[s, 0, CANON / 2 * (1 - s)], [0, s, CANON / 2 * (1 - s)]])
    img = cv2.warpAffine(img, M2, (CANON, CANON), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(img * (0.90 + 0.20 * rng.rand()), 0, 255)


def _to_tensor(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1)


class CropDataset(Dataset):
    def __init__(self, crops, labels, augment=False, seed=0):
        self.crops = crops.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self._aug(self.crops[i]) if self.augment else self.crops[i]
        return _to_tensor(x), self.labels[i]

    def _aug(self, img):
        return _augment(img, self.rng)


class HardPairDataset(Dataset):
    """Balanced (positive, hard-negative) pairs from build_dataset's train split.

    Positives = same-type (label 1); hard negatives = the mined NCC>=0.28
    cross-type pairs (kind 1). Balanced to min(|pos|, |hard|) so the pair term
    sees both directions of the confusable boundary."""

    def __init__(self, ca, cb, label, kind, augment=False, seed=0):
        rng = np.random.RandomState(seed)
        pos_i = np.where(label == 1)[0]
        hard_i = np.where(kind == 1)[0]
        rng.shuffle(pos_i)
        rng.shuffle(hard_i)
        n = min(len(pos_i), len(hard_i))
        idx = np.concatenate([pos_i[:n], hard_i[:n]])
        self.ca = ca[idx].astype(np.float32)
        self.cb = cb[idx].astype(np.float32)
        self.label = label[idx].astype(np.float32)
        self.augment = augment
        self.rng = np.random.RandomState(seed + 1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        a, b = self.ca[i], self.cb[i]
        if self.augment:
            a, b = _augment(a, self.rng), _augment(b, self.rng)
        return _to_tensor(a), _to_tensor(b), self.label[i]


# ---------------------------------------------------------------------------
# losses + model
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, feats, labels):
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


class MarginPairLoss(nn.Module):
    """Pull same-type embeddings together (cosine->1) and push different-type
    apart below (1-margin_gap). Operates on L2-normalised embeddings."""

    def __init__(self, margin_gap=0.3):
        super().__init__()
        self.tneg = 1.0 - margin_gap

    def forward(self, ea, eb, label):
        cos = (ea * eb).sum(1)
        pos = label == 1
        out = ea.sum() * 0
        if pos.any():
            out = out + (1.0 - cos[pos]).mean()
        if (~pos).any():
            out = out + F.relu(cos[~pos] - self.tneg).mean()
        return out


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


# ---------------------------------------------------------------------------
# eval helpers
# ---------------------------------------------------------------------------
def _load_crops(split):
    f = os.path.join(dsio.DATA_DIR, "crops", f"crops_{split}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    return d["crops"], d["label"], d["level"], d["board"]


def _load_pairs(split):
    out = []
    for fn in sorted(os.listdir(dsio.DATASET_DIR)):
        if fn.startswith(f"shard_{split}_") and fn.endswith(".npz"):
            d = np.load(os.path.join(dsio.DATASET_DIR, fn))
            out.append((d["ca"], d["cb"], d["label"].astype(int), d["kind"].astype(int)))
    if not out:
        return None
    return [np.concatenate(x) for x in zip(*out)]


@torch.no_grad()
def _embed_all(model, crops, batch=1024):
    model.eval()
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(crops[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
        e = F.normalize(model.embed(x), dim=1)
        out.append(e.cpu().numpy())
    return np.concatenate(out)


def _best_threshold(scores, ys):
    order = np.argsort(-scores)
    s, y = scores[order], ys[order]
    P = y.sum(); N = len(y) - P
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    j = tp / max(P, 1) - fp / max(N, 1)
    return float(s[int(j.argmax())])


def _eval_same_diff(emb, labels, boards, max_per_board=4000):
    scores, ys = [], []
    for b in np.unique(boards):
        idx = np.where(boards == b)[0]
        if len(idx) < 4:
            continue
        e = emb[idx]; lab = labels[idx]
        sim = e @ e.T
        n = len(idx)
        iu, ju = np.triu_indices(n, 1)
        s = sim[iu, ju]; y = (lab[iu] == lab[ju]).astype(int)
        if len(s) > max_per_board:
            sel = np.random.RandomState(0).choice(len(s), max_per_board, replace=False)
            s, y = s[sel], y[sel]
        scores.append(s); ys.append(y)
    scores = np.concatenate(scores); ys = np.concatenate(ys)
    auc = _auc(scores, ys)
    thr = _best_threshold(scores, ys)
    acc = float(((scores >= thr).astype(int) == ys).mean())
    same = scores[ys == 1]; diff = scores[ys == 0]
    return auc, acc, thr, same, diff


def _hardneg_acc(model, pairs, batch=1024):
    """Hard-negative accuracy on the pair test set: fraction of the confusable
    cross-type pairs whose embedding cosine falls below the (Youden) threshold."""
    ca, cb, lab, kind = pairs
    ea = _embed_all(model, ca); eb = _embed_all(model, cb)
    sc = (ea * eb).sum(1)
    thr = _best_threshold(sc, lab)
    h = kind == 1
    hard_acc = float((sc[h] < thr).mean()) if h.any() else float("nan")
    auc = _auc(sc, lab)
    acc = float(((sc >= thr).astype(int) == lab).mean())
    return auc, acc, thr, hard_acc, int(h.sum())


def _l13_type_count(emb, labels, boards, levels, thr=0.86):
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


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def _inf(loader):
    while True:
        for b in loader:
            yield b


def train(epochs=30, batch=512, lr=1e-3, temperature=0.07, seed=0, preset="default",
          hard_neg=False, pair_weight=1.0, margin_gap=0.3, tag=None):
    torch.manual_seed(seed); np.random.seed(seed)
    dsio.ensure_dirs(dsio.MODELS_DIR)
    cfg = PRESETS[preset]
    print(f"[supcon] preset={preset} widths={cfg['widths']} embed={cfg['embed_dim']} "
          f"proj={PROJ_DIM} temp={temperature} hard_neg={hard_neg} "
          f"pair_weight={pair_weight} margin_gap={margin_gap} device={DEVICE}")

    tr = _load_crops("train"); va = _load_crops("val"); te = _load_crops("test")
    ptr = _load_pairs("train")
    if tr is None:
        raise SystemExit("no crops; run build_crops.py first")
    if hard_neg and ptr is None:
        raise SystemExit("hard_neg requested but no pair dataset; run build_dataset.py first")
    print(f"[supcon] crops: train={len(tr[0])} val={len(va[0]) if va else 0} "
          f"test={len(te[0]) if te else 0}")

    crop_ds = CropDataset(tr[0], tr[1], augment=True, seed=seed)
    crop_loader = DataLoader(crop_ds, batch_size=batch, shuffle=True, drop_last=True)
    pair_iter = None
    if hard_neg:
        pds = HardPairDataset(ptr[0], ptr[1], ptr[2], ptr[3], augment=True, seed=seed)
        print(f"[supcon] hard-pair set: {len(pds)} pairs "
              f"({int((pds.label==1).sum())} pos, {int((pds.label==0).sum())} hard-neg)")
        pair_iter = _inf(DataLoader(pds, batch_size=batch, shuffle=True, drop_last=True))

    model = ContrastiveModel(cfg["widths"], cfg["embed_dim"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    supcon = SupConLoss(temperature).to(DEVICE)
    margin = MarginPairLoss(margin_gap).to(DEVICE)

    best_val, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0 = time.time(); tot_s = tot_p = n = 0
        for x, y in crop_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            feats = F.normalize(model.features(x), dim=1)
            loss_s = supcon(feats, y)
            if pair_iter is not None:
                a, b, lab = (v.to(DEVICE) for v in next(pair_iter))
                ea = F.normalize(model.embed(a), dim=1)
                eb = F.normalize(model.embed(b), dim=1)
                loss_p = margin(ea, eb, lab)
                loss = loss_s + pair_weight * loss_p
                tot_p += float(loss_p.item()) * len(y)
            else:
                loss = loss_s
            opt.zero_grad(); loss.backward(); opt.step()
            tot_s += float(loss_s.item()) * len(y); n += len(y)
        sched.step()
        val_auc = float("nan")
        if va is not None:
            ev = _embed_all(model, va[0])
            val_auc, _, _, _, _ = _eval_same_diff(ev, va[1], va[3])
            if val_auc > best_val:
                best_val = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        lp = (tot_p / n) if pair_iter is not None else 0.0
        print(f"[supcon] ep{ep+1:02d}/{epochs} supcon={tot_s/n:.4f} pair={lp:.4f} "
              f"val_auc={val_auc:.4f} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    tag = tag or f"supcon{('_hn' if hard_neg else '')}_{preset}_auc{best_val:.3f}"
    out = os.path.join(dsio.MODELS_DIR, f"pairnet_{tag}.pt")
    torch.save({"state_dict": model.state_dict(), "widths": list(cfg["widths"]),
                "embed_dim": cfg["embed_dim"], "proj_dim": PROJ_DIM, "canon": CANON,
                "kind": "supcon", "hard_neg": hard_neg,
                "pair_weight": pair_weight, "margin_gap": margin_gap}, out)
    print(f"[supcon] saved best (val_auc={best_val:.4f}) -> {out}")

    if te is not None:
        emb = _embed_all(model, te[0])
        auc, acc, thr, same, diff = _eval_same_diff(emb, te[1], te[3])
        print(f"\n[supcon] TEST same/different by cosine:")
        print(f"  AUC={auc:.4f} acc@{thr:.3f}={acc:.4f} | "
              f"same[min={same.min():.3f} med={np.median(same):.3f}] "
              f"diff[max={diff.max():.3f} med={np.median(diff):.3f}]")
        med, counts = _l13_type_count(emb, te[1], te[3], te[2])
        print(f"[supcon] L13 within-board type-count (cos thr=0.86): "
              f"median={med} per-board={counts} (true 42)")
        pte = _load_pairs("test")
        if pte is not None:
            hauc, hacc, hthr, hhard, nh = _hardneg_acc(model, pte)
            print(f"[supcon] PAIR test: AUC={hauc:.4f} acc@{hthr:.3f}={hacc:.4f} | "
                  f"hard-neg acc={hhard:.4f} (n={nh})")
            dsio.write_json_manifest(
                os.path.join(dsio.MODELS_DIR, f"eval_{tag}.json"),
                {"test_auc": auc, "test_acc": acc, "l13_type_count_median": med,
                 "pair_hardneg_acc": hhard, "pair_auc": hauc,
                 "model": os.path.basename(out), "hard_neg": hard_neg})
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
    ap.add_argument("--hard-neg", action=argparse.BooleanOptionalAction, default=False,
                    help="add the mined hard-negative pair term (off by default; tightens "
                         "the hard-neg margin but does not change hard-neg accuracy)")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--margin-gap", type=float, default=0.3)
    a = ap.parse_args()
    train(a.epochs, a.batch, a.lr, a.temperature, a.seed, a.preset,
          a.hard_neg, a.pair_weight, a.margin_gap)
