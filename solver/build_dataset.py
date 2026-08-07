"""Materialise a labelled same/different tile-pair dataset from harvested boards.

Labels are derived from the oracle's exact same-type pairs plus the NCC
separation structure (issue #3: within-pair/same-type NCC bottoms out ~0.71;
cross-type NCC tops out ~0.75). Crucially, NCC >= ~0.78 is therefore
*guaranteed* same-type (above the cross-type ceiling), so:

    positive  = a pair of crops with NCC >= POS_THR (~0.80)   -> clean same-type
                (this includes every oracle pair, plus extra high-NCC same-type
                 pairs that the oracle never happened to remove together)
    negative  = two crops in DIFFERENT type clusters (see below)

Type clusters per board: union-find over (a) the oracle's must-link pairs and
(b) cross-pair links at LINK_THR (~0.80). Because 0.80 sits above the 0.75
cross-type ceiling, a cross-link never merges two different types -- it only
repairs the 4-of-a-kind case (one type removed as two separate oracle pairs),
which plain pair-only union-find would split.

Negatives are mined by NCC band so the NN sees the hard cases:
    easy negative  : cross-cluster, NCC < EASY_HI  (~0.60)  -- obviously different
    hard negative  : cross-cluster, NCC in [HARD_LO, LINK_THR) (~0.62-0.80)
                     -- different type yet NCC-confusable (exactly what NCC fails on)

Split is by BOARD (shard) so a crop never appears in two splits.
"""
from __future__ import annotations

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import dsio
from gallery import color_ncc

CANON = dsio.CANON
POS_THR = 0.80      # NCC >= this => same type. On harvested canon crops same-type
                    # NCC has median ~0.88 (p25 ~0.81) while cross-type has median
                    # ~0.12 (p99 ~0.34), so 0.80 cleanly admits same-type and ~no
                    # cross-type. (NB: this clean gap is on HARVEST crops; the BOT
                    # fails at runtime because live perception jitter collapses it
                    # -- hence the trained, augmentation-robust NN.)
LINK_THR = 0.80     # cross-pair cluster link threshold (= POS_THR)
EASY_HI = 0.28      # cross-cluster NCC below this => "easy" (visually distant) neg
HARD_LO = 0.28      # cross-cluster NCC in [HARD_LO, LINK_THR) => "hard" neg: the
                    # most similar DIFFERENT-type pairs (top ~2% of cross-type),
                    # exactly the confusable cases. (A general VLM qwen3.5-4b CANNOT
                    # reliably tell these apart -- measured ~70% false-SAME -- which
                    # is itself the motivation for the trained NN.)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _ncc_matrix(crops):
    """Upper-triangular (n,n) NCC matrix (diag -inf)."""
    n = len(crops)
    S = np.full((n, n), -1.0, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            v = color_ncc(crops[i], crops[j])
            S[i, j] = S[j, i] = v
    return S


def _clusters(pairs_n, ncc, uf):
    """Add cross-pair links at LINK_THR, then relabel to 0..K-1."""
    n = ncc.shape[0]
    iu, ju = np.where(ncc >= LINK_THR)
    for i, j in zip(iu.tolist(), ju.tolist()):
        if i < j:
            uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    remap = {}
    out = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return np.asarray(out)


def build(target_pairs=10000, hard_neg_frac=0.45, val_boards=0.10,
          test_boards=0.10, max_neg_per_board=300, seed=0, out_dir=None):
    rng = random.Random(seed)
    out_dir = out_dir or dsio.DATASET_DIR
    dsio.ensure_dirs(out_dir)

    shards = list(dsio.iter_harvest_shards())
    if not shards:
        raise SystemExit(f"no harvest shards in {dsio.HARVEST_DIR}; run harvest.py first")
    print(f"[dataset] {len(shards)} boards loaded")

    # ---- split by board ----
    rng.shuffle(shards)
    n = len(shards)
    n_test = max(1, int(round(n * test_boards)))
    n_val = max(1, int(round(n * val_boards)))
    test_sh = shards[:n_test]
    val_sh = shards[n_test:n_test + n_val]
    train_sh = shards[n_test + n_val:]
    split_of = {}
    for sh, split in ((train_sh, "train"), (val_sh, "val"), (test_sh, "test")):
        for path, _ in sh:
            split_of[path] = split
    print(f"[dataset] boards: train={len(train_sh)} val={len(val_sh)} test={len(test_sh)}")

    bins = {"train": [], "val": [], "test": []}
    stat = {s: {"pos": 0, "easy": 0, "hard": 0} for s in bins}

    for path, sh in shards:
        split = split_of[path]
        pairs = sh["pairs"]                      # (P,2,CANON,CANON,3)
        level = int(sh["level"])
        P = pairs.shape[0]
        crops = pairs.reshape(P * 2, CANON, CANON, 3).astype(np.uint8)
        n = len(crops)

        # oracle must-links
        uf = _UF(n)
        for k in range(P):
            uf.union(2 * k, 2 * k + 1)
        ncc = _ncc_matrix(crops)
        clust = _clusters(P, ncc, uf)

        # ---- positives: all clean same-type pairs (NCC >= POS_THR) ----
        iu, ju = np.where(ncc >= POS_THR)
        for i, j in zip(iu.tolist(), ju.tolist()):
            if i < j:
                bins[split].append((crops[i], crops[j], 1, level, 2))
                stat[split]["pos"] += 1

        # ---- negatives: cross-cluster pairs, banded by NCC ----
        diff_i, diff_j = np.where((clust[None] != clust[:, None]))
        cand = [(i, j, float(ncc[i, j])) for i, j in zip(diff_i.tolist(), diff_j.tolist())
                if i < j]
        rng.shuffle(cand)
        easy = [(i, j) for (i, j, v) in cand if v < EASY_HI]
        hard = [(i, j) for (i, j, v) in cand if HARD_LO <= v < LINK_THR]
        rng.shuffle(easy)
        rng.shuffle(hard)
        emit = 0
        for (i, j) in hard:
            if emit >= max_neg_per_board:
                break
            bins[split].append((crops[i], crops[j], 0, level, 1))
            stat[split]["hard"] += 1
            emit += 1
        for (i, j) in easy:
            if emit >= max_neg_per_board:
                break
            bins[split].append((crops[i], crops[j], 0, level, 0))
            stat[split]["easy"] += 1
            emit += 1

    # ---- balance positives vs negatives per split ----
    out_shards = {}
    for split, items in bins.items():
        pos = [x for x in items if x[2] == 1]
        neg = [x for x in items if x[2] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        m = min(len(pos), len(neg))
        n_hard_want = int(m * hard_neg_frac)
        hard = [x for x in neg if x[4] == 1]
        easy = [x for x in neg if x[4] == 0]
        neg_bal = hard[:n_hard_want] + easy[:m - n_hard_want]
        bal = pos[:m] + neg_bal
        rng.shuffle(bal)
        out_shards[split] = bal
        print(f"[dataset] {split}: {len(pos)} pos / {len(neg)} neg available -> "
              f"balanced {m}+{len(neg_bal)} (hard={min(len(hard),n_hard_want)},"
              f"easy={min(len(easy),m-n_hard_want)})")

    written = []
    idx = 0
    for split in ("train", "val", "test"):
        items = out_shards[split]
        if not items:
            continue
        ca = np.stack([x[0] for x in items]).astype(np.uint8)
        cb = np.stack([x[1] for x in items]).astype(np.uint8)
        path = os.path.join(out_dir, f"shard_{split}_{idx:03d}.npz")
        np.savez_compressed(path, ca=ca, cb=cb,
                            label=np.array([x[2] for x in items], dtype=np.int8),
                            level=np.array([x[3] for x in items], dtype=np.int8),
                            kind=np.array([x[4] for x in items], dtype=np.int8))
        written.append({"path": os.path.basename(path), "split": split, "n": len(items)})
        print(f"[dataset] wrote {path} ({len(items)} pairs)")
        idx += 1

    manifest = {
        "target_pairs": target_pairs,
        "total_pairs": sum(w["n"] for w in written),
        "pos_thr": POS_THR, "link_thr": LINK_THR,
        "easy_hi": EASY_HI, "hard_lo": HARD_LO, "hard_neg_frac": hard_neg_frac,
        "shards": written, "stat": stat, "canon": CANON,
    }
    dsio.write_json_manifest(os.path.join(out_dir, "manifest.json"), manifest)
    print(f"[dataset] DONE total={manifest['total_pairs']} pairs")
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--hard-frac", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(target_pairs=a.target, hard_neg_frac=a.hard_frac, seed=a.seed)
