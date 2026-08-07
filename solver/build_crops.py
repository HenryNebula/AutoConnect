"""Build a TYPE-LABELLED crop dataset for supervised-contrastive training.

Where ``build_dataset`` materialises explicit same/different *pairs* (quadratic in
the pair count), contrastive learning trains on the **crop set** directly: each
crop carries a type label, and the loss uses all same/different pairs *inside each
mini-batch*. That scales linearly in the number of crops (10x data = 10x crops,
not 100x pairs).

Labels are exact and oracle-grounded, and -- importantly -- matched to what the
runtime bot actually needs: WITHIN-BOARD type identity. On each board the oracle
removed only genuine same-type pairs, so union-find over those must-links (garbage-
filtered at PAIR_NCC_MIN) plus high-NCC cross-links partitions the board's crops
into true type clusters. Each crop is labelled by (board, cluster); two crops
share a label iff they are the same type on the same board.

Output ``$AC_DATA_DIR/crops/``:
    crops_{split}.npz : crops (N,CANON,CANON,3) uint8,
                        label (N,) int64, level (N,) int8, board (N,) int32
    manifest.json
Split is by board (no crop leaks across splits).
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import dsio
from build_dataset import _UF, _ncc_matrix, _clusters, PAIR_NCC_MIN, LINK_THR

CANON = dsio.CANON


def build(val_boards=0.10, test_boards=0.10, seed=0, out_dir=None):
    rng = random.Random(seed)
    out_dir = out_dir or os.path.join(dsio.DATA_DIR, "crops")
    dsio.ensure_dirs(out_dir)

    shards = list(dsio.iter_harvest_shards())
    if not shards:
        raise SystemExit(f"no harvest shards in {dsio.HARVEST_DIR}; run harvest.py first")
    print(f"[crops] {len(shards)} boards loaded")

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
    print(f"[crops] boards: train={len(train_sh)} val={len(val_sh)} test={len(test_sh)}")

    bins = {"train": ([], [], [], []), "val": ([], [], [], []), "test": ([], [], [], [])}
    n_classes = {"train": 0, "val": 0, "test": 0}
    board_seq = {p: i for i, (p, _) in enumerate(shards)}

    for path, sh in shards:
        split = split_of[path]
        pairs = sh["pairs"]
        P = pairs.shape[0]
        level = int(sh["level"])
        bid = board_seq[path]
        crops = pairs.reshape(P * 2, CANON, CANON, 3).astype(np.uint8)
        nn = len(crops)
        ncc = _ncc_matrix(crops)
        uf = _UF(nn)
        for k in range(P):
            if float(ncc[2 * k, 2 * k + 1]) >= PAIR_NCC_MIN:
                uf.union(2 * k, 2 * k + 1)
        clust = _clusters(P, ncc, uf)                       # board-local cluster ids
        # global label = board * BIG + cluster (same board+type => same label)
        labels = bid * 100_000 + clust.astype(np.int64)
        n_classes[split] = max(n_classes[split], int(labels.max()) + 1)
        c, l, lv, b = bins[split]
        c.append(crops); l.append(labels)
        lv.append(np.full(nn, level, dtype=np.int8))
        b.append(np.full(nn, bid, dtype=np.int32))

    written = []
    for split in ("train", "val", "test"):
        c, l, lv, b = bins[split]
        if not c:
            continue
        crops = np.concatenate(c)
        labels = np.concatenate(l)
        levels = np.concatenate(lv)
        boards = np.concatenate(b)
        n_unique = len(np.unique(labels))
        path = os.path.join(out_dir, f"crops_{split}.npz")
        np.savez_compressed(path, crops=crops, label=labels, level=levels, board=boards)
        written.append({"split": split, "n_crops": len(crops), "n_classes": n_unique,
                        "path": os.path.basename(path)})
        print(f"[crops] {split}: {len(crops)} crops, {n_unique} classes -> {path}")

    dsio.write_json_manifest(os.path.join(out_dir, "manifest.json"),
                             {"splits": written, "canon": CANON,
                              "link_thr": LINK_THR, "pair_ncc_min": PAIR_NCC_MIN})
    print(f"[crops] DONE")
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(seed=a.seed)
