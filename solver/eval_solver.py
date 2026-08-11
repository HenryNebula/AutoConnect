"""Per-level CV-solver (classifier) quality for issue #3 -- same/different AUC
and hard-negative rejection, split by level, NN vs NCC, clean and under jitter.

Reads the labelled test shard directly (build_dataset already stored exact
oracle-grounded labels + level + kind per pair), so no NCC matrix or partition
is needed. This is the eval_robustness metric decomposed by level: it shows
WHICH levels' icon sets are hard, and whether the NN's jitter advantage
(eval_robustness: NCC 1.0->0.76 at j3, NN holds) is uniform across levels.

Columns:
  clean / jJ : same/different AUC at jitter level J (0 = clean, 3 = harshest)
  hard-acc   : rejection accuracy on the mined confusable cross-type pairs
               (kind==1; different type yet NCC-confusable -- exactly issue #3)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import dsio
from train_classifier import _auc
from eval_robustness import _load_model, _score, _jitter_pairs

HARD_KIND = 1   # build_dataset: kind 1 = hard (NCC-confusable cross-type) negative


def run(jitter=3, seed=0, model_path=None, shard="test"):
    if model_path is None:
        cands = sorted(f for f in os.listdir(dsio.MODELS_DIR)
                       if f.startswith("pairnet_micro_auc") and f.endswith(".pt"))
        model_path = os.path.join(dsio.MODELS_DIR, cands[0])
    model, kind = _load_model(model_path)
    mname = os.path.basename(model_path)

    # load the labelled test (or val) shard -- has level + kind per pair
    shard_path = os.path.join(dsio.DATASET_DIR, {"test": "shard_test_002.npz",
                                                 "val": "shard_val_001.npz"}[shard])
    d = np.load(shard_path, allow_pickle=True)
    ca, cb = d["ca"], d["cb"]
    y = d["label"].astype(int)
    lv = d["level"].astype(int)
    hk = d["kind"].astype(int) == HARD_KIND

    print(f"[eval-solver] model={mname} ({kind}); shard={shard} (n={len(y)}); jitter=j{jitter}\n")
    print(f"kind counts: pos={int((d['kind']==0).sum())} hard={int(hk.sum())} "
          f"easy={int((d['kind']==2).sum())}  (label=1 frac={y.mean():.3f})\n")

    # one-time jitter (independent per crop), same for NN and NCC
    ja, jb = _jitter_pairs(ca, cb, jitter, seed)

    # full-array scores (NN vectorised; NCC per-pair ~0.6s for 12k pairs), masked per level
    nn_clean = _score(model, kind, ca, cb)
    nn_jit = _score(model, kind, ja, jb)
    ncc_clean = _score(None, "ncc", ca, cb)
    ncc_jit = _score(None, "ncc", ja, jb)

    header = (f"{'lvl':>3} {'n':>5} {'hard':>5}   {'NN clean':>8} {'NCC clean':>9}   "
              f"{'NN j%d' % jitter:>6} {'NCC j%d' % jitter:>7}   {'NN hard':>7} {'NCC hard':>8}")
    print(header)
    print("-" * len(header))

    def acc(s, mask):
        m = mask & (y == 0)          # negatives only; correct = score < 0.5
        return float((s[m] < 0.5).mean()) if m.sum() else float("nan")

    agg = []
    for L in range(1, 14):
        m = lv == L
        if m.sum() == 0:
            continue
        nn_c = _auc(nn_clean[m], y[m]); ncc_c = _auc(ncc_clean[m], y[m])
        nn_j = _auc(nn_jit[m], y[m]); ncc_j = _auc(ncc_jit[m], y[m])
        hmask = (m & hk)
        print(f"{L:>3} {int(m.sum()):>5} {int(hmask.sum()):>5}   "
              f"{nn_c:>8.4f} {ncc_c:>9.4f}   {nn_j:>6.4f} {ncc_j:>7.4f}   "
              f"{acc(nn_clean, m):>7.4f} {acc(ncc_clean, m):>8.4f}")
        agg.append((nn_c, ncc_c, nn_j, ncc_j, int(m.sum())))

    print("-" * len(header))
    A = np.array([a[:4] for a in agg]); w = np.array([a[4] for a in agg], float)
    w /= w.sum()
    g = (A * w[:, None]).sum(0)
    print(f"{'ALL':>3} {int(w.sum() * len(y) / len(y)):>5} {'':>5}   "
          f"{g[0]:>8.4f} {g[1]:>9.4f}   {g[2]:>6.4f} {g[3]:>7.4f}")
    print(f"\n(weighted by pair count; clean/j# = same/diff AUC; hard = confusable-pair rejection @0.5)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jitter", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--shard", default="test", choices=["test", "val"])
    a = ap.parse_args()
    run(a.jitter, a.seed, a.model, a.shard)
