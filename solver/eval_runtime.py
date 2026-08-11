"""Definitive runtime eval for the tile-pair backbone (NN vs NCC).

The runtime condition the bot actually faces is a BRIGHT single-snapshot board
(~mean 170): the index.html veil isn't compositing into Page.captureScreenshot,
so the live capture is ~3.4x brighter than the dim (~mean 50) harvest/training
crops. An eval on the dim test shard says "NN is fine" while the live bot says
"NN is broken" -- only the deployment (bright) regime reveals the truth.

This eval reproduces the deployment regime OFFLINE: take full harvest boards
(any level, labelled layouts), and sweep brightness -- dim (training) vs bright
(deployment, x3.4) -- measuring the one metric that drives runtime cost:

  admitted @ thr = connectable pairs scored >= thr.
  Excess over `true` (NCC>=0.8 is guaranteed same-type) = false positives. Each
  false positive wastes an acRemovePair+verify cycle AND inflates every rollout
  step (the sampler checks connectivity per admitted pair). So `admitted` predicts
  both the actuate waste and the rollout time.

Also reports rollout-completion length (mean steps) as a check -- it turns out
~equal across backbones (~full clear), confirming the cost is per-step admitted
pairs, not completion length.

Backbones: NN(dim) = in-distribution (the bot's _ensure_sim brightness fix);
NN(bright) = out-of-distribution (the raw live regime); NCC = brightness-invariant.

Usage: AC_DATA_DIR=... python solver/eval_runtime.py   (offline, no live session)
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import dsio
from gallery import color_ncc
from gallery_nn import NNClassifier
from pairnet import CANON
import conn_fast as cf

THR = 0.7                  # operating threshold (the bot's NCC cand_thr)
BRIGHT_FACTOR = 3.4        # harvest(~50) -> deployment(~170)
LEVELS = (1, 5, 10, 13)
BOARDS_PER_LEVEL = 2


def _board_grid(sh):
    """Reconstruct (present, crops, R, C) from a harvest shard using pair_cells."""
    pairs = sh["pairs"]; cells = sh["pair_cells"]      # (P,2,2)
    P = pairs.shape[0]
    crops = pairs.reshape(P * 2, CANON, CANON, 3)
    pos = cells.reshape(P * 2, 2)
    R = int(pos[:, 0].max()) + 1; C = int(pos[:, 1].max()) + 1
    present = np.zeros((R, C), dtype=bool)
    tiles = np.zeros((R, C, CANON, CANON, 3), np.uint8)
    for k in range(P * 2):
        r, c = int(pos[k, 0]), int(pos[k, 1])
        present[r, c] = True
        tiles[r, c] = crops[k]
    flat = np.stack([tiles[r, c] for r in range(R) for c in range(C)]).astype(np.uint8)
    return present, flat, R, C


def _admitted(present, sim, pairs, thr):
    return int(sum(1 for r1, c1, r2, c2 in pairs if sim[r1, c1, r2, c2] >= thr))


def _ncc_matrix(flat, R, C):
    n = len(flat); nf = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            v = color_ncc(flat[i], flat[j]); nf[i, j] = nf[j, i] = v
    return nf.reshape(R, C, R, C)


def main():
    mp = os.path.join(dsio.MODELS_DIR, "pairnet_micro_auc1.000.pt")
    nn = NNClassifier(mp)
    if not nn.available:
        print("[eval-runtime] NN model not available; aborting."); return
    by_lv = {}
    for _, sh in dsio.iter_harvest_shards():
        by_lv.setdefault(int(sh["level"]), []).append(sh)

    print(f"[eval-runtime] NN={os.path.basename(mp)} | thr={THR} | "
          f"brightness sweep dim(~50) vs bright(x{BRIGHT_FACTOR}~170)\n")
    print(f"{'lvl':>3} {'tiles':>5} {'conn':>4} {'true':>4}   "
          f"{'NN(dim)':>8} {'NN(bright)':>11} {'NCC':>5}   "
          f"{'roll NN(dim)':>13} {'NCC':>7}")
    print("-" * 72)
    tot = {k: 0 for k in ("true", "nn_dim", "nn_bright", "ncc")}
    for L in LEVELS:
        boards = by_lv.get(L, [])[:BOARDS_PER_LEVEL]
        for sh in boards:
            present, flat, R, C = _board_grid(sh)
            if present.sum() < 4:
                continue
            pairs = cf.connectable_pairs(present)
            bright = np.clip(flat.astype(np.float32) * BRIGHT_FACTOR, 0, 255).astype(np.uint8)
            nn_dim = nn.sim_matrix(flat).reshape(R, C, R, C).astype(np.float32)
            nn_bright = nn.sim_matrix(bright).reshape(R, C, R, C).astype(np.float32)
            ncc = _ncc_matrix(flat, R, C)
            true_n = _admitted(present, ncc, pairs, 0.8)
            a_dim = _admitted(present, nn_dim, pairs, THR)
            a_brt = _admitted(present, nn_bright, pairs, THR)
            a_ncc = _admitted(present, ncc, pairs, THR)
            rs_nn = cf.rollout_mean_steps(present, nn_dim, THR, 40, 1)
            rs_ncc = cf.rollout_mean_steps(present, ncc, THR, 40, 1)
            tot["true"] += true_n; tot["nn_dim"] += a_dim
            tot["nn_bright"] += a_brt; tot["ncc"] += a_ncc
            print(f"{L:>3} {int(present.sum()):>5} {len(pairs):>4} {true_n:>4}   "
                  f"{a_dim:>8} {a_brt:>11} {a_ncc:>5}   "
                  f"{rs_nn:>13.1f} {rs_ncc:>7.1f}")
    print("-" * 72)
    print(f"{'TOT':>3} {'':>5} {'':>4} {tot['true']:>4}   "
          f"{tot['nn_dim']:>8} {tot['nn_bright']:>11} {tot['ncc']:>5}")
    print("\nadmitted = connectable pairs scored >= thr (excess over `true` = false positives).")
    print("NN(dim)=in-distribution (the bot's brightness fix); NN(bright)=raw live regime;")
    print("NCC=brightness-invariant. NN collapses on bright (deployment) but is precise on dim;")


if __name__ == "__main__":
    main()
