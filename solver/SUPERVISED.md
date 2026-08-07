# Supervised tile-pair classifier pipeline (issue #3)

The runtime bot classifies tiles with translation-tolerant colour NCC
(`gallery.color_ncc`). NCC cannot cleanly separate similar Pokémon icons:
within-pair (same-type) NCC bottoms out at ~0.71 while cross-type NCC reaches
~0.75, so the distributions **overlap** and no threshold works (issue #3). This
pipeline harvests exactly-labelled data from the oracle and trains a small
siamese CNN to replace NCC.

## Pipeline (4 stages)

```
 harvest.py ──▶ build_dataset.py ──▶ train_classifier.py ──▶ gallery_nn.py
  oracle           exact labels         siamese CNN           NN vs NCC eval +
  crop pairs       (pos + hard neg)     (drop-in for NCC)     runtime hook
```

All heavy artefacts live under `$AC_DATA_DIR` (default
`/media/ext4-data/autoconnect-data`), never in the repo:

```
$AC_DATA_DIR/
  harvest/board_{seq:05d}.npz   # raw oracle harvest, per board
  dataset/shard_{split}_{n}.npz # materialised labelled pairs
  models/pairnet_*.pt           # trained classifier
```

### 1. `harvest.py` — collect ground-truth crop pairs from the oracle

Observes the patched *oracle* SWF (which only ever removes genuine same-type
pairs via its built-in pair-finder) and harvests the exact same-type pair each
`acPlayOne` removes:

* the autonomous solver ascends L1→L13 (covering every level's image set);
* on each level we **freeze** the solver the instant a full board appears, then
  step `acPlayOne` one move at a time, diffing a before/after frame to pin the
  two removed cells (their before-crops are a confirmed same-type pair);
* **fully clear** every board so the level completes cleanly and the solver
  advances to the next.

Per-board shard: `pairs (P,2,40,40,3)`, `pair_cells`, `level`. The union of
pair-crops is the whole tile set, and the per-pair relation is an *exact*
same-type partition of that board.

Robustness notes (all learned the hard way):
* `detect_grid` mis-clusters sparse boards (66 tiles → 13×7), so we only lock
  the lattice on a **full** board and apply a stable per-move global
  `drift_correct` (the level "movement" mechanics do not actually shift the
  lattice);
* a removed cell is detected as a **top-2** pixel-diff cell (a fixed 40
  threshold missed ~30% of dark-icon removals that read as low as ~37).

### 2. `build_dataset.py` — exact labels, with hard negatives

Because cross-type NCC tops out at ~0.75, **NCC ≥ 0.80 is guaranteed same-type**.
Labels:

* **positive** = crop pair with NCC ≥ 0.80 (every oracle pair + extra same-type
  pairs the oracle never removed together) — clean.
* **negative** = two crops in different type clusters. Clusters = union-find over
  the oracle must-links **plus** cross-pair links at NCC ≥ 0.80 (repairs the
  4-of-a-kind case where one type is removed as two separate pairs).
  * *easy* negative: cross-cluster NCC < 0.60.
  * *hard* negative: cross-cluster NCC in [0.62, 0.80) — different type yet
    NCC-confusable, **exactly the cases NCC fails on**.

Split is **by board** so a crop never appears in two splits (no leakage).
Output is balanced ~50/50 (≈45% of negatives hard).

### 3. `train_classifier.py` — small siamese CNN (`pairnet.py`)

Shared 3-conv-block backbone → 64-d embedding; pair head on `|e_a−e_b|` and
`e_a⊙e_b` → same-probability. Trained with BCE + cosine LR, with translation /
scale / brightness augmentation (mirroring NCC's translation tolerance). Reports
held-out AUC/accuracy and the **same-vs-different separation** of NN and NCC side
by side, plus on the hard-negative subset specifically.

### 4. `gallery_nn.py` — evaluation + runtime hook

* `NNClassifier.sim(a, b)` is the drop-in for `color_ncc` (same-type probability
  in [0,1]; use a higher decision threshold, e.g. 0.8, in `bot._pick_move`).
* `eval_clustering()` clusters a level's harvested crops by NN vs NCC and reports
  the **type count** — the issue's headline metric (NCC over-segments level 13 to
  ~46–50 types vs the true 42; a good NN recovers ~42).

## Run it

```bash
# one-time: deps (torch CPU is enough for this tiny model)
uv sync

# 1. harvest (~8 min per L1->L13 ascent; --boards-per-level controls coverage)
AC_DATA_DIR=/media/ext4-data/autoconnect-data \
  python solver/harvest.py --boards-per-level 6

# 2. build the labelled dataset (~10k balanced pairs)
AC_DATA_DIR=/media/ext4-data/autoconnect-data python solver/build_dataset.py

# 3. train + evaluate
AC_DATA_DIR=/media/ext4-data/autoconnect-data python solver/train_classifier.py

# 4. NN vs NCC type-clustering comparison on level 13
AC_DATA_DIR=/media/ext4-data/autoconnect-data python solver/gallery_nn.py --level 13

# (optional) independent VLM label-noise check via qwen3.5-4b on :8000
AC_DATA_DIR=/media/ext4-data/autoconnect-data python solver/qwen_verify.py --n 60
```

The oracle SWF must be running (patched, with the ExternalInterface bridge) — see
`solver/session.sh start`.
