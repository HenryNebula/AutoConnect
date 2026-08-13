"""Shared dataset paths, formats and helpers for the supervised tile-pair
pipeline (harvest -> build_dataset -> train_classifier -> eval).

Everything heavy/regenerable lives OUTSIDE the repo under ``AC_DATA_DIR``
(default ``/media/ext4-data/autoconnect-data``) so the git tree stays small and
the data survives worktree churn.

Data layout::

    $AC_DATA_DIR/
      harvest/                       # raw oracle harvest (per-board shards)
        board_{seq:05d}.npz          # sequence-named; level stored inside
        manifest.json
      dataset/                       # materialised labelled pair dataset
        shard_{nnnnn}.npz
        manifest.json
      models/                        # trained classifiers
        pairnet_<tag>.pt

Shard formats
-------------
Harvest shard (one per board) ``.npz``::

    pairs       (P, 2, CANON, CANON, 3) uint8   same-type crop pairs (before-crops)
    pair_cells  (P, 2, 2) int32                 [[row,col],[row,col]] per pair
    level       int32                           1..13
    ascent      int32                           which ascent produced it
    grid_xs     (cols,) float32                 cached grid geometry (x centres)
    grid_ys     (rows,) float32
    grid_ts     float32                         tile stride (px)

The union of all pair crops on a board is the full tile set, and the per-pair
must-link relation is an exact same-type partition of that board (the oracle
only ever removes genuine same-type pairs). Negatives are derived later from
the cross-cluster (different-type) relation -- no NCC threshold guessing.
"""
from __future__ import annotations

import json
import os

CANON = 40  # canonical tile-crop edge (matches gallery.CANON / collect.CANON)

DATA_DIR = os.environ.get(
    "AC_DATA_DIR", "/media/ext4-data/autoconnect-data")
HARVEST_DIR = os.path.join(DATA_DIR, "harvest")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
MODELS_DIR = os.path.join(DATA_DIR, "models")
# Reference-gallery of labelled tile crops consumed by the runtime NCC backbone
# (gallery.GalleryClassifier). Like the model checkpoints, it is regenerable
# from harvest data (see bot.ensure_gallery), so it lives under AC_DATA_DIR.
GALLERY_PATH = os.path.join(DATA_DIR, "gallery_lvl1.npz")


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def harvest_shard_path(seq: int) -> str:
    # Sequence-based naming: robust to mid-ascent reloads/restarts (a Page.reload
    # drops the game back to level 1), which would otherwise collide (level,ascent)
    # keys. The true level is stored inside the shard.
    return os.path.join(HARVEST_DIR, f"board_{seq:05d}.npz")


def write_harvest_shard(path: str, **arrays) -> None:
    import numpy as np
    ensure_dirs(os.path.dirname(path))
    np.savez_compressed(path, **arrays)


def load_harvest_shard(path: str) -> dict:
    import numpy as np
    return dict(np.load(path, allow_pickle=True))


def iter_harvest_shards():
    """Yield (path, shard_dict) for every harvest shard on disk, sorted."""
    files = sorted(
        os.path.join(HARVEST_DIR, f)
        for f in os.listdir(HARVEST_DIR)
        if f.startswith("board_") and f.endswith(".npz"))
    for f in files:
        yield f, load_harvest_shard(f)


def write_json_manifest(path: str, obj: dict) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json_manifest(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)
