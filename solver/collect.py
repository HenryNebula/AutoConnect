"""OFFLINE ground-truth data collection (never used by the runtime bot).

Observes the working *oracle* SWF (which only ever removes legal same-type
pairs) and harvests confirmed same-type tile-crop pairs by diffing the board
before/after each oracle move. On level 1 there is no tile-drift, so the two
cells that flip present->empty are exactly the removed pair.

Output: a .npz of crop pairs (and singleton crops) used to build a reference
gallery. The runtime solver perceives the board purely from pixels; this file is
a trained model, not game state.
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cdp
import vision
import cv2

CANON = 40   # canonical tile-crop edge (gallery crops are resized to this)


def canon(crop: np.ndarray) -> np.ndarray:
    return cv2.resize(crop, (CANON, CANON), interpolation=cv2.INTER_AREA)


def _mk(e):
    return ("(function(){var e=document.getElementsByTagName('ruffle-embed')[0];"
            "if(!e||typeof e.%s!=='function')return undefined;try{return e.%s(%s);}"
            "catch(x){return 'ERR:'+x;}})()")


class Oracle:
    def __init__(self):
        self._ev = lambda e: cdp._send("Runtime.evaluate", {
            "expression": e, "returnByValue": True}).get("result", {}).get("value")

    def __call__(self, method, *args):
        a = ",".join(json.dumps(x) for x in args)
        return self._ev(_mk(method) % (method, method, a))

    def wait_ready(self, timeout=40.0):
        t = time.time()
        while time.time() - t < timeout:
            s = self("acStatus")
            if isinstance(s, str) and s.startswith("{"):
                return True
            time.sleep(0.4)
        return False

    def status(self):
        s = self("acStatus")
        return json.loads(s) if isinstance(s, str) and s.startswith("{") else None

    def reload(self):
        cdp._send("Page.enable", {})
        cdp._send("Page.reload", {"ignoreCache": True})
        time.sleep(3.5)
        self.wait_ready()
        self("acSetEnabled", False)

    def snap(self):
        path = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "_col.png")
        cdp.capture(path)
        return vision.load_img(path)


def detect_stable_grid(oc, retries=10, expect=(12, 8)):
    """Snap until detect_grid returns a stable expected grid; cache & return it."""
    last = None
    for _ in range(retries):
        img = oc.snap()
        g = vision.detect_grid(img)
        if g and (g["cols"], g["rows"]) == expect:
            return g
        last = g
        time.sleep(0.4)
    return last


def collect_level1(boards: int = 6, out_path: str | None = None):
    oc = Oracle()
    assert oc.wait_ready(), "oracle not ready"
    oc("acSetEnabled", False)
    pairs = []           # list of (crop1, crop2) same-type
    singles = []         # all present crops (for gallery density)
    for b in range(boards):
        # sustained acPlayOne hammering degrades Ruffle's renderer; reload
        # every couple of boards to reset it.
        if b and b % 2 == 0:
            oc.reload()
        oc("acReset")
        time.sleep(2.0)
        st = oc.status()
        if not st or st.get("level") != 1:
            continue
        grid = detect_stable_grid(oc)
        if not grid or (grid["cols"], grid["rows"]) != (12, 8):
            print(f"[collect] board {b}: grid unstable, skip")
            continue
        prev_present = None
        prev_tiles = None
        no_change = 0
        for _move in range(70):
            img = oc.snap()
            pres = vision.present_mask(img, grid)
            tiles = vision.extract_tiles(img, grid)
            if prev_present is not None:
                gone = prev_present & ~pres
                ys, xs = np.where(gone)
                if len(ys) == 2:
                    pairs.append((canon(prev_tiles[ys[0], xs[0]]),
                                  canon(prev_tiles[ys[1], xs[1]])))
                    no_change = 0
                elif len(ys) == 0:
                    no_change += 1
                else:
                    no_change = 0
            ys, xs = np.where(pres)
            for r, c in zip(ys, xs):
                singles.append(canon(tiles[r, c]))
            prev_present, prev_tiles = pres, tiles
            oc("acPlayOne")
            time.sleep(0.45)  # let the pop animation settle
            after = oc.status()
            if not after or after.get("level") != 1 or after.get("tilesLeft", 0) < 2:
                break
            if no_change > 3:
                break
        print(f"[collect] board {b}: {len(pairs)} pairs total so far")
    pairs_arr = np.stack(pairs) if pairs else np.zeros((0,), np.uint8)
    singles_arr = np.stack(singles) if singles else np.zeros((0,), np.uint8)
    if out_path:
        np.savez_compressed(out_path, pairs=pairs_arr, singles=singles_arr)
        print(f"[collect] saved {len(pairs)} pairs, {len(singles)} singles -> {out_path}")
    return pairs_arr, singles_arr


def collect_level13(boards: int, out_path: str):
    """Collect same-type pairs from level 13 (which uses all 42 icon types).

    Drift-aware: the game shifts tiles ~150ms after each move, so we capture
    the 'after' frame immediately (before the drift) and identify the removed
    pair by diff vs the 'before' frame. Their before-crops are a same-type pair.
    """
    import vision
    oc = Oracle()
    assert oc.wait_ready(), "oracle not ready"
    pairs = []
    for b in range(boards):
        # reach level 13 by letting the oracle auto-solve 1..12
        oc("acSetEnabled", False)
        oc("acReset"); time.sleep(1.5)
        oc("acSetEnabled", True)
        while True:
            st = oc.status()
            if not st:
                time.sleep(0.3); continue
            if st.get("level", 1) >= 13 and st.get("tilesLeft", 0) >= 90:
                break
            if st.get("clears", 0) > 0:
                break
            time.sleep(0.3)
        oc("acSetEnabled", False)
        time.sleep(1.5)
        st = oc.status()
        if not st or st.get("level") != 13:
            print(f"[collect13] could not reach level 13 (at {st.get('level') if st else '?'})")
            continue
        grid = detect_stable_grid(oc)
        if not grid or (grid["cols"], grid["rows"]) != (12, 8):
            print(f"[collect13] grid unstable; skip")
            continue
        no_change = 0
        for _move in range(70):
            img_before = oc.snap()
            tiles_before = vision.extract_tiles(img_before, grid)
            oc("acPlayOne")
            time.sleep(0.05)
            img_after = oc.snap()      # fast: well inside the ~150ms drift window
            # removed cells = changed a lot between before and after
            ts = grid["ts"]
            gone = []
            for r in range(grid["rows"]):
                for c in range(grid["cols"]):
                    cy, cx = grid["ys"][r], grid["xs"][c]
                    h = ts * 0.45
                    a = img_before[int(cy - h):int(cy + h), int(cx - h):int(cx + h)].astype(np.float32)
                    bb = img_after[int(cy - h):int(cy + h), int(cx - h):int(cx + h)].astype(np.float32)
                    if a.size and np.abs(a - bb).mean() > 40:
                        gone.append((r, c))
            if len(gone) == 2:
                pairs.append((canon(tiles_before[gone[0][0], gone[0][1]]),
                              canon(tiles_before[gone[1][0], gone[1][1]])))
                no_change = 0
            elif len(gone) == 0:
                no_change += 1
            else:
                no_change = 0
            after = oc.status()
            if not after or after.get("level") != 13 or after.get("tilesLeft", 0) < 2:
                break
            if no_change > 3:
                break
        print(f"[collect13] board {b}: {len(pairs)} pairs total")
    pairs_arr = np.stack(pairs) if pairs else np.zeros((0,), np.uint8)
    singles_arr = np.zeros((0,), np.uint8)
    if out_path:
        np.savez_compressed(out_path, pairs=pairs_arr, singles=singles_arr)
        print(f"[collect13] saved {len(pairs)} pairs -> {out_path}")
    return pairs_arr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=6)
    ap.add_argument("--out", type=str, default=os.path.join(
        os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "lvl1_pairs.npz"))
    ap.add_argument("--level13", type=int, default=0,
                    help="if >0, collect this many level-13 boards (all 42 types)")
    a = ap.parse_args()
    if a.level13:
        collect_level13(a.level13, a.out)
    else:
        collect_level1(a.boards, a.out)
