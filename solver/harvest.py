"""Oracle-driven supervised-data harvester for issue #3.

Observes the working *oracle* SWF (which only ever removes genuine same-type
pairs via its built-in pair-finder ``vum.cunufi``) and harvests, for every one
of the 13 levels' image sets, the exact same-type crop pairs the oracle removes.

Method (drift-aware, generalised from ``collect.collect_level13`` to ALL levels
-- every level except L1 shifts tiles after each move):
  freeze the oracle (``acSetEnabled False``) so the autonomous tick stays quiet;
  per move: snap a static *before* frame, call ``acPlayOne`` (one legal move),
  snap an *after* frame inside the drift window, and identify the two cells that
  changed a lot. Their before-frame crops are a confirmed same-type pair.

Each board is harvested to empty (which also clears it); ``acStep`` then advances
to the next level. A full ascent L1..L13 covers every per-level image set;
repeated ascents grow the per-type instance count.

Per-board shards are written immediately (a crash loses at most the current
board). The within-board union of pair-crops is the full tile set, and the
per-pair must-link relation is an EXACT same-type partition of that board --
negatives are derived later, with no NCC-threshold guessing.

This is OFFLINE ground-truth collection; it is never used by the runtime bot.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2

import cdp
import vision
import dsio

CANON = dsio.CANON

# per-move timing
REMOVE_SETTLE = 0.10   # s between acPlayOne and the after-snap: long enough for
                       # the removal to render, short enough to beat tile drift
DRIFT_SETTLE = 0.22    # s after the after-snap: let any per-move tile drift
                       # (L2 "下移" … L13 "中心靠拢") finish before the next frame
WIN_FRAC = 0.45        # central-window half-size as a fraction of tile stride
EXPECT = (12, 8)       # visible grid (cols, rows); 14×10 includes routing border


def canon(crop: np.ndarray) -> np.ndarray:
    return cv2.resize(crop, (CANON, CANON), interpolation=cv2.INTER_AREA)


def _mk(method):
    return ("(function(){var e=document.getElementsByTagName('ruffle-embed')[0];"
            "if(!e||typeof e.%s!=='function')return undefined;try{return e.%s(%s);}"
            "catch(x){return 'ERR:'+x;}})()")


class Oracle:
    """Thin ExternalInterface wrapper over the patched SWF."""

    def __init__(self):
        self._ev = lambda e: cdp._send("Runtime.evaluate", {
            "expression": e, "returnByValue": True}).get("result", {}).get("value")

    def _safe_ev(self, expr):
        # CDP reuses one persistent websocket; a page reload/navigation can stale
        # it. Reset and reconnect on any transport error so a blip doesn't kill
        # the whole harvest.
        try:
            return self._ev(expr)
        except Exception:
            cdp._ws = None
            try:
                return self._ev(expr)
            except Exception:
                return None

    def __call__(self, method, *args):
        a = ",".join(json.dumps(x) for x in args)
        return self._safe_ev(_mk(method) % (method, method, a))

    def status(self):
        s = self("acStatus")
        if isinstance(s, str) and s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None
        return None

    def wait_ready(self, timeout=60.0):
        t = time.time()
        while time.time() - t < timeout:
            cdp._ws = None            # fresh websocket each probe: dodges stale
            st = self.status()        # buffers left by page transitions
            if st is not None and st.get("scene") in ("play", "result"):
                self._ws_timeout(20.0)
                return True
            time.sleep(0.5)
        return False

    def _ws_timeout(self, t=20.0):
        """Give the persistent CDP socket a generous recv timeout -- a full-page
        screenshot (the biggest payload we send) can take >5s under renderer
        load, which would otherwise raise WebSocketTimeoutException."""
        try:
            cdp.connect()
            cdp._ws.settimeout(t)
        except Exception:
            cdp._ws = None

    def freeze(self):
        self("acSetEnabled", False)

    def snap(self):
        """Capture the viewport, resilient to transient websocket/page hiccups.
        Raises RuntimeError only after several retries (caller decides to give
        up the current board and reload)."""
        path = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "_harv.png")
        last = None
        for attempt in range(4):
            try:
                self._ws_timeout(20.0)
                cdp.capture(path)
                return vision.load_img(path)
            except Exception as ex:
                last = ex
                cdp._ws = None          # force a fresh connection next attempt
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"snap failed after retries: {repr(last)[:120]}")

    def reload(self):
        """Page.reload to a fresh level-1 board, then freeze the instant the
        board is laid out -- before the autonomous solver's 200ms tick can clear
        many tiles. A *full* board is required: detect_grid mis-clusters sparse
        boards (e.g. 66 tiles -> 13x7 instead of the true 12x8)."""
        for fn in (lambda: cdp._send("Page.enable", {}),
                   lambda: cdp._send("Page.reload", {"ignoreCache": True})):
            try:
                fn()
            except Exception:
                cdp._ws = None
                try:
                    fn()
                except Exception:
                    pass
        t0 = time.time()
        frozen = False
        while time.time() - t0 < 30:
            st = self.status()
            if st is not None and not frozen:        # EI up -> freeze at once
                self.freeze()
                frozen = True
            if st and st.get("level") == 1 and st.get("scene") == "play" \
                    and st.get("tilesLeft", 0) >= 90:
                break
            time.sleep(0.08)
        self.freeze()


def detect_stable_grid(oc, retries=14, expect=EXPECT):
    """Snap until detect_grid returns a stable expected grid; return it or None."""
    last = None
    for _ in range(retries):
        g = vision.detect_grid(oc.snap())
        if g and (g["cols"], g["rows"]) == expect:
            g2 = vision.detect_grid(oc.snap())
            if g2 and (g2["cols"], g2["rows"]) == expect \
                    and abs(g["ts"] - g2["ts"]) < 1.0:
                return g
        last = g
        time.sleep(0.35)
    return last if (last and (last["cols"], last["rows"]) == expect) else None


def _fit_lattice(coords, n, ts0):
    """Fit ``n`` equally-spaced lattice lines to 1-D centroid coords.

    Solves (phase x0, spacing ts) by grid search, minimising the total distance
    from each coord to its nearest line, with a heavy penalty for coords that
    fall outside [0, n) line indices. Robust to empty columns/rows (a line with
    no blobs is still placed correctly by the equal-spacing prior), which is
    exactly what ``vision.detect_grid`` gets wrong on partial boards."""
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) < max(3, n // 2):
        return None, None
    lo, hi = float(coords.min()), float(coords.max())
    ts_guess = (hi - lo) / (n - 1) if n > 1 else ts0
    best_score, best_x0, best_ts = float("inf"), None, None
    for ts in np.arange(max(20.0, ts_guess - 4.0), ts_guess + 4.0, 0.5):
        for x0 in np.arange(lo - ts, lo + ts, 1.0):
            k = np.round((coords - x0) / ts).astype(int)
            resid = np.abs(coords - (x0 + k * ts))
            bad = (k < 0) | (k >= n)
            score = float(resid.sum()) + float(bad.sum()) * ts
            if score < best_score:
                best_score, best_x0, best_ts = score, float(x0), float(ts)
    if best_x0 is None:
        return None, None
    # snap phase so line 0 is the leftmost (k=0) -> canonical x0
    lines = [best_x0 + c * best_ts for c in range(n)]
    return lines, best_ts


def fit_grid(img, expect=EXPECT, ts0=49.0, sat_thr=0.25, area=(120, 6000), box=70):
    """Detect the 12x8 lattice on a (possibly partial / drifted) board.

    Forces cols/rows to the expected counts via :func:`_fit_lattice`, so unlike
    ``vision.detect_grid`` it stays correct as tiles drift and the board thins.
    Returns a grid dict or None."""
    cols, rows = expect
    mask = (vision.saturation(img) > sat_thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    xs, ys = [], []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area[0] < a < area[1] and w < box and h < box:
            xs.append(float(cent[i, 0]))
            ys.append(float(cent[i, 1]))
    xl, ts_x = _fit_lattice(xs, cols, ts0)
    yl, ts_y = _fit_lattice(ys, rows, ts0)
    if xl is None or yl is None:
        return None
    return dict(cols=cols, rows=rows, xs=xl, ys=yl,
                ts=float((ts_x + ts_y) / 2.0))


def _blob_centroids(img, sat_thr=0.25, area=(120, 6000), box=70):
    mask = (vision.saturation(img) > sat_thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area[0] < a < area[1] and w < box and h < box:
            out.append((float(cent[i, 0]), float(cent[i, 1])))
    return out


def drift_correct(img, grid0):
    """Return ``grid0`` shifted by the median (dx, dy) from each tile blob to its
    nearest grid line. Stable when there is no drift (dx=dy=0, so the locked
    full-board grid is preserved -- L1 stays accurate) and tracks a uniform
    lattice translation on drift levels. ts is unchanged."""
    blobs = _blob_centroids(img)
    if not blobs:
        return grid0
    xs0 = np.asarray(grid0["xs"], dtype=np.float64)
    ys0 = np.asarray(grid0["ys"], dtype=np.float64)
    dxs, dys = [], []
    for bx, by in blobs:
        dxs.append(bx - xs0[np.argmin(np.abs(xs0 - bx))])
        dys.append(by - ys0[np.argmin(np.abs(ys0 - by))])
    dx, dy = float(np.median(dxs)), float(np.median(dys))
    return dict(grid0, xs=xs0 + dx, ys=ys0 + dy)


def changed_cells(img_before, img_after, grid, floor=30.0, frac=WIN_FRAC):
    """The two cells the oracle just removed.

    A removed cell goes icon -> empty (a ~full pixel change), which is always the
    LARGEST change on the board -- bigger than gravity shifts (icon -> different
    icon) or animation. So we take the top-2 cells by mean |Δpixel|, accepting
    them when the 2nd-highest clears ``floor``. A fixed threshold of 40 misses
    ~30% of removals (dark icons near the board background read as low as ~37),
    so we rank instead of threshold."""
    g = grid
    ts = g["ts"]
    h = ts * frac
    dif = np.abs(img_before.astype(np.int16) - img_after.astype(np.int16)).mean(2)
    scored = []
    for r in range(g["rows"]):
        for c in range(g["cols"]):
            cy, cx = g["ys"][r], g["xs"][c]
            win = dif[int(cy - h):int(cy + h), int(cx - h):int(cx + h)]
            if win.size:
                scored.append((float(win.mean()), r, c))
    if len(scored) < 2:
        return []
    scored.sort(reverse=True)
    if scored[1][0] >= floor:
        return [(scored[0][1], scored[0][2]), (scored[1][1], scored[1][2])]
    return []


def harvest_board(oc, level, grid0, max_moves=200):
    """Harvest same-type pairs from the current (frozen) board.

    Drift-aware: the 12x8 lattice is RE-FIT on every before-frame with
    :func:`fit_grid` (which forces 12x8 and tolerates partial/drifted boards), so
    crop extraction tracks tiles as they drift (every level except L1). The
    after-snap is taken before the ~150ms drift window so only the two removed
    cells read as changed. Drives the board to empty (clearing it). Returns
    (pairs, n_moves, last_grid).
    """
    if grid0 is None:
        grid0 = detect_stable_grid(oc)
    if grid0 is None:
        grid0 = fit_grid(oc.snap())
    if grid0 is None:
        return [], 0, None
    pairs = []
    moves = 0
    grid = grid0
    last_tl = None
    stall = 0              # consecutive moves with no drop in tilesLeft
    snap_fails = 0         # consecutive per-move transport failures
    for _ in range(max_moves):
        try:
            img_before = oc.snap()
            grid = drift_correct(img_before, grid0)   # track drift, stably
            tiles_before = vision.extract_tiles(img_before, grid)   # (R,C,ts,ts,3)
            oc("acPlayOne")         # ALWAYS advances the board (records when detected)
            time.sleep(REMOVE_SETTLE)
            img_after = oc.snap()
            ch = changed_cells(img_before, img_after, grid)
        except Exception as ex:
            # transient websocket/page hiccup: back off, reset the socket, retry
            snap_fails += 1
            cdp._ws = None
            print(f"[harvest] L{level} move transport error ({snap_fails}): "
                  f"{repr(ex)[:80]}")
            if snap_fails >= 4:
                break            # give up this board; run() will reload
            time.sleep(1.0)
            continue
        snap_fails = 0
        if len(ch) == 2:
            (r1, c1), (r2, c2) = ch
            try:
                ca = canon(tiles_before[r1, c1])
                cb = canon(tiles_before[r2, c2])
                pairs.append((ca, cb, (r1, c1), (r2, c2)))
                moves += 1
            except Exception:
                pass
        st = oc.status()
        if not st:
            break
        tl = st.get("tilesLeft", 99)
        # IMPORTANT: keep going until the board is truly empty so the level
        # completes cleanly and the solver advances. Stopping early leaves a
        # half-cleared board whose level never completes -> the ascent stalls.
        if st.get("level") != level or tl < 2:
            break
        if tl == last_tl:
            stall += 1
        else:
            stall, last_tl = 0, tl
        if stall >= 6:        # tilesLeft stuck -> board can't clear; give up
            break
        time.sleep(DRIFT_SETTLE)   # let any per-move drift settle before next snap
    return pairs, moves, grid


def wait_next_board(oc, from_level, prev_clears, timeout=95.0):
    """Let the autonomous solver advance past ``from_level`` and freeze on the
    next level's (near-)full board. The board was fully cleared by harvest_board,
    so the level completes cleanly and the solver advances; we just detect 'new
    level, full board' and freeze. Returns the new level (or current on timeout)."""
    oc("acSetEnabled", True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = oc.status()
        if not st:
            time.sleep(0.1)
            continue
        lv, tl, scene = st.get("level"), st.get("tilesLeft"), st.get("scene")
        advanced = (lv is not None and lv > from_level) or \
            (from_level == 13 and lv == 1 and st.get("clears", 0) > prev_clears)
        if advanced and scene == "play" and tl is not None and tl >= 86:
            oc.freeze()
            return lv
        time.sleep(0.1)
    oc.freeze()
    st = oc.status() or {}
    return st.get("level")


def run(boards_per_level=6, max_runs=12, max_level=13):
    """Harvest ``boards_per_level`` full boards per level across all 13 image sets.

    Strategy: harvest the current full board to empty (clearing it), then let the
    autonomous solver advance to the next level and freeze on its full board.
    The solver does the ascending (reliably); we only catch full boards, so
    detect_grid always sees a dense 12x8 lattice. Repeat runs until every level
    has enough boards."""
    oc = Oracle()
    oc.freeze()
    assert oc.wait_ready(), "oracle not ready (is the patched SWF loaded on :8765?)"
    dsio.ensure_dirs(dsio.HARVEST_DIR)

    counts = {L: 0 for L in range(1, max_level + 1)}
    seq = 0
    run_i = 0
    t_start = time.time()
    while min(counts.values()) < boards_per_level and run_i < max_runs:
        run_i += 1
        oc.reload()                       # fresh renderer + full level-1 board
        level = 1
        while level and 1 <= level <= max_level and \
                min(counts.values()) < boards_per_level:
            want = counts[level] < boards_per_level
            try:
                pairs, moves, grid = harvest_board(oc, level, None)
            except Exception as ex:
                # unrecoverable for this board (renderer/transport): reload to a
                # fresh level-1 and restart this run. Already-saved boards keep
                # their counts, so nothing is lost.
                print(f"[harvest] L{level} run={run_i} HARVEST CRASH "
                      f"{repr(ex)[:100]}; reloading")
                oc.reload()
                level = 1
                continue
            if want and len(pairs) >= 10:
                _save_shard(seq, level, run_i, pairs, grid)
                counts[level] += 1
                seq += 1
            el = time.time() - t_start
            tot = sum(counts.values())
            tag = "save" if (want and len(pairs) >= 10) else \
                ("skip(have)" if not want else f"sparse({len(pairs)})")
            print(f"[harvest] seq={seq} L{level} run={run_i} "
                  f"pairs={len(pairs):2d} moves={moves:2d} {tag} | "
                  f"boards={tot} per-L={list(counts.values())} {el:.0f}s")
            if level == max_level:
                break                      # next is a new run (wrap)
            prev_clears = (oc.status() or {}).get("clears", 0)
            try:
                level = wait_next_board(oc, level, prev_clears)
            except Exception as ex:
                print(f"[harvest] advance crash L{level}: {repr(ex)[:80]}; reloading")
                oc.reload()
                level = 1
            if not level:
                break

    manifest = {
        "boards_total": sum(counts.values()),
        "counts_per_level": counts,
        "runs": run_i,
        "canon": CANON,
        "expect_grid": list(EXPECT),
        "seconds": time.time() - t_start,
    }
    dsio.write_json_manifest(os.path.join(dsio.HARVEST_DIR, "manifest.json"), manifest)
    print(f"[harvest] DONE boards={sum(counts.values())} "
          f"counts={counts} {manifest['seconds']:.0f}s")
    return manifest


def _save_shard(seq, level, ascent, pairs, grid):
    if not pairs:
        return
    # pair k -> arr[k,0]=crop_a, arr[k,1]=crop_b (interleaved, NOT [all-a ++ all-b]
    # which a .reshape(P,2,...) would mis-pair as a0~a1, b0~b1, ...).
    arr = np.stack([np.stack([p[0], p[1]]) for p in pairs])  # (P,2,CANON,CANON,3)
    cells = np.array([[[*p[2]], [*p[3]]] for p in pairs], dtype=np.int32)  # (P,2,2)
    dsio.write_harvest_shard(
        dsio.harvest_shard_path(seq),
        pairs=arr.astype(np.uint8),
        pair_cells=cells,
        level=np.int32(level),
        ascent=np.int32(ascent),
        grid_xs=np.asarray(grid["xs"], dtype=np.float32),
        grid_ys=np.asarray(grid["ys"], dtype=np.float32),
        grid_ts=np.float32(grid["ts"]),
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boards-per-level", type=int, default=6)
    ap.add_argument("--max-runs", type=int, default=12)
    ap.add_argument("--max-level", type=int, default=13)
    a = ap.parse_args()
    run(a.boards_per_level, a.max_runs, a.max_level)
