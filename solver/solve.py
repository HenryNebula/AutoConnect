"""Sense -> solve -> act loop for the 宠物连连看 board (Linux/Xvfb).

Connectivity rule (Lianliankan): two tiles are connectable iff a path with
<=2 turns (<=3 straight segments) exists through EMPTY cells, where the path
may route around the outside of the grid (the border around the board counts
as empty). We implement this with the classic O(W+H) corner/line-clear check
on a border-padded passable map.

Per level we classify the board ONCE (stable labels) and then track removals
locally (present mask), re-checking connectivity as tiles vanish.
"""
from __future__ import annotations
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import perception as perc  # owned by the perception subagent
import actuator


def pad_passable(present: np.ndarray) -> np.ndarray:
    """present (R,C) bool: True where a tile EXISTS.
    Returns passable (R+2, C+2) bool: True where EMPTY (passable), with a
    True border so paths can route around the board edge. Tile (r,c) -> (r+1,c+1).
    """
    return ~np.pad(present, 1, mode="constant", constant_values=False)


def _line_clear(p: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
    """Cells strictly between two same-row/col points are all passable (True)."""
    if r1 == r2:
        a, b = min(c1, c2), max(c1, c2)
        return bool(p[r1, a + 1:b].all())
    if c1 == c2:
        a, b = min(r1, r2), max(r1, r2)
        return bool(p[a + 1:b, c1].all())
    return False


def connectable(p: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
    """<=2-turn connection on the padded passable map p. Endpoints are tiles."""
    # 0-turn: straight line
    if (r1 == r2 or c1 == c2) and _line_clear(p, r1, c1, r2, c2):
        return True
    R, C = p.shape

    def one_turn_from(sr, sc):
        # can we get from (sr,sc) to end with at most one extra turn (<=2 total)?
        # corner candidates on the end's row/col: (sr,c2) and (r2,sc)
        # 1-turn: a single L via one corner
        for (cr, cc) in ((sr, c2), (r2, sc)):
            if 0 <= cr < R and 0 <= cc < C and p[cr, cc]:
                if _line_clear(p, sr, sc, cr, cc) and _line_clear(p, cr, cc, r2, c2):
                    return True
        # 2-turn: go straight from start to a pivot k (passable), then 1-turn k->end
        # scan along start's row
        for cc in range(C):
            if cc == sc:
                continue
            if p[sr, cc] and _line_clear(p, sr, sc, sr, cc):
                # one extra turn from (sr,cc) to end
                for (cr2, cc2) in ((sr, c2), (r2, cc)):
                    if 0 <= cr2 < R and 0 <= cc2 < C and p[cr2, cc2]:
                        if _line_clear(p, sr, cc, cr2, cc2) and _line_clear(p, cr2, cc2, r2, c2):
                            return True
        # scan along start's col
        for cr in range(R):
            if cr == sr:
                continue
            if p[cr, sc] and _line_clear(p, sr, sc, cr, sc):
                for (cr2, cc2) in ((cr, c2), (r2, sc)):
                    if 0 <= cr2 < R and 0 <= cc2 < C and p[cr2, cc2]:
                        if _line_clear(p, cr, sc, cr2, cc2) and _line_clear(p, cr2, cc2, r2, c2):
                            return True
        return False

    return one_turn_from(r1, c1)


def find_move(labels: np.ndarray, present: np.ndarray):
    """Return a connectable same-label pair ((r1,c1),(r2,c2)) or None."""
    R, C = labels.shape
    p = pad_passable(present)
    idxs = [(r, c) for r in range(R) for c in range(C) if present[r, c]]
    for i in range(len(idxs)):
        r1, c1 = idxs[i]
        for j in range(i + 1, len(idxs)):
            r2, c2 = idxs[j]
            if labels[r1, c1] != labels[r2, c2]:
                continue
            if connectable(p, r1 + 1, c1 + 1, r2 + 1, c2 + 1):
                return (r1, c1), (r2, c2)
    return None


def perceive_board(path: str):
    img = perc.load_img(path)
    grid = perc.get_grid(img)
    tiles = perc.extract_tiles(img, grid)
    labels = perc.classify(tiles)
    present = labels != perc.EMPTY if hasattr(perc, "EMPTY") else (labels != -1)
    return grid, labels, present


def solve_level(settle: float = 0.12, max_steps: int = 400, verbose: bool = True):
    """Capture the current board and clear it by clicking connectable pairs.
    Returns True if the board is cleared."""
    cap = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "solve_cap.png")
    actuator.capture(cap)
    grid, labels, present = perceive_board(cap)
    n_tiles = int(present.sum())
    if verbose:
        print(f"[solve] grid {grid['cols']}x{grid['rows']} tile={grid['tile']} "
              f"x0={grid['x0']} y0={grid['y0']} tiles={n_tiles}")
    steps = 0
    while n_tiles > 0 and steps < max_steps:
        mv = find_move(labels, present)
        if mv is None:
            if verbose:
                print(f"[solve] no move at step {steps} ({n_tiles} tiles left) -> stuck/misread")
            break
        (r1, c1), (r2, c2) = mv
        actuator.click_tile(r1, c1, grid)
        time.sleep(settle)
        actuator.click_tile(r2, c2, grid)
        time.sleep(settle)
        present[r1, c1] = False
        present[r2, c2] = False
        n_tiles = int(present.sum())
        steps += 1
        if verbose and steps % 10 == 0:
            print(f"[solve] step {steps}, {n_tiles} tiles left")
    cleared = n_tiles == 0
    if verbose:
        print(f"[solve] done steps={steps} cleared={cleared} left={n_tiles}")
    return cleared


if __name__ == "__main__":
    ok = solve_level()
    sys.exit(0 if ok else 1)
