"""My own Lianliankan connectivity solver + deadlock-aware move selection.

Connectivity rule: two same-icon tiles are connectable iff a path with at most
2 turns (<=3 straight segments) exists through EMPTY cells, where the path may
route around the outside of the board (the border counts as empty). Implemented
on a border-padded passable map with the classic straight-line / one-corner /
two-corner checks.

Deadlock prevention: among all currently-connectable same-icon pairs, prefer a
move that -- after removal -- still leaves the board with at least one
connectable pair (one-ply lookahead), unless the move clears the board. This
avoids the "remove the only remaining pair's bridge" blunders that strand a
greedy solver. If no move preserves a follow-up, the board is in a forced
deadlock and the caller should reshuffle.
"""
from __future__ import annotations
import numpy as np


def pad_passable(present: np.ndarray) -> np.ndarray:
    """present (R,C) bool: True where a tile EXISTS.
    Returns passable (R+2, C+2) bool: True where EMPTY, with a True border so
    paths can route around the board. Tile (r,c) -> padded (r+1, c+1)."""
    return ~np.pad(present, 1, mode="constant", constant_values=False)


def _h_clear(p: np.ndarray, r: int, c1: int, c2: int) -> bool:
    """All cells strictly between (r,c1) and (r,c2) on row r are passable."""
    if c1 == c2:
        return True
    a, b = (c1, c2) if c1 < c2 else (c2, c1)
    return bool(p[r, a + 1:b].all())


def _v_clear(p: np.ndarray, c: int, r1: int, r2: int) -> bool:
    """All cells strictly between (r1,c) and (r2,c) on col c are passable."""
    if r1 == r2:
        return True
    a, b = (r1, r2) if r1 < r2 else (r2, r1)
    return bool(p[a + 1:b, c].all())


def connectable(p: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
    """<=2-turn connection on the padded passable map p. PADDED coords."""
    # 0-turn (straight). Do NOT early-return on same row/col: a blocked line
    # may still connect via a 2-turn detour (handled below).
    if r1 == r2 and _h_clear(p, r1, c1, c2):
        return True
    if c1 == c2 and _v_clear(p, c1, r1, r2):
        return True
    # 1-turn: L via corner (r1,c2) or (r2,c1). Corner must be passable.
    if p[r1, c2] and _h_clear(p, r1, c1, c2) and _v_clear(p, c2, r1, r2):
        return True
    if p[r2, c1] and _v_clear(p, c1, r1, r2) and _h_clear(p, r2, c1, c2):
        return True
    R, C = p.shape
    # 2-turn via a horizontal middle segment on row R: A->(R,c1)->(R,c2)->B.
    for R_ in range(R):
        if p[R_, c1] and p[R_, c2] and \
           _v_clear(p, c1, r1, R_) and _h_clear(p, R_, c1, c2) and _v_clear(p, c2, R_, r2):
            return True
    # 2-turn via a vertical middle segment on col C: A->(r1,C)->(r2,C)->B.
    for C_ in range(C):
        if p[r1, C_] and p[r2, C_] and \
           _h_clear(p, r1, c1, C_) and _v_clear(p, C_, r1, r2) and _h_clear(p, r2, C_, c2):
            return True
    return False


def all_connectable_pairs_anylabel(present: np.ndarray):
    """Yield ((r1,c1),(r2,c2)) for EVERY connectable pair of present tiles,
    ignoring icon labels (used as a last-resort fallback when over-segmentation
    hides the last same-type pair)."""
    R, C = present.shape
    p = pad_passable(present)
    cells = [(r, c) for r in range(R) for c in range(C) if present[r, c]]
    for i in range(len(cells)):
        r1, c1 = cells[i]
        for j in range(i + 1, len(cells)):
            r2, c2 = cells[j]
            if connectable(p, r1 + 1, c1 + 1, r2 + 1, c2 + 1):
                yield (r1, c1), (r2, c2)


def all_connectable_pairs(labels: np.ndarray, present: np.ndarray):
    """Yield ((r1,c1),(r2,c2)) for every connectable same-label present pair."""
    R, C = labels.shape
    p = pad_passable(present)
    cells = [(r, c) for r in range(R) for c in range(C) if present[r, c]]
    for i in range(len(cells)):
        r1, c1 = cells[i]
        l1 = labels[r1, c1]
        for j in range(i + 1, len(cells)):
            r2, c2 = cells[j]
            if labels[r2, c2] != l1:
                continue
            if connectable(p, r1 + 1, c1 + 1, r2 + 1, c2 + 1):
                yield (r1, c1), (r2, c2)


def has_any_move(labels: np.ndarray, present: np.ndarray) -> bool:
    """True if at least one connectable same-label pair exists."""
    for _ in all_connectable_pairs(labels, present):
        return True
    return False


def find_any_move(labels: np.ndarray, present: np.ndarray):
    for mv in all_connectable_pairs(labels, present):
        return mv
    return None


def _move_safe(labels: np.ndarray, present: np.ndarray, mv) -> bool:
    """After applying mv, does >=1 connectable pair remain (or board cleared)?"""
    (r1, c1), (r2, c2) = mv
    present2 = present.copy()
    present2[r1, c1] = False
    present2[r2, c2] = False
    if not present2.any():
        return True                      # board cleared -> always safe
    return has_any_move(labels, present2)


def choose_move(labels: np.ndarray, present: np.ndarray,
                lookahead: bool = True):
    """Pick a connectable pair. With lookahead, prefer moves that keep a
    follow-up move; return (move, forced_deadlock).
    forced_deadlock is True iff NO connectable move preserves a follow-up
    (every move strands the board) -> caller should reshuffle."""
    pairs = list(all_connectable_pairs(labels, present))
    if not pairs:
        return None, True
    if not lookahead:
        return pairs[0], False
    safe = [mv for mv in pairs if _move_safe(labels, present, mv)]
    if safe:
        return safe[0], False
    # every move strands -> forced deadlock approaching; still return a move so
    # the caller can decide (it will likely need a reshuffle soon)
    return pairs[0], True


def find_move(labels: np.ndarray, present: np.ndarray, lookahead: bool = True):
    """Compatibility wrapper: returns a move or None (None => true deadlock)."""
    mv, forced = choose_move(labels, present, lookahead)
    return mv


# ---------------------------------------------------------------------------
# self-test: verify connectable() against a brute-force BFS (<=2 turns)
# ---------------------------------------------------------------------------
def _bfs_connectable(present: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> bool:
    """Brute-force: BFS over (row,col,dir,turns) from r1,c1 on padded map."""
    R, C = present.shape
    P = pad_passable(present)
    from collections import deque
    # state: (r,c,dir,turns). dir 0..3 = N,E,S,W or -1 start. End at (r2,c2).
    sr, sc, tr, tc = r1 + 1, c1 + 1, r2 + 1, c2 + 1
    # we can pass THROUGH passable cells; endpoints sr,sc/tr,tc are tiles.
    best = {}
    q = deque()
    for d in range(4):
        q.append((sr, sc, d, 0))
    while q:
        r, c, d, t = q.popleft()
        if t > 2:
            continue
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < R + 2 and 0 <= nc < C + 2):
            continue
        reached_end = (nr == tr and nc == tc)
        if not reached_end and not P[nr, nc]:
            continue
        nt = t
        # stepping doesn't add a turn at the start; turns counted at arrival
        key = (nr, nc, d)
        if key in best and best[key] <= t:
            continue
        best[key] = t
        if reached_end:
            return True
        for nd in range(4):
            nnt = t + (1 if nd != d else 0)
            if nnt <= 2:
                q.append((nr, nc, nd, nnt))
    return False


if __name__ == "__main__":
    import random
    np.random.seed(0)
    fails = 0
    for trial in range(400):
        R, C = 6, 6
        present = np.random.rand(R, C) > 0.35
        # ensure even count per arbitrary label grouping not needed for conn test
        if present.sum() < 2:
            continue
        cells = [(r, c) for r in range(R) for c in range(C) if present[r, c]]
        a, b = random.sample(cells, 2)
        fast = connectable(pad_passable(present), a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1)
        slow = _bfs_connectable(present, a[0], a[1], b[0], b[1])
        if fast != slow:
            fails += 1
            if fails <= 3:
                print(f"MISMATCH {a}-{b}: fast={fast} bfs={slow}")
    print(f"connectable self-test: {fails} mismatches over 400 random trials")
