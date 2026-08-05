"""Perception layer for the 宠物连连看 (Pokémon Lianliankan / tile-matching) board.

The game SWF is rendered by Ruffle so that the tile grid is packed (tiles touch,
no gap lines) and the pixel period is *non-integer* (~60 x 58.5 px).  That makes
naive pixel comparison of two same-icon tiles fail: each instance is sampled with
a different sub-pixel phase.  We therefore:

  1. ``get_grid``        - locate the grid from pixels: top frame line, full-width
                           columns, and the button border gives the (non-integer)
                           row period.
  2. ``extract_tiles``   - sub-pixel tile extraction (remap) at each cell centre.
  3. ``classify``        - translation-tolerant NCC (Normalised Cross-Correlation
                           via ``cv2.matchTemplate``) so sub-pixel / 1-px jitter
                           between two instances of the same icon is absorbed,
                           then agglomerative clustering.
  4. ``build_passable_map`` - True where a cell is empty / background.

All geometry is derived from pixels, so it survives the board state changing.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter

# --- defaults / constants ---------------------------------------------------
COLS = 12
ROWS = 8
# frame line that separates the title area from the tile grid is a near-uniform
# gray stripe; the first tile sits directly below it.
FRAME_GRAY = np.array([225, 227, 225], dtype=np.float32)


# ===========================================================================
#  loading
# ===========================================================================
def load_img(path: str) -> np.ndarray:
    """Load an image as RGB uint8 ``(H, W, 3)`` array."""
    return np.array(Image.open(path).convert("RGB"))


# ===========================================================================
#  1. grid geometry
# ===========================================================================
def _find_top_frame(img: np.ndarray, y_lo: int = 35, y_hi: int = 80) -> int:
    """Find the y of the uniform gray frame line that tops the tile grid.

    The frame row is >90 % close to ``FRAME_GRAY`` across the full width.
    Falls back to ``y_lo`` if not found.
    """
    dist = np.abs(img.astype(np.float32) - FRAME_GRAY).sum(axis=2)
    near = dist < 30                       # pixels close to frame gray
    frac = near[y_lo:y_hi].mean(axis=1)    # fraction per row
    cand = np.where(frac > 0.9)[0]
    if len(cand):
        return int(cand[0]) + y_lo
    return y_lo


def _find_grid_bottom(img: np.ndarray, top: int) -> int:
    """Bottom of the tile area = the menu-button border, minus a small margin.

    The button strip is topped by a strong full-width horizontal edge (the button
    rectangles).  We find the strongest such row in the lower half of the window
    and return a couple of pixels above it as the grid bottom.
    """
    gray = img.mean(axis=2)
    h = gray.shape[0]
    dy = np.abs(np.diff(gray, axis=0)).sum(axis=1)            # per-row edge energy
    # the button strip sits below the grid and begins with a persistent band of
    # strong horizontal edges (the button rectangles).  Scan downwards from a few
    # rows below the expected grid and return the first such band.
    lo = top + ROWS * 48
    hi = min(h - 8, top + ROWS * 72)
    window = dy[lo:hi]
    if len(window) == 0:
        return h - 8
    thr = window.mean() + 1.2 * window.std()
    strong = window > thr
    run = 0
    for k, s in enumerate(strong):
        run = run + 1 if s else 0
        if run >= 4:                                           # persistent band => buttons
            button_top = lo + k - run + 1
            return max(top + 8, button_top - 3)
    # fallback: strongest row
    return max(top + 8, lo + int(np.argmax(window)) - 3)


def _tile_centres(grid: dict) -> np.ndarray:
    """Return ``(ROWS*COLS, 2)`` array of (cy, cx) tile centres from a grid dict."""
    g = grid
    ys = g["y0"] + (np.arange(g["rows"]) + 0.5) * g["tile_h"]
    xs = g["x0"] + (np.arange(g["cols"]) + 0.5) * g["tile_w"]
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def get_grid(img: np.ndarray) -> dict:
    """Detect the tile grid purely from pixels.

    Returns ``dict(rows, cols, tile_w, tile_h, x0, y0)`` where ``tile_w``/``tile_h``
    are the per-tile pixel size (may be non-integer - the SWF renders at a
    non-integer scale) and ``x0``/``y0`` the top-left of the first tile.
    """
    h, w = img.shape[:2]
    frame_y = _find_top_frame(img)
    y0 = frame_y + 1                       # tiles start just below the frame line
    bottom = _find_grid_bottom(img, y0)

    cols = COLS
    rows = ROWS
    x0 = 0
    tile_w = w / cols                      # 720 / 12 == 60 (full width)

    # tile height: the SWF renders at a non-integer scale (~58.5 px), so the grid
    # height (top frame -> button border) divided by the row count gives the
    # period directly.  classify() uses translation-tolerant matching, so a
    # sub-pixel period error of a few px/row is fully absorbed.
    tile_h = (bottom - y0) / rows

    return dict(rows=rows, cols=cols,
                tile_w=float(tile_w), tile_h=float(tile_h),
                tile=int(round(min(tile_w, tile_h))),     # square extract size
                x0=float(x0), y0=float(y0))


# ===========================================================================
#  2. tile extraction (sub-pixel)
# ===========================================================================
def extract_tiles(img: np.ndarray, grid: dict) -> np.ndarray:
    """Extract every tile as a square ``(rows, cols, tile, tile, 3)`` uint8 array.

    Uses bilinear remap at each cell centre so the non-integer grid period does
    not introduce per-instance aliasing.
    """
    g = grid
    ts = int(round(min(g["tile_w"], g["tile_h"])))          # square output size
    rows, cols = g["rows"], g["cols"]
    out = np.zeros((rows, cols, ts, ts, 3), dtype=np.uint8)
    f32 = img.astype(np.float32)
    yy = g["y0"] + (np.arange(rows) + 0.5) * g["tile_h"]
    xx = g["x0"] + (np.arange(cols) + 0.5) * g["tile_w"]
    for r, cy in enumerate(yy):
        for c, cx in enumerate(xx):
            map_y, map_x = np.mgrid[0:ts, 0:ts].astype(np.float32)
            map_y += cy - ts / 2.0
            map_x += cx - ts / 2.0
            sub = cv2.remap(f32, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            out[r, c] = np.clip(sub, 0, 255).astype(np.uint8)
    return out


# ===========================================================================
#  3. classification
# ===========================================================================
def _ncc_matrix(tiles: np.ndarray, img: np.ndarray | None = None,
                grid: dict | None = None) -> np.ndarray:
    """Full ``(N, N)`` translation-tolerant grayscale NCC matrix between tiles.

    For each unordered pair we slide one tile's small central template over the
    other's larger grayscale region (``cv2.matchTemplate`` ``TM_CCOEFF_NORMED``)
    and keep the max.  That absorbs the residual ~1 px sub-pixel jitter between
    two instances of the same icon (the grid period is non-integer).

    If the source ``img`` + ``grid`` are supplied the regions are cropped directly
    from the image at the exact sub-pixel cell centres (sharpest, least aliasing);
    otherwise the extracted ``tiles`` array is used.  Diagonal set to 1.0.
    """
    rows, cols, t = tiles.shape[:3]
    r_half, t_half = 27, 10
    if img is not None and grid is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        pad = r_half + 4
        gp = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        centres = _tile_centres(grid) + pad
        regs = [gp[int(cy) - r_half:int(cy) + r_half, int(cx) - r_half:int(cx) + r_half]
                for cy, cx in centres]
        tmps = [gp[int(cy) - t_half:int(cy) + t_half, int(cx) - t_half:int(cx) + t_half]
                for cy, cx in centres]
    else:                                  # fall back to the extracted tiles
        gray = np.stack([cv2.cvtColor(tiles[r, c], cv2.COLOR_RGB2GRAY)
                         for r in range(rows) for c in range(cols)]).astype(np.float32)
        m = t // 2 - r_half
        regs = gray[:, m:m + 2 * r_half, m:m + 2 * r_half]
        m2 = t // 2 - t_half
        tmps = gray[:, m2:m2 + 2 * t_half, m2:m2 + 2 * t_half]
    n = len(regs)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ri = regs[i]
        for j in range(i + 1, n):
            v = float(cv2.matchTemplate(ri, tmps[j], cv2.TM_CCOEFF_NORMED).max())
            S[i, j] = S[j, i] = v
    np.fill_diagonal(S, 1.0)
    return S


def _is_background(tiles: np.ndarray) -> np.ndarray:
    """Boolean ``(rows, cols)`` mask: True where a cell is empty / background.

    Empty cells are very uniform (the board background) - low per-cell std - and
    also low-saturation.  A fresh, full board has *no* background cells.
    """
    gray = tiles.mean(axis=-1).astype(np.float32)           # (R, C, t, t)
    per_cell_std = gray.reshape(gray.shape[0], gray.shape[1], -1).std(axis=-1)
    # full-bleed icons have std >= ~16; an empty/background cell is very uniform
    # (std < ~10).  Keep the threshold in that gap so low-variance icons are not
    # mistaken for background, while real empty cells still register.
    thr = max(13.0, 0.20 * per_cell_std.mean())
    return per_cell_std < thr


def classify(tiles: np.ndarray, grid: dict | None = None,
             img: np.ndarray | None = None) -> np.ndarray:
    """Label each tile by icon (same label == same icon); background == -1.

    Returns ``(rows, cols)`` int array.  Same label == same icon; empty/background
    cells get ``-1``.  Works on the extracted ``tiles`` alone (translation-tolerant
    NCC + agglomerative clustering); ``grid`` / ``img`` are accepted for API
    compatibility but not required.
    """
    rows, cols = tiles.shape[:2]
    bg = _is_background(tiles)
    flat_bg = bg.ravel()
    n_icon = int((~flat_bg).sum())

    S = _ncc_matrix(tiles, img=img, grid=grid)
    # background tiles: maximally far from everything, zero distance among themselves
    D = np.clip(1.0 - S, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    if flat_bg.any():
        D[np.ix_(flat_bg, flat_bg)] = 0.0
        D[np.ix_(flat_bg, ~flat_bg)] = 2.0
        D[np.ix_(~flat_bg, flat_bg)] = 2.0

    # complete linkage keeps clusters tight (no chaining between look-alikes).
    # Pick the threshold that minimises tiles in odd-sized groups - the
    # Lianliankan even-pair property (singletons count as odd: an unmatched tile
    # is an odd-count cluster of size 1).
    Z = linkage(squareform(D, checks=False), method="complete")
    icon_idx = np.where(~flat_bg)[0]
    best_thr, best_odd, best_lab = None, None, None
    for thr in np.arange(0.05, 0.60, 0.005):
        lab = fcluster(Z, t=thr, criterion="distance")
        sizes = Counter(lab[i] for i in icon_idx).values()
        odd_tiles = sum(s for s in sizes if s % 2 == 1)        # incl. singletons
        if best_odd is None or odd_tiles < best_odd:
            best_odd, best_thr, best_lab = odd_tiles, float(thr), lab
    labels = best_lab

    # re-base label ids to consecutive integers; mark background as -1
    out = np.full(rows * cols, -1, dtype=int)
    for new_id, old_id in enumerate(sorted(set(labels[i] for i in icon_idx))):
        for i in icon_idx:
            if labels[i] == old_id:
                out[i] = new_id
    return out.reshape(rows, cols)


# ===========================================================================
#  4. passable map
# ===========================================================================
def build_passable_map(labels: np.ndarray, empty_label: int = -1) -> np.ndarray:
    """Boolean ``(rows, cols)``: True where the cell is empty / passable.

    A cell is passable when it is background (no tile).  Once a pair of tiles is
    removed during play, those cells become background and the path-finder may
    route through them.
    """
    return labels == empty_label


# ===========================================================================
#  montage (verification artefact)
# ===========================================================================
def make_montage(tiles: np.ndarray, labels: np.ndarray, out_path: str,
                 cell: int = 56) -> None:
    """Draw every tile in grid position with its cluster id, save to ``out_path``."""
    rows, cols = tiles.shape[:2]
    pad = 2
    W = cols * (cell + pad) + pad
    H = rows * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (32, 32, 32))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for r in range(rows):
        for c in range(cols):
            x0 = pad + c * (cell + pad)
            y0 = pad + r * (cell + pad)
            tile_img = Image.fromarray(tiles[r, c]).resize((cell, cell))
            canvas.paste(tile_img, (x0, y0))
            lab = int(labels[r, c])
            txt = "·" if lab < 0 else str(lab)
            # coloured badge so the id is readable over any icon
            draw.rectangle([x0, y0, x0 + 16, y0 + 14], fill=(0, 0, 0))
            draw.text((x0 + 3, y0 + 1), txt, fill=(255, 255, 0))
    canvas.save(out_path)


# ===========================================================================
#  CLI / demo
# ===========================================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/henryhuang/.claude/jobs/06608b45/tmp/fresh_board.png"
    montage_out = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/henryhuang/.claude/jobs/06608b45/tmp/verify_montage.png"

    image = load_img(path)
    grid = get_grid(image)
    print("GRID:", grid)

    tiles = extract_tiles(image, grid)
    print("tiles shape:", tiles.shape)

    labels = classify(tiles, grid=grid, img=image)
    print("labels shape:", labels.shape, "dtype:", labels.dtype)

    # cluster-count breakdown (excluding background label -1)
    non_bg = labels[labels >= 0]
    counts = Counter(non_bg.tolist())
    sizes = sorted(counts.values(), reverse=True)
    n_bg = int((labels < 0).sum())
    n_odd = sum(1 for s in sizes if s % 2 == 1)
    n_even = sum(1 for s in sizes if s % 2 == 0 and s > 0)
    print(f"distinct icon clusters: {len(counts)}")
    print(f"background cells: {n_bg}")
    print(f"cluster sizes: {sizes}")
    print(f"even-count clusters: {n_even} / {len(counts)} ; "
          f"odd-count clusters: {n_odd}  -> even-validation "
          f"{'PASSED' if n_odd == 0 else 'FAILED (partial)'}")

    pmap = build_passable_map(labels)
    print(f"passable cells: {int(pmap.sum())} / {pmap.size}")

    make_montage(tiles, labels, montage_out)
    print("saved montage:", montage_out)
