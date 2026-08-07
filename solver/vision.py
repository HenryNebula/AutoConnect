"""Legitimate, CV-only board perception for 宠物连连看 (Pokémon Lianliankan).

Nothing from the game's internal state is ever read here. The grid is located
purely from pixels (colourful-icon connected components -> centroid grid), tiles
are extracted by sub-pixel sampling, and icons are grouped by visual similarity
(translation-tolerant NCC + agglomerative clustering, threshold chosen to honour
the Lianliankan even-pair property). Empty cells are detected by colourfulness,
not by game state.

Geometry is derived every call, so it survives window moves / rescales.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter

EMPTY = -1


# ===========================================================================
# loading
# ===========================================================================
def load_img(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def saturation(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    mx = f.max(2)
    mn = f.min(2)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)


# ===========================================================================
# 1. grid geometry  (icon-blob centroids -> rows/cols)
# ===========================================================================
def _cluster(vals, gap: float):
    vals = sorted(vals)
    out, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(float(np.mean(cur)))
            cur = [v]
    out.append(float(np.mean(cur)))
    return out


def detect_grid(img: np.ndarray, expect_cols: range = range(10, 15),
                expect_rows: range = range(7, 10),
                sat_thr: float = 0.25) -> dict | None:
    """Detect the tile grid from colourful-icon blobs.

    Blobs are clustered into columns/rows with a half-tile merge gap (robust to
    multi-blob icons), then each column/row centre is the median blob position.
    Returns dict(cols, rows, xs, ys, ts) or None.
    """
    H, W = img.shape[:2]
    mask = (saturation(img) > sat_thr).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    xs, ys = [], []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 120 < a < 6000 and w < 70 and h < 70:
            xs.append(float(cent[i, 0]))
            ys.append(float(cent[i, 1]))
    if len(xs) < 16:
        return None
    # stage 1: coarse cluster with a small fixed gap to dedupe within-tile
    # blobs (multiple colour regions of one icon) -> provisional columns/rows.
    coarse_x = _cluster(xs, gap=12.0)
    coarse_y = _cluster(ys, gap=12.0)
    if len(coarse_x) < 2 or len(coarse_y) < 2:
        return None
    ts = float(np.median([np.median(np.diff(coarse_x)), np.median(np.diff(coarse_y))]))
    # stage 2: re-cluster with a half-tile gap (robust: merges any sub-tile
    # blobs split by stage 1, never merges adjacent tiles ~ts apart).
    cols_x = _cluster(xs, gap=0.42 * ts)
    rows_y = _cluster(ys, gap=0.42 * ts)
    cols, rows = len(cols_x), len(rows_y)
    if cols not in expect_cols or rows not in expect_rows:
        return None
    ts = float(np.median([np.median(np.diff(cols_x)), np.median(np.diff(rows_y))]))
    return dict(cols=cols, rows=rows, xs=cols_x, ys=rows_y, ts=ts)


# ===========================================================================
# 2. tile extraction (sub-pixel, centred on the detected grid)
# ===========================================================================
def extract_tiles(img: np.ndarray, grid: dict) -> np.ndarray:
    """(rows, cols, ts, ts, 3) uint8 array of square tiles, one per cell."""
    ts = int(round(grid["ts"]))
    ts = max(ts, 16)
    rows, cols = grid["rows"], grid["cols"]
    out = np.zeros((rows, cols, ts, ts, 3), dtype=np.uint8)
    f32 = img.astype(np.float32)
    half = ts / 2.0
    for r in range(rows):
        for c in range(cols):
            cy, cx = grid["ys"][r], grid["xs"][c]
            map_y, map_x = np.mgrid[0:ts, 0:ts].astype(np.float32)
            map_y += cy - half
            map_x += cx - half
            sub = cv2.remap(f32, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
            out[r, c] = np.clip(sub, 0, 255).astype(np.uint8)
    return out


# ===========================================================================
# 3. present-mask (colourfulness) + classification (NCC clustering)
# ===========================================================================
def cell_stats(img: np.ndarray, grid: dict, ts_frac: float = 0.66):
    """Return (stds (R,C), mean_rgb (R,C,3)) per cell over the central window."""
    rows, cols = grid["rows"], grid["cols"]
    ts = grid["ts"]
    half = ts * ts_frac / 2.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = img.astype(np.float32)
    stds = np.zeros((rows, cols), dtype=np.float32)
    mean_rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cy, cx = grid["ys"][r], grid["xs"][c]
            y0, y1 = int(round(cy - half)), int(round(cy + half))
            x0, x1 = int(round(cx - half)), int(round(cx + half))
            wg = gray[max(0, y0):y1, max(0, x0):x1]
            wc = f[max(0, y0):y1, max(0, x0):x1]
            stds[r, c] = wg.std() if wg.size else 0.0
            mean_rgb[r, c] = wc.reshape(-1, 3).mean(0) if wc.size else 0.0
    return stds, mean_rgb


def present_mask(img: np.ndarray, grid: dict, bg_color=None,
                 ts_frac: float = 0.66, std_thr: float = 9.0,
                 dist_thr: float = 28.0) -> np.ndarray:
    """Boolean (rows, cols): True where an icon is present.

    A cell is present when it has internal detail (std > ``std_thr``) OR its
    colour differs from the empty-cell background (dist > ``dist_thr``). The
    second clause catches the dark/low-contrast Pokémon icons that are nearly
    uniform (low std) but whose colour still differs from the background.
    ``bg_color`` is the empty-cell colour (tracked adaptively by the caller);
    if unknown, only the std test is used.
    """
    stds, mean_rgb = cell_stats(img, grid, ts_frac)
    present = stds > std_thr
    if bg_color is not None:
        bg = np.asarray(bg_color, dtype=np.float32)
        dist = np.abs(mean_rgb - bg).sum(2)
        present = present | (dist > dist_thr)
    return present


def _ncc_matrix(tiles: np.ndarray, grid: dict, img: np.ndarray) -> np.ndarray:
    """(N,N) translation-tolerant grayscale NCC between tile centres.

    Regions/templates are cropped from the image at the exact sub-pixel centres
    so the non-integer grid period introduces no per-instance aliasing.
    """
    rows, cols = grid["rows"], grid["cols"]
    r_half, t_half = 24, 9
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pad = r_half + 4
    gp = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    centres = []
    for r in range(rows):
        for c in range(cols):
            centres.append((grid["ys"][r] + pad, grid["xs"][c] + pad))
    regs = [gp[int(cy) - r_half:int(cy) + r_half, int(cx) - r_half:int(cx) + r_half]
            for cy, cx in centres]
    tmps = [gp[int(cy) - t_half:int(cy) + t_half, int(cx) - t_half:int(cx) + t_half]
            for cy, cx in centres]
    n = len(regs)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ri = regs[i]
        for j in range(i + 1, n):
            v = float(cv2.matchTemplate(ri, tmps[j], cv2.TM_CCOEFF_NORMED).max())
            S[i, j] = S[j, i] = v
    np.fill_diagonal(S, 1.0)
    return S


def classify(img: np.ndarray, grid: dict, bg_color=None) -> np.ndarray:
    """(rows, cols) int labels; same label == same icon, empty cell == EMPTY.

    Groups present tiles by translation-tolerant NCC + complete-link clustering,
    choosing the distance threshold that minimises odd-count clusters (the
    Lianliankan even-pair property).
    """
    rows, cols = grid["rows"], grid["cols"]
    present = present_mask(img, grid, bg_color=bg_color)
    flat_present = present.ravel()
    n_icon = int(flat_present.sum())
    out = np.full(rows * cols, EMPTY, dtype=int)
    if n_icon < 2:
        return out.reshape(rows, cols)
    S = _ncc_matrix(np.empty(0), grid, img)
    D = np.clip(1.0 - S, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    # push background rows/cols away
    bg = ~flat_present
    if bg.any():
        D[np.ix_(bg, bg)] = 0.0
        D[np.ix_(bg, ~bg)] = 2.0
        D[np.ix_(~bg, bg)] = 2.0
    icon_idx = np.where(~bg)[0]
    Dsub = D[np.ix_(icon_idx, icon_idx)]
    Z = linkage(squareform(Dsub, checks=False), method="complete")
    best_odd, best_lab = None, None
    for thr in np.arange(0.05, 0.60, 0.005):
        lab = fcluster(Z, t=thr, criterion="distance")
        sizes = Counter(lab.tolist()).values()
        odd = sum(s for s in sizes if s % 2 == 1)
        if best_odd is None or odd < best_odd:
            best_odd, best_lab = odd, lab
    # re-base to consecutive ids
    for new_id, old_id in enumerate(sorted(set(best_lab.tolist()))):
        for k, gi in enumerate(icon_idx):
            if best_lab[k] == old_id:
                out[gi] = new_id
    return out.reshape(rows, cols)


def perceive(img: np.ndarray) -> tuple[dict, np.ndarray] | None:
    """One-shot: detect grid + classify. Returns (grid, labels) or None."""
    grid = detect_grid(img)
    if grid is None:
        return None
    labels = classify(img, grid)
    return grid, labels


def perceive_with_grid(img: np.ndarray, grid: dict) -> np.ndarray:
    """Classify using a *cached* grid (the geometry is a level/window property
    and is stable across board states; re-detecting every frame is unreliable
    because move animations create spurious blobs). Returns label array."""
    return classify(img, grid)


def labels_to_present(labels: np.ndarray) -> np.ndarray:
    return labels != EMPTY


# ===========================================================================
# montage (verification artefact)
# ===========================================================================
def make_montage(img: np.ndarray, grid: dict, labels: np.ndarray, out_path: str,
                 cell: int = 48) -> None:
    rows, cols = grid["rows"], grid["cols"]
    ts = int(round(grid["ts"]))
    half = ts / 2.0
    pad = 2
    W = cols * (cell + pad) + pad
    H = rows * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (32, 32, 32))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    f32 = img.astype(np.float32)
    for r in range(rows):
        for c in range(cols):
            cy, cx = grid["ys"][r], grid["xs"][c]
            map_y, map_x = np.mgrid[0:ts, 0:ts].astype(np.float32)
            map_y += cy - half
            map_x += cx - half
            sub = np.clip(cv2.remap(f32, map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE), 0, 255).astype(np.uint8)
            x0 = pad + c * (cell + pad)
            y0 = pad + r * (cell + pad)
            canvas.paste(Image.fromarray(sub).resize((cell, cell)), (x0, y0))
            lab = int(labels[r, c])
            draw.rectangle([x0, y0, x0 + 18, y0 + 14], fill=(0, 0, 0))
            draw.text((x0 + 3, y0 + 1), "·" if lab < 0 else str(lab), fill=(255, 255, 0))
    canvas.save(out_path)


if __name__ == "__main__":
    import sys, os
    path = sys.argv[1] if len(sys.argv) > 1 else "full_board.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "vision_montage.png"
    image = load_img(path)
    res = perceive(image)
    if res is None:
        print("no grid detected")
        raise SystemExit(1)
    grid, labels = res
    cnt = Counter(labels[labels >= 0].tolist())
    sizes = sorted(cnt.values(), reverse=True)
    print("grid:", grid["cols"], "x", grid["rows"], "ts=%.1f" % grid["ts"])
    print("present:", int((labels >= 0).sum()), "empty:", int((labels < 0).sum()))
    print("distinct icons:", len(cnt), "sizes:", sizes)
    print("odd-count clusters:", sum(1 for s in sizes if s % 2 == 1))
    make_montage(image, grid, labels, out)
    print("saved", out)
