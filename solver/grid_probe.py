"""Board perception for 宠物连连看 (Pokémon Lianliankan).

Works in SCREENSHOT pixel coordinates (full 720x560 window, incl. the ~56px
top chrome bar). Derives the tile grid from boundary energy, then classifies
tiles by clustering visual features. Empty/background cells get label EMPTY.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import AgglomerativeClustering

EMPTY = -1


def load_img(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _profile(region_gray: np.ndarray, axis: int) -> np.ndarray:
    """axis=1 -> per-COLUMN boundary energy (length W); axis=0 -> per-ROW (length H)."""
    if axis == 1:
        return np.abs(np.diff(region_gray, axis=0)).sum(0)
    return np.abs(np.diff(region_gray, axis=1)).sum(1)


def _period(profile: np.ndarray, lo: int = 44, hi: int = 80) -> int:
    p = profile.astype(np.float64)
    p -= p.mean()
    if p.std() < 1e-6:
        return 60
    p /= p.std()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    k = lo + int(np.argmax(ac[lo:hi]))
    return k


def _board_peaks(profile: np.ndarray, period: int) -> list[int]:
    """Longest run of ~period-spaced peaks = board boundary lines."""
    prof = uniform_filter1d(profile.astype(float), 5)
    pk, _ = find_peaks(prof, distance=int(period * 0.55),
                       prominence=prof.std() * 0.35)
    pk = list(pk)
    if len(pk) < 2:
        return pk
    best_run = []
    run = [pk[0]]
    for x in pk[1:]:
        if abs((x - run[-1]) - period) <= 0.18 * period:
            run.append(x)
        else:
            if len(run) > len(best_run):
                best_run = run
            run = [x]
    if len(run) > len(best_run):
        best_run = run
    return best_run


def get_grid(img: np.ndarray) -> dict:
    H, W = img.shape[:2]
    gray = img.mean(2)
    # exclude the top chrome bar (~56px) and a small margin everywhere
    x0r, x1r, y0r, y1r = 6, W - 6, 64, H - 6
    col = _profile(gray[y0r:y1r, x0r:x1r], axis=1)  # length x1r-x0r
    row = _profile(gray[y0r:y1r, x0r:x1r], axis=0)
    pc = _period(col)
    pr = _period(row)
    tile = round((pc + pr) / 2)
    colp = _board_peaks(col, pc)
    rowp = _board_peaks(row, pr)
    if len(colp) >= 2:
        x0 = x0r + colp[0]
        cols = len(colp) - 1
        tile_w = int(round(np.median(np.diff(colp))))
    else:
        x0, cols, tile_w = x0r, W // tile, tile
    if len(rowp) >= 2:
        y0 = y0r + rowp[0]
        rows = len(rowp) - 1
        tile_h = int(round(np.median(np.diff(rowp))))
    else:
        y0, rows, tile_h = y0r, H // tile, tile
    return dict(rows=rows, cols=cols, tile=int(round((tile_w + tile_h) / 2)),
                tile_w=tile_w, tile_h=tile_h, x0=x0, y0=y0)


def extract_tiles(img: np.ndarray, grid: dict) -> np.ndarray:
    rows, cols = grid["rows"], grid["cols"]
    tw, th = grid.get("tile_w", grid["tile"]), grid.get("tile_h", grid["tile"])
    x0, y0 = grid["x0"], grid["y0"]
    out = np.zeros((rows, cols, th, tw, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = img[y0 + r * th: y0 + (r + 1) * th,
                            x0 + c * tw: x0 + (c + 1) * tw]
    return out


def _feature(tile: np.ndarray) -> np.ndarray:
    small = np.asarray(Image.fromarray(tile).convert("L").resize((12, 12)), dtype=float)
    col = np.asarray(Image.fromarray(tile).convert("RGB").resize((6, 6)), dtype=float).mean(-1)
    return np.concatenate([small.flatten() / 255.0, col.flatten() / 255.0])


def classify(tiles: np.ndarray) -> np.ndarray:
    rows, cols = tiles.shape[:2]
    feats = np.stack([_feature(tiles[r, c]) for r in range(rows) for c in range(cols)])
    # empty cells: very low internal variance (flat background)
    var = tiles.reshape(rows * cols, -1).std(1)
    labels = np.full(rows * cols, EMPTY, dtype=int)
    present = var > 14
    if present.sum() < 2:
        return labels.reshape(rows, cols)
    feats_p = feats[present]
    # normalise per-feature
    mu, sd = feats_p.mean(0), feats_p.std(0) + 1e-6
    feats_n = (feats_p - mu) / sd
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=4.5, linkage="average")
    sub = cl.fit_predict(feats_n)
    labels[present] = sub
    return labels.reshape(rows, cols)


def build_passable_map(labels: np.ndarray, empty_label: int = EMPTY) -> np.ndarray:
    return labels == empty_label


if __name__ == "__main__":
    import sys, os
    img = load_img(sys.argv[1])
    g = get_grid(img)
    print("grid:", g)
    tiles = extract_tiles(img, g)
    labels = classify(tiles)
    uniq, cnt = np.unique(labels[labels != EMPTY], return_counts=True)
    print("distinct icons:", len(uniq), "counts:", sorted(cnt.tolist(), reverse=True))
    print("odd-count clusters:", int((cnt % 2).sum()))
    print("empty cells:", int((labels == EMPTY).sum()))
    # montage for visual check
    ts = 40
    mon = Image.new("RGB", (ts * g["cols"] + 4, ts * g["rows"] + 4), (40, 40, 40))
    from PIL import ImageDraw
    d = ImageDraw.Draw(mon)
    for r in range(g["rows"]):
        for c in range(g["cols"]):
            t = Image.fromarray(tiles[r, c]).resize((ts - 2, ts - 2))
            mon.paste(t, (c * ts + 3, r * ts + 3))
            lab = labels[r, c]
            d.text((c * ts + 3, r * ts + 3), str(lab if lab >= 0 else "·"), (255, 255, 0))
    mon.save(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "grid_montage.png"))
    print("saved grid_montage.png")
