"""Objective grid search: find (period, offset) maximizing the even-cluster
property. The correct Lianliankan grid yields tile clusters that are almost all
even-counted (icons come in pairs). No vision needed.
"""
import sys
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

img = np.array(Image.open(sys.argv[1]).convert("RGB"))
H, W = img.shape[:2]
gray = img.mean(2)
x0r, x1r, y0r, y1r = 8, W - 8, 62, H - 18
col = np.abs(np.diff(gray[y0r:y1r, x0r:x1r], axis=0)).sum(0)   # len x1r-x0r
row = np.abs(np.diff(gray[y0r:y1r, x0r:x1r], axis=1)).sum(1)   # len y1r-y0r


def best_lines(profile, P):
    """Best offset (max boundary energy at ox+i*P) and the boundary positions."""
    best_o, best_s = 0, -1.0
    for o in range(P):
        s = 0.0
        x = o
        while x < len(profile):
            s += profile[x]
            x += P
        if s > best_s:
            best_s, best_o = s, o
    lines = []
    x = best_o
    while x < len(profile):
        lines.append(x)
        x += P
    return best_o, lines


def feat(t):
    g = np.asarray(Image.fromarray(t).convert("L").resize((14, 14)), dtype=float).flatten() / 255.0
    c = np.asarray(Image.fromarray(t).convert("RGB").resize((8, 8)), dtype=float).reshape(-1, 3).mean(1) / 255.0
    return np.concatenate([g, c])


results = []
for P in range(48, 66):
    ox, xlines = best_lines(col, P)
    oy, ylines = best_lines(row, P)
    cols, rows = len(xlines) - 1, len(ylines) - 1
    if cols < 4 or rows < 4 or cols > 18 or rows > 14:
        continue
    # extract tiles
    tiles = []
    for r in range(rows):
        for c in range(cols):
            xa, ya = x0r + xlines[c], y0r + ylines[r]
            tiles.append(img[ya:ya + P, xa:xa + P])
    tiles = np.stack(tiles)
    var = tiles.reshape(len(tiles), -1).std(1)
    present = var > 14
    if present.sum() < 8:
        continue
    feats = np.stack([feat(t) for t in tiles[present]])
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    fn = (feats - mu) / sd
    for thr in (3.5, 4.5, 6.0):
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=thr, linkage="average").fit_predict(fn)
        _, cnt = np.unique(cl, return_counts=True)
        even = int((cnt % 2 == 0).sum())
        score = even / len(cnt)
        results.append((score, P, thr, rows, cols, ox, oy, len(cnt), int((cnt % 2).sum()), int(present.sum())))

results.sort(reverse=True)
print("top grids (score,P,thr,rows,cols,ox,oy,#clusters,odd,nonempty):")
for r in results[:8]:
    print("  %.3f P=%d thr=%.1f %dx%d ox=%d oy=%d clusters=%d odd=%d nonempty=%d" % r)
