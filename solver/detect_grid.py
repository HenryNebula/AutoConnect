"""Detect the tile grid by finding uniform-color gap lines (low cross-line variance)."""
import sys
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu

img = np.array(Image.open(sys.argv[1]).convert("RGB"))
H, W, _ = img.shape
out = sys.argv[2]
# generous board region (excludes title at very top)
y0, y1, x0, x1 = 70, 552, 8, 712
gray = img.mean(2)


def gap_centers(profile, min_w=2, merge=8):
    thr = threshold_otsu(profile) * 0.8
    low = profile < thr
    centers = []
    i, n = 0, len(low)
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if j - i >= min_w:
                centers.append((i + j - 1) / 2.0)
            i = j
        else:
            i += 1
    merged = []
    for c in centers:
        if merged and c - merged[-1] < merge:
            merged[-1] = (merged[-1] + c) / 2
        else:
            merged.append(c)
    return merged


row_std = gray[y0:y1, x0:x1].std(1)   # per row across the board width
col_std = gray[y0:y1, x0:x1].std(0)   # per col across the board height
y_gaps = [y0 + g for g in gap_centers(row_std)]
x_gaps = [x0 + g for g in gap_centers(col_std)]
rows = max(len(y_gaps) - 1, 0)
cols = max(len(x_gaps) - 1, 0)
print(f"rows={rows} cols={cols}")
print("y_gaps:", [round(g) for g in y_gaps])
print("x_gaps:", [round(g) for g in x_gaps])
if len(y_gaps) > 1:
    print("row period median:", round(float(np.median(np.diff(y_gaps))), 1))
if len(x_gaps) > 1:
    print("col period median:", round(float(np.median(np.diff(x_gaps))), 1))

# overlay gap lines (green) + tile centers (red)
pil = Image.open(sys.argv[1]).convert("RGB")
d = ImageDraw.Draw(pil)
for x in x_gaps:
    d.line([(x, y0), (x, y1)], fill=(0, 255, 0), width=1)
for y in y_gaps:
    d.line([(x0, y), (x1, y)], fill=(0, 255, 0), width=1)
for r in range(rows):
    for c in range(cols):
        cx = (x_gaps[c] + x_gaps[c + 1]) / 2
        cy = (y_gaps[r] + y_gaps[r + 1]) / 2
        d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 0, 0))
pil.save(out)
print("saved:", out)
