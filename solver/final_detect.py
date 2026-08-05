"""Fix period=60, find best grid offset by boundary alignment; overlay + montage."""
import sys
import numpy as np
from PIL import Image, ImageDraw

img = np.array(Image.open(sys.argv[1]).convert("RGB"))
H, W, _ = img.shape
P = int(sys.argv[4]) if len(sys.argv) > 4 else 60
overlay_out = sys.argv[2] if len(sys.argv) > 2 else "overlay.png"
montage_out = sys.argv[3] if len(sys.argv) > 3 else "montage.png"
gray = img.mean(2)
col = np.abs(np.diff(gray, axis=1)).sum(0)   # length W-1
row = np.abs(np.diff(gray, axis=0)).sum(1)   # length H-1


def best_offset(profile, P, lo, hi):
    best_o, best_s = 0, -1
    for o in range(P):
        s = 0
        x = o
        while x < len(profile) and lo <= x <= hi:
            s += profile[x]
            x += P
        if s > best_s:
            best_s, best_o = s, o
    return best_o


ox = best_offset(col, P, 10, W - 10)
oy = best_offset(row, P, 30, H - 40)
xlines = [x for x in range(ox, W, P) if 8 <= x <= W - 8]
ylines = [y for y in range(oy, H, P) if 25 <= y <= H - 35]
cols, rows = len(xlines) - 1, len(ylines) - 1
print(f"P={P} ox={ox} oy={oy} cols={cols} rows={rows}")
print("xlines:", xlines)
print("ylines:", ylines)

# overlay
pil = Image.open(sys.argv[1]).convert("RGB")
d = ImageDraw.Draw(pil)
for x in xlines:
    d.line([(x, ylines[0]), (x, ylines[-1])], fill=(255, 0, 255), width=1)
for y in ylines:
    d.line([(xlines[0], y), (xlines[-1], y)], fill=(255, 0, 255), width=1)
pil.save(overlay_out)
# montage
TS = 48
mon = Image.new("RGB", (TS * cols + 2, TS * rows + 2), (255, 255, 255))
for r in range(rows):
    for c in range(cols):
        x0, y0 = xlines[c], ylines[r]
        tile = pil.crop((x0 + 1, y0 + 1, x0 + P - 1, y0 + P - 1)).resize((TS, TS))
        mon.paste(tile, (c * TS + 1, r * TS + 1))
mon.save(montage_out)
print("saved", overlay_out, montage_out)
