import sys
import numpy as np
from PIL import Image

path = sys.argv[1]
img = np.array(Image.open(path).convert("RGB"))
H, W, _ = img.shape
gray = img.mean(2)
col_e = np.abs(np.diff(gray, axis=0)).sum(0)   # length W
row_e = np.abs(np.diff(gray, axis=1)).sum(1)   # length H


def spark(a, bins=70):
    step = len(a) / bins
    out = [a[int(i * step):int((i + 1) * step)].mean() for i in range(bins)]
    m = max(out) or 1
    return "".join(" .:-=+*#%@"[min(9, int(v / m * 9))] for v in out)


print(f"W={W} H={H}")
print("col energy (L->R):", spark(col_e))
print("row energy (T->B):", spark(row_e))
for frac in [0.3, 0.5, 0.7]:
    ry = np.where(row_e > row_e.max() * frac)[0]
    cx = np.where(col_e > col_e.max() * frac)[0]
    print(f"frac={frac}: rows {ry.min()}-{ry.max()} ({ry.max()-ry.min()}px)  "
          f"cols {cx.min()}-{cx.max()} ({cx.max()-cx.min()}px)")

# autocorrelation period within the likely board band (rows 60..520)
def period(p, lo, hi):
    p = p.astype(float); p -= p.mean()
    if p.std() < 1e-6: return None
    p /= p.std()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    k = lo + int(np.argmax(ac[lo:hi]))
    return k, round(float(ac[k]), 2)

rb = row_e[60:520]
cb = col_e[20:700]
print("row period (band 60-520):", period(rb, 25, 110))
print("col period (band 20-700):", period(cb, 25, 110))
