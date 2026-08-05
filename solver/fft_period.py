"""Objective tile-period measurement via FFT on gradient profiles."""
import sys
import numpy as np
from PIL import Image

img = np.array(Image.open(sys.argv[1]).convert("RGB")).astype(float)
gray = img.mean(2)
reg = gray[40:540, 20:700]          # central board region
col = np.abs(np.diff(reg, axis=1)).sum(0)   # per-column boundary strength
row = np.abs(np.diff(reg, axis=0)).sum(1)   # per-row boundary strength


def top_periods(p, lo=30, hi=95, n=6):
    p = p - p.mean()
    p /= p.std() + 1e-9
    F = np.abs(np.fft.rfft(p))
    res = []
    for period in range(lo, hi + 1):
        k = len(p) / period
        k0 = int(k); f = k - k0
        mag = (1 - f) * F[k0] + f * F[min(k0 + 1, len(F) - 1)]
        res.append((period, float(mag)))
    res.sort(key=lambda x: -x[1])
    return res[:n]


print("W (col) top periods:", [(p, round(m)) for p, m in top_periods(col)])
print("H (row) top periods:", [(p, round(m)) for p, m in top_periods(row)])
