"""Perception for the 宠物连连看 (Pokémon Lianliankan) board.

The game runs in a Chrome --app window on an Xvfb display. Ruffle letterboxes the
SWF (black background), so:
  1. find_stage()        -> bounding box of the (non-black) SWF stage content
  2. detect_grid()       -> tile rows/cols + per-tile pixel size within the stage
  3. crop_tiles()        -> (rows, cols, ts, ts, 3) array of tile images
  4. classify_tiles()    -> label each tile by icon via feature clustering

All geometry is derived from pixels so it survives window moves / scaling.
"""
from __future__ import annotations
import sys
import numpy as np
from PIL import Image


def load_img(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def find_stage(img: np.ndarray, black_thresh: int = 18) -> tuple[int, int, int, int]:
    """Bounding box (x0,y0,x1,y1) of non-black (SWF) content."""
    mask = img.max(axis=2) > black_thresh
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("no content found (blank/black image)")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _energy_profile(region: np.ndarray, axis: int) -> np.ndarray:
    """Sum of absolute gradient along the perpendicular axis -> reveals gaps.
    axis=0 returns a 1D array of length = region height (row energy)."""
    gray = region.mean(axis=2)
    if axis == 0:  # row energy: horizontal gradient summed across columns
        g = np.abs(np.diff(gray, axis=1)).sum(axis=1)
    else:  # col energy: vertical gradient summed across rows
        g = np.abs(np.diff(gray, axis=0)).sum(axis=0)
    return g


def detect_period(profile: np.ndarray, min_sz: int = 20, max_sz: int = 120):
    """Estimate the dominant tile period (px) in a 1D energy profile via autocorrelation."""
    p = profile.astype(np.float64)
    p -= p.mean()
    if p.std() < 1e-6:
        return None
    p /= p.std()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    best_k, best_v = None, -1.0
    for k in range(min_sz, min(max_sz, len(ac))):
        if ac[k] > best_v:
            best_k, best_v = k, ac[k]
    return best_k


def detect_grid(stage_img: np.ndarray, min_sz=20, max_sz=120):
    """Return dict with rows, cols, tile_h, tile_w, and the board bbox inside the stage."""
    row_e = _energy_profile(stage_img, axis=0)
    col_e = _energy_profile(stage_img, axis=1)
    th = detect_period(row_e, min_sz, max_sz)
    tw = detect_period(col_e, min_sz, max_sz)
    if th is None or tw is None:
        raise RuntimeError(f"could not detect tile period: th={th} tw={tw}")
    H, W = stage_img.shape[:2]
    rows = round(H / th)
    cols = round(W / tw)
    return dict(rows=rows, cols=cols, tile_h=th, tile_w=tw, H=H, W=W)


def main(path: str):
    img = load_img(path)
    x0, y0, x1, y1 = find_stage(img)
    print(f"window {img.shape[1]}x{img.shape[0]}  stage=({x0},{y0})-({x1},{y1}) "
          f"size {x1-x0}x{y1-y0}")
    stage = img[y0:y1, x0:x1]
    g = detect_grid(stage)
    print(f"grid: {g}")
    Image.fromarray(stage).save("/".join(path.split("/")[:-1]) + "/stage.png")
    print("saved stage.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cap99b.png")
