"""Linux controller: capture frames and click tiles in the Chrome app window.

The game runs in a Chrome --app window on Xvfb display :99 (managed by
session.sh). xdotool XTest clicks are silently ignored by the Ruffle canvas,
so clicks go through the Chrome DevTools Protocol (Input.dispatchMouseEvent).

Coordinate model (everything in VIEWPORT pixels):
  * capture()   -> CDP Page.captureScreenshot (viewport only; excludes the
    browser chrome / flag-warning banner).
  * click_xy()  -> CDP Input.dispatchMouseEvent in the SAME viewport space.
Perception runs on the captured viewport image, so tile pixel coordinates map
directly to click coordinates with NO offset.
"""
from __future__ import annotations
import os
import subprocess
import time

import cdp

DISPLAY = os.environ.get("SOLVE_DISPLAY", ":99")
SESSION = os.path.join(os.path.dirname(__file__), "session.sh")


def get_wid() -> int:
    """Window id of the game window (via session.sh)."""
    out = subprocess.run(["bash", SESSION, "windowid"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[-1].strip())


def window_geometry(wid: int) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the window."""
    out = subprocess.run(["xdotool", "getwindowgeometry", "--shell", str(wid)],
                         capture_output=True, text=True,
                         env={**os.environ, "DISPLAY": DISPLAY}).stdout
    g = {"X": 0, "Y": 0, "WIDTH": 0, "HEIGHT": 0}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k in g:
                g[k] = int(v)
    return (g["X"], g["Y"], g["WIDTH"], g["HEIGHT"])


def click_xy(x: int, y: int, settle: float = 0.05) -> None:
    """Click at VIEWPORT coordinate (x, y) via CDP."""
    cdp.click(x, y, settle=settle)


def tile_center(row: int, col: int, grid: dict) -> tuple[int, int]:
    """Viewport pixel center of tile (row, col)."""
    x0, y0, tile = grid["x0"], grid["y0"], grid["tile"]
    return (x0 + col * tile + tile // 2, y0 + row * tile + tile // 2)


def click_tile(row: int, col: int, grid: dict) -> None:
    x, y = tile_center(row, col, grid)
    click_xy(x, y)


def capture(path: str) -> str:
    """Capture the viewport to `path` via CDP."""
    return cdp.capture(path)


if __name__ == "__main__":
    import sys
    wid = get_wid()
    print("wid", wid, "geom", window_geometry(wid))
    if len(sys.argv) > 1:
        p = capture(sys.argv[1])
        print("captured", p)
