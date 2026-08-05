"""Top-level harness: solve the whole game 3 times in a row.

Flow per attempt:
  1. ensure we are on a board (click 开始游戏 on the title screen if needed)
  2. solve_level() until the board is cleared
  3. handle the level-complete / game-complete screen to advance/restart
  4. count a completion; repeat until N completed
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import actuator
import solve
import perception as perc

JOB_TMP = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp")
TARGET_COMPLETIONS = 3


def is_board(path: str) -> bool:
    """Heuristic: a real board has many high-variance (icon) tiles."""
    try:
        img = perc.load_img(path)
        grid = perc.get_grid(img)
        tiles = perc.extract_tiles(img, grid)
        labels = perc.classify(tiles)
        present = labels != (getattr(perc, "EMPTY", -1))
        return int(present.sum()) >= 6
    except Exception:
        return False


def fresh_cap(name="run_cap.png") -> str:
    p = os.path.join(JOB_TMP, name)
    actuator.capture(p)
    return p


def attempt_once(verbose=True) -> bool:
    """Clear one board (one level). Returns True if cleared."""
    return solve.solve_level(verbose=verbose)


def main():
    completions = 0
    fails = 0
    while completions < TARGET_COMPLETIONS and fails < 20:
        p = fresh_cap()
        if not is_board(p):
            print(f"[run] not on a board -> click 开始游戏")
            # yellow 开始游戏 button near center-bottom of the title screen
            actuator.click_xy(360, 360)
            time.sleep(1.0)
            continue
        ok = attempt_once()
        if ok:
            completions += 1
            print(f"[run] *** completion {completions}/{TARGET_COMPLETIONS} ***")
            time.sleep(1.5)
            # advance past a level-complete screen (click center)
            actuator.click_xy(360, 300)
            time.sleep(1.5)
        else:
            fails += 1
            print(f"[run] attempt failed (#{fails}); re-capturing")
            time.sleep(1.0)
    print(f"[run] DONE completions={completions} fails={fails}")
    return completions >= TARGET_COMPLETIONS


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
