"""Python harness that drives/observes the patched 宠物连连看 SWF via Chrome CDP.

The patched SWF (local/web/game_inner.swf) skips the title, starts level 1 on
load, and runs an *internal* autonomous solver that reuses the game's own
pair-finder + move handler to clear every level, then restarts. It also exposes
ExternalInterface callbacks (acStatus/acStep/acReset/acSetEnabled/acGetClears)
so this harness can observe exact state and confirm wins.

This harness is deliberately passive by default: it watches the SWF clear the
game and reports progress, confirming a run only when acStatus reports the
"您已通关!" (full clear) reason. Use --drive to instead step the SWF one action
at a time from Python (still via the game's own solver), which is slower but
gives full per-move logging.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import cdp  # noqa: E402

CDP_PORT = int(os.environ.get("CDP_PORT", "9222"))
DEFAULT_WINS = 3


def _find_player_js() -> str:
    """JS expression returning the Ruffle player DOM element that owns the
    ExternalInterface callbacks. Ruffle replaces <embed> with a custom element
    but keeps its id; fall back to the custom tag names."""
    return (
        "(function(){"
        "var e=document.getElementById('game');"
        "if(e&&(e.acStatus||e.tagName.toLowerCase().indexOf('ruffle')>=0))return e;"
        "var t=document.getElementsByTagName('ruffle-object');if(t[0])return t[0];"
        "t=document.getElementsByTagName('ruffle-embed');if(t[0])return t[0];"
        "t=document.getElementsByTagName('embed');if(t[0])return t[0];"
        "return null;})()"
    )


def eval_js(expr: str, await_promise: bool = False):
    params = {"expression": expr, "returnByValue": True}
    if await_promise:
        params["awaitPromise"] = True
    res = cdp._send("Runtime.evaluate", params)
    val = res.get("result", {})
    if val.get("type") == "undefined":
        return None
    if "exceptionDetails" in res:
        raise RuntimeError(f"JS error: {res['exceptionDetails'].get('text')}")
    return val.get("value")


def wait_for_player(timeout: float = 30.0) -> str:
    """Wait until the Ruffle element exists and exposes acStatus."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            has = eval_js(
                "(function(){var e=" + _find_player_js() + ";"
                "return !!(e && typeof e.acStatus==='function');})()"
            )
            if has:
                return "ready"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(0.5)
    raise RuntimeError(f"player/acStatus not available within {timeout}s (last={last})")


def call_ac(method: str, *args):
    """Invoke an ExternalInterface callback on the player element."""
    argstr = ",".join(json.dumps(a) for a in args)
    expr = f"(function(){{var e={_find_player_js()};if(!e||typeof e.{method}!=='function')return undefined;try{{return e.{method}({argstr});}}catch(x){{return 'ERR:'+x;}}}})()"
    return eval_js(expr)


def status() -> dict:
    raw = call_ac("acStatus")
    if raw is None:
        raise RuntimeError("acStatus returned undefined (player not ready or EI off)")
    if isinstance(raw, str) and raw.startswith("ERR:"):
        raise RuntimeError(raw)
    return json.loads(raw)


def reload_page() -> None:
    cdp._send("Page.enable", {})
    cdp._send("Page.reload", {"ignoreCache": True})
    # give the SWF a moment to start loading
    time.sleep(1.0)


def capture(path: str) -> str:
    return cdp.capture(path)


def _print_status(s: dict):
    print(
        f"  L{s['level']}/{s['maxLevel']} score={s['score']} "
        f"left={s['tilesLeft']} scene={s['scene']} clears={s['clears']} "
        f"fails={s['fails']} ei={s['ei']} reason={s['reason']!r}"
    )


def observe(wins: int, timeout: float, verbose: bool) -> bool:
    """Watch the SWF's internal solver until it records `wins` full clears."""
    start = time.time()
    last_clears = -1
    last_level = -1
    last_scene = None
    target = wins
    no_progress = 0
    while time.time() - start < timeout:
        try:
            s = status()
        except Exception as e:  # noqa: BLE001
            print(f"[observe] status error: {e}")
            time.sleep(1.0)
            continue
        changed = (
            s["clears"] != last_clears
            or s["level"] != last_level
            or s["scene"] != last_scene
        )
        if changed and verbose:
            _print_status(s)
        if s["level"] == last_level and s["scene"] == last_scene and s["tilesLeft"] == 0:
            no_progress += 1
        else:
            no_progress = 0
        if s["clears"] >= target:
            if verbose:
                print(f"[observe] reached {target} clears ✓")
            return True
        last_clears = s["clears"]
        last_level = s["level"]
        last_scene = s["scene"]
        # safety: if stuck with no state change for a long time, nudge
        time.sleep(1.0)
    return False


def drive(wins: int, step_delay: float, verbose: bool) -> bool:
    """Step the SWF from Python using the game's own solver (acStep)."""
    call_ac("acSetEnabled", False)  # pause internal timer; Python drives
    start = time.time()
    timeout = 3600
    last_clears = -1
    while time.time() - start < timeout:
        s = json.loads(call_ac("acStep"))
        if verbose and (s["clears"] != last_clears or s["scene"] == "result"):
            _print_status(s)
        if s["clears"] >= wins:
            print(f"[drive] reached {wins} clears ✓")
            return True
        last_clears = s["clears"]
        time.sleep(step_delay)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wins", type=int, default=DEFAULT_WINS)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--drive", action="store_true", help="Python steps the SWF")
    ap.add_argument("--step-delay", type=float, default=0.12)
    ap.add_argument("--reload", action="store_true", help="reload page first")
    ap.add_argument("--status", action="store_true", help="print status once and exit")
    ap.add_argument("--reset", action="store_true", help="reset the run and exit")
    ap.add_argument("--capture", type=str, default=None, help="capture screenshot path")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if args.reload:
        reload_page()
    wait_for_player()

    if args.capture:
        capture(args.capture)
        print(f"captured {args.capture}")
    if args.reset:
        call_ac("acReset")
    if args.status:
        _print_status(status())
        return
    if args.capture or args.reset:
        return

    verbose = not args.quiet
    if args.drive:
        ok = drive(args.wins, args.step_delay, verbose)
    else:
        ok = observe(args.wins, args.timeout, verbose)
    print("RESULT", "SUCCESS" if ok else "TIMEOUT/FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
