"""Click into the game via Chrome DevTools Protocol (Input.dispatchMouseEvent).

xdotool XTest clicks are silently ignored by the Ruffle canvas, but CDP
dispatches TRUSTED mouse events straight into the page viewport, which the
canvas receives. Coordinates are CSS pixels relative to the 720x560 viewport,
i.e. identical to window-relative pixel coordinates.
"""
from __future__ import annotations
import json
import os
import time
import urllib.request

# Configurable so multiple game sessions can run isolated (e.g. a second agent's
# harvest on 9222 while tests use 9223). Set CDP_PORT in the environment.
CDP_PORT = int(os.environ.get("CDP_PORT", "9222"))
# Chrome/Ruffle can go unresponsive for well over 5 s during level transitions,
# and the websocket can drop across page reloads. So recv uses a generous
# timeout and _send retries with a reconnect rather than crashing on one stall.
RECV_TIMEOUT = 30.0
MAX_RETRIES = 3


def _page_ws_url() -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=5) as r:
        targets = json.load(r)
    for t in targets:
        if t.get("type") == "page" and "127.0.0.1" in t.get("url", ""):
            return t["webSocketDebuggerUrl"]
    # fall back to any page target
    for t in targets:
        if t.get("type") == "page":
            return t["webSocketDebuggerUrl"]
    raise RuntimeError("no page target found in CDP")


_ws = None
_mid = 0


def connect():
    global _ws
    import websocket
    if _ws is None:
        _ws = websocket.create_connection(_page_ws_url(), timeout=RECV_TIMEOUT)
    return _ws


def _reset_connection():
    """Drop the current websocket so the next connect() re-fetches the page
    target (its URL can change across navigations/reloads)."""
    global _ws
    if _ws is not None:
        try:
            _ws.close()
        except Exception:  # noqa: BLE001
            pass
    _ws = None


def _send(method: str, params: dict | None = None):
    global _mid
    import websocket
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            connect()
            _mid += 1
            my_id = _mid
            _ws.send(json.dumps({"id": my_id, "method": method, "params": params or {}}))
            while True:
                resp = json.loads(_ws.recv())
                if resp.get("id") == my_id:
                    if "error" in resp:
                        # Legitimate method error (e.g. a JS exception) -- don't retry.
                        raise RuntimeError(f"CDP error on {method}: {resp['error']}")
                    return resp.get("result", {})
        except (websocket.WebSocketTimeoutException, websocket.WebSocketException, OSError) as e:
            # Transient: the renderer stalled mid-transition, or the socket
            # dropped on a page reload. Reconnect and retry.
            last_err = e
            _reset_connection()
            time.sleep(0.5 * (attempt + 1))
            continue
    raise RuntimeError(f"CDP {method} failed after {MAX_RETRIES} retries: {last_err}")


def move(x: float, y: float):
    _send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})


def click(x: float, y: float, settle: float = 0.05):
    common = {"x": x, "y": y, "button": "left", "clickCount": 1, "pointerType": "mouse"}
    move(x, y)
    _send("Input.dispatchMouseEvent", {**common, "type": "mousePressed"})
    time.sleep(settle)
    _send("Input.dispatchMouseEvent", {**common, "type": "mouseReleased"})


def eval_js(expr: str):
    return _send("Runtime.evaluate", {"expression": expr, "returnByValue": True}).get("result", {}).get("value")


def capture(path: str, fmt: str = "png") -> str:
    """Capture the viewport (page content only, no browser chrome/banner)."""
    import base64
    res = _send("Page.captureScreenshot", {"format": fmt})
    with open(path, "wb") as f:
        f.write(base64.b64decode(res["data"]))
    return path


def viewport_size() -> tuple[int, int]:
    return (int(eval_js("window.innerWidth")), int(eval_js("window.innerHeight")))


if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image
    import subprocess
    cap = "/tmp/cdp_cap.png"

    def snap():
        subprocess.run(["bash", "solver/session.sh", "capture", cap], check=True)
        return np.array(Image.open(cap).convert("L")).astype(int)

    before = snap()
    x, y = (float(a) for a in sys.argv[1:3])
    print("CDP click", (x, y))
    click(x, y)
    time.sleep(2.0)
    after = snap()
    d = np.abs(after - before)
    print("diff mean=%.3f max=%d frac>10=%.4f" % (d.mean(), d.max(), (d > 10).mean()))
