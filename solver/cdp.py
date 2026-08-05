"""Click into the game via Chrome DevTools Protocol (Input.dispatchMouseEvent).

xdotool XTest clicks are silently ignored by the Ruffle canvas, but CDP
dispatches TRUSTED mouse events straight into the page viewport, which the
canvas receives. Coordinates are CSS pixels relative to the 720x560 viewport,
i.e. identical to window-relative pixel coordinates.
"""
from __future__ import annotations
import json
import time
import urllib.request

CDP_PORT = 9222


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
        _ws = websocket.create_connection(_page_ws_url(), timeout=5)
    return _ws


def _send(method: str, params: dict | None = None):
    global _mid
    connect()
    _mid += 1
    my_id = _mid
    _ws.send(json.dumps({"id": my_id, "method": method, "params": params or {}}))
    while True:
        resp = json.loads(_ws.recv())
        if resp.get("id") == my_id:
            if "error" in resp:
                raise RuntimeError(f"CDP error on {method}: {resp['error']}")
            return resp.get("result", {})


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
