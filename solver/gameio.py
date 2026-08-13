"""GameIO: the boundary between the bot's brain and the game it is playing.

The brain (``solver/bot.py``: perceive via CV, classify, solve with ``conn``,
act) is pure compute + control logic. Every interaction with the actual game --
capturing a frame, reading state, executing a removal, reshuffling, advancing a
level, drawing the demo overlay -- goes through a ``GameIO``. Two
implementations:

- ``CDPGameIO`` (here): drives the local Chrome over the Chrome DevTools
  Protocol -- the original headless/headed rig. Preserves today's behaviour
  exactly (this is a behaviour-preserving extraction of bot.py's former
  ``_ev``/``_player_call``/``cdp.*`` calls).
- ``WSGameIO`` (``solver/ws_gameio.py``): round-trips to a browser frontend
  over WebSocket, so the same brain runs as a LAN backend.

The brain never imports ``cdp`` for game I/O; it calls ``self.io.<method>``.
This is what makes the brain runnable unchanged against either a local Chrome
or a remote browser. The only ``cdp`` use left in bot.py is the lossy
click-cell fallback (reached only when the acRemovePair EI is absent, which
never happens with the patched SWF).

Game state is read/returned as parsed ``acStatus`` dicts. ``remove_pair`` does
the SWF's 1-based (x=col, y=row) conversion internally and polls ``acStatus``
for up to ``verify_wait`` so the brain's acceptance check is a single call.
``reshuffle``/``advance`` use the airtight no-``cunufi`` EI (acReshuffle /
acAdvance) -- never acPlayOne/acStep, which can reach the builtin pair-finder.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import cdp
import vision


def _embed_call_expr(method: str, args=()) -> str:
    """JS expression that invokes an ExternalInterface method on the ruffle-embed
    element, returning undefined if unregistered or 'ERR:<x>' on throw."""
    a = ",".join(json.dumps(x) for x in args)
    return ("(function(){var e=document.getElementsByTagName('ruffle-embed')[0];"
            "if(!e||typeof e.%s!=='function')return undefined;"
            "try{return e.%s(%s);}catch(x){return 'ERR:'+x;}})()" % (method, method, a))


def _parse_status(s):
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:  # noqa: BLE001
            return {}
    return {}


class GameIO:
    """Abstract game I/O surface used by the Bot brain. See module docstring."""

    def capture(self) -> np.ndarray:
        raise NotImplementedError

    def status(self) -> dict:
        raise NotImplementedError

    def remove_pair(self, r1, c1, r2, c2) -> int:
        """Execute the caller-chosen removal and return the post-removal
        tilesLeft. 0-based (r, c) in; the SWF's 1-based conversion is internal."""
        raise NotImplementedError

    def reshuffle(self) -> dict:
        raise NotImplementedError

    def advance(self) -> dict:
        raise NotImplementedError

    def set_enabled(self, v: bool) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def draw_overlay(self, boxes: list) -> None:
        """Cosmetic: render {x,y,w,h,color,label?} boxes. Default no-op."""
        pass

    def hide_overlay(self) -> None:
        pass

    def has_ei(self, method: str) -> bool:
        raise NotImplementedError

    def wait_ready(self, timeout: float = 40.0) -> bool:
        raise NotImplementedError


class CDPGameIO(GameIO):
    """GameIO backed by the local Chrome over the Chrome DevTools Protocol."""

    def __init__(self, verify_wait: float = 0.40):
        self.verify_wait = verify_wait

    # -- raw CDP helpers (the former bot.py module-level _ev / _player_call) --
    @staticmethod
    def _ev(expr):
        r = cdp._send("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return r.get("result", {}).get("value")

    def _player_call(self, method, *args):
        return self._ev(_embed_call_expr(method, args))

    # -- GameIO --
    def capture(self) -> np.ndarray:
        p = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "_bot.png")
        cdp.capture(p)
        return vision.load_img(p)

    def status(self) -> dict:
        return _parse_status(self._player_call("acStatus"))

    def remove_pair(self, r1, c1, r2, c2) -> int:
        # SWF board coords are 1-based x=col, y=row (myicon_x{X}y{Y}).
        before = self.status().get("tilesLeft", 99)
        self._player_call("acRemovePair", c1 + 1, r1 + 1, c2 + 1, r2 + 1)
        deadline = time.time() + self.verify_wait
        while time.time() < deadline:
            cur = self.status().get("tilesLeft", 99)
            if cur < before:                 # the removal landed
                return cur
            time.sleep(0.03)
        return self.status().get("tilesLeft", 99)

    def reshuffle(self) -> dict:
        # acReshuffle -> createNewMap(null) + acStatus (no builtin pair-finder).
        return _parse_status(self._player_call("acReshuffle"))

    def advance(self) -> dict:
        # acAdvance -> acHandleResult() + acStatus (dismiss result overlay only).
        return _parse_status(self._player_call("acAdvance"))

    def set_enabled(self, v: bool) -> None:
        self._player_call("acSetEnabled", bool(v))

    def reset(self) -> None:
        self._player_call("acReset")

    def has_ei(self, method: str) -> bool:
        expr = ("(function(){var e=document.getElementsByTagName('ruffle-embed')[0];"
                "return !!(e && typeof e.%s==='function');})()" % method)
        return self._ev(expr) is True

    def wait_ready(self, timeout: float = 40.0) -> bool:
        t = time.time()
        while time.time() - t < timeout:
            if self.has_ei("acStatus"):
                return True
            time.sleep(0.4)
        return False

    def draw_overlay(self, boxes: list) -> None:
        # Inject/update #ac-demo-overlay on document.body. position:fixed at max
        # z-index so it paints above Ruffle's shadow-DOM dimming veil. Cosmetic;
        # never raise.
        js = """(function(boxes){
var host=document.body;
var ov=document.getElementById('ac-demo-overlay');
if(!ov){ov=document.createElement('div');ov.id='ac-demo-overlay';
ov.style.cssText='position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483647;margin:0;padding:0;';
host.appendChild(ov);}
ov.innerHTML='';
for(var i=0;i<boxes.length;i++){
var b=boxes[i],d=document.createElement('div');
d.style.cssText='position:absolute;box-sizing:border-box;left:'+b.x.toFixed(1)+'px;top:'+b.y.toFixed(1)+'px;width:'+b.w.toFixed(1)+'px;height:'+b.h.toFixed(1)+'px;border:3px solid '+b.color+';box-shadow:0 0 6px '+b.color+',0 0 6px '+b.color+';background:transparent;';
if(b.label){var lbl=document.createElement('div');lbl.textContent=b.label;
lbl.style.cssText='position:absolute;top:-18px;left:0;color:'+b.color+';font:bold 14px monospace;text-shadow:0 0 3px #000,0 0 3px #000,0 0 3px #000;';
d.appendChild(lbl);}
ov.appendChild(d);}
})(""" + json.dumps(boxes) + ")"
        try:
            cdp.eval_js(js)
        except Exception:  # noqa: BLE001
            pass

    def hide_overlay(self) -> None:
        try:
            cdp.eval_js("(function(){var o=document.getElementById('ac-demo-overlay');"
                        "if(o)o.innerHTML='';})()")
        except Exception:  # noqa: BLE001
            pass
