"""``WSGameIO`` -- a ``GameIO`` that round-trips the brain's synchronous I/O
over a WebSocket to a browser frontend (see ``server.py`` + ``local/web/app.js``).

Threading model
---------------
The Bot (the "brain") runs in a worker thread and makes **blocking** calls
(``capture``, ``remove_pair``, ...). The WebSocket itself lives on the asyncio
event loop of the server thread. The bridge between the two is:

* an outgoing ``queue.Queue`` (thread-safe): the worker thread pushes request
  messages; an asyncio task (``outgoing_pump``) drains it from the loop thread
  via ``loop.run_in_executor(None, queue.get)`` and ``await ws.send_text(...)``.
* a ``dict[id, concurrent.futures.Future]`` of pending requests: the worker
  creates a Future per call and blocks on ``future.result(timeout=...)``; the
  server's WS-receive coroutine resolves the matching Future by id via
  ``future.set_result(...)`` / ``set_exception(...)`` (both thread-safe on
  ``concurrent.futures.Future``).
* a ``threading.Event`` for the one-shot browser ``{"type":"ready"}`` signal,
  which ``wait_ready`` blocks on.

A per-call timeout (default 30 s) guarantees a dropped browser can never hang
the brain forever; on disconnect, ``shutdown()`` fails every still-pending
Future and pushes a sentinel through the queue so the pump exits cleanly.
"""
from __future__ import annotations

import base64
import io
import itertools
import json
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as _FutureTimeout
from typing import Any

import numpy as np
import cv2
from PIL import Image

import gameio  # frozen GameIO contract (Branch B)

# Canonical capture width. vision.detect_grid's blob-size thresholds
# (area<6000, w/h<70) are absolute pixels, tuned for a ~720px-wide board. A
# full-size browser renders the Ruffle canvas 2-3x larger, so captures are
# normalized to this width; the scale factor maps brain coords back onto the
# live canvas for the overlay (see capture / draw_overlay).
CANON_CAPTURE_W = 720


# Request methods the patched SWF + Ruffle expose once "ready" is signalled.
# The browser only sends ``ready`` after ``typeof acStatus === 'function'``; at
# that point all of these are registered (the SWF registers them together in
# one ExternalInterface block), so ``has_ei`` is a static allow-list rather than
# a per-call round-trip.
_EI_METHODS = (
    "acStatus",
    "acRemovePair",
    "acReshuffle",
    "acAdvance",
    "acReset",
    "acSetEnabled",
)

#: Default per-call timeout (seconds). Generous: the SWF can stall a few
#: seconds during level transitions, but a dropped browser must not hang the
#: brain indefinitely.
DEFAULT_TIMEOUT = 30.0


class WSGameIO(gameio.GameIO):
    """``GameIO`` adapter that drives a browser game over a WebSocket.

    Construct one per WS connection from the server's event-loop thread, then
    call the synchronous ``GameIO`` methods from the brain's worker thread.
    """

    def __init__(self, ws, loop, timeout: float = DEFAULT_TIMEOUT):
        # ``ws`` is the Starlette/FastAPI WebSocket; we only use send_text from
        # the loop thread (inside ``outgoing_pump``), never directly from the
        # worker thread. ``loop`` is the running asyncio loop.
        self._ws = ws
        self._loop = loop
        self._timeout = timeout

        self._outgoing: "queue.Queue[dict | None]" = queue.Queue()
        self._pending: dict[int, Future] = {}
        self._pending_lock = threading.Lock()
        self._id_counter = itertools.count(1)

        # Signalled once when the browser sends ``{"type":"ready"}``.
        self._ready = threading.Event()

        # Set by ``shutdown()`` so later ``_request`` calls fail fast instead
        # of enqueueing onto a dead socket.
        self._closed = False

        # Capture -> 720-wide scale factor (brain-space -> canvas-space), so the
        # bbox overlay aligns with tiles on the browser's full-size canvas.
        self._cap_scale = 1.0

    # ---- public GameIO surface -------------------------------------------

    def capture(self) -> np.ndarray:
        """Grab the Ruffle canvas as an RGB HxWx3 uint8 ndarray.

        The browser replies with ``{"w","h","data":"data:image/png;base64,..."}``;
        we decode the PNG into the same bright, veil-free array shape the brain
        gets from ``Page.captureScreenshot`` today (the dimming veil is a DOM
        layer above the canvas, never on the canvas bitmap).
        """
        res = self._request({"type": "capture"})
        data_url = res.get("data", "") if isinstance(res, dict) else ""
        if not data_url.startswith("data:image/png;base64,"):
            raise RuntimeError(
                f"capture: expected a PNG data URL, got {type(res).__name__}"
            )
        buf = base64.b64decode(data_url.split(",", 1)[1])
        pil = Image.open(io.BytesIO(buf)).convert("RGB")
        arr = np.asarray(pil, dtype=np.uint8)
        if not getattr(self, "_logged_cap", False):
            print(f"[ws_gameio] raw capture {arr.shape} mean={arr.mean():.0f} "
                  f"(bright board ~170, blank ~0)", flush=True)
            self._logged_cap = True
        # Normalize to the canonical width the CV was tuned on: detect_grid's
        # blob thresholds are absolute pixels, so a full-size browser canvas
        # (tiles 2-3x larger) would be rejected. Remember the scale so draw_overlay
        # can map brain coords back onto the live canvas.
        oh, ow = arr.shape[:2]
        ch = max(1, round(oh * CANON_CAPTURE_W / ow))
        arr = cv2.resize(arr, (CANON_CAPTURE_W, ch), interpolation=cv2.INTER_AREA)
        self._cap_scale = ow / float(CANON_CAPTURE_W)
        return arr

    def status(self) -> dict:
        return self._request({"type": "status"})

    def remove_pair(self, r1: int, c1: int, r2: int, c2: int) -> int:
        res = self._request(
            {"type": "removePair", "r1": int(r1), "c1": int(c1),
             "r2": int(r2), "c2": int(c2)}
        )
        # Browser returns the final post-removal tilesLeft (an int).
        return int(res)

    def reshuffle(self) -> dict:
        return self._request({"type": "reshuffle"})

    def advance(self) -> dict:
        return self._request({"type": "advance"})

    def set_enabled(self, v: bool) -> None:
        self._request({"type": "setEnabled", "v": bool(v)})

    def reset(self) -> None:
        self._request({"type": "reset"})

    def draw_overlay(self, boxes: list[dict]) -> None:
        # Brain coords are in the 720-wide capture space; scale back onto the
        # browser's full-size canvas so the overlay lines up with the tiles.
        s = self._cap_scale
        out = [{**b, "x": b["x"] * s, "y": b["y"] * s,
                "w": b["w"] * s, "h": b["h"] * s} for b in (boxes or [])]
        self._request({"type": "overlay", "boxes": out})

    def hide_overlay(self) -> None:
        self._request({"type": "hideOverlay"})

    def has_ei(self, method: str) -> bool:
        # After "ready", the full EI surface is registered (the SWF registers
        # every callback in one ExternalInterface block). A live probe would
        # need an extra round-trip per call and the brain only ever asks about
        # the methods below.
        return method in _EI_METHODS

    def wait_ready(self, timeout: float = 40.0) -> bool:
        return self._ready.wait(timeout=timeout)

    # ---- bridge: worker-thread side --------------------------------------

    def _request(self, msg: dict, timeout: float | None = None) -> Any:
        """Send ``msg`` to the browser and block on its reply.

        Adds the next monotonic id, registers a Future, enqueues, and waits.
        Raises on browser error, timeout, or disconnect. The check-then-add
        against ``_closed`` happens under ``_pending_lock`` so a ``shutdown``
        racing us from another (loop) thread can never leave a Future in
        ``_pending`` that nobody will ever resolve (which would force the
        worker to wait out the full per-call timeout).
        """
        if timeout is None:
            timeout = self._timeout
        mid = next(self._id_counter)
        msg = {**msg, "id": mid}
        fut: Future = Future()
        with self._pending_lock:
            if self._closed:
                raise RuntimeError("WSGameIO closed (browser disconnected)")
            self._pending[mid] = fut
        # Push to the outgoing queue; the loop-thread pump turns this into
        # ``await ws.send_text(json.dumps(msg))``.
        self._outgoing.put(msg)

        try:
            return fut.result(timeout=timeout)
        except _FutureTimeout:
            raise TimeoutError(
                f"WSGameIO request {msg.get('type')!r} (id={mid}) timed out "
                f"after {timeout:.0f}s with no browser reply"
            )
        finally:
            with self._pending_lock:
                self._pending.pop(mid, None)

    # ---- bridge: loop-thread side (called only from server.py) -----------

    def on_reply(self, msg: dict) -> None:
        """Resolve the Future for a ``{"id","ok",...}`` reply. Loop-thread only."""
        mid = msg.get("id")
        if not isinstance(mid, int):
            return  # unsolicited (handled by on_event) or malformed
        with self._pending_lock:
            fut = self._pending.get(mid)
        if fut is None or fut.done():
            return  # late reply after timeout/disconnect -- drop silently
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            err = msg.get("error", "browser error")
            fut.set_exception(RuntimeError(str(err)))

    def on_event(self, msg: dict) -> None:
        """Handle an unsolicited browser message (no id). Loop-thread only."""
        if msg.get("type") == "ready":
            self._ready.set()

    def shutdown(self) -> None:
        """Fail all pending calls and release the outgoing pump. Loop-thread
        only. Idempotent. ``_closed`` is set under ``_pending_lock`` so a
        concurrently-running ``_request`` either sees ``_closed`` and aborts
        or has already added its Future (which we then fail) -- never both."""
        with self._pending_lock:
            self._closed = True
            futs = list(self._pending.values())
            self._pending.clear()
        for f in futs:
            if not f.done():
                f.set_exception(RuntimeError("browser disconnected"))
        # Sentinel so ``outgoing_pump`` exits its blocking ``queue.get``.
        self._outgoing.put(None)

    # ---- outgoing pump: drain queue -> ws.send_text (loop-thread) --------

    async def outgoing_pump(self) -> None:
        """Drain ``_outgoing`` and forward each message over the WebSocket.

        Runs as an asyncio task on the server's event loop. ``queue.get`` is
        blocking, so it is dispatched to the default executor; each entry is
        either a request dict or the ``None`` shutdown sentinel.
        """
        while True:
            msg = await self._loop.run_in_executor(None, self._outgoing.get)
            if msg is None:
                return  # shutdown()
            try:
                await self._ws.send_text(json.dumps(msg))
            except Exception as e:  # noqa: BLE001 -- socket gone; bail out
                print(f"[ws_gameio] send failed: {e!r}", flush=True)
                return
