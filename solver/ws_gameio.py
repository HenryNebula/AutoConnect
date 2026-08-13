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
from PIL import Image

import gameio  # frozen GameIO contract (Branch B)


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
        return np.asarray(pil, dtype=np.uint8)

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
        self._request({"type": "overlay", "boxes": list(boxes or [])})

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
        Raises on browser error, timeout, or disconnect.
        """
        if self._closed:
            raise RuntimeError("WSGameIO closed (browser disconnected)")

        mid = next(self._id_counter)
        msg = {**msg, "id": mid}
        fut: Future = Future()
        with self._pending_lock:
            self._pending[mid] = fut
        # Push to the outgoing queue; the loop-thread pump turns this into
        # ``await ws.send_text(json.dumps(msg))``.
        self._outgoing.put(msg)

        try:
            return fut.result(timeout=timeout or self._timeout)
        except _FutureTimeout:
            raise TimeoutError(
                f"WSGameIO request {msg.get('type')!r} (id={mid}) timed out "
                f"after {timeout or self._timeout:.0f}s with no browser reply"
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
        only. Idempotent."""
        self._closed = True
        with self._pending_lock:
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
