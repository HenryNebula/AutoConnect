"""FastAPI WebSocket backend for the CV solver.

Serves the game frontend (``local/web``) over HTTP and exposes a ``/ws``
endpoint that drives the existing Bot brain from any browser on the LAN.
On connect: build a ``WSGameIO`` (the bridge to the browser), spin up the
brain in a worker thread, and pump browser replies back into it. One active
session at a time; a second connecting client is told ``{"type":"busy"}``.

Run with either:
    uvicorn solver.server:app --host 0.0.0.0 --port 8765
    python solver/server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

# Make sibling modules (bot, ws_gameio, gameio) importable regardless of
# whether this file is run as ``solver.server`` or ``python solver/server.py``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from ws_gameio import WSGameIO
from bot import Bot

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "local" / "web"

app = FastAPI(title="AutoConnect CV solver backend")

# One active session at a time. Acquired on WS connect, released on disconnect
# (after the brain thread has joined or timed out).
_session_lock = threading.Lock()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    # Reject a second concurrent client (keep it simple -- no takeover).
    if not _session_lock.acquire(blocking=False):
        try:
            await ws.send_text(json.dumps({"type": "busy"}))
        finally:
            await ws.close()
        print("[server] rejected 2nd WS (session busy)", flush=True)
        return

    loop = asyncio.get_running_loop()
    wsio = WSGameIO(ws, loop)
    pump_task = asyncio.create_task(wsio.outgoing_pump())

    max_level = int(os.environ.get("AC_MAX_LEVEL", "13"))
    # ``Bot(..., io=wsio)`` is the post-merge signature (Branch B adds the
    # ``io=`` kwarg). On this branch Bot takes no ``io=``; we still write the
    # call against the post-merge contract and verify locally via a stub.
    bot = Bot(verbose=True, io=wsio)
    bot.demo = True  # push per-move bbox overlays through io.draw_overlay

    worker = threading.Thread(
        target=_run_brain,
        args=(bot, wsio, max_level),
        name="bot-brain",
        daemon=True,
    )
    worker.start()

    try:
        # Receive loop lives on the event-loop thread; it resolves the Futures
        # the brain is blocking on. Runs until the browser disconnects.
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[server] dropping non-JSON frame: {raw[:80]!r}", flush=True)
                continue
            if not isinstance(msg, dict):
                continue
            # Replies carry an id and are routed to the matching Future.
            # ``ready`` is an unsolicited event with no id.
            if "id" in msg:
                wsio.on_reply(msg)
            else:
                wsio.on_event(msg)
    except WebSocketDisconnect:
        print("[server] browser disconnected", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[server] receive loop error: {e!r}", flush=True)
    finally:
        # Wake any blocked brain call, stop the pump, drain the brain thread.
        wsio.shutdown()
        try:
            await asyncio.wait_for(pump_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pump_task.cancel()
        worker.join(timeout=5.0)
        _session_lock.release()
        print("[server] session torn down", flush=True)


def _run_brain(bot: Bot, wsio: WSGameIO, max_level: int) -> None:
    """Worker-thread entry: wait for the browser, then play. Never raises --
    a brain crash is logged, never propagated (the thread is daemon)."""
    try:
        if not wsio.wait_ready(timeout=40.0):
            print("[server] browser did not signal ready within 40s -- aborting",
                  flush=True)
            return
        print("[server] browser ready; starting brain", flush=True)
        # ``restart_between`` is accepted today and post-merge; in browser mode
        # the renderer is owned by the page, so a per-run Chrome/Xvfb restart
        # never applies.
        bot.play_game(runs=1, max_level=max_level, restart_between=False)
        print("[server] brain finished play_game", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[server] brain crashed: {e!r}", flush=True)


# Static mount registered AFTER /ws so the catch-all doesn't shadow it.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("AC_HOST", "0.0.0.0")
    port = int(os.environ.get("AC_PORT", "8765"))
    print(f"[server] serving {WEB_DIR} on http://{host}:{port}/  (WS at /ws)",
          flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
