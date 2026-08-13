/*
 * AutoConnect browser client.
 *
 * Connects to the FastAPI backend's /ws endpoint, drives the Ruffle game from
 * the brain's requests (canvas capture for perception, acRemovePair /
 * acReshuffle / acAdvance for control), and draws the bbox overlay the brain
 * pushes for the visual demo. Plain script (no module / no build step); loaded
 * via <script src="app.js?v=1" defer> in index.html.
 *
 * Protocol (frozen):
 *   brain -> browser request : {"id":int,"type":<capture|status|removePair|
 *                                          reshuffle|advance|setEnabled|reset|
 *                                          overlay|hideOverlay>, ...}
 *   browser -> brain reply   : {"id":int,"ok":true,"result":<...>}
 *                                | {"id":int,"ok":false,"error":"..."}
 *   browser -> brain event   : {"type":"ready"}  (once, after acStatus exists)
 *
 * Unsolicited server -> browser: {"type":"busy"} (2nd client; we just log it).
 */
(function () {
  'use strict';

  var ws = null;
  var reconnectDelayMs = 1000;
  var MAX_RECONNECT_MS = 10000;

  // Overlay canvas, created lazily on first "overlay" request, sized to match
  // the Ruffle canvas's backing store and CSS-positioned exactly over it so
  // box coords (in captured-canvas pixel space) line up on screen.
  var overlayCanvas = null;
  var overlayCtx = null;
  var overlaySyncTimer = null;

  // ---- DOM helpers --------------------------------------------------------

  function getEmbed() {
    // There is exactly one <embed> upgraded by Ruffle into a ruffle-embed.
    return document.getElementsByTagName('ruffle-embed')[0]
        || document.getElementById('game');
  }

  function getRuffleCanvas() {
    var e = getEmbed();
    if (!e || !e.shadowRoot) return null;
    return e.shadowRoot.querySelector('canvas');
  }

  // ---- WS plumbing --------------------------------------------------------

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  function replyOk(id, result) { send({ id: id, ok: true, result: result }); }
  function replyErr(id, err)   { send({ id: id, ok: false, error: String(err) }); }

  function connect() {
    var host = location.hostname || 'localhost';
    var port = location.port || '8765';
    var url = 'ws://' + host + ':' + port + '/ws';
    console.log('[ac] connecting ' + url);
    try {
      ws = new WebSocket(url);
    } catch (e) {
      console.warn('[ac] WebSocket construction failed', e);
      scheduleReconnect();
      return;
    }
    ws.onopen = function () {
      console.log('[ac] WS connected');
      reconnectDelayMs = 1000;
      waitReadyAndSignal();
    };
    ws.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      onMessage(msg);
    };
    ws.onclose = function () {
      console.log('[ac] WS closed');
      ws = null;
      scheduleReconnect();
    };
    ws.onerror = function (e) {
      console.warn('[ac] WS error', e);
    };
  }

  function scheduleReconnect() {
    setTimeout(connect, reconnectDelayMs);
    if (reconnectDelayMs < MAX_RECONNECT_MS) reconnectDelayMs *= 2;
  }

  // ---- unsolicited + request dispatch ------------------------------------

  function onMessage(msg) {
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'busy') {
      console.warn('[ac] server reports session busy (another client is driving)');
      return;
    }
    if (msg.id === undefined || msg.id === null) return;  // unsolicited other
    // Dispatch; each handler is responsible for replying exactly once.
    try {
      switch (msg.type) {
        case 'capture':     return handleCapture(msg.id);
        case 'status':      return replyOk(msg.id, JSON.parse(getEmbed().acStatus()));
        case 'reshuffle':   return replyOk(msg.id, JSON.parse(getEmbed().acReshuffle()));
        case 'advance':     return replyOk(msg.id, JSON.parse(getEmbed().acAdvance()));
        case 'reset':       getEmbed().acReset();               return replyOk(msg.id, null);
        case 'setEnabled':  getEmbed().acSetEnabled(!!msg.v);   return replyOk(msg.id, null);
        case 'removePair':  return handleRemovePair(msg);
        case 'overlay':     return handleOverlay(msg.id, msg.boxes);
        case 'hideOverlay': clearOverlay();                     return replyOk(msg.id, null);
        default:            return replyErr(msg.id, 'unknown type: ' + msg.type);
      }
    } catch (e) {
      replyErr(msg.id, (e && e.message) || String(e));
    }
  }

  // ---- capture ------------------------------------------------------------
  // The dimming veil is a DOM layer ABOVE the canvas, not painted onto its
  // bitmap, so cv.toDataURL returns the bright board the brain expects.

  function handleCapture(id) {
    var cv = getRuffleCanvas();
    if (!cv) throw new Error('ruffle canvas not found');
    var data = cv.toDataURL('image/png');
    replyOk(id, { w: cv.width, h: cv.height, data: data });
  }

  // ---- removePair ---------------------------------------------------------
  // Brain sends 0-based r,c. The SWF wants 1-based x=col+1, y=row+1. After the
  // call we locally poll acStatus().tilesLeft until it drops below the
  // pre-call value (or ~0.5 s elapses) and return that final count.

  function handleRemovePair(msg) {
    var e = getEmbed();
    var before = readTilesLeft(e);
    e.acRemovePair(msg.c1 + 1, msg.r1 + 1, msg.c2 + 1, msg.r2 + 1);
    pollTilesDrop(e, before, 500, function (final) {
      replyOk(msg.id, final);
    });
  }

  function readTilesLeft(e) {
    try { return JSON.parse(e.acStatus()).tilesLeft; }
    catch (x) { return Number.POSITIVE_INFINITY; }
  }

  function pollTilesDrop(e, before, budgetMs, cb) {
    var t0 = Date.now();
    (function check() {
      var now = readTilesLeft(e);
      if (now < before) return cb(now);
      if (Date.now() - t0 > budgetMs) return cb(now);
      setTimeout(check, 25);
    })();
  }

  // ---- ready handshake ----------------------------------------------------
  // On WS open, poll until EI is registered, then send the one-shot ready.

  function waitReadyAndSignal() {
    var t0 = Date.now();
    (function check() {
      var e = getEmbed();
      if (e && typeof e.acStatus === 'function') {
        console.log('[ac] acStatus ready -- signalling brain');
        send({ type: 'ready' });
        return;
      }
      if (Date.now() - t0 > 60000) {
        console.warn('[ac] gave up waiting for acStatus after 60s');
        return;
      }
      setTimeout(check, 200);
    })();
  }

  // ---- overlay canvas -----------------------------------------------------

  function ensureOverlay() {
    if (overlayCanvas) return;
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.id = 'ac-brain-overlay';
    overlayCanvas.style.position = 'fixed';
    overlayCanvas.style.left = '0';
    overlayCanvas.style.top = '0';
    overlayCanvas.style.pointerEvents = 'none';
    overlayCanvas.style.background = 'transparent';
    overlayCanvas.style.zIndex = '2147483647';  // above Ruffle's shadow veil
    document.body.appendChild(overlayCanvas);
    overlayCtx = overlayCanvas.getContext('2d');
    // Track Ruffle canvas resizes (letterboxing / window resize) periodically.
    overlaySyncTimer = setInterval(syncOverlaySize, 250);
  }

  function syncOverlaySize() {
    if (!overlayCanvas) return;
    var cv = getRuffleCanvas();
    if (!cv) return;
    var rect = cv.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    // Backing store matches the Ruffle canvas 1:1 so capture-space coords draw
    // at the right pixel; CSS box matches the on-screen rect so it sits on top.
    if (overlayCanvas.width !== cv.width)  overlayCanvas.width  = cv.width;
    if (overlayCanvas.height !== cv.height) overlayCanvas.height = cv.height;
    overlayCanvas.style.left   = rect.left + 'px';
    overlayCanvas.style.top    = rect.top + 'px';
    overlayCanvas.style.width  = rect.width + 'px';
    overlayCanvas.style.height = rect.height + 'px';
  }

  function handleOverlay(id, boxes) {
    ensureOverlay();
    syncOverlaySize();
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!boxes) boxes = [];
    for (var i = 0; i < boxes.length; i++) {
      var b = boxes[i];
      if (!b) continue;
      var color = b.color || '#00FFFF';
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 3;
      overlayCtx.strokeRect(b.x, b.y, b.w, b.h);
      if (b.label) {
        overlayCtx.font = 'bold 14px monospace';
        overlayCtx.fillStyle = color;
        overlayCtx.textBaseline = 'bottom';
        // small dark shadow so the label reads on any backdrop
        overlayCtx.shadowColor = '#000';
        overlayCtx.shadowBlur = 3;
        overlayCtx.fillText(b.label, b.x, Math.max(b.y - 2, 12));
        overlayCtx.shadowBlur = 0;
      }
    }
    replyOk(id, null);
  }

  function clearOverlay() {
    if (!overlayCtx || !overlayCanvas) return;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  // ---- boot ---------------------------------------------------------------

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', connect);
  } else {
    connect();
  }
})();
