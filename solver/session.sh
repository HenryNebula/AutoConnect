#!/usr/bin/env bash
# Manage a headless game session on a virtual X display (:99) so automation has
# full capture+click control, independent of the (often locked) user session on :1.
# Usage: solver/session.sh {start|stop|status|windowid|capture [outfile]}
set -uo pipefail

DISPLAY_NUM=99
if [ "${AC_HEADED:-0}" = "1" ]; then
  # Headed: land the --app window on the user's real graphical session so it can
  # be watched / screen-recorded. CDP capture+click are display-agnostic, so the
  # only change vs the Xvfb path is which X display chrome lands on (no Xvfb).
  export DISPLAY="${DISPLAY:-:1}"
  AC_HEADED_ACTIVE=1
else
  export DISPLAY=":$DISPLAY_NUM"
  AC_HEADED_ACTIVE=0
fi
JOBTMP="${CLAUDE_JOB_DIR:-/tmp}/tmp"
PORT_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/local/.serve_port"
PORT="${AC_PORT:-$(cat "$PORT_FILE" 2>/dev/null || echo 8765)}"
GEOM_W=720; GEOM_H=560
mkdir -p "$JOBTMP"

xvfb_running() { pgrep -f "Xvfb :${DISPLAY_NUM} " >/dev/null 2>&1; }
chrome_wid()   {
  # prefer the app window by title; fall back to the largest chrome-class window
  local wid; wid="$(xdotool search --name 'AutoConnect' 2>/dev/null | head -1)"
  if [ -z "$wid" ]; then
    wid="$(xdotool search --class chrome 2>/dev/null | while read -r w; do
      g="$(xdotool getwindowgeometry "$w" 2>/dev/null | awk '/Geometry/{print $2}')"
      echo "$g $w"
    done | sort -rh | head -1 | awk '{print $2}')"
  fi
  echo "$wid"
}

start_xvfb() {
  if xvfb_running; then echo "Xvfb already up"; return 0; fi
  setsid nohup Xvfb ":$DISPLAY_NUM" -screen 0 1280x960x24 -nolisten tcp -ac \
    > "$JOBTMP/xvfb.log" 2>&1 < /dev/null &
  # wait for the server socket
  for _ in $(seq 1 30); do
    if [ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then echo "Xvfb up"; return 0; fi
    sleep 0.2
  done
  echo "Xvfb failed to start"; return 1
}

start_chrome() {
  if [ -n "$(chrome_wid)" ]; then echo "chrome already up"; return 0; fi
  setsid nohup google-chrome \
    --user-data-dir="$JOBTMP/ac_xvfb_profile" \
    --disable-gpu --disable-software-rasterizer --no-sandbox \
    --no-first-run --no-default-browser-check --disable-features=Translate \
    --window-position=0,0 --window-size=${GEOM_W},${GEOM_H} \
    --force-device-scale-factor=1 \
    --remote-debugging-port=9222 --remote-allow-origins=* \
    --app="http://127.0.0.1:${PORT}/" \
    > "$JOBTMP/game_chrome.log" 2>&1 < /dev/null &
  echo "chrome launched"
}

case "${1:-status}" in
  start)
    if [ "$AC_HEADED_ACTIVE" != "1" ]; then
      start_xvfb || exit 1
    fi
    start_chrome
    ;;
  stop)
    for p in $(pgrep -f "ac_xvfb_profile" 2>/dev/null); do kill "$p" 2>/dev/null; done
    for p in $(pgrep -f "Xvfb :${DISPLAY_NUM} " 2>/dev/null); do kill "$p" 2>/dev/null; done
    echo "stopped"
    ;;
  status)
    echo "xvfb: $(xvfb_running && echo UP || echo down)"
    wid="$(chrome_wid)"; echo "chrome_wid: ${wid:-none}"
    ;;
  windowid) chrome_wid ;;
  capture)
    out="${2:-$JOBTMP/cap.png}"
    wid="$(chrome_wid)"
    [ -z "$wid" ] && { echo "no window"; exit 1; }
    xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
    import -window "$wid" "$out" && echo "$out"
    ;;
  *) echo "unknown command: $1"; exit 1 ;;
esac
