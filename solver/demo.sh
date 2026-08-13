#!/usr/bin/env bash
# One-shot HEADED demo launcher. Serves the patched SWF, opens a VISIBLE Chrome
# app window (CDP on :9222) on the user's real display, waits for the game's
# ExternalInterface bridge, then runs the CV bot with --demo so each chosen pair
# is outlined with bboxes in the window BEFORE it is cleared -- ready to watch
# and screen-record.
#
# Usage: solver/demo.sh [levels]      (default: 3 levels, 1 run)
#   solver/demo.sh          # 3 levels
#   solver/demo.sh 13       # a full clear
#
# For finer control (backbone, pause, frame saving) run the bot directly:
#   AC_HEADED=1 DISPLAY=:1 bash solver/session.sh start
#   <venv-python> solver/bot.py --demo --runs 1 --max-level N [--ncc|--pause 1.0|--save-frames]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$HERE/local/web"
PORT="${AC_PORT:-8765}"
LEVELS="${1:-3}"

# Visible window lands on the user's graphical session (:1 on this box).
# Override by exporting AC_DISPLAY (or DISPLAY) before running.
export DISPLAY="${DISPLAY:-${AC_DISPLAY:-:1}}"
export AC_HEADED=1           # session.sh: skip Xvfb, keep the --app + CDP Chrome
export AC_PORT="$PORT"       # keep session.sh and the server on the same port

# Prefer the project venv python (direnv sets UV_PROJECT_ENVIRONMENT); the venv
# has all deps (numpy, cv2, websocket-client). Fall back to `uv run python`.
if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  PY="$UV_PROJECT_ENVIRONMENT/bin/python"
else
  PY="uv run python"
fi

# 1) serve local/web (patched game_inner.swf + Ruffle) if nothing is on PORT
if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "[demo] serving $WEB_DIR on http://127.0.0.1:$PORT"
  ( cd "$WEB_DIR" && nohup python3 -m http.server "$PORT" --bind 127.0.0.1 \
      > "$HERE/local/server.log" 2>&1 & )
  sleep 1
fi

# 2) headed Chrome app window with CDP (AC_HEADED makes session.sh skip Xvfb).
# Stop any existing automation session first -- two chromes can't share CDP
# port 9222 or the ac_xvfb_profile lock, so the headed window needs a clean slate.
echo "[demo] stopping any existing session (free port 9222 + profile lock)"
bash "$HERE/solver/session.sh" stop >/dev/null 2>&1 || true
echo "[demo] starting headed session on DISPLAY=$DISPLAY"
bash "$HERE/solver/session.sh" start

# tear the session down on exit / Ctrl-C (leaves the HTTP server running)
cleanup() { echo "[demo] stopping session"; bash "$HERE/solver/session.sh" stop; }
trap cleanup EXIT

# 3) wait for the SWF's EI bridge (ruffle-embed.acStatus) to register
echo "[demo] waiting for EI bridge..."
if ! $PY - <<'PYEOF'; then
import sys
sys.path.insert(0, "solver")
import bot
sys.exit(0 if bot.Bot._wait_player(timeout=60) else 1)
PYEOF
  echo "[demo] EI bridge did not come up. Check $HERE/local/server.log and: bash solver/session.sh status"
  exit 1
fi
echo "[demo] EI bridge ready; running bot in demo mode"

# 4) run the bot: --demo paints the per-pair bbox overlay before each clear
$PY "$HERE/solver/bot.py" --demo --runs 1 --max-level "$LEVELS"
