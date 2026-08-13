#!/usr/bin/env bash
# One-command HEADLESS smoke test: serve the patched SWF, bring up the Xvfb +
# Chrome session, wait for the ExternalInterface bridge, and run the CV bot
# through a level. Use this as the regression test.
#
# It exercises the REAL runtime -- bot.py clearing tiles via the airtight
# acRemovePair handle -- NOT driver.py --wins, which only watches the SWF's
# autonomous acTick self-solver (disabled by design; see solver/README.md).
#
# Usage: solver/smoke.sh [levels] [backbone]
#   solver/smoke.sh            # 1 level, NCC backbone  (no torch/NN needed)
#   solver/smoke.sh 3          # 3 levels, NCC
#   solver/smoke.sh 3 nn       # 3 levels, PairNet NN backbone
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$HERE/local/web"
PORT="${AC_PORT:-8765}"
LEVELS="${1:-1}"
BACKBONE="${2:-ncc}"

# Project venv python (direnv sets UV_PROJECT_ENVIRONMENT); fall back to uv.
if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  PY="$UV_PROJECT_ENVIRONMENT/bin/python"
else
  PY="uv run python"
fi

# 1) serve local/web (patched game_inner.swf + Ruffle) if nothing is on PORT
if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "[smoke] serving $WEB_DIR on http://127.0.0.1:$PORT"
  ( cd "$WEB_DIR" && nohup python3 -m http.server "$PORT" --bind 127.0.0.1 \
      > "$HERE/local/server.log" 2>&1 & )
  sleep 1
fi

# 2) headless session (Xvfb :99 + Chrome --app, CDP 9222). Clean slate first --
#    two chromes can't share CDP port 9222 or the ac_xvfb_profile lock.
echo "[smoke] starting headless session"
bash "$HERE/solver/session.sh" stop >/dev/null 2>&1 || true
bash "$HERE/solver/session.sh" start

cleanup() { echo "[smoke] stopping session"; bash "$HERE/solver/session.sh" stop; }
trap cleanup EXIT

# 3) wait for the SWF's EI bridge (ruffle-embed.acStatus) to register
echo "[smoke] waiting for EI bridge..."
if ! $PY - <<'PYEOF'; then
import sys
sys.path.insert(0, "solver")
import bot
sys.exit(0 if bot.Bot._wait_player(timeout=60) else 1)
PYEOF
  echo "[smoke] FAIL: EI bridge did not come up (check $HERE/local/server.log)"
  exit 1
fi

# 4) run the bot
FLAGS="--runs 1 --max-level $LEVELS --rollout"
if [ "$BACKBONE" = "nn" ]; then
  echo "[smoke] backbone = PairNet NN"
else
  FLAGS="$FLAGS --ncc"
  echo "[smoke] backbone = colour-NCC"
fi
echo "[smoke] running: bot.py $FLAGS"
$PY "$HERE/solver/bot.py" $FLAGS
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "[smoke] PASS"
  exit 0
fi
echo "[smoke] FAIL (bot exit $rc)"
exit "$rc"
