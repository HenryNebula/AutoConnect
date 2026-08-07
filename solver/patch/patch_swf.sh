#!/usr/bin/env bash
# patch_swf.sh — apply the AutoConnect patches (solver/patch/Mafokem.as) to the
# built SWF, incrementally. This is the ONLY supported way to change the SWF.
#
# What it does: recompile just the main timeline (kawai2_fla.Mafokem) from
# solver/patch/Mafokem.as and re-import it into the existing built SWF
# (solver/patch/game_inner.swf), then copy the result to the served location
# (local/web/game_inner.swf). See README.md for why this is incremental.
#
# Usage:
#   ./solver/patch/patch_swf.sh            # patch + deploy
#   python solver/driver.py --reload       # then reload so the browser loads it
#
# Requires: java (JDK 11+) and FFdec at solver/tools/ffdec/ (run get_ffdec.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFDEC="${FFDEC:-$HERE/../tools/ffdec/ffdec-cli.jar}"
TARGET="$HERE/game_inner.swf"                       # canonical built SWF (PRECARIOUS — see README)
SRC="$HERE/Mafokem.as"                              # patch source: hooks + solver
SERVED="${SERVED:-$HERE/../../local/web/game_inner.swf}"

[ -f "$FFDEC" ] || { echo "ERROR: FFdec not found at $FFDEC (run solver/patch/get_ffdec.sh)"; exit 1; }
[ -f "$TARGET" ] || { echo "ERROR: built SWF missing at $TARGET."; \
                       echo "       It cannot be rebuilt from scratch (see README.md). Restore it from git."; exit 1; }
[ -f "$SRC" ]    || { echo "ERROR: patch source missing at $SRC"; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/kawai2_fla"
cp "$SRC" "$WORK/kawai2_fla/Mafokem.as"

echo "[patch_swf] recompiling kawai2_fla.Mafokem from $SRC into $TARGET ..."
java -jar "$FFDEC" -onerror abort -importScript "$TARGET" "$WORK/out.swf" "$WORK" >/dev/null

# An identical output means Mafokem.as is already compiled into the SWF (a
# re-run with no source edits) — nothing to do. (A genuine name-mismatch against
# the wrong base SWF would also be a no-op, but TARGET is fixed to the canonical
# matching SWF below, so that can't happen via this script. See README.md.)
if cmp -s "$TARGET" "$WORK/out.swf"; then
  echo "[patch_swf] SWF already up to date (Mafokem.as unchanged since last patch). Nothing to do."
  exit 0
fi

cp "$WORK/out.swf" "$TARGET"
echo "[patch_swf] updated $TARGET"
if [ -d "$(dirname "$SERVED")" ]; then
  cp "$TARGET" "$SERVED"
  echo "[patch_swf] deployed to $SERVED"
fi
echo "[patch_swf] done. Reload with: python solver/driver.py --reload"
