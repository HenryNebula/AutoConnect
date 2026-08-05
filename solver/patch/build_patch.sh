#!/usr/bin/env bash
# Regenerate the patched, self-solving SWF (solver/patch/game_inner.swf) from
# the pristine game shipped in static/game.swf.
#
# Pipeline:
#   1. extract the inner (real) AS3 SWF from the outer Flex wrapper
#      (DefineBinaryData tag 1_ASEmbed3_87521_GameClass.bin)
#   2. rename the obfuscated identifiers to ASCII so FFdec's AS3 text compiler
#      can recompile them (the raw Brahmic-range names crash it)
#   3. import the patched main timeline (solver/patch/Mafokem.as) which:
#        - skips the title and starts level 1 on load
#        - installs an autonomous solver (reuses the game's own pair-finder +
#          move handler) plus ExternalInterface hooks
#        - makes reshuffles free so a deadlock never ends the run
#
# Requires: java (JDK 11+) and FFdec (run ./get_ffdec.sh first).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FFDEC="${FFDEC:-$HERE/../tools/ffdec/ffdec-cli.jar}"
OUTER="${OUTER:-$HERE/../../static/game.swf}"
OUT="${OUT:-$HERE/game_inner.swf}"

[ -f "$FFDEC" ] || { echo "ERROR: FFdec not found at $FFDEC (run $HERE/get_ffdec.sh)"; exit 1; }
[ -f "$OUTER" ] || { echo "ERROR: pristine game not found at $OUTER"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[1/3] extracting inner SWF from $OUTER ..."
java -jar "$FFDEC" -export binaryData "$WORK/bin" "$OUTER" >/dev/null
INNER="$WORK/bin/1_ASEmbed3_87521_GameClass.bin"
[ -f "$INNER" ] || { echo "ERROR: inner SWF not found in export"; exit 1; }

echo "[2/3] renaming obfuscated identifiers to ASCII ..."
java -jar "$FFDEC" -renameInvalidIdentifiers randomWord "$INNER" "$WORK/renamed.swf" >/dev/null

echo "[3/3] importing patched main timeline (Mafokem.as) ..."
mkdir -p "$WORK/src/kawai2_fla"
cp "$HERE/Mafokem.as" "$WORK/src/kawai2_fla/Mafokem.as"
java -jar "$FFDEC" -onerror abort -importScript "$WORK/renamed.swf" "$OUT" "$WORK/src" >/dev/null

echo "Built $OUT"
