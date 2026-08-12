#!/usr/bin/env bash
# Build the conn_fast C++ (pybind11) extension: connectivity + Monte-Carlo
# rollout lookahead + exact endgame solver. Output is solver/conn_fast<EXT>.so
# (gitignored). Fallback if pybind11/g++ can't build for this Python: nanobind.
#
# Usage:  PY=/media/ext4-data/venvs/autoconnect/bin/python bash solver/cpp/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python3}"

if ! "$PY" -c 'import pybind11' >/dev/null 2>&1; then
  echo "[build] pybind11 not importable in $PY; install with: uv add --dev pybind11" >&2
  exit 1
fi

EXT="$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
OUT="$ROOT/solver/conn_fast$EXT"

# shellcheck disable=SC2046
g++ -O3 -shared -std=c++17 -fPIC \
    $("$PY" -m pybind11 --includes) \
    "$HERE/conn_fast.cpp" \
    -o "$OUT"

echo "[build] wrote $OUT"
