#!/usr/bin/env bash
# Download JPEXS Free Flash Decompiler (FFdec) CLI, used to patch the SWF.
# Output: solver/tools/ffdec/ffdec-cli.jar
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOLS="$HERE/../tools"
VER="26.2.1"
mkdir -p "$TOOLS"
URL="https://github.com/jindrapetrik/jpexs-decompiler/releases/download/version${VER}/ffdec_${VER}.zip"
echo "Downloading FFdec $VER ..."
curl -sL "$URL" -o "$TOOLS/ffdec.zip"
rm -rf "$TOOLS/ffdec"
unzip -q -o "$TOOLS/ffdec.zip" -d "$TOOLS/ffdec"
rm -f "$TOOLS/ffdec.zip"
echo "FFdec CLI ready at $TOOLS/ffdec/ffdec-cli.jar"
