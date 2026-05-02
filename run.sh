#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$WORK_DIR/build"
BIN_DIR="$WORK_DIR/bin"

ACTION="${1:-run}"
DEBUG_ARG="${2:-0}"

export VISION_ROOT="$WORK_DIR"

if [[ "$ACTION" != "build" && "$ACTION" != "rebuild" && "$ACTION" != "run" ]]; then
  echo "Usage: $0 {build|rebuild|run [debug(0/1)]}"
  exit 1
fi

if [[ "$ACTION" == "rebuild" ]]; then
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR" "$BIN_DIR"
cmake -S "$WORK_DIR" -B "$BUILD_DIR" -G Ninja
cmake --build "$BUILD_DIR" -j"$(nproc)"

if [[ -e "$BIN_DIR/config" ]]; then
  rm -rf "$BIN_DIR/config"
fi
ln -sf "$WORK_DIR/config" "$BIN_DIR/config"

if [[ "$ACTION" == "run" ]]; then
  "$BIN_DIR/laser_aim" "$DEBUG_ARG"
fi
