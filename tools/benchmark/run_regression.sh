#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${1:-$ROOT_DIR/config/benchmark/classic_acceptance_manifest.json}"
BASELINE="${2:-$ROOT_DIR/config/benchmark/classic_baseline.json}"
REPORT="${3:-/tmp/classic_regression_report.json}"
MODE="${4:-check}"

ARGS=(
  "--manifest" "$MANIFEST"
  "--baseline" "$BASELINE"
  "--report" "$REPORT"
)

if [[ "$MODE" == "update-baseline" ]]; then
  ARGS+=("--update-baseline")
fi

python3 "$ROOT_DIR/tools/benchmark/run_regression.py" "${ARGS[@]}"

