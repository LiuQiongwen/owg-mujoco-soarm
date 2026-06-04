#!/bin/bash
# Score-spread uncertainty gating ablation.
# Runs S3 baseline + 3 × Stage-4 gate conditions (175 trials each),
# then parses all four result files into a comparison table.
#
# Usage:
#   bash scripts/run_gate_ablation.sh
#   bash scripts/run_gate_ablation.sh parse-only   # skip runs, just parse

set -euo pipefail
cd "$(dirname "$0")/.."

TIMEOUT=90
OUT_DIR=results
mkdir -p "$OUT_DIR"

if [ "${1:-}" != "parse-only" ]; then

  echo "========================================"
  echo " [1/4] Stage 3 baseline"
  echo "========================================"
  bash scripts/quick_eval.sh full 3 "$TIMEOUT" 0.0 \
      2>&1 | tee "$OUT_DIR/gate_s3.txt"

  echo ""
  echo "========================================"
  echo " [2/4] Stage 4 — no gate (delta=0.0)"
  echo "========================================"
  bash scripts/quick_eval.sh full 4 "$TIMEOUT" 0.0 \
      2>&1 | tee "$OUT_DIR/gate_nogate.txt"

  echo ""
  echo "========================================"
  echo " [3/4] Stage 4 — gate delta=0.1"
  echo "========================================"
  bash scripts/quick_eval.sh full 4 "$TIMEOUT" 0.1 \
      2>&1 | tee "$OUT_DIR/gate_01.txt"

  echo ""
  echo "========================================"
  echo " [4/4] Stage 4 — gate delta=0.2"
  echo "========================================"
  bash scripts/quick_eval.sh full 4 "$TIMEOUT" 0.2 \
      2>&1 | tee "$OUT_DIR/gate_02.txt"

fi

echo ""
echo "========================================"
echo " Parsing results → gate_uncertainty_ablation.txt"
echo "========================================"
conda run -n owg-mujoco python scripts/parse_gate_results.py \
    --s3  "$OUT_DIR/gate_s3.txt" \
    --g0  "$OUT_DIR/gate_nogate.txt" \
    --g01 "$OUT_DIR/gate_01.txt" \
    --g02 "$OUT_DIR/gate_02.txt" \
    --out "$OUT_DIR/gate_uncertainty_ablation.txt"

echo ""
echo "Done. Results in $OUT_DIR/gate_uncertainty_ablation.txt"
