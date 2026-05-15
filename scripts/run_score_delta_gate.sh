#!/bin/bash
# Experiment 3: score-delta uncertainty gate
#
# Design: skip LGGSN reranking when max(score)−min(score) < δ.
# Rationale: tiny spread → scores saturated → ranking arbitrary → revert to GR-ConvNet.
# Unlike the σ_H gate (Section 4.3), this operates on model output, not input geometry.
#
# Usage:
#   bash scripts/run_score_delta_gate.sh            # retrospective only (fast, no PyBullet)
#   bash scripts/run_score_delta_gate.sh live        # live PyBullet run (~60 min, 25 seeds × 6 obj × 5 δ)
#   bash scripts/run_score_delta_gate.sh live-fast   # live run, seeds 1–10 only

set -euo pipefail

MODE=${1:-retro}
CONDA_ENV=owg2
LOG_DIR=logs

mkdir -p "$LOG_DIR"

# ── δ thresholds to sweep ──────────────────────────────────────────────────────
# Range chosen to span [no gate, fire-on-all] given observed spreads (max ~0.01).
DELTAS=("0.0000" "0.0001" "0.0005" "0.0010" "0.0020" "0.0050")

if [ "$MODE" = "retro" ]; then
  echo "=== Experiment 3: score-delta gate — retrospective simulation ==="
  echo "    (uses lggsn_scores_all already in logs/batch_s3s4_v2_25seed.jsonl)"
  conda run -n "$CONDA_ENV" python scripts/summarize_score_delta_gate.py
  exit 0
fi

# ── live run ───────────────────────────────────────────────────────────────────
if [ "$MODE" = "live" ]; then
  SEEDS_ENV=""
  SUFFIX="score_delta_gate_25seed"
elif [ "$MODE" = "live-fast" ]; then
  SEEDS_ENV="SEEDS_MAX=10"
  SUFFIX="score_delta_gate_10seed"
else
  echo "Unknown mode: $MODE  (use: retro | live | live-fast)"
  exit 1
fi

# Build comma-separated strategy list: baseline + one strategy per δ
STRATEGY_LIST="margin_0.00"
for d in "${DELTAS[@]}"; do
  STRATEGY_LIST="${STRATEGY_LIST},delta_gate_${d}"
done

echo "=== Experiment 3: score-delta gate — live run ==="
echo "    strategies : $STRATEGY_LIST"
echo "    output     : $LOG_DIR/batch_s3s4_${SUFFIX}.jsonl"
echo "    started    : $(date)"

EVAL_STRATEGIES="$STRATEGY_LIST" \
OUT_SUFFIX="$SUFFIX" \
conda run -n "$CONDA_ENV" python batch_s3s4.py

echo "    finished   : $(date)"

echo ""
echo "=== Summarising results ==="
conda run -n "$CONDA_ENV" python scripts/summarize_score_delta_gate.py \
  --live "$LOG_DIR/batch_s3s4_${SUFFIX}.jsonl"
