#!/usr/bin/env bash
set -e

# 用法：
#   bash scripts/run_phase1.sh [SEED] [TOPK] [VIS] [EXTRA]
# 例子：
#   bash scripts/run_phase1.sh 123 40 0 "--fast --cube-scale 0.08"

SEED="${1:-123}"
TOPK="${2:-40}"
VIS="${3:-0}"
EXTRA="${4:-"--fast"}"

TS=$(date +"%Y%m%d-%H%M%S")
CSV="grasp_6dof/out/grasp_bench.csv"
OUT="grasp_6dof/dataset/validated_grasps_panda.json"

echo "[RUN] seed=$SEED topk=$TOPK vis=$VIS extra='$EXTRA'"

python grasp_6dof/validate_grasps_panda.py \
  --obj cube.urdf \
  --grasps grasp_6dof/dataset/sample_grasps.json \
  --out "$OUT" \
  --csv "$CSV" \
  --seed "$SEED" \
  --topk "$TOPK" \
  --vis "$VIS" \
  $EXTRA

