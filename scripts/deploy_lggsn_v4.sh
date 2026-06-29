#!/bin/bash
# Deploy LGGSN v4: train on full v4 dataset, update config, run benchmark.
# Run after collect_lggsn_data_v4.py has finished.
#
# Usage:
#   bash scripts/deploy_lggsn_v4.sh          # train + update config
#   bash scripts/deploy_lggsn_v4.sh bench    # train + update config + benchmark

set -e
CKPT="grasp_6dof/models/lggsn_pairwise_live_v4.pt"
DATA="grasp_6dof/dataset/lggsn_candidates_v4.jsonl"

if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA not found — run collect_lggsn_data_v4.py first"
  exit 1
fi

ROWS=$(wc -l < "$DATA")
echo "Data: $DATA  ($ROWS rows)"

# ── 1. Train ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Training LGGSN v4 ==="
conda run -n owg-mujoco python train_lggsn_v4.py

if [ ! -f "$CKPT" ]; then
  echo "ERROR: training failed — checkpoint not found"
  exit 1
fi

# ── 2. Update config to use v4 (17-dim) ──────────────────────────────────────
echo ""
echo "=== Updating config to v4 ==="

# env.yaml: lggsn_input_dim 14 → 17
sed -i 's/lggsn_input_dim: 14/lggsn_input_dim: 17/' config/mujoco/env.yaml
echo "  config/mujoco/env.yaml  lggsn_input_dim: 17  ✓"

# policy.py: default checkpoint path
sed -i "s|lggsn_pairwise_live_v2.pt|lggsn_pairwise_live_v4.pt|g" owg/policy.py demo.py
echo "  owg/policy.py + demo.py  → lggsn_pairwise_live_v4.pt  ✓"

echo ""
echo "=== v4 deployed ==="
echo "  Checkpoint : $CKPT"
echo "  Config dim : 17"

# ── 3. Benchmark (optional) ───────────────────────────────────────────────────
if [ "${1:-}" = "bench" ]; then
  echo ""
  echo "=== Running benchmark (stage 4, 20 trials per object) ==="
  bash scripts/quick_eval.sh fast 4
fi
