#!/bin/bash
# Phase 1 pilot, round 2: same as run_mpc_correction_pilot.sh but with the
# bilateral-classification correction model (mpc_correction_bilateral_v1.pt)
# instead of the original gap-regression model (mpc_correction_v1.pt), which
# the round-1 pilot showed net HURT success rate (-9.3pp pooled). Reuses the
# already-collected pilot_baseline_{obj}.jsonl (deterministic, no need to
# rerun) -- only runs the new correction condition.
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("Pear" "TomatoSoupCan" "CrackerBox")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
CKPT="grasp_6dof/models/mpc_correction_bilateral_v1.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/worldmodel_trajs"
mkdir -p "$OUT_DIR"

for obj in "${OBJECTS[@]}"; do
  raw="$OUT_DIR/pilot_mpc_correction_bilateral_${obj}.jsonl"
  > "$raw"
  succ=0
  for seed in "${SEEDS[@]}"; do
    OUTPUT=$(MPC_CORRECTION="$CKPT" timeout 90 conda run -n tango python demo.py \
      --stage 4 --prompt "$obj" --seed "$seed" --once --verbose 0 \
      --gate-delta 0.0 --mc-gate-delta 0.0 --no-semantic 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      SUCCESS="timeout"
    elif echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS="true"
      succ=$((succ+1))
    else
      SUCCESS="false"
    fi
    echo "{\"condition\":\"mpc_correction_bilateral\",\"object\":\"$obj\",\"seed\":$seed,\"success\":\"$SUCCESS\"}" >> "$raw"
  done
  echo "[pilot-mpc_correction_bilateral-$obj] === DONE: $succ/${#SEEDS[@]} ==="
done
