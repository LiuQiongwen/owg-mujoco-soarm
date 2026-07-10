#!/bin/bash
# Phase 1 pilot (see /home/lina/.claude/plans/floating-crunching-yeti.md):
# does the MPC-style pre-close correction search improve on the current best
# candidate generator (random-CoM Baseline, 79.1% pooled per Phase 0's locked
# reference numbers) when stacked on top of it? Same 3-object convention used
# throughout this session (Pear/TomatoSoupCan/CrackerBox), n=25 seeds,
# baseline vs baseline+MPC_CORRECTION -- correction is additive, not a
# replacement candidate generator.
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("Pear" "TomatoSoupCan" "CrackerBox")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
CKPT="grasp_6dof/models/mpc_correction_v1.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/worldmodel_trajs"
mkdir -p "$OUT_DIR"

run_condition () {
  local cond="$1"  # "baseline" or "mpc_correction"
  for obj in "${OBJECTS[@]}"; do
    raw="$OUT_DIR/pilot_${cond}_${obj}.jsonl"
    > "$raw"
    succ=0
    for seed in "${SEEDS[@]}"; do
      if [ "$cond" = "mpc_correction" ]; then
        OUTPUT=$(MPC_CORRECTION="$CKPT" timeout 90 conda run -n tango python demo.py \
          --stage 4 --prompt "$obj" --seed "$seed" --once --verbose 0 \
          --gate-delta 0.0 --mc-gate-delta 0.0 --no-semantic 2>&1)
      else
        OUTPUT=$(timeout 90 conda run -n tango python demo.py \
          --stage 4 --prompt "$obj" --seed "$seed" --once --verbose 0 \
          --gate-delta 0.0 --mc-gate-delta 0.0 --no-semantic 2>&1)
      fi
      EXIT_CODE=$?
      if [ $EXIT_CODE -eq 124 ]; then
        SUCCESS="timeout"
      elif echo "$OUTPUT" | grep -q "Done pick"; then
        SUCCESS="true"
        succ=$((succ+1))
      else
        SUCCESS="false"
      fi
      echo "{\"condition\":\"$cond\",\"object\":\"$obj\",\"seed\":$seed,\"success\":\"$SUCCESS\"}" >> "$raw"
    done
    echo "[pilot-$cond-$obj] === DONE: $succ/${#SEEDS[@]} ==="
  done
}

run_condition "baseline"
run_condition "mpc_correction"
