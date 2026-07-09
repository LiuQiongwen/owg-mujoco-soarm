#!/bin/bash
# Diagnostic: does per-object-stratified OT coupling (C2OT discrete-class fix,
# trained via CFM_STRATIFY_OT=1 into cfm_allobj_ot_stratified.pt) recover
# baseline-competitive performance, vs. the original condition-agnostic
# OT-CFM checkpoint that was found to significantly underperform baseline?
# Same 3 objects/seeds as the DDPM and Remove-OT checks for direct comparison.
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("Pear" "TomatoSoupCan" "CrackerBox")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
CKPT="grasp_6dof/models/cfm_allobj_ot_stratified.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$OUT_DIR"

for obj in "${OBJECTS[@]}"; do
  raw="$OUT_DIR/stratifiedOT_check_${obj}.jsonl"
  > "$raw"
  succ=0
  for seed in "${SEEDS[@]}"; do
    OUTPUT=$(timeout 90 conda run -n tango python demo.py \
      --stage 4 --prompt "$obj" --seed "$seed" --once --verbose 0 \
      --gate-delta 0.0 --mc-gate-delta 0.0 --cfm-ckpt "$CKPT" --no-semantic 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      SUCCESS="timeout"
    elif echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS="true"
      succ=$((succ+1))
    else
      SUCCESS="false"
    fi
    echo "{\"condition\":\"stratifiedOT\",\"object\":\"$obj\",\"seed\":$seed,\"success\":\"$SUCCESS\"}" >> "$raw"
  done
  echo "[stratifiedOT-check-$obj] === DONE: $succ/${#SEEDS[@]} ==="
done
