#!/bin/bash
# Hyperparameter-robustness check for EBM v2's adversarial negative mining:
# same InfoNCE + static/uniform/hard-negative recipe (train_ebm_grasp.py),
# but with a stronger hard-negative-mining schedule (K_HARD=6 vs 4, HARD_POP=64
# vs 32, HARD_ITERS=6 vs 3, K_STATIC/K_UNIFORM reduced to 3 each to compensate)
# -- checkpoint grasp_6dof/models/ebm_allobj_v2b.pt. Same 3 objects/seeds as
# every other diagnostic this session (Pear/TomatoSoupCan/CrackerBox, n=25) for
# direct, apples-to-apples comparison against the already-reported EBM v2
# numbers in ebm_v2_check_*.jsonl. Purpose: the paper's Limitations section
# flags EBM v2 as "one hyperparameter setting, not swept" -- this checks
# whether the parity-with-baseline result (and the one-object win) is
# specific to that one setting or holds under a meaningfully different
# mining schedule.
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("Pear" "TomatoSoupCan" "CrackerBox")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
CKPT="grasp_6dof/models/ebm_allobj_v2b.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$OUT_DIR"

for obj in "${OBJECTS[@]}"; do
  raw="$OUT_DIR/ebm_v2b_check_${obj}.jsonl"
  > "$raw"
  succ=0
  for seed in "${SEEDS[@]}"; do
    OUTPUT=$(timeout 90 conda run -n tango python demo.py \
      --stage 4 --prompt "$obj" --seed "$seed" --once --verbose 0 \
      --gate-delta 0.0 --mc-gate-delta 0.0 --ebm-ckpt "$CKPT" --no-semantic 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      SUCCESS="timeout"
    elif echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS="true"
      succ=$((succ+1))
    else
      SUCCESS="false"
    fi
    echo "{\"condition\":\"ebm_v2b\",\"object\":\"$obj\",\"seed\":$seed,\"success\":\"$SUCCESS\"}" >> "$raw"
  done
  echo "[ebm-v2b-check-$obj] === DONE: $succ/${#SEEDS[@]} ==="
done
