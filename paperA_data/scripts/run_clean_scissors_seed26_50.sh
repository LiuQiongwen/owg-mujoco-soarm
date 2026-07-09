#!/bin/bash
# Completes Scissors to n=50 for consistency with the other 6 objects.
# Scissors' own 2026-07-09 recheck (22/25 both conditions,
# scissors_recheck_{baseline,otcfm}.jsonl) already used the fixed worktree
# code for seeds 1-25 (its CFM path is irrelevant -- it always falls back to
# the same random-CoM sampler in both conditions -- but the spawn-orientation
# fix in demo.py still applies). This adds seeds 26-50, run from the main
# repo root (now that the fix is merged there too, no worktree cd needed).
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

SEEDS=(26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$OUT_DIR"

for cond in baseline otcfm; do
  raw="$OUT_DIR/clean_scissors_seed26_50_${cond}.jsonl"
  > "$raw"
  succ=0
  for seed in "${SEEDS[@]}"; do
    if [ "$cond" = "otcfm" ]; then
      EXTRA="--cfm-ckpt $CKPT"
    else
      EXTRA=""
    fi
    OUTPUT=$(timeout 90 conda run -n tango python demo.py \
      --stage 4 --prompt "Scissors" --seed "$seed" --once --verbose 0 \
      --gate-delta 0.0 --mc-gate-delta 0.0 --no-semantic $EXTRA 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      SUCCESS="timeout"
    elif echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS="true"
      succ=$((succ+1))
    else
      SUCCESS="false"
    fi
    echo "{\"condition\":\"$cond\",\"object\":\"Scissors\",\"seed\":$seed,\"success\":\"$SUCCESS\"}" >> "$raw"
  done
  echo "[clean-scissors-26-50-$cond] === DONE: $succ/${#SEEDS[@]} ==="
done
