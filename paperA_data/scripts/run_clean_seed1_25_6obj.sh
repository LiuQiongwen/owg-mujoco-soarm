#!/bin/bash
# Part of the full Table I/II/III rebuild after merging the seeding fix
# (commit range: worktree-fix-eval-seeding -> main, 2026-07-09). Runs
# seeds 1-25 for the 6 non-Scissors objects using the now-fixed main
# branch (no worktree cd -- this runs from the actual production code
# path, same as scripts/quick_eval.sh). Combined with the already-collected
# seed 26-50 data (paperA_data/phase0_diag_extended/seed26_50_*.jsonl,
# which was -- by coincidence of following the Tier-1 script pattern --
# already run via the fixed worktree code), this gives a clean n=50
# per object per condition, all on the fixed code.
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("Banana" "TomatoSoupCan" "Pear" "MustardBottle" "CrackerBox" "PowerDrill")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$OUT_DIR"

for obj in "${OBJECTS[@]}"; do
  for cond in baseline otcfm; do
    raw="$OUT_DIR/clean_seed1_25_${obj}_${cond}.jsonl"
    > "$raw"
    succ=0
    for seed in "${SEEDS[@]}"; do
      if [ "$cond" = "otcfm" ]; then
        EXTRA="--cfm-ckpt $CKPT"
      else
        EXTRA=""
      fi
      OUTPUT=$(timeout 90 conda run -n tango python demo.py \
        --stage 4 --prompt "$obj" --seed "$seed" --once --verbose 0 \
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
      echo "{\"condition\":\"$cond\",\"object\":\"$obj\",\"seed\":$seed,\"success\":\"$SUCCESS\"}" >> "$raw"
    done
    echo "[clean-seed1-25-$obj-$cond] === DONE: $succ/${#SEEDS[@]} ==="
  done
done
