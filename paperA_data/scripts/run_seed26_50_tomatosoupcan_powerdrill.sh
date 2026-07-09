#!/bin/bash
# Extends Table II's TomatoSoupCan/PowerDrill Baseline + OT-CFM+LGGSN
# conditions from 25 to 50 seeds (seeds 26-50 only -- does not re-run 1-25).
#
# Exact flags reverse-engineered from the original seed-1-25 source logs'
# own printed headers (confirmed to match Table II's published counts
# exactly: TomatoSoupCan 23/25 + 25/25, PowerDrill 19/25 + 22/25):
#   Baseline: logs/eval_baseline_nosem.log
#     "Stage 4 | gate-delta=0.0 | mc-gate=0.0 | ckpt=default | cfm=none"
#   OT-CFM:   logs/eval_cfm_ot_nosem_current.log
#     "Stage 4 | gate-delta=0.0 | mc-gate=0.0 | ckpt=default | cfm=grasp_6dof/models/cfm_allobj_ot.pt"
#   Both "_nosem" (--no-semantic), both quick_eval.sh's owg-mujoco env,
#   which was renamed to `tango` (conda-meta/history: `conda rename -n
#   owg-mujoco tango`) -- using tango here is not a confound, it's the
#   same environment under its current name.
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

OBJECTS=("TomatoSoupCan" "PowerDrill")
SEEDS=(26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$OUT_DIR"

for obj in "${OBJECTS[@]}"; do
  for cond in baseline otcfm; do
    raw="$OUT_DIR/seed26_50_${obj}_${cond}.jsonl"
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
    echo "[seed26-50-$obj-$cond] === DONE: $succ/${#SEEDS[@]} ==="
  done
done
