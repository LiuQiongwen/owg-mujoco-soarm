#!/bin/bash
# Re-check Scissors baseline + OT-CFM (25 seeds each) to resolve a discrepancy
# found in git history: logs/eval_scissors_fix_summary.log (2026-06-26) logged
# 25/25=100% for both conditions with individual per-seed [Y] entries, but
# commit cf58a7d (2026-06-28) claims a re-run found 23/25=92% and called the
# 06-26 result "imputed" (which it wasn't -- it has real per-seed logs). A
# smoke-test seed=1 re-run today (2026-07-09) already disagrees with the
# 06-26 log at that exact seed, so neither historical number is trusted here
# -- this is a fresh, from-scratch measurement.
set -u
cd /lena/projects/OWG-main/.claude/worktrees/fix-eval-seeding

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"
OUT_DIR="/lena/projects/OWG-main/paperA_data/phase0_diag_extended"
mkdir -p "$OUT_DIR"

for cond in baseline otcfm; do
  raw="$OUT_DIR/scissors_recheck_${cond}.jsonl"
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
  echo "[scissors-recheck-$cond] === DONE: $succ/${#SEEDS[@]} ==="
done
