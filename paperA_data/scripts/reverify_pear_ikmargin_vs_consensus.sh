#!/bin/bash
# Re-verification (2026-07-10): the original Pear ikmargin-vs-consensus data
# (6.0% vs 68.0%, phase1_v2/ikmargin_Pear.jsonl + phase1_matched_n10/consensus_n10_Pear.jsonl)
# was generated 2026-07-08 09:37-10:46, over a full day before the seeding-bug
# fix (commit 10814cd) was formally committed to main -- from a worktree whose
# exact code state at that timestamp cannot be reconstructed from git history.
# --consensus-n/--ikmargin-n structurally require sample_poses's seed= param
# (only ever added in 10814cd) to run at all, so the worktree must have had
# *some* working version of the fix already, but whether it was the same as
# what eventually got committed is unverified. Re-running here on confirmed
# main-branch code (current HEAD, includes 10814cd) to resolve the ambiguity
# before this finding is used as the basis for real-robot validation.
#
# Identical protocol to paperA_data/scripts/run_consensus_n10_matched.sh /
# run_ikmargin_n10_tomatosoupcan.sh: orient_seed in {5..9} x gen-seed base in
# {1,11,...,91} = 50 trials/strategy, same checkpoint.
set -u
cd /lena/projects/OWG-main

CUPTI_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
NCCL_SO=/home/lina/miniforge3/envs/tango/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_PRELOAD="$CUPTI_SO:$NCCL_SO"

ORIENT_SEEDS=(5 6 7 8 9)
BASES_N10=(1 11 21 31 41 51 61 71 81 91)
CKPT="grasp_6dof/models/cfm_allobj_ot.pt"
RESULTS_DIR="/lena/projects/OWG-main/paperA_data/phase1_matched_n10"
mkdir -p "$RESULTS_DIR"

run_strategy () {
  local strategy="$1"   # "ikmargin" or "consensus"
  local flag="$2"       # "--ikmargin-n" or "--consensus-n"
  local raw="$RESULTS_DIR/reverify_${strategy}_Pear.jsonl"
  > "$raw"
  for oseed in "${ORIENT_SEEDS[@]}"; do
    obj_success=0
    for base in "${BASES_N10[@]}"; do
      OUTPUT=$(timeout 90 conda run -n tango python demo.py \
        --stage 4 --prompt Pear --seed "$oseed" --gen-seed "$base" "$flag" 10 \
        --once --verbose 0 --gate-delta 0.0 --mc-gate-delta 0.0 --cfm-ckpt "$CKPT" --no-semantic 2>&1)
      EXIT_CODE=$?
      if [ $EXIT_CODE -eq 124 ]; then
        SUCCESS="timeout"
      elif echo "$OUTPUT" | grep -q "Done pick"; then
        SUCCESS="true"
        obj_success=$((obj_success+1))
      else
        SUCCESS="false"
      fi
      echo "{\"strategy\":\"$strategy\",\"object\":\"Pear\",\"orient_seed\":$oseed,\"ensemble_base\":$base,\"success\":\"$SUCCESS\"}" >> "$raw"
    done
    echo "[reverify-$strategy-Pear] orient_seed=$oseed: $obj_success/${#BASES_N10[@]}"
  done
  echo "[reverify-$strategy-Pear] === DONE ==="
}

run_strategy "ikmargin" "--ikmargin-n"
run_strategy "consensus" "--consensus-n"
