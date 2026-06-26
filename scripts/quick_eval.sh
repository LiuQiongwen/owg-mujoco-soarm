#!/bin/bash
# 用法：
#   bash quick_eval.sh                                      → 快速 35次（autoresearch用）
#   bash quick_eval.sh full                                 → 完整 175次（论文用）
#   bash quick_eval.sh full 4 90 0.1                       → Stage4, timeout=90s, gate-delta=0.1
#   bash quick_eval.sh full 4 90 0.0 0.05                 → Stage4, MC gate delta=0.05
#   bash quick_eval.sh fast 4 90 0.0 0.0 path.pt          → use custom LGGSN checkpoint
#   bash quick_eval.sh fast 4 90 0.0 0.0 "" cfm.pt        → OT-CFM candidates + default LGGSN

MODE=${1:-fast}
OBJECTS=("Banana" "TomatoSoupCan" "Pear" "MustardBottle" "Scissors" "CrackerBox" "PowerDrill")

if [ "$MODE" = "full" ]; then
  SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
  echo "=== FULL EVAL (175次) ==="
else
  SEEDS=(1 2 3 4 5)
  echo "=== FAST EVAL (35次) ==="
fi

STAGE=${2:-4}
TIMEOUT=${3:-90}        # 每个 trial 最多等待秒数，默认 90s
GATE_DELTA=${4:-0.0}    # score-spread gate threshold; 0.0 = disabled
MC_GATE_DELTA=${5:-0.0} # MC Dropout uncertainty gate; 0.0 = disabled
LGGSN_CKPT_ARG=${6:-}   # optional LGGSN checkpoint path override
CFM_CKPT_ARG=${7:-}     # optional OT-CFM checkpoint path for candidate generation
NO_SEMANTIC=${8:-0}     # set to 1 to skip GPT grounding (OWG_NO_SEMANTIC=1)
NO_RANKER=${9:-0}       # set to 1 to disable LGGSN ranking (OWG_NO_RANKER=1)
GRC6DOF=${10:-0}        # set to 1 to use GR-ConvNet 2D→6-DoF lifting (OWG_GRC6DOF=1)
SUCCESS=0
TOTAL=0
GATE_FIRED=0
MC_GATE_FIRED=0

CKPT_LABEL=${LGGSN_CKPT_ARG:-default}
CFM_LABEL=${CFM_CKPT_ARG:-none}
echo "Stage $STAGE | gate-delta=${GATE_DELTA} | mc-gate=${MC_GATE_DELTA} | ckpt=${CKPT_LABEL} | cfm=${CFM_LABEL} | timeout=${TIMEOUT}s | $(date)"
for obj in "${OBJECTS[@]}"; do
  OBJ_SUCCESS=0
  OBJ_GATE=0
  OBJ_MC_GATE=0
  for seed in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    # Build env overrides and extra args
    ENV_VARS=""
    EXTRA_ARGS=""
    [ -n "$LGGSN_CKPT_ARG" ] && ENV_VARS="LGGSN_CKPT=$LGGSN_CKPT_ARG"
    [ -n "$CFM_CKPT_ARG"   ] && EXTRA_ARGS="--cfm-ckpt $CFM_CKPT_ARG"
    [ "$NO_SEMANTIC" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --no-semantic"
    [ "$NO_RANKER"   = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --no-ranker"
    [ "$GRC6DOF"     = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --grconvnet-6dof"

    if [ -n "$ENV_VARS" ]; then
      OUTPUT=$(timeout "$TIMEOUT" conda run -n owg-mujoco env $ENV_VARS python demo.py \
        --stage "$STAGE" --prompt "$obj" --seed "$seed" --once --verbose 0 \
        --gate-delta "$GATE_DELTA" --mc-gate-delta "$MC_GATE_DELTA" $EXTRA_ARGS 2>&1)
    else
      OUTPUT=$(timeout "$TIMEOUT" conda run -n owg-mujoco python demo.py \
        --stage "$STAGE" --prompt "$obj" --seed "$seed" --once --verbose 0 \
        --gate-delta "$GATE_DELTA" --mc-gate-delta "$MC_GATE_DELTA" $EXTRA_ARGS 2>&1)
    fi
    EXIT_CODE=$?
    FIRED=$(echo "$OUTPUT" | grep -F "[GATE]" | wc -l)
    GATE_FIRED=$((GATE_FIRED + FIRED))
    OBJ_GATE=$((OBJ_GATE + FIRED))
    MC_FIRED=$(echo "$OUTPUT" | grep -F "[MC-GATE]" | grep -F "skipping" | wc -l)
    MC_GATE_FIRED=$((MC_GATE_FIRED + MC_FIRED))
    OBJ_MC_GATE=$((OBJ_MC_GATE + MC_FIRED))
    if [ $EXIT_CODE -eq 124 ]; then
      echo "  [T] $obj seed=$seed  (timeout ${TIMEOUT}s)"
    elif echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS=$((SUCCESS + 1))
      OBJ_SUCCESS=$((OBJ_SUCCESS + 1))
      echo "  [✓] $obj seed=$seed"
    else
      echo "  [✗] $obj seed=$seed"
    fi
  done
  N_SEEDS=${#SEEDS[@]}
  echo "  --- $obj: ${OBJ_SUCCESS}/${N_SEEDS}  gate=${OBJ_GATE}/${N_SEEDS}  mc_gate=${OBJ_MC_GATE}/${N_SEEDS}"
done

RATE=$(echo "scale=1; $SUCCESS * 100 / $TOTAL" | bc)
echo ""
echo "=== RESULT: $SUCCESS / $TOTAL ($RATE%)  gate=${GATE_FIRED}/${TOTAL}  mc_gate=${MC_GATE_FIRED}/${TOTAL} ==="
