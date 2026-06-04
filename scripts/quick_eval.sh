#!/bin/bash
# 用法：
#   bash quick_eval.sh                       → 快速 35次（autoresearch用）
#   bash quick_eval.sh full                  → 完整 175次（论文用）
#   bash quick_eval.sh full 4 90 0.1        → Stage4, timeout=90s, gate-delta=0.1
#   bash quick_eval.sh full 4 90 0.0 0.05  → Stage4, MC gate delta=0.05

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
SUCCESS=0
TOTAL=0
GATE_FIRED=0
MC_GATE_FIRED=0

echo "Stage $STAGE | gate-delta=${GATE_DELTA} | mc-gate=${MC_GATE_DELTA} | timeout=${TIMEOUT}s | $(date)"
for obj in "${OBJECTS[@]}"; do
  OBJ_SUCCESS=0
  OBJ_GATE=0
  OBJ_MC_GATE=0
  for seed in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    OUTPUT=$(timeout "$TIMEOUT" conda run -n owg-mujoco python demo.py \
      --stage "$STAGE" --prompt "$obj" --seed "$seed" --once --verbose 0 \
      --gate-delta "$GATE_DELTA" --mc-gate-delta "$MC_GATE_DELTA" 2>&1)
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
