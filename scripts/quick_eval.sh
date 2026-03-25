#!/bin/bash
# 用法：
#   bash quick_eval.sh        → 快速 20次（autoresearch用）
#   bash quick_eval.sh full   → 完整 100次（论文用）

MODE=${1:-fast}
OBJECTS=("Banana" "TomatoSoupCan" "Pear" "MustardBottle")

if [ "$MODE" = "full" ]; then
  SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
  echo "=== FULL EVAL (100次) ==="
else
  SEEDS=(1 2 3 4 5)
  echo "=== FAST EVAL (20次) ==="
fi

STAGE=${2:-4}
SUCCESS=0
TOTAL=0

echo "Stage $STAGE | $(date)"
for obj in "${OBJECTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    OUTPUT=$(conda run -n owg2 python demo.py \
      --stage "$STAGE" --prompt "$obj" --seed "$seed" --once --verbose 0 2>&1)
    if echo "$OUTPUT" | grep -q "Done pick"; then
      SUCCESS=$((SUCCESS + 1))
      echo "  [✓] $obj seed=$seed"
    else
      echo "  [✗] $obj seed=$seed"
    fi
  done
done

RATE=$(echo "scale=1; $SUCCESS * 100 / $TOTAL" | bc)
echo ""
echo "=== RESULT: $SUCCESS / $TOTAL ($RATE%) ==="
