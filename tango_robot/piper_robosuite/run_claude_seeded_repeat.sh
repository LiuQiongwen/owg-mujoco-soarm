#!/bin/bash
# Repeats "pick up the yellow bottle" against the IDENTICAL fixed-seed scene
# (verified reproducible: seed 777 -> same pear/can/mustard xpos across runs)
# to isolate whether the earlier decline was scene-specific bad luck (bad
# framing/occlusion/lighting on that particular random reset) versus a
# stable property of Claude's response to this image + instruction.
set -e
cd "$(dirname "$0")/../.."

SEED=777
N=5
# Baseline (already tested, 3/5 correct): CLAUDE_EFFORT=low CLAUDE_THINKING=disabled
# This run tests whether more reasoning stabilizes the borderline case.
export CLAUDE_EFFORT="${CLAUDE_EFFORT:-medium}"
export CLAUDE_THINKING="${CLAUDE_THINKING:-adaptive}"

echo "Testing with CLAUDE_EFFORT=$CLAUDE_EFFORT CLAUDE_THINKING=$CLAUDE_THINKING"
echo

for i in $(seq 1 $N); do
    echo "=================================================="
    echo "RUN $i / $N (seed=$SEED, effort=$CLAUDE_EFFORT, thinking=$CLAUDE_THINKING)"
    echo "=================================================="
    VLM_BACKEND=claude conda run -n tango python3 -m tango_robot.piper_robosuite.piper_language_grounded_pick \
        --seed $SEED \
        --instruction "pick up the yellow bottle" \
        --use-real-gpt 2>&1 | grep -E "^\[grounding\]|^\[execution\]|^\[ABORT\]|^🟢 Grounder raw response"
    echo
done
