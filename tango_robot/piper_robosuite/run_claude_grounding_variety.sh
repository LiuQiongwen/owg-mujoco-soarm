#!/bin/bash
# Exercises Claude's actual language grounding on the pear/can/mustard scene
# with non-literal instructions (color, category, negation) instead of just
# the object's literal name -- the fast-path string-match mode can't handle
# any of these, so this is a genuine test of the VLM grounding step, not
# just plumbing verification.
set -e
cd "$(dirname "$0")/../.."

run() {
    echo "=================================================="
    echo "INSTRUCTION: $1"
    echo "=================================================="
    VLM_BACKEND=claude conda run -n tango python3 -m tango_robot.piper_robosuite.piper_language_grounded_pick \
        --instruction "$1" \
        --use-real-gpt 2>&1 | grep -E "^\[grounding\]|^\[execution\]|^\[ABORT\]"
    echo
}

run "pick up the yellow bottle"
run "grab the fruit"
run "pick up the object that is not a fruit and not a bottle"
run "get the tallest object on the table"
