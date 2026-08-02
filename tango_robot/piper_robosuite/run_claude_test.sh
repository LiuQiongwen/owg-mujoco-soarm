#!/bin/bash
set -e
VLM_BACKEND=claude conda run -n tango python3 -m tango_robot.piper_robosuite.piper_language_grounded_pick \
    --instruction "pick up the mustard bottle" \
    --use-real-gpt
