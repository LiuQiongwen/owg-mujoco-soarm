"""
Wrapper for `lerobot-calibrate` that works around this environment's two
unrelated import-time issues (see _lerobot_groot_patch.py): the installed
lerobot==0.4.4's broken GR00TN15Config dataclass, and this project's own
datasets/episode.py shadowing the real HuggingFace `datasets` package.

Usage (same args as the real lerobot-calibrate):
    conda run -n tango python paperA_data/scripts/lerobot_calibrate.py \\
        --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 \\
        --teleop.id=my_leader
"""
import sys
sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
import _lerobot_groot_patch  # noqa: F401  (side effect: patches sys.modules)

sys.path.insert(0, "/lena/projects/OWG-main")
from lerobot.scripts.lerobot_calibrate import main

if __name__ == "__main__":
    sys.exit(main())
