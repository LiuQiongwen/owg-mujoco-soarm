"""Wrapper for `lerobot-teleoperate` -- see lerobot_calibrate.py for why this
wrapper exists (unrelated lerobot/datasets import issues in this env)."""
import sys
sys.path.insert(0, "/lena/projects/OWG-main/paperA_data/scripts")
import _lerobot_groot_patch  # noqa: F401

sys.path.insert(0, "/lena/projects/OWG-main")
from lerobot.scripts.lerobot_teleoperate import main

if __name__ == "__main__":
    sys.exit(main())
