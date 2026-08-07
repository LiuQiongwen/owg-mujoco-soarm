"""Print the measured SO-101 jaw pad geometry and its sanity checks.

The derivation itself lives in `tango_robot/jaw_pads.py` so that
`_build_scene_xml` can import it without depending on `scripts/` being on the
path.  This is the auditable front end: run it to see what the pads are, how
flat the fitted faces are, and whether the two pad normals actually oppose.

Run:  conda run -n tango python scripts/derive_jaw_pads.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tango_robot.jaw_pads import (  # noqa: E402
    FIXED_FINGER,
    MOVING_FINGER,
    derive,
    pad_geom_xml,
)


def main():
    pads = derive(verbose=True)
    c = pads["_check"]
    print("\ncheck at the near-closed pose:")
    print(f"  normals dot            {c['normals_dot']:+.3f}   (want ~ -1)")
    print(f"  face centres apart     {c['centre_sep_m']*1000:.1f} mm")
    print(f"  fixed normal . sep     {c['fixed_normal_dot_sep']:+.3f}   (want ~ +1)")
    print(f"  shared axial band      {c['axial_band_mm'][0]:.1f}"
          f"..{c['axial_band_mm'][1]:.1f} mm")
    print(f"  face flatness (rms)    fixed {pads['fixed']['flatness']*1000:.2f} mm, "
          f"moving {pads['moving']['flatness']*1000:.2f} mm")
    print("\nMJCF geom attributes:")
    for body, g in pad_geom_xml(pads).items():
        print(f"  body {body}:\n    {g}")


if __name__ == "__main__":
    main()
