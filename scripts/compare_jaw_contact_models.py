"""Paired comparison of the three jaw contact configurations.

Step 3C of the 4 -> 3 -> 1 -> 2 plan.  The plan called for a two-arm A/B varying
contact geometry alone, holding the approach fixed so any label flip would be
attributable to the collider.  That design turned out to be confounded, and the
third arm exists to break the confound:

  proxy_spheres        legacy.  Contact via two 6 mm spheres at the finger mesh
                       frame origins; jaw-midpoint IK aims the midpoint of those
                       same origins at the grasp target.
  measured_pads        contact via pads measured off the finger meshes; approach
                       UNCHANGED, i.e. still aiming the sphere midpoint.
  measured_pads_aimed  contact via the same pads, and the IK aims the PAD
                       midpoint -- the physically coherent configuration.

The confound: the legacy IK target is not the gripping point.  It is the midpoint
of the finger meshes' frame origins, measured 52-57 mm from the fingers' actual
gripping faces.  Aiming it at an object parks the finger roots on the object with
the fingers extending past it -- which is also where the proxy spheres sit, so
the approach error and the contact error are anchored to the same wrong place and
hide each other.  Swapping only the collider therefore moves contact to a surface
the arm was never aiming at, and `measured_pads` scores worse for a reason that
has nothing to do with pad quality.

Held constant across all three arms:

  * same control commands (GRIP_CLOSED / GRIP_OPEN / move_gripper's map are
    untouched -- opening calibration is deliberately deferred to step 1)
  * same candidate (straight down at the settled centroid)
  * same scene, same seed, same spawn
  * same weld logic and the same definition of success

proxy_spheres vs measured_pads is bit-identical in approach (the spheres stay put
and merely stop colliding; the IK midpoint matches to 0.000000 mm), so that pair
isolates the collider.  measured_pads_aimed additionally moves the IK target, so
read it against proxy_spheres as "corrected geometry end to end", not as a
single-variable contrast.

Usage
-----
  conda run -n tango python scripts/compare_jaw_contact_models.py \
      --objects ScissorsC HammerC MediumClampC BananaC TomatoSoupCanC \
      --seeds 0 1 2 3 4 --out outputs/jaw_contact_ab.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.env_soarm import (  # noqa: E402
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS_WELD,
    JAW_CONTACT_MEASURED_PADS,
    JAW_CONTACT_MEASURED_PADS_AIMED,
    JAW_CONTACT_PROXY_SPHERES,
    TABLE_TOP_Z,
)

ARMS = [JAW_CONTACT_PROXY_SPHERES, JAW_CONTACT_MEASURED_PADS,
        JAW_CONTACT_MEASURED_PADS_AIMED]
SHORT = {JAW_CONTACT_PROXY_SPHERES: "proxy",
         JAW_CONTACT_MEASURED_PADS: "pads",
         JAW_CONTACT_MEASURED_PADS_AIMED: "pads_aimed"}

DEFAULT_OBJECTS = ["ScissorsC", "HammerC", "MediumClampC", "BananaC",
                   "TomatoSoupCanC"]
LABELS = ["bilateral_contact", "weld_triggered", "lifted", "success"]


def fixed_spawn(seed: int):
    """Same deterministic spawn as BenchmarkRunner._fixed_spawn."""
    rng = np.random.default_rng(seed)
    return [float(rng.uniform(-0.06, 0.06)),
            -0.40 + float(rng.uniform(-0.04, 0.04)),
            TABLE_TOP_Z + 0.12]


def _pyb(v):
    return None if v is None else bool(v)


def run_arm(env, obj_key: str, seed: int, opening_m: float) -> dict:
    env.reset_robot()
    env.remove_all_obj()
    obj_id = env.load_obj(obj_key, name=obj_key, pos=fixed_spawn(seed))
    env._steps(240)
    env.wait_until_all_still(max_wait_epochs=200)

    p0 = env.get_obj_pos(obj_id).copy()
    ok, _ = env._execute_grasp(
        pos=(float(p0[0]), float(p0[1]), float(p0[2])),
        roll=0.0,
        gripper_opening_length=float(opening_m),
        obj_height=float(p0[2] - TABLE_TOP_Z),
    )
    m = dict(env.last_grasp_metrics or {})
    p1 = env.get_obj_pos(obj_id)
    return {
        "bilateral_contact": _pyb(m.get("bilateral_contact")),
        "weld_triggered": _pyb(m.get("weld_triggered")),
        "lifted": _pyb(m.get("lifted")),
        "success": bool(ok),
        "left_contacts": int(m.get("left_contacts", 0)),
        "right_contacts": int(m.get("right_contacts", 0)),
        "min_true_opening_m": m.get("close_min_true_opening_m"),
        # SETTLED opening at the post-close snapshot (before the weld decision),
        # as opposed to min_true_opening_m's minimum over the whole close+settle
        # trace. The trace minimum catches transient compliant-contact overshoot
        # and is NOT the quantity a close-probe criterion (DISF-style: does the
        # jaw stall wider than its free-closing target) should be built on.
        "settled_true_opening_m": m.get("true_opening_m"),
        "proxy_obj_dist_fixed_m": m.get("close_min_proxy_obj_dist_fixed_m"),
        "proxy_obj_dist_moving_m": m.get("close_min_proxy_obj_dist_moving_m"),
        "pad_obj_dist_fixed_m": m.get("pad_obj_dist_fixed_m"),
        "pad_obj_dist_moving_m": m.get("pad_obj_dist_moving_m"),
        "dz": float(p1[2] - p0[2]),
    }


def flip_reason(a: dict, b: dict) -> str:
    """Why this trial's labels moved, in terms of what each collider saw."""
    if a["success"] == b["success"] and a["bilateral_contact"] == b["bilateral_contact"]:
        return "stable"
    if a["bilateral_contact"] and not b["bilateral_contact"]:
        pf = b.get("pad_obj_dist_fixed_m")
        pm = b.get("pad_obj_dist_moving_m")
        if pf is not None and pm is not None:
            far = "fixed" if pf > pm else "moving"
            return f"pads never both touched ({far} pad {max(pf, pm)*1000:.1f} mm clear)"
        return "pads never both touched"
    if b["bilateral_contact"] and not a["bilateral_contact"]:
        return "proxy missed a contact the pads make"
    return "contact agreed, outcome differed downstream"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--opening", type=float, default=0.065)
    ap.add_argument("--out", default="outputs/jaw_contact_ab.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    envs = {m: EnvironmentSoArm(obj_names=args.objects, vis=False,
                                grasp_mode=GRASP_MODE_PHYSICS_WELD,
                                enable_jaw_metrology=True,
                                jaw_contact_model=m)
            for m in ARMS}

    records = []
    try:
        with out.open("w") as fh:
            for obj_key in args.objects:
                for seed in args.seeds:
                    arms = {SHORT[m]: run_arm(envs[m], obj_key, seed, args.opening)
                            for m in ARMS}
                    rec = {"object": obj_key, "seed": seed, "arms": arms,
                           "flips_pads": {k: arms["proxy"][k] != arms["pads"][k]
                                          for k in LABELS},
                           "flips_aimed": {k: arms["proxy"][k] != arms["pads_aimed"][k]
                                           for k in LABELS},
                           "reason_pads": flip_reason(arms["proxy"], arms["pads"]),
                           "reason_aimed": flip_reason(arms["proxy"], arms["pads_aimed"])}
                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"[{obj_key} seed={seed}] " + "  ".join(
                        f"{k}(bi={int(bool(v['bilateral_contact']))},"
                        f"succ={int(v['success'])})" for k, v in arms.items()),
                        flush=True)
    finally:
        for e in envs.values():
            e.close()

    summarize(records)
    print(f"\nwrote {len(records)} paired trials to {out}")


def summarize(records):
    n = len(records)
    print("\n" + "=" * 78)
    print(f"success over {n} paired trials")
    print("=" * 78)
    for k in ("proxy", "pads", "pads_aimed"):
        c = sum(r["arms"][k]["success"] for r in records)
        print(f"  {k:12s} {c:3d}/{n}  ({100*c/max(n,1):5.1f}%)")

    print("\nper-object success")
    print(f"  {'object':16s} {'proxy':>7} {'pads':>7} {'pads_aimed':>12}")
    for o in sorted({r["object"] for r in records}):
        rs = [r for r in records if r["object"] == o]
        cells = [f"{sum(x['arms'][k]['success'] for x in rs)}/{len(rs)}"
                 for k in ("proxy", "pads", "pads_aimed")]
        print(f"  {o:16s} {cells[0]:>7} {cells[1]:>7} {cells[2]:>12}")

    for tag, key in (("pads (collider only)", "flips_pads"),
                     ("pads_aimed (collider + IK target)", "flips_aimed")):
        print(f"\nlabel flip rate vs legacy proxy -- {tag}")
        for k in LABELS:
            f = sum(r[key][k] for r in records)
            print(f"  {k:20s} {f:3d}/{n}  = {100*f/max(n,1):5.1f}%")

    print("\nflip reasons vs legacy")
    for tag, key in (("pads", "reason_pads"), ("pads_aimed", "reason_aimed")):
        print(f"  [{tag}]")
        for reason, c in Counter(r[key] for r in records).most_common():
            print(f"    {c:3d}  {reason}")

    print("\npaired diff table (rows where any arm disagrees on success)")
    print(f"{'scene':22s} {'proxy':>6} {'pads':>6} {'aimed':>6}  reason (pads_aimed)")
    for r in records:
        ss = [r["arms"][k]["success"] for k in ("proxy", "pads", "pads_aimed")]
        if len(set(ss)) == 1:
            continue
        print(f"{r['object']+'/s'+str(r['seed']):22s} "
              f"{int(ss[0]):6d} {int(ss[1]):6d} {int(ss[2]):6d}  {r['reason_aimed']}")


if __name__ == "__main__":
    main()
