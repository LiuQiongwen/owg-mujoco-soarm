"""Attribute grasp failures to jaw command range vs. proxy collision geometry.

Runs the LEGACY configuration unchanged -- same GRIP_CLOSED/GRIP_OPEN, same
linear move_gripper map, same 6 mm proxy spheres, same success rule -- and only
adds read-only measurement (`enable_jaw_metrology=True`).  The point is to find
out how much of the existing failure mass is explained by each distortion
BEFORE any of them is fixed.

Each trial is sorted into one of four buckets:

  A  control-lower-bound   the real jaw can never close far enough: the minimum
                           true fingertip opening reached exceeds the object's
                           local thickness at the jaw.  GRIP_CLOSED (0.05 rad)
                           holds the jaw 19.5 mm open; nothing thinner is
                           pinchable, whatever the candidate says.
  B  proxy-false-negative  the real pads reached the object surface on both
                           sides, but the proxy spheres registered no bilateral
                           contact -- the collider missed a grasp the geometry
                           supports.
  C  proxy-embedded        a proxy sphere is buried inside the object (signed
                           distance below -PROXY_EMBED_M) while the real pad
                           that sphere stands in for is still clear.  The weld,
                           and therefore `success`, then rests on interpenetration
                           rather than on contact any real finger could make.
  D  other                 opening and contact both plausible; look at the
                           candidate, friction or dynamics instead.
  E  approach-miss         neither real pad ever came within APPROACH_TOL of the
                           object -- the jaw was never in a position to grasp.

Distance conventions
--------------------
`min_proxy_obj_dist_*` is EXACT and signed: it comes from mj_geomDistance on the
geoms the solver collides, and was verified equal to MuJoCo's own contact.dist
to within 0.03 mm.  `min_true_pad_dist_*` is a sampled, non-negative proximity
measure for the counterfactual real pad, which no longer exists in the model;
it cannot express penetration, so it is only ever read as "how far away".

Usage
-----
  conda run -n tango python scripts/diagnose_jaw_opening_failures.py \
      --objects ScissorsC HammerC MediumClampC TomatoSoupCanC \
      --seeds 0 1 2 3 4 --out outputs/jaw_diag.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.env_soarm import (  # noqa: E402
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS_WELD,
    TABLE_TOP_Z,
)

# Default set: two known-failing thin objects plus two controls, all from the
# CoACD tier so collision geometry is comparable across the four.
DEFAULT_OBJECTS = ["ScissorsC", "HammerC", "MediumClampC", "TomatoSoupCanC"]

PAD_TOUCH_TOL_M = 0.002    # a real pad this close counts as "would have touched"
APPROACH_TOL_M = 0.010     # beyond this on both sides, the jaw never engaged
PROXY_EMBED_M = 0.003      # signed proxy distance below -this = buried, not touching

BUCKETS = ["A_control_lower_bound", "B_proxy_false_negative",
           "C_proxy_embedded", "D_other", "E_approach_miss"]


def fixed_spawn(seed: int, spread_xy: float = 0.06,
                centre_y: float = -0.40, drop_z_off: float = 0.12):
    """Same deterministic spawn as BenchmarkRunner._fixed_spawn."""
    rng = np.random.default_rng(seed)
    return [float(rng.uniform(-spread_xy, spread_xy)),
            centre_y + float(rng.uniform(-0.04, 0.04)),
            TABLE_TOP_Z + drop_z_off]


def classify(rec: dict) -> str:
    thick = rec.get("object_local_thickness_m")
    min_open = rec.get("min_true_opening_m")
    pf = rec.get("min_true_pad_dist_fixed_m")
    pm = rec.get("min_true_pad_dist_moving_m")
    proxy_bi = bool(rec.get("proxy_bilateral_ever"))

    real_would_touch = (pf is not None and pm is not None
                        and pf <= PAD_TOUCH_TOL_M and pm <= PAD_TOUCH_TOL_M)

    xf = rec.get("min_proxy_obj_dist_fixed_m")
    xm = rec.get("min_proxy_obj_dist_moving_m")
    embedded = [(x, p) for x, p in ((xf, pf), (xm, pm))
                if x is not None and x < -PROXY_EMBED_M]
    # C outranks E: a sphere buried centimetres inside the object, standing in
    # for a pad that is still clear, is decisive evidence about the collider
    # whether or not the approach also missed.  Ordering it after E would
    # relabel the strongest cases as "never engaged".
    if embedded and any(p is None or p > PAD_TOUCH_TOL_M for _, p in embedded):
        return "C_proxy_embedded"
    if (pf is not None and pm is not None
            and pf > APPROACH_TOL_M and pm > APPROACH_TOL_M):
        return "E_approach_miss"
    if thick is not None and min_open is not None and min_open > thick:
        return "A_control_lower_bound"
    if real_would_touch and not proxy_bi:
        return "B_proxy_false_negative"
    return "D_other"


def _pybool(v):
    return None if v is None else bool(v)


def run_trial(env, obj_key: str, seed: int, opening_m: float) -> dict:
    env.reset_robot()
    env.remove_all_obj()
    obj_id = env.load_obj(obj_key, name=obj_key, pos=fixed_spawn(seed))
    env._steps(240)
    env.wait_until_all_still(max_wait_epochs=200)

    pos_before = env.get_obj_pos(obj_id).copy()
    # Grasp straight down at the settled centroid: removes candidate-selection
    # variance so the trial isolates the gripper, not the ranker.
    ok, _ = env._execute_grasp(
        pos=(float(pos_before[0]), float(pos_before[1]), float(pos_before[2])),
        roll=0.0,
        gripper_opening_length=float(opening_m),
        obj_height=float(pos_before[2] - TABLE_TOP_Z),
    )
    m = dict(env.last_grasp_metrics or {})
    pos_after = env.get_obj_pos(obj_id)

    rec = {
        "object": obj_key,
        "seed": seed,
        "requested_opening_m": float(opening_m),
        # GRIP_REDUCTION-scaled value actually handed to move_gripper
        "commanded_opening_m": m.get("commanded_opening_m"),
        # what the code believes vs. what the jaw did
        "claimed_opening_m": m.get("claimed_opening_m"),
        "true_opening_final_m": m.get("true_opening_m"),
        "min_true_opening_m": m.get("close_min_true_opening_m"),
        "true_opening_at_contact_m": m.get("close_true_opening_at_first_bilateral_m"),
        "proxy_gap_final_m": m.get("proxy_gap_m"),
        "object_local_thickness_m": m.get("object_local_thickness_m"),
        "jaw_mid_to_obj_surface_m": m.get("jaw_mid_to_obj_surface_m"),
        # proxy-sphere contact -- what the success rule actually reads
        "proxy_left_contact": bool(m.get("left_contacts", 0)),
        "proxy_right_contact": bool(m.get("right_contacts", 0)),
        "proxy_bilateral_ever": int(m.get("close_proxy_bilateral_ever", 0)),
        "bilateral_contact": bool(m.get("bilateral_contact", False)),
        # exact signed proxy-collider distance (negative = sphere inside object)
        "min_proxy_obj_dist_fixed_m": m.get("close_min_proxy_obj_dist_fixed_m"),
        "min_proxy_obj_dist_moving_m": m.get("close_min_proxy_obj_dist_moving_m"),
        # real pad geometry -- the counterfactual the proxy cannot answer
        "min_true_pad_dist_fixed_m": m.get("close_min_true_pad_dist_fixed_m"),
        "min_true_pad_dist_moving_m": m.get("close_min_true_pad_dist_moving_m"),
        "pad_dist_at_contact_fixed_m": m.get("close_pad_dist_at_first_bilateral_fixed_m"),
        "pad_dist_at_contact_moving_m": m.get("close_pad_dist_at_first_bilateral_moving_m"),
        # outcome -- coerced off numpy scalars so the record stays JSON-safe
        "weld_triggered": _pybool(m.get("weld_triggered")),
        "lifted": _pybool(m.get("lifted")),
        "success": bool(ok),
        "dz": float(pos_after[2] - pos_before[2]),
    }
    rec["bucket"] = classify(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--opening", type=float, default=0.065,
                    help="requested opening in metres (legacy sampler draws 0.04-0.09)")
    ap.add_argument("--out", default="outputs/jaw_opening_diagnostic.jsonl")
    ap.add_argument("--reclassify", action="store_true",
                    help="re-bucket an existing --out file in place and re-print "
                         "the summary; runs no simulation")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.reclassify:
        records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        for r in records:
            r["bucket"] = classify(r)
        with out.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        summarize(records)
        print(f"\nreclassified {len(records)} trials in {out}")
        return

    env = EnvironmentSoArm(obj_names=args.objects, vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True)
    if env._jaw_metrology is None or not env._jaw_metrology.available:
        raise SystemExit("jaw metrology unavailable -- cannot run this diagnostic")

    records = []
    try:
        with out.open("w") as fh:
            for obj_key in args.objects:
                for seed in args.seeds:
                    rec = run_trial(env, obj_key, seed, args.opening)
                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"[{obj_key} seed={seed}] {rec['bucket']:22s} "
                          f"succ={str(rec['success']):5s} "
                          f"open={_mm(rec['min_true_opening_m'])} "
                          f"thick={_mm(rec['object_local_thickness_m'])} "
                          f"proxy=({_mm(rec['min_proxy_obj_dist_fixed_m'])},"
                          f"{_mm(rec['min_proxy_obj_dist_moving_m'])}) "
                          f"pad=({_mm(rec['min_true_pad_dist_fixed_m'])},"
                          f"{_mm(rec['min_true_pad_dist_moving_m'])})",
                          flush=True)
    finally:
        env.close()

    summarize(records)
    print(f"\nwrote {len(records)} trials to {out}")


def _mm(v):
    return "  n/a" if v is None else f"{v*1000:5.1f}mm"


def summarize(records):
    print("\n" + "=" * 78)
    print("per-object buckets")
    print("=" * 78)
    objs = sorted({r["object"] for r in records})
    buckets = BUCKETS
    print(f"{'object':16s} {'n':>3} {'succ':>5} " +
          " ".join(f"{b.split('_')[0]:>4}" for b in buckets))
    for o in objs:
        rs = [r for r in records if r["object"] == o]
        counts = [sum(1 for r in rs if r["bucket"] == b) for b in buckets]
        print(f"{o:16s} {len(rs):3d} {sum(r['success'] for r in rs):5d} " +
              " ".join(f"{c:4d}" for c in counts))

    print("\n" + "=" * 78)
    print("opening: what was asked for vs. what the jaw did (mm)")
    print("=" * 78)
    print(f"{'object':16s} {'cmded':>8} {'claimed':>8} {'min true':>9} "
          f"{'thickness':>10}")
    for o in objs:
        rs = [r for r in records if r["object"] == o]

        def avg(k):
            vs = [r[k] for r in rs if r.get(k) is not None]
            return f"{np.mean(vs)*1000:8.1f}" if vs else "     n/a"

        print(f"{o:16s} {avg('commanded_opening_m'):>8} {avg('claimed_opening_m'):>8} "
              f"{avg('min_true_opening_m'):>9} {avg('object_local_thickness_m'):>10}")

    print("\n" + "=" * 78)
    print("collider vs. real pad (mm; proxy is signed, negative = inside object)")
    print("=" * 78)
    print(f"{'object':16s} {'proxy fixed':>12} {'proxy moving':>13} "
          f"{'pad fixed':>10} {'pad moving':>11}")
    for o in objs:
        rs = [r for r in records if r["object"] == o]

        def avg(k):
            vs = [r[k] for r in rs if r.get(k) is not None]
            return f"{np.mean(vs)*1000:+8.1f}" if vs else "     n/a"

        print(f"{o:16s} {avg('min_proxy_obj_dist_fixed_m'):>12} "
              f"{avg('min_proxy_obj_dist_moving_m'):>13} "
              f"{avg('min_true_pad_dist_fixed_m'):>10} "
              f"{avg('min_true_pad_dist_moving_m'):>11}")


if __name__ == "__main__":
    main()
