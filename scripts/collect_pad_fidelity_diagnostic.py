"""Collect the read-only pad-contact fidelity diagnostic across scenes.

Read-only, opt-in, does not touch GRIP_CLOSED/GRIP_OPEN, move_gripper(),
weld triggering, contact solver parameters, or any success label -- see
tango_robot/pad_fidelity.py's module docstring. Runs the SAME deterministic
spawn/candidate protocol as scripts/compare_jaw_contact_models.py, so results
are directly comparable to that A/B and to outputs/jaw_contact_ab.jsonl.

Usage:
  conda run -n tango python scripts/collect_pad_fidelity_diagnostic.py \
      --objects ScissorsC HammerC MediumClampC BananaC TomatoSoupCanC \
      --seeds 0 1 2 3 4 --jaw-contact-model measured_pads_aimed \
      --out outputs/pad_fidelity.jsonl
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
    JAW_CONTACT_MEASURED_PADS,
    JAW_CONTACT_MEASURED_PADS_AIMED,
    TABLE_TOP_Z,
)
from tango_robot.pad_fidelity import PadFidelityConfig  # noqa: E402

DEFAULT_OBJECTS = ["ScissorsC", "HammerC", "MediumClampC", "BananaC",
                   "TomatoSoupCanC"]


def fixed_spawn(seed: int):
    """Same deterministic spawn as compare_jaw_contact_models.py / step 3D."""
    rng = np.random.default_rng(seed)
    return [float(rng.uniform(-0.06, 0.06)),
            -0.40 + float(rng.uniform(-0.04, 0.04)),
            TABLE_TOP_Z + 0.12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--opening", type=float, default=0.065)
    ap.add_argument("--jaw-contact-model",
                    choices=(JAW_CONTACT_MEASURED_PADS, JAW_CONTACT_MEASURED_PADS_AIMED),
                    default=JAW_CONTACT_MEASURED_PADS_AIMED)
    ap.add_argument("--contact-tol-m", type=float, default=PadFidelityConfig().contact_tol_m)
    ap.add_argument("--plausible-penetration-max-m", type=float,
                    default=PadFidelityConfig().plausible_penetration_max_m)
    ap.add_argument("--persistence-steps", type=int,
                    default=PadFidelityConfig().persistence_steps)
    ap.add_argument("--out", default="outputs/pad_fidelity.jsonl")
    args = ap.parse_args()

    cfg = PadFidelityConfig(contact_tol_m=args.contact_tol_m,
                            plausible_penetration_max_m=args.plausible_penetration_max_m,
                            persistence_steps=args.persistence_steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = EnvironmentSoArm(obj_names=args.objects, vis=False,
                           grasp_mode=GRASP_MODE_PHYSICS_WELD,
                           enable_jaw_metrology=True,
                           jaw_contact_model=args.jaw_contact_model,
                           enable_pad_fidelity_trace=True,
                           pad_fidelity_config=cfg)

    n = 0
    try:
        with out.open("w") as fh:
            for obj_key in args.objects:
                for seed in args.seeds:
                    env.reset_robot()
                    env.remove_all_obj()
                    oid = env.load_obj(obj_key, name=obj_key, pos=fixed_spawn(seed))
                    env._steps(240)
                    env.wait_until_all_still(max_wait_epochs=200)
                    p = env.get_obj_pos(oid).copy()
                    env._execute_grasp(
                        pos=(float(p[0]), float(p[1]), float(p[2])), roll=0.0,
                        gripper_opening_length=float(args.opening),
                        obj_height=float(p[2] - TABLE_TOP_Z))
                    m = env.last_grasp_metrics or {}
                    s = m.get("pad_fidelity_summary")
                    if s is None:
                        raise RuntimeError(
                            f"{obj_key}/s{seed}: no pad_fidelity_summary -- "
                            "the recording hook did not fire")
                    s["seed"] = seed
                    fh.write(json.dumps(s) + "\n")
                    fh.flush()
                    n += 1
                    print(f"[{obj_key} seed={seed}] verdict={s['geometric_verdict']:28s} "
                          f"legacy_success={s['legacy_success']} "
                          f"engaged={s['bilateral_engagement_samples']} "
                          f"excessive={s['excessive_penetration_samples']}", flush=True)
    finally:
        env.close()

    print(f"\nwrote {n} trials to {out}  (cfg: contact_tol={cfg.contact_tol_m*1000:.1f}mm "
          f"plausible_max={cfg.plausible_penetration_max_m*1000:.1f}mm "
          f"persistence={cfg.persistence_steps} samples)")


if __name__ == "__main__":
    main()
