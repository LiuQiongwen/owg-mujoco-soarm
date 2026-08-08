"""Identify the mediator of the P2 aim-offset effect, by time order.

Phase A ruled out contact-local geometry: support width, opening margin,
centring error and antipodal score are all CONSTANT across the offsets that
move success 3/12 -> 12/12. The surviving hypothesis is kinematic -- the
offset moves the descend target into a part of the workspace Piper can
actually realise more accurately.

This re-runs the exact P2 grid (same objects, seeds, offsets) recording the
chain in execution order:

    offset
      -> descend IK position residual
      -> descend IK orientation residual
      -> min joint-limit margin at the descend solution
      -> ||eef_achieved - eef_commanded|| once descend has settled
      -> rel_dist_at_descend
      -> gripper_q_at_close
      -> success

The tracking error is the decisive one: IK residual says whether a solution
EXISTS near the target, tracking error says whether the arm actually got
there. Geometry is already known to be flat across this grid, so any
monotone term here is a candidate mediator.

Zero production diff -- offset injected via the same descend-only ArmIK
subclass used in P2.

Run:  conda run -n tango python scripts/piper_mediator_chain.py [n_seeds]
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for
from scripts.piper_cross_section_intervention import AimOffsetArmIK, _ACTIVE
from scripts.piper_execution_trace import ExecutionTracer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "piper_mediator_chain.jsonl"

OBJECTS = ["cracker", "mustard", "pear"]
OFFSETS_M = [-0.015, -0.0075, 0.0, 0.0075, 0.015]


class ChainTracer(ExecutionTracer):
    """Captures the achieved eef pose at descend settle, so commanded-vs-
    achieved tracking error can be measured (IK residual alone only says a
    solution exists, not that the arm reached it)."""

    def __init__(self, env, obj_name):
        super().__init__(env, obj_name)
        self.eef_at_descend_settle = None
        self.rel_dist_mm = None

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh" and self.eef_at_descend_settle is None:
            m, d = self._env.sim.model._model, self._env.sim.data._data
            self.eef_at_descend_settle = d.site_xpos[m.site("robot0_eef_site").id].copy()
            self.rel_dist_mm = float(
                np.linalg.norm(d.xpos[self.body_id] - self.eef_at_descend_settle)) * 1000


def run_trial(obj_name, seed, offset):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        original = ppp.ArmIK
        ppp.ArmIK = AimOffsetArmIK
        tracer = ChainTracer(env, obj_name)
        holder = {}
        _ACTIVE["offset"] = offset
        _ACTIVE["log"] = {}
        real_init = AimOffsetArmIK.__init__

        def _cap(self, e, _o=real_init, _t=tracer, _h=holder):
            _o(self, e)
            self._phase_tracker = _t
            _h["ik"] = self

        AimOffsetArmIK.__init__ = _cap
        try:
            res = ppp.run_pick_and_place(
                env, obj_name, use_oriented_grasp=True, verbose=False,
                candidate_selection=None, wrist_friendly_orientation=True,
                step_hook=tracer)
        finally:
            AimOffsetArmIK.__init__ = real_init
            ppp.ArmIK = original
            _ACTIVE["offset"] = 0.0
            _ACTIVE["log"] = None

        calls = getattr(holder.get("ik"), "_calls", [])
        desc = [c for c in calls if (c["phase"] or "") == "descend"]
        rec = {"object": obj_name, "seed": seed, "offset_mm": offset * 1000,
               "success": bool(res.get("success"))}
        if desc:
            c = desc[-1]
            q = np.array(c["qpos"])
            margins = [min(q[i] - lo, hi - q[i])
                       for i, (lo, hi) in enumerate(ppp.REAL_JOINT_LIMITS)]
            rec["descend_ik_pos_residual_mm"] = c["err_cm"] * 10
            rec["descend_ik_ori_residual_deg"] = c.get("ori_err_deg")
            rec["descend_min_joint_margin_rad"] = float(min(margins))
            rec["descend_converged"] = bool(c["converged"])
            # INVALID -- kept only so the column is not silently reused.
            # _SolveRecorder logs the target it was CALLED with, but
            # AimOffsetArmIK applies the offset inside _solve_impl, so
            # c["target_pos"] is the UNOFFSET target. Differencing the
            # achieved eef against it measures the deliberate offset, not
            # tracking error. Confirmed by
            # scripts/piper_decompose_descend_error.py: this term reads
            # ~15.0mm at a +/-15mm offset, i.e. exactly the offset.
            if tracer.eef_at_descend_settle is not None:
                rec["descend_tracking_error_mm_INVALID"] = float(np.linalg.norm(
                    tracer.eef_at_descend_settle - np.array(c["target_pos"]))) * 1000
        rec["rel_dist_at_descend_mm"] = tracer.rel_dist_mm
        s_close = tracer.snaps.get("lift")
        rec["gripper_q_at_close"] = s_close["gripper_q"][0] if s_close else None
        rec["bilateral_at_close"] = s_close["bilateral"] if s_close else None
        return rec
    finally:
        env.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seeds = list(range(5001, 5001 + n))
    rows = []
    with OUT.open("w") as fh:
        for obj in OBJECTS:
            for off in OFFSETS_M:
                for s in seeds:
                    try:
                        r = run_trial(obj, s, off)
                    except Exception as e:
                        print(f"  {obj} {off} {s} ERROR {type(e).__name__}: {e}")
                        continue
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                print(f"[{obj:8s} {off*1000:+6.1f}mm] done")

    keys = ["descend_ik_pos_residual_mm", "descend_ik_ori_residual_deg",
            "descend_min_joint_margin_rad", "descend_tracking_error_mm",
            "rel_dist_at_descend_mm", "gripper_q_at_close"]
    for obj in OBJECTS:
        print(f"\n=== {obj} ===")
        print(f"{'off':>7s} {'succ':>6s} " + " ".join(f"{k[:11]:>12s}" for k in keys))
        for off in OFFSETS_M:
            sub = [r for r in rows if r["object"] == obj
                   and abs(r["offset_mm"] - off * 1000) < 1e-6]
            if not sub:
                continue
            cells = []
            for k in keys:
                v = [r[k] for r in sub if r.get(k) is not None]
                cells.append(f"{np.mean(v):12.3f}" if v else f"{'-':>12s}")
            print(f"{off*1000:+7.1f} {sum(r['success'] for r in sub):3d}/{len(sub):<2d} "
                  + " ".join(cells))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
