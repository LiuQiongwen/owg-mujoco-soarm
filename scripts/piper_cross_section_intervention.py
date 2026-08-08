"""P2: cross-section / aim-point intervention.

Tests the chain P1.1+P1.2 assembled, at the one link that is manipulable:

    candidate-local geometry  (local support width at the aim point)
        -> landing accuracy   (object-to-eef distance at end of descend)
        -> captured width     (gripper_q_at_close)
        -> success

Design: hold scene, object, seed, orientation and controller fixed, and
vary ONLY where along the object the descend target aims. The offset is
applied along the grasp frame's local Y -- perpendicular to the jaw-closing
axis and horizontal -- because that is the axis that selects WHICH
cross-section the jaws will close on (moving along a bottle's taper, a
box's face, a pear's curve). Offsetting along local X would instead
de-centre the grasp between the fingers, which is a different (centring)
question.

`local_support_width_at_aim` is computed at candidate time, from the object
mesh, BEFORE the descend executes -- so if it predicts landing accuracy it
is usable as a genuine pre-execution feature, which is the point of the
whole exercise (P1.2 found no promoted pre-execution signal in the features
available so far).

Zero production diff: the offset is injected by subclassing ArmIK and
overriding solve() for descend-prefixed phases only -- the same mechanism
validated in scripts/piper_tcp_correction_ab.py, and the same lesson
applied (an earlier version of that script offset EVERY phase and produced
a spurious collapse).

Diagnostic only. Nothing here is a proposed production change.

Run:  conda run -n tango python scripts/piper_cross_section_intervention.py [n_seeds]
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
from scripts.piper_tcp_correction_ab import (
    _SolveRecorder, _ORIGINAL_ARMIK_SOLVE, _is_capture_phase, scene_objects_for,
)
from scripts.piper_execution_trace import ExecutionTracer, _mesh_world

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "piper_cross_section_intervention.jsonl"

OBJECTS = ["cracker", "mustard", "pear"]     # the three failure-bearing objects
OFFSETS_M = [-0.015, -0.0075, 0.0, 0.0075, 0.015]

# Two Y-bands, because they answer different questions and the first smoke
# test showed they behave very differently:
#   JAW band  -- the fingers' own footprint. This is what physically blocks
#                closure, but it is INSENSITIVE to small aim offsets, since a
#                +/-20mm window still contains the object's widest part after
#                a 15mm shift (observed: identical 51.6mm at 0 and +15mm).
#   LOCAL band -- a thin slice at the aim point. This is the actual "which
#                cross-section did I aim at" quantity the intervention varies.
BAND_HALF_Y_JAW_M = 0.020
BAND_HALF_Y_LOCAL_M = 0.005
BAND_HALF_Z_M = 0.015

_ACTIVE = {"offset": 0.0, "log": None}


class AimOffsetArmIK(_SolveRecorder, ppp.ArmIK):
    """Shifts ONLY descend-phase targets, along the grasp frame's local Y."""

    def _solve_impl(self, target_pos, seed_qpos, target_mat, iters, phase):
        tp = np.asarray(target_pos, dtype=float)
        if _is_capture_phase(phase) and _ACTIVE["offset"] != 0.0:
            tp = tp + np.asarray(target_mat, dtype=float)[:, 1] * _ACTIVE["offset"]
        if _is_capture_phase(phase) and _ACTIVE["log"] is not None:
            _ACTIVE["log"].setdefault("aim_targets", []).append(tp.tolist())
        return _ORIGINAL_ARMIK_SOLVE(self, tp, seed_qpos, target_mat=target_mat, iters=iters)


def support_widths(env, obj_name, aim_pos, grasp_mat):
    """Object cross-section width along the jaw-closing axis at the aim
    point, for both bands. Computed from the object mesh at candidate time,
    before descend executes -- a pre-execution quantity."""
    m, d = env.sim.model._model, env.sim.data._data
    pts = _mesh_world(m, d, env.object_body_ids[obj_name])
    if not len(pts):
        return None, None
    loc = (pts - np.asarray(aim_pos)) @ np.asarray(grasp_mat)
    inz = np.abs(loc[:, 2]) <= BAND_HALF_Z_M

    def width(half_y):
        b = loc[inz & (np.abs(loc[:, 1]) <= half_y)]
        return float(b[:, 0].max() - b[:, 0].min()) if len(b) >= 5 else 0.0

    return width(BAND_HALF_Y_JAW_M), width(BAND_HALF_Y_LOCAL_M)


class WidthTracer(ExecutionTracer):
    """ExecutionTracer + the pre-execution support width at the aim point,
    captured the first time descend begins (before the descend motion runs)."""

    def __init__(self, env, obj_name, offset):
        super().__init__(env, obj_name)
        self.offset = offset
        self.support_width_m = None
        self.local_width_m = None
        self.rel_dist_at_descend_mm = None

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend" and self.support_width_m is None:
            env = self._env
            m, d = env.sim.model._model, env.sim.data._data
            bid = self.body_id
            quat = d.xquat[bid].copy()
            ref = ppp.true_centroid_xy(d.xpos[bid].copy(), quat, self._obj_name)
            gm = ppp.compute_grasp_orientation(env, self._obj_name)
            aim = ref + np.array([0.0, 0.0, ppp.GRASP_HEIGHT_OFFSET]) + gm[:, 1] * self.offset
            self.support_width_m, self.local_width_m = support_widths(
                env, self._obj_name, aim, gm)
        if name == "descend_refresh" and self.rel_dist_at_descend_mm is None:
            env = self._env
            m, d = env.sim.model._model, env.sim.data._data
            eef = d.site_xpos[m.site("robot0_eef_site").id]
            self.rel_dist_at_descend_mm = float(
                np.linalg.norm(d.xpos[self.body_id] - eef)) * 1000


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
        tracer = WidthTracer(env, obj_name, offset)
        _ACTIVE["offset"] = offset
        _ACTIVE["log"] = {}
        real_init = AimOffsetArmIK.__init__

        def _cap(self, e, _o=real_init, _t=tracer):
            _o(self, e)
            self._phase_tracker = _t

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

        s_close = tracer.snaps.get("lift")
        return {
            "object": obj_name, "seed": seed, "offset_mm": offset * 1000,
            "success": bool(res.get("success")),
            "jaw_band_width_mm": (tracer.support_width_m * 1000
                                   if tracer.support_width_m is not None else None),
            "local_width_mm": (tracer.local_width_m * 1000
                                if tracer.local_width_m is not None else None),
            "rel_dist_at_descend_mm": tracer.rel_dist_at_descend_mm,
            "gripper_q_at_close": s_close["gripper_q"][0] if s_close else None,
            "bilateral_at_close": s_close["bilateral"] if s_close else None,
        }
    finally:
        env.close()


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seeds = list(range(5001, 5001 + n_seeds))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w") as fh:
        for obj in OBJECTS:
            for off in OFFSETS_M:
                for s in seeds:
                    try:
                        r = run_trial(obj, s, off)
                    except Exception as e:
                        print(f"  {obj} off={off} seed={s} ERROR {type(e).__name__}: {e}")
                        continue
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    n += 1
                sub = "..."
                print(f"[{obj:8s} offset={off*1000:+6.1f}mm] done ({n} total)")

    rows = [json.loads(l) for l in OUT.open()]
    print(f"\n{len(rows)} trials -> {OUT}")
    print(f"\n{'object':9s} {'offset':>8s} {'succ':>7s} {'jawW':>7s} {'locW':>7s} "
          f"{'rel_mm':>8s} {'grip_q':>9s}")
    for obj in OBJECTS:
        for off in OFFSETS_M:
            sub = [r for r in rows if r["object"] == obj and abs(r["offset_mm"] - off * 1000) < 1e-6]
            if not sub:
                continue
            def mean(k):
                v = [r[k] for r in sub if r.get(k) is not None]
                return float(np.mean(v)) if v else float("nan")
            print(f"{obj:9s} {off*1000:+8.1f} {sum(r['success'] for r in sub):3d}/{len(sub):<3d} "
                  f"{mean('jaw_band_width_mm'):7.1f} {mean('local_width_mm'):7.1f} {mean('rel_dist_at_descend_mm'):8.1f} "
                  f"{mean('gripper_q_at_close'):9.4f}")


if __name__ == "__main__":
    main()
