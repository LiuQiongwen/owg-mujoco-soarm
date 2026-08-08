"""Is OBJECT_CENTROID_OFFSET_LOCAL["pear"] applied in the wrong direction?

P2 found an ASYMMETRIC aim-offset effect on pear (-15mm -> 3/12,
+15mm -> 12/12, 50/0 concordant seed-pairs, p=8.9e-16). A "aim at the
thicker cross-section" mechanism would be symmetric about the object
centre; a systematic aim error would look exactly like this. The constant
in question is [-0.0014, +0.0155] -- a 15.5mm correction along the object's
local y, almost exactly the winning offset.

Three arms, same 12 seeds, everything else identical:

    A current    [-0.0014, +0.0155]     (production)
    B zero       [ 0.0,     0.0    ]     (correction removed)
    C negated    [+0.0014, -0.0155]     (direction reversed)

Reading:
  C ~ 12/12 and B in between  -> the correction has the wrong SIGN/frame
  B ~ 12/12                   -> the correction should not exist at all
  no difference               -> the P2 effect is not this constant

Deliberately NOT recording a "failure stage" label: that construct produced
the transit_high tautology earlier in this investigation. Outcome-adjacent
continuous quantities are recorded instead.

Zero production diff -- the constant is patched in the module dict for the
duration of each trial and restored in a finally block.

Run:  conda run -n tango python scripts/piper_pear_centroid_offset_test.py [n_seeds]
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
from scripts.piper_tcp_correction_ab import LegacyArmIK, scene_objects_for
from scripts.piper_execution_trace import ExecutionTracer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "piper_pear_centroid_test.jsonl"

ARMS = {
    "A_current": np.array([-0.0014, 0.0155]),
    "B_zero": np.array([0.0, 0.0]),
    "C_negated": np.array([0.0014, -0.0155]),
}


class AimTracer(ExecutionTracer):
    """Records the descend aim target and the object's pose at that moment,
    so aim error is measured against the object rather than inferred."""

    def __init__(self, env, obj_name):
        super().__init__(env, obj_name)
        self.aim_xy = None
        self.obj_xy_at_descend = None
        self.rel_dist_mm = None
        self.obj_pos_start = None

    def set_phase(self, name):
        super().set_phase(name)
        env = self._env
        m, d = env.sim.model._model, env.sim.data._data
        if self.obj_pos_start is None:
            self.obj_pos_start = d.xpos[self.body_id].copy()
        if name == "descend" and self.aim_xy is None:
            quat = d.xquat[self.body_id].copy()
            ref = ppp.true_centroid_xy(d.xpos[self.body_id].copy(), quat, self._obj_name)
            self.aim_xy = ref[:2].copy()
            self.obj_xy_at_descend = d.xpos[self.body_id][:2].copy()
        if name == "descend_refresh" and self.rel_dist_mm is None:
            eef = d.site_xpos[m.site("robot0_eef_site").id]
            self.rel_dist_mm = float(np.linalg.norm(d.xpos[self.body_id] - eef)) * 1000


def run_trial(seed, offset_vec):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for("pear"),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    saved = ppp.OBJECT_CENTROID_OFFSET_LOCAL.get("pear")
    try:
        env.reset()
        ppp.OBJECT_CENTROID_OFFSET_LOCAL["pear"] = np.asarray(offset_vec, dtype=float)
        original = ppp.ArmIK
        ppp.ArmIK = LegacyArmIK
        tracer = AimTracer(env, "pear")
        real_init = LegacyArmIK.__init__

        def _cap(self, e, _o=real_init, _t=tracer):
            _o(self, e)
            self._phase_tracker = _t

        LegacyArmIK.__init__ = _cap
        try:
            res = ppp.run_pick_and_place(
                env, "pear", use_oriented_grasp=True, verbose=False,
                candidate_selection=None, wrist_friendly_orientation=True,
                step_hook=tracer)
        finally:
            LegacyArmIK.__init__ = real_init
            ppp.ArmIK = original

        m, d = env.sim.model._model, env.sim.data._data
        s_close = tracer.snaps.get("lift")
        s_tray = tracer.snaps.get("transit_above_tray")
        aim_err = (float(np.linalg.norm(tracer.aim_xy - tracer.obj_xy_at_descend)) * 1000
                   if tracer.aim_xy is not None else None)
        final_disp = (float(np.linalg.norm(d.xpos[tracer.body_id] - tracer.obj_pos_start)) * 1000
                      if tracer.obj_pos_start is not None else None)
        return {
            "seed": seed,
            "success": bool(res.get("success")),
            "aim_to_obj_xy_mm": aim_err,
            "rel_dist_at_descend_mm": tracer.rel_dist_mm,
            "gripper_q_at_close": s_close["gripper_q"][0] if s_close else None,
            "bilateral_at_close": s_close["bilateral"] if s_close else None,
            "lift_height_gain_mm": ((s_tray["obj_pos"][2] - s_close["obj_pos"][2]) * 1000
                                     if (s_close and s_tray) else None),
            "final_obj_displacement_mm": final_disp,
        }
    finally:
        ppp.OBJECT_CENTROID_OFFSET_LOCAL["pear"] = saved
        env.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seeds = list(range(5001, 5001 + n))   # same seeds as the P2 sweep
    rows = []
    with OUT.open("w") as fh:
        for arm, vec in ARMS.items():
            for s in seeds:
                try:
                    r = run_trial(s, vec)
                except Exception as e:
                    print(f"  {arm} seed={s} ERROR {type(e).__name__}: {e}")
                    continue
                r["arm"] = arm
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            k = [x for x in rows if x["arm"] == arm]
            print(f"[{arm:11s}] {sum(x['success'] for x in k)}/{len(k)}")

    def mean(k, arm):
        v = [r[k] for r in rows if r["arm"] == arm and r.get(k) is not None]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{'arm':12s} {'succ':>7s} {'aim_err':>9s} {'rel_mm':>8s} "
          f"{'grip_q':>9s} {'bilat':>7s} {'lift_mm':>8s}")
    for arm in ARMS:
        k = [r for r in rows if r["arm"] == arm]
        bl = [r["bilateral_at_close"] for r in k if r["bilateral_at_close"] is not None]
        print(f"{arm:12s} {sum(r['success'] for r in k):3d}/{len(k):<3d} "
              f"{mean('aim_to_obj_xy_mm', arm):9.2f} {mean('rel_dist_at_descend_mm', arm):8.1f} "
              f"{mean('gripper_q_at_close', arm):9.4f} {sum(bl):3d}/{len(bl):<3d} "
              f"{mean('lift_height_gain_mm', arm):8.1f}")

    # Paired McNemar-style comparison against the production arm.
    base = {r["seed"]: r["success"] for r in rows if r["arm"] == "A_current"}
    print("\npaired vs A_current (same seed):")
    for arm in ("B_zero", "C_negated"):
        cur = {r["seed"]: r["success"] for r in rows if r["arm"] == arm}
        both = sorted(set(base) & set(cur))
        gain = sum(cur[s] and not base[s] for s in both)
        loss = sum(base[s] and not cur[s] for s in both)
        print(f"  {arm:11s} improved {gain}, worsened {loss}, unchanged {len(both)-gain-loss}")


if __name__ == "__main__":
    main()
