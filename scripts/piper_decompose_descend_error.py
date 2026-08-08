"""Decompose the ~25mm descend eef error into its three possible layers.

The smoke test showed descend IK residual ~0.05mm but commanded-vs-achieved
eef error ~25mm. That is NOT yet evidence of controller undershoot -- at
least two very different causes produce the same number:

  A. the joint controller never reaches q*        -> real tracking undershoot
  B. the joints DO reach q*, but the eef pose being compared against is not
     the settled state (frame / phase-timing / measurement semantics)

Distinguished by measuring all three layers on the same trial:

  IK      : target_pos            vs FK(q_commanded)      -- solver accuracy
  CONTROL : q_commanded           vs q_achieved           -- joint tracking
  GEOMETRY: FK(q_achieved)        vs eef_actual           -- frame consistency

Reading:
  q error small AND FK(q_achieved) == eef_actual AND eef error large
      -> the COMMANDED pose was never physically consistent with the target,
         i.e. a semantics problem, not a controller problem
  q error large AND FK(q_achieved) == eef_actual
      -> genuine controller undershoot
  FK(q_achieved) != eef_actual
      -> the measurement itself is wrong

Also samples the eef error at several points after descend to check whether
the arm is still settling when the next phase begins -- the phase
termination question, which is cheaper to answer than tuning PD gains.

Run:  conda run -n tango python scripts/piper_decompose_descend_error.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for
from scripts.piper_cross_section_intervention import AimOffsetArmIK, _ACTIVE
from scripts.piper_execution_trace import ExecutionTracer


class DecomposeTracer(ExecutionTracer):
    def __init__(self, env, obj_name):
        super().__init__(env, obj_name)
        self.at_settle = None
        self.post_settle = []
        self._watch = 0

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh" and self.at_settle is None:
            m, d = self._env.sim.model._model, self._env.sim.data._data
            self.at_settle = {
                "q": np.array([d.qpos[m.joint(n).qposadr[0]] for n in ppp.JOINTS]),
                "eef": d.site_xpos[m.site("robot0_eef_site").id].copy(),
                "qvel_max": float(np.max(np.abs(
                    [d.qvel[m.joint(n).dofadr[0]] for n in ppp.JOINTS]))),
            }
            self._watch = 1

    def __call__(self, env):
        super().__call__(env)
        if self._watch and len(self.post_settle) < 3:
            m, d = env.sim.model._model, env.sim.data._data
            self.post_settle.append(d.site_xpos[m.site("robot0_eef_site").id].copy())


def fk(env, q):
    """Forward kinematics on a saved state -- never leaks into the sim."""
    m, d = env.sim.model._model, env.sim.data._data
    adr = [m.joint(n).qposadr[0] for n in ppp.JOINTS]
    saved = d.qpos.copy()
    for a, v in zip(adr, q):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)
    out = d.site_xpos[m.site("robot0_eef_site").id].copy()
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return out


def run(obj_name, seed, offset):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        original = ppp.ArmIK
        ppp.ArmIK = AimOffsetArmIK
        tracer = DecomposeTracer(env, obj_name)
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

        calls = [c for c in getattr(holder["ik"], "_calls", [])
                 if (c["phase"] or "") == "descend"]
        if not calls or tracer.at_settle is None:
            return None
        c = calls[-1]
        q_cmd = np.array(c["qpos"])
        target = np.array(c["target_pos"])
        q_ach = tracer.at_settle["q"]
        eef_actual = tracer.at_settle["eef"]

        fk_cmd = fk(env, q_cmd)
        fk_ach = fk(env, q_ach)

        return {
            "obj": obj_name, "seed": seed, "offset_mm": offset * 1000,
            "success": bool(res.get("success")),
            "ik_layer_mm": float(np.linalg.norm(fk_cmd - target)) * 1000,
            "control_layer_q_rad": float(np.linalg.norm(q_cmd - q_ach)),
            "control_layer_maxjoint_rad": float(np.max(np.abs(q_cmd - q_ach))),
            "geometry_layer_mm": float(np.linalg.norm(fk_ach - eef_actual)) * 1000,
            "total_eef_error_mm": float(np.linalg.norm(eef_actual - target)) * 1000,
            "qvel_max_at_settle": tracer.at_settle["qvel_max"],
            "post_settle_drift_mm": ([float(np.linalg.norm(p - eef_actual)) * 1000
                                      for p in tracer.post_settle] or None),
        }
    finally:
        env.close()


def main():
    print(f"{'obj':8s}{'off':>7s}{'succ':>6s}{'IK':>8s}{'CTRL_q':>9s}"
          f"{'GEOM':>8s}{'TOTAL':>9s}{'qvel':>8s}")
    print(f"{'':8s}{'mm':>7s}{'':>6s}{'mm':>8s}{'rad':>9s}{'mm':>8s}{'mm':>9s}{'rad/s':>8s}")
    print("-" * 64)
    for obj, off in [("pear", -0.015), ("pear", 0.015),
                     ("cracker", -0.015), ("cracker", 0.015)]:
        for seed in (5001, 5002):
            r = run(obj, seed, off)
            if r is None:
                continue
            print(f"{r['obj']:8s}{r['offset_mm']:+7.1f}{str(r['success']):>6s}"
                  f"{r['ik_layer_mm']:8.3f}{r['control_layer_q_rad']:9.4f}"
                  f"{r['geometry_layer_mm']:8.3f}{r['total_eef_error_mm']:9.2f}"
                  f"{r['qvel_max_at_settle']:8.3f}")
            if r["post_settle_drift_mm"]:
                print(f"{'':8s}post-settle eef drift over next steps: "
                      f"{[round(x,2) for x in r['post_settle_drift_mm']]} mm")


if __name__ == "__main__":
    main()
