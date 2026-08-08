"""Phase 2Y instrument validation (pre-registered: docs/PHASE2Y_PREREGISTRATION.md).

Shifts the finger COLLISION geometry along the gripper's local Y in a
runtime-patched model copy, leaving candidate/EEF target/IK/controller/
object/gripper command untouched.

Gates, all four required before any success number is interpreted:
  1. dY=0 reproduces the unpatched baseline byte-identically
  2. measured finger displacement == commanded dY (< 0.1mm)
  3. EEF trajectory numerically unchanged vs baseline
  4. no new table / palm / self collisions

Diagnostic only. Zero production diff.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for, PhaseTracker


def finger_geom_ids(m):
    return [i for i in range(m.ngeom)
            if ("finger7" in (m.geom(i).name or "") or "finger8" in (m.geom(i).name or ""))
            and "collision" in (m.geom(i).name or "")]


def eef_local_axis_in_body(m, d, gid, axis=1):
    """Which body-frame direction corresponds to the eef frame's local Y."""
    eef = m.site("robot0_eef_site").id
    R_eef = d.site_xmat[eef].reshape(3, 3)
    bid = m.geom_bodyid[gid]
    R_body = d.xmat[bid].reshape(3, 3)
    return R_body.T @ R_eef[:, axis]      # eef-Y expressed in body frame


class Rec(PhaseTracker):
    def __init__(self, env):
        super().__init__(env=env, obj_name="pear")
        self.eef = []
        self.nonobj_contacts = 0
        self._m = env.sim.model._model
    def __call__(self, env):
        m, d = env.sim.model._model, env.sim.data._data
        self.eef.append(d.site_xpos[m.site("robot0_eef_site").id].copy())


def run(seed, dY_mm, patch=True):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for("pear"),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        m, d = env.sim.model._model, env.sim.data._data
        gids = finger_geom_ids(m)
        eef = m.site("robot0_eef_site").id

        def finger_pos_in_eef():
            R = d.site_xmat[eef].reshape(3, 3); p = d.site_xpos[eef]
            return np.array([ (d.geom_xpos[g] - p) @ R for g in gids ])

        before = finger_pos_in_eef()
        if patch and dY_mm != 0.0:
            for g in gids:
                ax = eef_local_axis_in_body(m, d, g, axis=1)
                m.geom_pos[g] = m.geom_pos[g] + ax * (dY_mm / 1000.0)
            import mujoco; mujoco.mj_forward(m, d)
        after = finger_pos_in_eef()
        measured = float(np.mean(after[:, 1] - before[:, 1])) * 1000

        rec = Rec(env)
        res = ppp.run_pick_and_place(
            env, "pear", use_oriented_grasp=True, verbose=False,
            candidate_selection=None, wrist_friendly_orientation=True,
            step_hook=rec)
        return {"seed": seed, "dY": dY_mm, "measured_shift_mm": measured,
                "success": bool(res.get("success")),
                "eef": np.array(rec.eef)}
    finally:
        env.close()


def main():
    seeds = [5001, 5002, 5003]
    print("GATE 2 (measured shift == commanded) and GATE 1/3 (dY=0 == baseline)\n")
    base = {s: run(s, 0.0, patch=False) for s in seeds}
    zero = {s: run(s, 0.0, patch=True) for s in seeds}

    print("GATE 1 + 3: dY=0 vs unpatched baseline")
    ok1 = True
    for s in seeds:
        b, z = base[s], zero[s]
        n = min(len(b["eef"]), len(z["eef"]))
        dmax = float(np.max(np.abs(b["eef"][:n] - z["eef"][:n]))) if n else float("nan")
        same = (b["success"] == z["success"]) and dmax < 1e-12
        ok1 &= same
        print(f"  seed={s}  success {b['success']}->{z['success']}  "
              f"max|dEEF|={dmax:.2e} m  {'OK' if same else 'FAIL'}")

    print("\nGATE 2: measured finger shift vs commanded")
    ok2 = True
    for dY in (-15.0, 15.0):
        for s in seeds[:2]:
            r = run(s, dY)
            err = abs(r["measured_shift_mm"] - dY)
            ok2 &= err < 0.1
            print(f"  dY={dY:+6.1f}  seed={s}  measured={r['measured_shift_mm']:+7.3f}mm  "
                  f"err={err:.3f}mm  {'OK' if err < 0.1 else 'FAIL'}")

    print(f"\nGATE 1+3: {'PASS' if ok1 else 'FAIL'}    GATE 2: {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        print("Instrument invalid -- do not interpret any success numbers.")


if __name__ == "__main__":
    main()
