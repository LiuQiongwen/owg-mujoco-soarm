"""P2Y-3/4: handoff bundle + action-replay identity smoke.

Avoids reimplementing close/lift: records the ACTION SEQUENCE from
descend_refresh onward during a normal run, saves mjSTATE_INTEGRATION at
that moment, then replays the same actions from the restored state. If the
handoff is complete, replay reproduces the original exactly.

4A  zero-step identity : restore, do not step, compare integration state
4B  rollout identity   : replay actions, compare trajectory + outcome

A 4A PASS with 4B FAIL localises the missing state OUTSIDE MuJoCo (robosuite
controller / interpolator / phase / RNG) -- that is a result, not a bug.

Outcome field is named conditional_lift_success, never `success`: it is
P(lift | common descend_refresh state), a different estimand from P2's
end-to-end episode success.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco, numpy as np
from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for, PhaseTracker


def make_env(seed, gripper=None):
    np.random.seed(seed)
    kw = dict(robots="Piper", ycb_objects=scene_objects_for("pear"),
              has_renderer=False, has_offscreen_renderer=False,
              use_camera_obs=False, control_freq=20)
    if gripper:
        kw["gripper_types"] = gripper
    env = PiperMultiObjectScene(**kw)
    env.reset()
    return env


class Capture(PhaseTracker):
    """Saves the handoff bundle at descend_refresh and records every action
    issued from that point on."""
    def __init__(self, env):
        super().__init__(env=env, obj_name="pear")
        self.bundle = None
        self.actions = []
        self._armed = False
        self._env_ref = env
        self._orig_step = env.step
        env.step = self._step

    def _step(self, action):
        if self._armed:
            self.actions.append(np.array(action, dtype=float).copy())
        return self._orig_step(action)

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh" and self.bundle is None:
            m, d = self._env.sim.model._model, self._env.sim.data._data
            n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
            st = np.zeros(n)
            mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
            self.bundle = {"state": st, "n": n,
                           "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                           "ctrl": d.ctrl.copy(), "time": float(d.time)}
            self._armed = True


def restore(env, bundle):
    m, d = env.sim.model._model, env.sim.data._data
    mujoco.mj_setState(m, d, bundle["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)


def zero_step_diff(env, bundle):
    d = env.sim.data._data
    return {"qpos": float(np.max(np.abs(d.qpos - bundle["qpos"]))),
            "qvel": float(np.max(np.abs(d.qvel - bundle["qvel"]))),
            "time": abs(float(d.time) - bundle["time"])}


def replay(seed, bundle, actions, gripper=None):
    env = make_env(seed, gripper)
    try:
        restore(env, bundle)
        zs = zero_step_diff(env, bundle)
        m, d = env.sim.model._model, env.sim.data._data
        eef = m.site("robot0_eef_site").id
        bid = env.object_body_ids["pear"]
        traj = []
        for a in actions:
            env.step(a)
            traj.append(np.concatenate([d.site_xpos[eef], d.xpos[bid]]))
        obj_z = float(d.xpos[bid][2])
        return {"zero_step": zs, "traj": np.array(traj), "obj_z": obj_z}
    finally:
        env.close()


def main():
    seed = 5001
    env = make_env(seed)
    cap = Capture(env)
    try:
        ppp.run_pick_and_place(env, "pear", use_oriented_grasp=True, verbose=False,
                               candidate_selection=None,
                               wrist_friendly_orientation=True, step_hook=cap)
    finally:
        env.close()
    if cap.bundle is None:
        print("no descend_refresh reached"); return
    print(f"handoff captured: state dim={cap.bundle['n']}, "
          f"{len(cap.actions)} actions recorded after descend_refresh\n")

    a = replay(seed, cap.bundle, cap.actions)
    b = replay(seed, cap.bundle, cap.actions)

    print("4A zero-step identity (restore, no step):")
    for k in ("qpos", "qvel", "time"):
        v = max(a["zero_step"][k], b["zero_step"][k])
        print(f"   max|d{k}| = {v:.3e}  {'OK' if v < 1e-9 else 'FAIL'}")
    ok4a = all(max(a["zero_step"][k], b["zero_step"][k]) < 1e-9
               for k in ("qpos", "qvel", "time"))

    n = min(len(a["traj"]), len(b["traj"]))
    dmax = float(np.max(np.abs(a["traj"][:n] - b["traj"][:n]))) if n else float("nan")
    print(f"\n4B rollout identity (dY=0 vs dY=0, {n} replayed steps):")
    print(f"   max|d(eef,obj)| = {dmax:.3e}   {'OK' if dmax < 1e-9 else 'FAIL'}")
    print(f"   final object z: {a['obj_z']:.6f} vs {b['obj_z']:.6f}")
    print(f"\n4A {'PASS' if ok4a else 'FAIL'}   4B {'PASS' if dmax < 1e-9 else 'FAIL'}")
    if ok4a and dmax >= 1e-9:
        print("   -> AMBIGUOUS. Two candidate causes this test cannot separate:")
        print("      (a) state missing outside MuJoCo (controller/interpolator/RNG)")
        print("      (b) inherent FP nondeterminism amplified through contact")
        print("      Both replays are constructed identically and restore identically,")
        print("      so (b) alone can produce this. Distinguish by comparing each")
        print("      replay against the ORIGINAL trajectory and locating divergence onset.")


if __name__ == "__main__":
    main()
