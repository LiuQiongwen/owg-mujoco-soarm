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
from scripts.piper_phase2y_driver import restore_reconstructed_root


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
    def __init__(self, env, boundary_snapshotter=None):
        super().__init__(env=env, obj_name="pear")
        self.bundle = None
        self.actions = []
        self.action_phases = []
        self.traj = []
        self.ncon = []
        self._armed = False
        self._env_ref = env
        self._boundary_snapshotter = boundary_snapshotter
        self._orig_step = env.step
        env.step = self._step

    def _step(self, action):
        if self._armed:
            if self.bundle is None:
                # Capture at the actual action boundary, not in set_phase().
                # solve_multi_seed runs between those two points and may use
                # internal save/restore operations of its own.
                m, d = self._env.sim.model._model, self._env.sim.data._data
                n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
                st = np.zeros(n)
                mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
                self.bundle = {"state": st, "n": n,
                               "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                               "ctrl": d.ctrl.copy(), "time": float(d.time),
                               "gripper_current_action": np.asarray(
                                   self._env.robots[0].gripper["right"].current_action,
                                   dtype=float,
                               ).copy()}
                if self._boundary_snapshotter is not None:
                    self._boundary_snapshotter(self._env, self.bundle)
            self.actions.append(np.array(action, dtype=float).copy())
            self.action_phases.append(self.current_phase)
        r = self._orig_step(action)
        if self._armed:
            m, d = self._env.sim.model._model, self._env.sim.data._data
            self.traj.append(np.concatenate([
                d.site_xpos[m.site("robot0_eef_site").id],
                d.xpos[self._env.object_body_ids["pear"]]]))
            self.ncon.append(int(d.ncon))
        return r

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh" and not self._armed:
            self._armed = True


def restore(env, bundle, forward=True):
    m, d = env.sim.model._model, env.sim.data._data
    mujoco.mj_setState(m, d, bundle["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    if forward:
        mujoco.mj_forward(m, d)


def zero_step_diff(env, bundle):
    d = env.sim.data._data
    return {"qpos": float(np.max(np.abs(d.qpos - bundle["qpos"]))),
            "qvel": float(np.max(np.abs(d.qvel - bundle["qvel"]))),
            "time": abs(float(d.time) - bundle["time"])}


def replay(seed, bundle, actions, gripper=None, restore_gripper_state=False,
           reset_controller=False, forward_after_restore=True,
           force_controller_sync=False):
    env = make_env(seed, gripper)
    try:
        return replay_in_env(env, bundle, actions, restore_gripper_state,
                             reset_controller, forward_after_restore,
                             force_controller_sync)
    finally:
        env.close()


def replay_in_env(env, bundle, actions, restore_gripper_state=True,
                  reset_controller=False, forward_after_restore=True,
                  force_controller_sync=False):
    if restore_gripper_state and force_controller_sync:
        if not forward_after_restore:
            raise ValueError("validated reconstruction requires forward_after_restore=True")
        restore_reconstructed_root(env, bundle)
    else:
        restore(env, bundle, forward=forward_after_restore)
    if restore_gripper_state and not force_controller_sync:
        env.robots[0].gripper["right"].current_action = \
            bundle["gripper_current_action"].copy()
    if force_controller_sync and not restore_gripper_state:
        # mj_setState updates MuJoCo data but not robosuite's cached
        # joint_pos / joint_vel / mass matrix. A freshly reset controller
        # has new_update=False, so its first set_goal() otherwise consumes
        # reset-state caches and emits a different step-0 torque command.
        env.robots[0].composite_controller.update_state()
        for controller in env.robots[0].composite_controller.part_controllers.values():
            controller.update(force=True)
    if reset_controller:
        env.robots[0].composite_controller.update_state()
        env.robots[0].composite_controller.reset()
    zs = zero_step_diff(env, bundle)
    m, d = env.sim.model._model, env.sim.data._data
    eef = m.site("robot0_eef_site").id
    bid = env.object_body_ids["pear"]
    traj, ncon = [], []
    for a in actions:
        env.step(a)
        traj.append(np.concatenate([d.site_xpos[eef], d.xpos[bid]]))
        ncon.append(int(d.ncon))
    return {"zero_step": zs, "traj": np.array(traj), "ncon": ncon,
            "obj_z": float(d.xpos[bid][2]),
            "restore_gripper_state": restore_gripper_state,
            "reset_controller": reset_controller,
            "forward_after_restore": forward_after_restore}


def main():
    seed = 5001
    env = make_env(seed)
    cap = Capture(env)
    try:
        ppp.run_pick_and_place(env, "pear", use_oriented_grasp=True, verbose=False,
                               candidate_selection=None,
                               wrist_friendly_orientation=True, step_hook=cap)
        cap._armed = False
        env.step = cap._orig_step
        cap.actions = tuple(cap.actions)
        same_env = replay_in_env(env, cap.bundle, cap.actions)
    finally:
        env.close()
    if cap.bundle is None:
        print("no descend_refresh reached"); return
    print(f"handoff captured: state dim={cap.bundle['n']}, "
          f"{len(cap.actions)} actions recorded after descend_refresh\n")

    conditions = {
        "same-env": same_env,
        "fresh": replay(seed, cap.bundle, cap.actions),
        "controller-reset": replay(seed, cap.bundle, cap.actions, reset_controller=True),
        "gripper-state": replay(seed, cap.bundle, cap.actions, restore_gripper_state=True),
        "gripper+controller": replay(seed, cap.bundle, cap.actions,
                                      restore_gripper_state=True, reset_controller=True),
        "gripper-no-extra-fwd": replay(seed, cap.bundle, cap.actions,
                                        restore_gripper_state=True,
                                        forward_after_restore=False),
        "gripper+forced-sync": replay(seed, cap.bundle, cap.actions,
                                       restore_gripper_state=True,
                                       force_controller_sync=True),
    }
    a, b = conditions["fresh"], conditions["controller-reset"]

    print("4A zero-step identity (restore, no step):")
    for k in ("qpos", "qvel", "time"):
        v = max(a["zero_step"][k], b["zero_step"][k])
        print(f"   max|d{k}| = {v:.3e}  {'OK' if v < 1e-9 else 'FAIL'}")
    ok4a = all(max(a["zero_step"][k], b["zero_step"][k]) < 1e-9
               for k in ("qpos", "qvel", "time"))

    n = min(len(r["traj"]) for r in conditions.values())
    o_orig = np.array(cap.traj)
    print(f"\n4B rollout identity against original ({n} replayed steps):")
    condition_max = {}
    for label, result in conditions.items():
        condition_max[label] = float(np.max(np.abs(o_orig[:n] - result["traj"][:n])))
        print(f"   {label:20s} max|d(eef,obj)|={condition_max[label]:.3e} "
              f"final_obj_z={result['obj_z']:.6f}")
    dmax = condition_max["same-env"]
    # TEST 1: divergence ONSET, original vs replay -- immediate departure
    # implies missing state, delayed departure implies FP amplification.
    def onset(t1, t2, eps=1e-9):
        n = min(len(t1), len(t2))
        dd = np.max(np.abs(np.asarray(t1)[:n] - np.asarray(t2)[:n]), axis=1)
        idx = np.argmax(dd > eps) if (dd > eps).any() else None
        return (None if idx is None else int(idx)), dd

    for label, r in conditions.items():
        t1, t2 = o_orig, r["traj"]
        i, dd = onset(t1, t2)
        if i is None:
            print(f"\nTEST1 {label:20s}: identical throughout")
        else:
            first_contact = next((k for k, c in enumerate(cap.ncon) if c > cap.ncon[0]), None)
            print(f"\nTEST1 {label:20s}: first |d|>1e-9 at step {i}/{len(dd)} "
                  f"(|d| there = {dd[i]:.2e}, final = {dd[-1]:.2e})")
            print(f"       ncon at onset = {cap.ncon[i] if i < len(cap.ncon) else '?'}, "
                  f"ncon at step0 = {cap.ncon[0]}, first ncon increase at step {first_contact}")

    print(f"\n4A {'PASS' if ok4a else 'FAIL'}   "
          f"4B(same-instance) {'PASS' if dmax < 1e-9 else 'FAIL'}")
    cross = condition_max["gripper+forced-sync"]
    print(f"   4D(cross-instance reconstructed) {'PASS' if cross < 1e-9 else 'FAIL'}")
    if ok4a and dmax < 1e-9 and cross < 1e-9:
        print("   -> RESOLVED: cross-instance reconstruction requires both")
        print("      PiperGripper.current_action and a forced refresh of every")
        print("      part-controller cache after mj_setState. reset_goal() alone")
        print("      does not refresh stale reset-state joint caches.")
    if not ok4a or dmax >= 1e-9 or cross >= 1e-9:
        raise SystemExit("P2Y reconstruction regression gate failed")


if __name__ == "__main__":
    main()
