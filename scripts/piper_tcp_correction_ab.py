"""Paired A/B: does the calibrated T_eef_capture correction change historical
Piper pick-and-place conclusions?

Zero production-code diff. `tango_robot/piper_robosuite/piper_pick_and_place.py`
is imported and used entirely as-is. The correction is injected via a
subclass of `ArmIK` that overrides `solve()` to apply T_eef_capture's inverse
before delegating to the unmodified parent implementation, then monkeypatches
`piper_pick_and_place.ArmIK` to that subclass for the duration of the
"corrected" arm's trials only -- `ik = ArmIK(env)` inside
`run_pick_and_place` resolves this name from the module's global namespace at
CALL time (ordinary Python late binding), so the substitution is transparent
to every one of run_pick_and_place's 8 phases and every descend variant,
without editing the file on disk. Restored to the original class immediately
after.

Scope, per the explicit instruction this was scoped against: ONLY the IK
target changes. `use_oriented_grasp=True` (the validated wrist-fix condition,
per PIPER_FINDINGS_SUMMARY.md), `candidate_selection=None` (deterministic
nominal target, no extra RNG-sampled-pool noise), no compliant/force-
compliant/two-stage/CR-CFM descend variants, no changes to joint6 handling.
Everything else identical between arms, same object, same seed.

Objects: Cracker (wrist-fix's strongest validated effect, p=1.8e-5/p=0.027,
n=152 in the existing investigation) and Pear (wrist-fix's validated null
result, 6/8 vs 6/8) -- the two objects PIPER_FINDINGS_SUMMARY.md already
treats as decisively characterized, so any TCP-correction effect is being
measured against an already-understood baseline, not a fresh unknown.

Per trial, records (not just final success):
  candidate_target, actual_capture_center at the closest-to-object solve call
  (a robust proxy for "the descend phase" that doesn't depend on knowing
  phase_log's exact internal call ordering), capture_position_error,
  success, ik_converged, joint6 at descend, IK phase log.

Run:  conda run -n tango python scripts/piper_tcp_correction_ab.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa: registers Piper/PiperGripper
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_tcp_correction_ab.jsonl"

# Captured at import time, BEFORE any monkeypatching happens in main(). Must
# not be looked up as `ppp.ArmIK.solve` inside _solve_impl -- by the time
# that runs, `ppp.ArmIK` has been reassigned to the subclass itself (that's
# the whole point of the patch), so a call-time lookup would recurse into
# the mixin's own solve() forever instead of reaching the real DLS solver.
_ORIGINAL_ARMIK_SOLVE = ppp.ArmIK.solve

# From scripts/calibrate_piper_capture_frame.py -- confirmed rigid, 15/15
# samples, std 0.000mm.
LOCAL_OFFSET = np.array([0.0, 0.0, -0.0656])

OBJECTS = ["cracker", "pear"]
SEEDS = list(range(1041, 1051))   # 10 seeds/object -- small-scale first pass,
                                  # not a replacement for the existing n=152
                                  # confirmatory run


def scene_objects_for(obj_name):
    """Matches piper_experiment_runner.py's own scene-composition rule
    exactly: pear/can/mustard (small horizontal_radius) are spawned together
    as the original pilot's fixed trio; every other object (cracker
    included -- large horizontal_radius) is spawned solo, since
    PiperMultiObjectScene's placement sampler cannot reliably fit
    banana/cracker/drill/clamp alongside other objects (documented in
    piper_multi_object_scene.py's DEFAULT_SCENE_OBJECTS comment -- exceeds
    its 5000-retry placement budget). Reusing the historical rule, not a
    fresh choice, since the A/B must reproduce the exact scene conditions
    the earlier findings were measured under."""
    return ["pear", "can", "mustard"] if obj_name in ("pear", "can", "mustard") else [obj_name]


class PhaseTracker:
    """Passed to run_pick_and_place as `step_hook`. This is the file's own
    designed extension point -- `_set_phase(step_hook, name)` calls
    `step_hook.set_phase(name)` right before every ik.solve_multi_seed call
    inside `solve_and_move`, and `move_to*` helpers call `step_hook(env)`
    per physics step. Purely a passive observer: `__call__` is a no-op, and
    `set_phase` only records the name. No production code path branches on
    step_hook's identity, so supplying this changes nothing about how the
    trial executes -- it only gives an outside script a legitimate way to
    know which named phase ("transit_high"/"approach"/"descend"/"lift"/...)
    the upcoming solve() call belongs to, without parsing phase_log or
    guessing from geometry.

    grep confirmed (piper_pick_and_place.py) the only phases whose target is
    actually a fresh candidate-grasp-pose-to-IK-target conversion are
    "descend"(+"_retry*") and "descend_refresh" -- everything else
    (transit_high/approach/retry_liftN/lift/transit_above_tray/
    lower_into_tray) is an independently-computed hover/transit/tray
    waypoint that was never meant to carry capture-frame semantics."""

    def __init__(self, env=None, obj_name=None):
        self.current_phase = None
        self._env = env
        self._obj_name = obj_name
        self.bilateral_contact_post_close = None

    def __call__(self, env):
        pass

    def set_phase(self, name):
        self.current_phase = name
        # Snapshot bilateral contact the first time "lift" begins solving --
        # by construction (see solve_and_move's call order) this fires right
        # after the preceding close command (move_to(..., GRIPPER_CLOSE,
        # steps=250, ...)) has already run and before any lift motion has
        # moved the arm, i.e. exactly "did closing the gripper actually
        # achieve bilateral contact on the object" -- checking at the
        # trial's final state instead would catch the object already
        # released (open/retract are later phases), which isn't the
        # question being asked.
        if name == "lift" and self.bilateral_contact_post_close is None and self._env is not None:
            self.bilateral_contact_post_close = _bilateral_contact(self._env, self._obj_name)


def _is_capture_phase(phase_name):
    return phase_name is not None and phase_name.startswith("descend")


def _bilateral_contact(env, obj_name):
    """Post-hoc, read-only: are BOTH fingers touching the target object in
    the env's current (final, post-trial) sim state? Reuses
    ppp._object_contact_geoms (the file's own existing helper) for object
    geom ids, and resolves finger geoms by substring match on their
    robosuite-assigned name -- same technique already used and validated in
    scripts/calibrate_piper_capture_frame.py. Not part of run_pick_and_place
    itself; this is purely an outside observation added for this A/B's
    reporting, touching no production state."""
    obj_geoms = ppp._object_contact_geoms(env, obj_name)
    model = env.sim.model._model
    data = env.sim.data._data

    def geoms_matching(substrings):
        return {i for i in range(model.ngeom)
               if all(s in (model.geom(i).name or "") for s in substrings)}

    left = geoms_matching(["finger7"])
    right = geoms_matching(["finger8"])
    touch_left = touch_right = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        pair = {g1, g2}
        if pair & obj_geoms:
            if pair & left:
                touch_left = True
            if pair & right:
                touch_right = True
    return bool(touch_left and touch_right)


def _failure_stage(phase_log, success):
    if success:
        return None
    if not phase_log:
        return "no_phases_logged"
    for name, entry in phase_log.items():
        if isinstance(entry, dict) and entry.get("converged") is False:
            return f"ik_no_converge:{name}"
    return "post_ik"  # every logged IK solve converged; failure happened
                       # during physical execution (grasp slip, drop, etc.)


class _SolveRecorder:
    """Mixin: records every solve() call's target, phase, and the eef pose
    actually converged to, on the class instance, for post-hoc capture-error
    analysis. Shared by both the legacy and corrected variants so both are
    instrumented identically -- the recording itself must not be a source
    of asymmetry."""

    def solve(self, target_pos, seed_qpos, target_mat=ppp.DOWN_ORIENTATION, iters=3000):
        phase = getattr(getattr(self, "_phase_tracker", None), "current_phase", None)
        result, converged, err = self._solve_impl(target_pos, seed_qpos, target_mat, iters, phase)
        # Read the converged eef pose (already restored to `saved` inside
        # solve(), so re-derive via forward kinematics at `result` rather
        # than reading live sim state post-call).
        saved = self._get_qpos()
        self._set_qpos(result)
        eef_pos = self.data.site_xpos[self.eef_site_id].copy()
        eef_R = self.data.site_xmat[self.eef_site_id].reshape(3, 3).copy()
        self._set_qpos(saved)
        capture_pos = eef_pos + eef_R @ LOCAL_OFFSET
        if not hasattr(self, "_calls"):
            self._calls = []
        self._calls.append({
            "phase": phase,
            "target_pos": np.asarray(target_pos).tolist(),
            "eef_pos": eef_pos.tolist(),
            "capture_pos": capture_pos.tolist(),
            "converged": bool(converged),
            "err_cm": float(err * 100),
            "qpos": np.asarray(result).tolist(),
        })
        return result, converged, err


class LegacyArmIK(_SolveRecorder, ppp.ArmIK):
    """P0: unmodified target semantics at every phase (target_pos IS the
    eef_site target), just wrapped for recording."""

    def _solve_impl(self, target_pos, seed_qpos, target_mat, iters, phase):
        return _ORIGINAL_ARMIK_SOLVE(self, target_pos, seed_qpos, target_mat=target_mat, iters=iters)


class CorrectedArmIK(_SolveRecorder, ppp.ArmIK):
    """P1: ONLY at the descend/descend_refresh grasp-commit call, target_pos/
    target_mat are interpreted as the DESIRED CAPTURE-FRAME pose and
    T_eef_capture's inverse (see
    docs/PIPER_CAPTURE_FRAME_CALIBRATION_20260807.md) is applied before
    delegating to the unmodified ArmIK.solve. Every other phase (transit,
    approach, lift, tray placement, retract) passes through unmodified,
    identical to the legacy arm -- those targets were never expressed in
    capture-frame terms in the first place, so correcting them would move
    the arm somewhere the original design never intended."""

    def _solve_impl(self, target_pos, seed_qpos, target_mat, iters, phase):
        if _is_capture_phase(phase):
            eef_target_pos = np.asarray(target_pos) - np.asarray(target_mat) @ LOCAL_OFFSET
        else:
            eef_target_pos = target_pos
        return _ORIGINAL_ARMIK_SOLVE(self, eef_target_pos, seed_qpos, target_mat=target_mat, iters=iters)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for obj in OBJECTS:
            for seed in SEEDS:
                for label, arm_cls in (("legacy", LegacyArmIK), ("corrected", CorrectedArmIK)):
                    np.random.seed(seed)
                    env = PiperMultiObjectScene(
                        robots="Piper",
                        ycb_objects=scene_objects_for(obj),
                        has_renderer=False, has_offscreen_renderer=False,
                        use_camera_obs=False, control_freq=20,
                    )
                    try:
                        env.reset()
                        original_armik = ppp.ArmIK
                        ppp.ArmIK = arm_cls
                        ik_holder = {}
                        tracker = PhaseTracker(env=env, obj_name=obj)
                        # Capture the ik instance run_pick_and_place builds
                        # internally by wrapping the class constructor, and
                        # attach the phase tracker to it so _SolveRecorder
                        # can tag every call with the phase name that was
                        # active when it fired (see PhaseTracker docstring).
                        real_init = arm_cls.__init__

                        def _capturing_init(self, env, _orig=real_init, _holder=ik_holder, _tracker=tracker):
                            _orig(self, env)
                            self._phase_tracker = _tracker
                            _holder["ik"] = self

                        arm_cls.__init__ = _capturing_init
                        try:
                            result = ppp.run_pick_and_place(
                                env, obj, use_oriented_grasp=True, verbose=False,
                                candidate_selection=None,
                                wrist_friendly_orientation=True,
                                step_hook=tracker,
                            )
                        finally:
                            arm_cls.__init__ = real_init
                            ppp.ArmIK = original_armik

                        ik = ik_holder.get("ik")
                        calls = getattr(ik, "_calls", [])
                        # The descend-phase call, precisely: the LAST solve()
                        # tagged with a phase name starting "descend" (covers
                        # "descend"/"descend_retryN"/"descend_refresh" -- the
                        # refresh, right before the gripper closes, is the
                        # most representative "final aimed capture point").
                        capture_calls = [c for c in calls if _is_capture_phase(c["phase"])]
                        descend_call = capture_calls[-1] if capture_calls else None

                        capture_error_m = (
                            float(np.linalg.norm(np.array(descend_call["target_pos"])
                                                 - np.array(descend_call["capture_pos"])))
                            if descend_call else None)
                        joint6 = descend_call["qpos"][5] if descend_call else None
                        min_object_distance_m = (
                            min(np.linalg.norm(np.array(c["capture_pos"])
                                               - np.array(result.get("spawn_pos", [0, 0, 0])))
                               for c in calls)
                            if calls else None)
                        bilateral_contact = tracker.bilateral_contact_post_close

                        rec = {
                            "object": obj, "seed": seed, "arm": label,
                            "success": bool(result.get("success")),
                            "failure_stage": _failure_stage(result.get("phases"), result.get("success")),
                            "dist_to_tray": result.get("dist_to_tray"),
                            "n_solve_calls": len(calls),
                            "candidate_target": (descend_call["target_pos"] if descend_call else None),
                            "actual_capture_center_at_pregrasp": (descend_call["capture_pos"]
                                                                   if descend_call else None),
                            "capture_position_error_m": capture_error_m,
                            "joint6": joint6,
                            "min_object_distance_m": min_object_distance_m,
                            "bilateral_contact": bilateral_contact,
                            "phases": result.get("phases"),
                        }
                    finally:
                        env.close()

                    records.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    ce = rec["capture_position_error_m"]
                    ce_str = f"{ce*1000:.1f}mm" if ce is not None else "n/a"
                    j6 = f"{rec['joint6']:+.3f}rad" if rec["joint6"] is not None else "n/a"
                    print(f"[{obj:8s} seed={seed} {label:9s}] success={rec['success']!s:5s} "
                          f"capture_error={ce_str:>10s} joint6={j6} "
                          f"bilateral={rec['bilateral_contact']!s:5s} stage={rec['failure_stage']}")

    summarize(records)
    print(f"\nwrote {len(records)} trials to {OUT}")


def summarize(records):
    print("\n" + "=" * 90)
    print("sanity check: does capture_error drop from ~65.6mm (legacy) to ~0 (corrected)?")
    print("=" * 90)
    for arm in ("legacy", "corrected"):
        vs = [r["capture_position_error_m"] for r in records
             if r["arm"] == arm and r["capture_position_error_m"] is not None]
        if vs:
            print(f"  {arm:10s} n={len(vs):3d}  mean={np.mean(vs)*1000:7.2f}mm  "
                  f"median={np.median(vs)*1000:7.2f}mm  "
                  f"range=[{min(vs)*1000:.2f}, {max(vs)*1000:.2f}]mm")

    print("\n" + "=" * 90)
    print("success rate, paired by (object, seed)")
    print("=" * 90)
    for obj in OBJECTS:
        legacy = {r["seed"]: r["success"] for r in records if r["object"] == obj and r["arm"] == "legacy"}
        corrected = {r["seed"]: r["success"] for r in records if r["object"] == obj and r["arm"] == "corrected"}
        seeds = sorted(set(legacy) & set(corrected))
        n_leg = sum(legacy[s] for s in seeds)
        n_cor = sum(corrected[s] for s in seeds)
        both = sum(legacy[s] and corrected[s] for s in seeds)
        leg_only = sum(legacy[s] and not corrected[s] for s in seeds)
        cor_only = sum(corrected[s] and not legacy[s] for s in seeds)
        neither = sum(not legacy[s] and not corrected[s] for s in seeds)
        print(f"  {obj:10s} legacy={n_leg}/{len(seeds)}  corrected={n_cor}/{len(seeds)}  "
              f"(both_succeed={both} legacy_only={leg_only} corrected_only={cor_only} neither={neither})")
        # How many legacy FAILURES were purely reference misalignment: the
        # trial fails under P0 but succeeds under P1 at the SAME seed --
        # the most important quantity per the requested analysis.
        pure_reference_failures = sum(
            (not legacy[s]) and corrected[s] for s in seeds)
        print(f"    -> of {len(seeds)-n_leg} legacy failures, {pure_reference_failures} "
              f"were purely reference misalignment (fail under P0, succeed under P1 at same seed)")

    print("\n" + "=" * 90)
    print("joint6 comparison (does the wrist-fix/joint6 conclusion still hold post-correction?)")
    print("=" * 90)
    for obj in OBJECTS:
        for arm in ("legacy", "corrected"):
            vs = [r["joint6"] for r in records
                 if r["object"] == obj and r["arm"] == arm and r["joint6"] is not None]
            if vs:
                print(f"  {obj:10s} {arm:10s} n={len(vs):3d}  mean_joint6={np.mean(vs):+.4f}rad  "
                      f"std={np.std(vs):.4f}  range=[{min(vs):+.4f}, {max(vs):+.4f}]")

    print("\n" + "=" * 90)
    print("bilateral contact rate (both fingers touching object at end of trial)")
    print("=" * 90)
    for obj in OBJECTS:
        for arm in ("legacy", "corrected"):
            vs = [r["bilateral_contact"] for r in records if r["object"] == obj and r["arm"] == arm]
            if vs:
                print(f"  {obj:10s} {arm:10s} bilateral={sum(vs)}/{len(vs)}")


if __name__ == "__main__":
    main()
