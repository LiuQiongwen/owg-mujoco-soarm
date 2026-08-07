"""P1: outcome-conditioned rollout dataset for Piper, against PIPER_BASELINE_V1.

Extends scripts/piper_execution_trace.py from 36 rollouts to a proper
dataset, and adds per-timestep trajectory recording so the
trajectory-prefix question can be answered: at what point in execution do
success and failure first become separable?

Three guards are enforced IN CODE, not by discipline, because each
corresponds to an error this investigation actually made:

  1. OUTCOME_DERIVED -- variables that the outcome is defined by (notably
     dist_to_tray, which `success` is literally thresholded from) are
     refused entry to any analysis. Attempting to score one raises.
  2. No phase-blame labels. There is no "first failed phase" field; that
     construct manufactured the transit_high tautology.
  3. Within-object + leave-one-object-out are computed alongside every
     pooled statistic, since object-level confounding already faked three
     strong-looking pooled separators.

Object set spans the shape categories requested: box (cracker), cylinder
(can), tapered bottle (mustard), sphere-ish (pear), irregular (drill).
`clamp` is excluded on measured evidence -- 150mm wide at grasp height
against a measured 100mm opening (outputs/piper_object_grasp_geometry.json),
i.e. mechanically infeasible, not a failure to learn from.

Run:  conda run -n tango python scripts/piper_outcome_dataset.py [n_seeds]
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
from scripts.piper_execution_trace import ExecutionTracer, _mesh_world, _quat_angle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "piper_outcome_dataset.jsonl"
TRAJ_OUT = ROOT / "outputs" / "piper_outcome_trajectories.jsonl"

# Verified spawnable under PIPER_BASELINE_V1 (3/3 seeds each). `banana` and
# `drill` are EXCLUDED because they cannot be spawned at all under the frozen
# baseline -- PiperMultiObjectScene's placement sampler rejects their
# horizontal_radius against the current table size (0/3 seeds, ValueError
# "high - low < 0"). That also means the historical
# experiment_results_oriented_banana_*.json / _drill_* runs are NOT
# reproducible under this baseline and must not be used as comparisons.
# `clamp` spawns fine but is mechanically infeasible (150mm at grasp height
# vs a measured 100mm opening), so its failures carry no mechanism
# information -- excluded from the learning set by design, not oversight.
# Remaining shape coverage: box, cylinder, tapered bottle, round.
OBJECTS = ["cracker", "can", "mustard", "pear"]

# Variables the outcome is DEFINED by. Refused entry to analysis -- see
# docs/PIPER_EXECUTION_TRACE_20260807.md (dist_to_tray scored a perfect
# AUC 0.00 purely because success is thresholded from it).
OUTCOME_DERIVED = {"dist_to_tray", "success", "final_pos"}

TRAJ_EVERY = 10   # record one trajectory sample per N env steps


class TrajectoryTracer(ExecutionTracer):
    """ExecutionTracer + a downsampled per-timestep trajectory, so
    success/failure separability can be evaluated as a function of how much
    of the rollout has been observed."""

    def __init__(self, env, obj_name):
        super().__init__(env, obj_name)
        self.traj = []
        self._step = 0
        m = env.sim.model._model
        self._eef_site = m.site("robot0_eef_site").id
        self._arm_qadr = [m.joint(n).qposadr[0] for n in ppp.JOINTS]
        self._arm_dadr = [m.joint(n).dofadr[0] for n in ppp.JOINTS]
        self._j7 = m.joint(next(j for j in range(m.njnt)
                                if "joint7" in (m.joint(j).name or ""))).qposadr[0]

    def __call__(self, env):
        self._step += 1
        if self._step % TRAJ_EVERY:
            return
        m, d = env.sim.model._model, env.sim.data._data
        left = right = False
        pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if not (pair & self.obj_geoms):
                continue
            if pair & self.left_geoms:
                left = True
            if pair & self.right_geoms:
                right = True
            pen = min(pen, float(c.dist))
        eef = d.site_xpos[self._eef_site]
        obj = d.xpos[self.body_id]
        self.traj.append({
            "t": self._step,
            "phase": self.current_phase,
            "eef": [round(float(x), 5) for x in eef],
            "obj": [round(float(x), 5) for x in obj],
            "obj_quat": [round(float(x), 5) for x in d.xquat[self.body_id]],
            "rel": [round(float(a - b), 5) for a, b in zip(obj, eef)],
            "grip_q": round(float(d.qpos[self._j7]), 5),
            "l": left, "r": right,
            "pen_mm": round(pen * 1000, 3),
            "qvel_max": round(float(np.max(np.abs(d.qvel[self._arm_dadr]))), 4),
        })


def run_trial(obj_name, seed):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        original = ppp.ArmIK
        ppp.ArmIK = LegacyArmIK
        tracer = TrajectoryTracer(env, obj_name)
        holder = {}
        real_init = LegacyArmIK.__init__

        def _cap(self, e, _o=real_init, _t=tracer, _h=holder):
            _o(self, e)
            self._phase_tracker = _t
            _h["ik"] = self

        LegacyArmIK.__init__ = _cap
        try:
            res = ppp.run_pick_and_place(
                env, obj_name, use_oriented_grasp=True, verbose=False,
                candidate_selection=None, wrist_friendly_orientation=True,
                step_hook=tracer)
        finally:
            LegacyArmIK.__init__ = real_init
            ppp.ArmIK = original

        ik = holder["ik"]
        calls = getattr(ik, "_calls", [])
        limits = ppp.REAL_JOINT_LIMITS

        rec = {"object": obj_name, "seed": seed,
               "success": bool(res.get("success")),
               "spawn_pos": res.get("spawn_pos"),
               "grasp_yaw": res.get("grasp_yaw"),
               "candidate_grasp_yaw": res.get("candidate_grasp_yaw")}

        for p in ("transit_high", "approach", "descend", "descend_refresh",
                  "lift", "transit_above_tray", "lower_into_tray"):
            cs = [c for c in calls if (c["phase"] or "").startswith(p)]
            if not cs:
                continue
            c = cs[-1]
            q = np.array(c["qpos"])
            margins = [min(q[i] - lo, hi - q[i]) for i, (lo, hi) in enumerate(limits)]
            rec[f"{p}_pos_err_cm"] = c["err_cm"]
            rec[f"{p}_ori_err_deg"] = c.get("ori_err_deg")
            rec[f"{p}_min_joint_margin_rad"] = float(min(margins))
            rec[f"{p}_joint6_rad"] = float(q[5])
            rec[f"{p}_converged"] = bool(c["converged"])

        s_desc = tracer.snaps.get("descend")
        s_close = tracer.snaps.get("lift")
        s_tray = tracer.snaps.get("transit_above_tray")
        if s_close:
            rec.update({
                "bilateral_at_close": s_close["bilateral"],
                "n_sides_contact_at_close": int(s_close["left_contact"]) + int(s_close["right_contact"]),
                "min_contact_dist_at_close_mm": s_close["min_contact_dist_m"] * 1000,
                "finger_obj_overlap_at_close_mm": (
                    s_close["finger_obj_vertical_overlap_m"] * 1000
                    if s_close["finger_obj_vertical_overlap_m"] is not None else None),
                "gripper_q_at_close": s_close["gripper_q"][0],
                "obj_z_at_close_m": s_close["obj_pos"][2],
            })
        if s_desc and s_close:
            p0, p1 = np.array(s_desc["obj_pos"]), np.array(s_close["obj_pos"])
            rec["pre_close_drift_mm"] = float(np.linalg.norm(p1 - p0)) * 1000
            rec["pre_close_rotation_deg"] = _quat_angle(s_desc["obj_quat"], s_close["obj_quat"])
        if s_close and s_tray:
            rec["lift_height_gain_mm"] = (s_tray["obj_pos"][2] - s_close["obj_pos"][2]) * 1000

        # Contact-onset timing from the trajectory (upstream of close).
        first_contact = next((s["t"] for s in tracer.traj if s["l"] or s["r"]), None)
        first_bilateral = next((s["t"] for s in tracer.traj if s["l"] and s["r"]), None)
        rec["first_contact_step"] = first_contact
        rec["first_bilateral_step"] = first_bilateral
        rec["n_traj_samples"] = len(tracer.traj)

        return rec, {"object": obj_name, "seed": seed,
                     "success": rec["success"], "traj": tracer.traj}
    finally:
        env.close()


def assert_no_outcome_derived(keys):
    bad = OUTCOME_DERIVED & set(keys)
    if bad:
        raise ValueError(
            f"outcome-derived variables refused entry to analysis: {sorted(bad)}. "
            "These are defined by the outcome and predict it by construction.")


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seeds = list(range(4001, 4001 + n_seeds))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with OUT.open("w") as fh, TRAJ_OUT.open("w") as tf:
        for obj in OBJECTS:
            for seed in seeds:
                try:
                    rec, traj = run_trial(obj, seed)
                except Exception as e:
                    print(f"  {obj} seed={seed} ERROR {type(e).__name__}: {e}")
                    continue
                assert_no_outcome_derived([k for k in rec if k != "success"])
                fh.write(json.dumps(rec) + "\n")
                tf.write(json.dumps(traj) + "\n")
                fh.flush(); tf.flush()
                n_ok += 1
                if n_ok % 10 == 0:
                    print(f"  ... {n_ok} rollouts")
            print(f"[{obj}] done")

    rows = [json.loads(l) for l in OUT.open()]
    print(f"\ncollected {len(rows)} rollouts -> {OUT}")
    print(f"trajectories -> {TRAJ_OUT}")
    by = {}
    for r in rows:
        by.setdefault(r["object"], []).append(r["success"])
    for o, v in by.items():
        print(f"  {o:9s} {sum(v):3d}/{len(v):3d}  {100*sum(v)/len(v):5.1f}%")


if __name__ == "__main__":
    main()
