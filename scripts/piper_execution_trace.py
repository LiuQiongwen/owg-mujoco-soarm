"""Outcome-conditioned execution trace for Piper -- record first, attribute later.

Motivation (docs/PIPER_TCP_PREMISE_RETRACTION_20260807.md): three proposed
failure mechanisms in a row (transit_high non-convergence, the "65.6mm TCP
offset", grasp-height-above-object) were each proposed first and tested
second, and each was withdrawn. The first was a tautological label, the
second a measurement artifact, the third a coincidence. So this script
deliberately inverts the order: it records CONTINUOUS per-phase quantities
over matched success/failure rollouts and only afterwards asks which
variables actually separate the two.

Explicitly NOT here: any "first failed phase" label. That construct is what
produced the transit_high tautology (transit_high is the first IK phase and
essentially never converges, so it was assigned to every failure regardless
of cause). Nothing in this script assigns blame to a phase.

Separation is scored by AUC (rank-based, distribution-free): 0.5 = the
variable carries no information about the outcome, 1.0 or 0.0 = perfect
separation in either direction. Reported alongside group means so a large
AUC driven by one outlier is visible rather than hidden.

Runs against CURRENT production code (post-revert), on the three objects
with meaningful and differing historical rates, so both classes are
well-populated.

Run:  conda run -n tango python scripts/piper_execution_trace.py
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
    LegacyArmIK, PhaseTracker, scene_objects_for,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "piper_execution_trace.jsonl"

OBJECTS = ["cracker", "pear", "mustard"]
SEEDS = list(range(3001, 3013))   # 12 seeds x 3 objects = 36 rollouts


def _geom_sets(model):
    def matching(*subs):
        return {i for i in range(model.ngeom)
               if all(s in (model.geom(i).name or "") for s in subs)}
    return matching("finger7", "collision"), matching("finger8", "collision")


def _mesh_world(model, data, body_id):
    pts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id or model.geom_dataid[gid] < 0:
            continue
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vl = model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
        pts.append(vl @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


def _quat_angle(q0, q1):
    d = float(np.clip(abs(np.dot(np.asarray(q0), np.asarray(q1))), -1.0, 1.0))
    return float(np.degrees(2 * np.arccos(d)))


class ExecutionTracer(PhaseTracker):
    """Snapshots continuous state at phase boundaries. Purely observational --
    __call__ stays a no-op on the per-step path so this doesn't perturb
    timing-sensitive behaviour, and nothing here feeds back into control."""

    def __init__(self, env, obj_name):
        super().__init__(env=env, obj_name=obj_name)
        self.snaps = {}
        m = env.sim.model._model
        self.left_geoms, self.right_geoms = _geom_sets(m)
        self.obj_geoms = ppp._object_contact_geoms(env, obj_name)
        self.body_id = env.object_body_ids[obj_name]

    def _snapshot(self):
        env = self._env
        m, d = env.sim.model._model, env.sim.data._data
        obj_pts = _mesh_world(m, d, self.body_id)

        fz = []
        for gid in (self.left_geoms | self.right_geoms):
            mid = m.geom_dataid[gid]
            if mid < 0:
                continue
            adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            vl = m.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
            fz.append(vl @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid])
        fz = np.concatenate(fz, axis=0) if fz else np.zeros((0, 3))

        left = right = False
        min_pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if not (pair & self.obj_geoms):
                continue
            if pair & self.left_geoms:
                left = True
            if pair & self.right_geoms:
                right = True
            min_pen = min(min_pen, float(c.dist))

        overlap = None
        if len(obj_pts) and len(fz):
            overlap = float(min(obj_pts[:, 2].max(), fz[:, 2].max())
                            - max(obj_pts[:, 2].min(), fz[:, 2].min()))

        j7 = next(j for j in range(m.njnt) if "joint7" in (m.joint(j).name or ""))
        j8 = next(j for j in range(m.njnt) if "joint8" in (m.joint(j).name or ""))

        return {
            "obj_pos": d.xpos[self.body_id].copy().tolist(),
            "obj_quat": d.xquat[self.body_id].copy().tolist(),
            "left_contact": left, "right_contact": right,
            "bilateral": bool(left and right),
            "min_contact_dist_m": min_pen,
            "finger_obj_vertical_overlap_m": overlap,
            "gripper_q": [float(d.qpos[m.joint(j7).qposadr[0]]),
                          float(d.qpos[m.joint(j8).qposadr[0]])],
        }

    def set_phase(self, name):
        super().set_phase(name)
        if name in ("descend", "descend_refresh", "lift", "transit_above_tray") \
                and name not in self.snaps:
            self.snaps[name] = self._snapshot()


def run_trial(obj_name, seed):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        original = ppp.ArmIK
        ppp.ArmIK = LegacyArmIK          # pure pass-through recorder
        tracer = ExecutionTracer(env, obj_name)
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
        m = env.sim.model._model
        limits = ppp.REAL_JOINT_LIMITS

        def phase_metrics(prefix):
            cs = [c for c in calls if (c["phase"] or "").startswith(prefix)]
            if not cs:
                return {}
            c = cs[-1]
            q = np.array(c["qpos"])
            margins = [min(q[i] - lo, hi - q[i]) for i, (lo, hi) in enumerate(limits)]
            return {
                f"{prefix}_pos_err_cm": c["err_cm"],
                f"{prefix}_ori_err_deg": c.get("ori_err_deg"),
                f"{prefix}_min_joint_margin_rad": float(min(margins)),
                f"{prefix}_joint6_rad": float(q[5]),
                f"{prefix}_converged": bool(c["converged"]),
            }

        rec = {"object": obj_name, "seed": seed, "success": bool(res.get("success")),
               "dist_to_tray": res.get("dist_to_tray")}
        for p in ("transit_high", "approach", "descend", "lift", "transit_above_tray"):
            rec.update(phase_metrics(p))

        s_desc = tracer.snaps.get("descend")
        s_close = tracer.snaps.get("lift")
        s_tray = tracer.snaps.get("transit_above_tray")

        if s_close:
            rec["bilateral_at_close"] = s_close["bilateral"]
            rec["left_contact_at_close"] = s_close["left_contact"]
            rec["right_contact_at_close"] = s_close["right_contact"]
            rec["n_sides_contact_at_close"] = int(s_close["left_contact"]) + int(s_close["right_contact"])
            rec["min_contact_dist_at_close_mm"] = s_close["min_contact_dist_m"] * 1000
            rec["finger_obj_overlap_at_close_mm"] = (
                s_close["finger_obj_vertical_overlap_m"] * 1000
                if s_close["finger_obj_vertical_overlap_m"] is not None else None)
            rec["gripper_q_at_close"] = s_close["gripper_q"][0]
            rec["obj_z_at_close_m"] = s_close["obj_pos"][2]

        if s_desc and s_close:
            p0 = np.array(s_desc["obj_pos"]); p1 = np.array(s_close["obj_pos"])
            rec["pre_close_drift_mm"] = float(np.linalg.norm(p1 - p0)) * 1000
            rec["pre_close_drift_xy_mm"] = float(np.linalg.norm((p1 - p0)[:2])) * 1000
            rec["pre_close_rotation_deg"] = _quat_angle(s_desc["obj_quat"], s_close["obj_quat"])

        if s_close and s_tray:
            rec["lift_height_gain_mm"] = (s_tray["obj_pos"][2] - s_close["obj_pos"][2]) * 1000
            rec["bilateral_at_tray"] = s_tray["bilateral"]

        return rec
    finally:
        env.close()


def auc(pos, neg):
    """Rank-based AUC: P(random success value > random failure value), ties
    counted at 0.5. Distribution-free, so a single wild outlier can't
    manufacture separation the way a mean difference can."""
    if not pos or not neg:
        return None
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def analyze(records):
    succ = [r for r in records if r["success"]]
    fail = [r for r in records if not r["success"]]
    print("\n" + "=" * 96)
    print(f"outcome-conditioned separation:  {len(succ)} successes vs {len(fail)} failures")
    print("=" * 96)
    if not succ or not fail:
        print("  need both classes populated to compute separation")
        return

    keys = set()
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float, bool)) and k not in ("seed",):
                keys.add(k)
    keys.discard("success")

    rows = []
    for k in sorted(keys):
        p = [float(r[k]) for r in succ if r.get(k) is not None]
        n = [float(r[k]) for r in fail if r.get(k) is not None]
        if len(p) < 3 or len(n) < 3:
            continue
        a = auc(p, n)
        rows.append((abs(a - 0.5), a, k, np.mean(p), np.mean(n), len(p), len(n)))

    rows.sort(reverse=True)
    print(f"{'variable':42s} {'AUC':>6s} {'succ_mean':>11s} {'fail_mean':>11s}   n")
    print("-" * 96)
    for _, a, k, mp, mn, np_, nn in rows:
        flag = "  <<<" if abs(a - 0.5) >= 0.25 else ""
        print(f"{k:42s} {a:6.2f} {mp:11.3f} {mn:11.3f}   {np_}/{nn}{flag}")
    print("\nAUC 0.5 = no information about outcome; >=0.75 or <=0.25 flagged.")
    print("A flagged variable is a CANDIDATE separator only -- it still needs a")
    print("directed intervention before any causal claim (three prior mechanisms")
    print("failed exactly at that step).")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as fh:
        for obj in OBJECTS:
            for seed in SEEDS:
                try:
                    rec = run_trial(obj, seed)
                except Exception as e:
                    print(f"  {obj} seed={seed} ERROR {type(e).__name__}: {e}")
                    continue
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[{obj:8s} seed={seed}] success={rec['success']!s:5s} "
                      f"bilateral={rec.get('bilateral_at_close')!s:5s} "
                      f"drift={rec.get('pre_close_drift_mm', float('nan')):6.1f}mm "
                      f"overlap={rec.get('finger_obj_overlap_at_close_mm', float('nan')):6.1f}mm")

    by_obj = {}
    for r in records:
        by_obj.setdefault(r["object"], []).append(r["success"])
    print("\nper-object success:")
    for o, v in by_obj.items():
        print(f"  {o:9s} {sum(v)}/{len(v)}")

    analyze(records)
    print(f"\nwrote {len(records)} rollouts to {OUT}")


if __name__ == "__main__":
    main()
