"""Phase 2Y instrument qualification: all four gates, frozen thresholds.

Gate 1  dY=0 vs baseline  <= frozen thresholds (all four metrics)
Gate 2  measured finger shift == commanded dY (<0.1mm)
Gate 3  PRE-CONTACT EEF trajectory <= frozen thresholds
        (windowed to before first finger-object contact: requiring
         post-contact equivalence would classify the mechanism under test
         as a confound)
Gate 4  no new unintended contacts / proximity violations vs baseline
        (finger<->table / palm / opposite-finger / arm)

Thresholds come from calib/phase2y_noise_floor.json and are NOT recomputed.
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco, numpy as np
from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for, PhaseTracker
from scripts.piper_phase2y_smoke import finger_geom_ids, eef_local_axis_in_body

FLOOR = json.loads((ROOT / "calib" / "phase2y_noise_floor.json").read_text())["floor"]
TH = {k: FLOOR[k]["threshold_1.25x"] for k in
      ("max_pos_m", "rms_pos_m", "max_ori_deg", "refresh_pos_m")}
SEEDS = [5001, 5002, 5003]
PROBE_EVERY = 10


def geom_groups(m, fingers):
    g = {"table": set(), "palm": set(), "arm": set()}
    for i in range(m.ngeom):
        n = (m.geom(i).name or "")
        if i in fingers:
            continue
        if "table" in n:
            g["table"].add(i)
        elif "robot0_link" in n or "base_link" in n:
            g["arm"].add(i)
        elif "gripper" in n or "eef" in n or "hand" in n:
            g["palm"].add(i)
    return g


class Q(PhaseTracker):
    def __init__(self, env, fingers, groups, obj_geoms):
        super().__init__(env=env, obj_name="pear")
        self.f, self.g, self.obj = fingers, groups, obj_geoms
        self.pos, self.mat = [], []
        self.refresh = None
        self.first_obj_contact = None
        self.touch = {k: None for k in list(groups) + ["opp_finger"]}
        self.mindist = {k: np.inf for k in groups}
        self._s = 0

    def __call__(self, env):
        m, d = env.sim.model._model, env.sim.data._data
        sid = m.site("robot0_eef_site").id
        self.pos.append(d.site_xpos[sid].copy())
        self.mat.append(d.site_xmat[sid].reshape(3, 3).copy())
        self._s += 1
        for i in range(d.ncon):
            c = d.contact[i]; pair = {c.geom1, c.geom2}
            if pair & self.f:
                if pair & self.obj and self.first_obj_contact is None:
                    self.first_obj_contact = len(self.pos) - 1
                for k, gs in self.g.items():
                    if pair & gs and self.touch[k] is None:
                        self.touch[k] = self._s
                if len(pair & self.f) == 2 and self.touch["opp_finger"] is None:
                    self.touch["opp_finger"] = self._s
        if self._s % PROBE_EVERY == 0:
            for k, gs in self.g.items():
                for fg in self.f:
                    for og in list(gs)[:6]:
                        dist = float(mujoco.mj_geomDistance(m, d, fg, og, 0.5, None))
                        self.mindist[k] = min(self.mindist[k], dist)

    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh" and self.refresh is None:
            m, d = self._env.sim.model._model, self._env.sim.data._data
            self.refresh = d.site_xpos[m.site("robot0_eef_site").id].copy()


def run(seed, dY_mm, patch):
    np.random.seed(seed)
    env = PiperMultiObjectScene(robots="Piper", ycb_objects=scene_objects_for("pear"),
                                has_renderer=False, has_offscreen_renderer=False,
                                use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        m, d = env.sim.model._model, env.sim.data._data
        fg = set(finger_geom_ids(m)); groups = geom_groups(m, fg)
        objg = ppp._object_contact_geoms(env, "pear")
        eef = m.site("robot0_eef_site").id

        def fpos():
            R = d.site_xmat[eef].reshape(3, 3); p = d.site_xpos[eef]
            return np.array([(d.geom_xpos[g] - p) @ R for g in sorted(fg)])

        b4 = fpos()
        if patch and dY_mm != 0.0:
            for g in fg:
                m.geom_pos[g] = m.geom_pos[g] + eef_local_axis_in_body(m, d, g, 1) * (dY_mm / 1000)
            mujoco.mj_forward(m, d)
        measured = float(np.mean(fpos()[:, 1] - b4[:, 1])) * 1000

        q = Q(env, fg, groups, objg)
        res = ppp.run_pick_and_place(env, "pear", use_oriented_grasp=True, verbose=False,
                                     candidate_selection=None,
                                     wrist_friendly_orientation=True, step_hook=q)
        return {"seed": seed, "dY": dY_mm, "measured": measured,
                "success": bool(res.get("success")), "pos": np.array(q.pos),
                "mat": np.array(q.mat), "refresh": q.refresh,
                "first_obj_contact": q.first_obj_contact,
                "touch": q.touch, "mindist": q.mindist}
    finally:
        env.close()


def cmp_traj(a, b, upto=None):
    n = min(len(a["pos"]), len(b["pos"]))
    if upto: n = min(n, upto)
    dp = np.linalg.norm(a["pos"][:n] - b["pos"][:n], axis=1)
    ang = [np.degrees(np.arccos(np.clip((np.trace(a["mat"][i].T @ b["mat"][i]) - 1) / 2, -1, 1)))
           for i in range(n)]
    ref = (float(np.linalg.norm(a["refresh"] - b["refresh"]))
           if a["refresh"] is not None and b["refresh"] is not None else 0.0)
    return {"max_pos_m": float(dp.max()), "rms_pos_m": float(np.sqrt((dp**2).mean())),
            "max_ori_deg": float(max(ang)), "refresh_pos_m": ref}


def check(c, keys=("max_pos_m","rms_pos_m","max_ori_deg","refresh_pos_m")):
    return all(c[k] <= TH[k] for k in keys), {k: (c[k], TH[k], c[k] <= TH[k]) for k in keys}


def main():
    print("frozen thresholds:", {k: f"{v:.3e}" for k, v in TH.items()}, "\n")
    base = {s: run(s, 0.0, False) for s in SEEDS}

    print("GATE 1  dY=0 vs baseline")
    g1 = True
    for s in SEEDS:
        z = run(s, 0.0, True)
        ok, det = check(cmp_traj(base[s], z)); g1 &= ok and (z["success"] == base[s]["success"])
        print(f"  seed={s} succ {base[s]['success']}->{z['success']}  " +
              "  ".join(f"{k.split('_')[0]}={v:.2e}{'OK' if p else 'FAIL'}" for k,(v,t,p) in det.items()))

    print("\nGATE 2 / 3 / 4  (dY = -15, +15)")
    g2 = g3 = g4 = True
    for dY in (-15.0, 15.0):
        for s in SEEDS:
            r = run(s, dY, True)
            e = abs(r["measured"] - dY); g2 &= e < 0.1
            w = r["first_obj_contact"] or len(r["pos"])
            ok3, d3 = check(cmp_traj(base[s], r, upto=w), ("max_pos_m","rms_pos_m","max_ori_deg"))
            g3 &= ok3
            newt = [k for k, v in r["touch"].items()
                    if v is not None and base[s]["touch"].get(k) is None]
            g4 &= not newt
            print(f"  dY={dY:+6.1f} seed={s} shift_err={e:.3f}mm "
                  f"preC_max={d3['max_pos_m'][0]:.2e}({'OK' if ok3 else 'FAIL'}) "
                  f"window={w} newcontacts={newt or 'none'} "
                  f"mindist={{{', '.join(f'{k}:{v:.3f}' for k,v in r['mindist'].items())}}}")

    print(f"\nGATE 1 {'PASS' if g1 else 'FAIL'}   GATE 2 {'PASS' if g2 else 'FAIL'}   "
          f"GATE 3 {'PASS' if g3 else 'FAIL'}   GATE 4 {'PASS' if g4 else 'FAIL'}")
    print("INSTRUMENT QUALIFIED" if (g1 and g2 and g3 and g4)
          else "NOT QUALIFIED -- full sweep remains blocked")


if __name__ == "__main__":
    main()
