"""Baseline-vs-baseline reproducibility calibration for Phase 2Y gates.

Per Amendment 1: a null criterion must first be shown achievable by the
unmodified system against itself (R6). Establishes the empirical floor that
Gates 1 and 3 are judged against, using four metrics rather than a single
max, since one instantaneous spike dominates a max.

Threshold rule, FROZEN here before any treatment data:
    threshold = observed baseline-baseline maximum x 1.25
Chosen over a percentile estimate because n=10-20 pairs cannot support a
credible 99th-percentile claim.
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite import piper_pick_and_place as ppp
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for, PhaseTracker

OUT = ROOT / "calib" / "phase2y_noise_floor.json"
N_PAIRS = 10


class Rec(PhaseTracker):
    def __init__(self, env):
        super().__init__(env=env, obj_name="pear")
        self.pos, self.mat = [], []
        self.at_refresh = None
    def __call__(self, env):
        m, d = env.sim.model._model, env.sim.data._data
        sid = m.site("robot0_eef_site").id
        self.pos.append(d.site_xpos[sid].copy())
        self.mat.append(d.site_xmat[sid].reshape(3, 3).copy())
    def set_phase(self, name):
        super().set_phase(name)
        if name == "descend_refresh" and self.at_refresh is None:
            m, d = self._env.sim.model._model, self._env.sim.data._data
            self.at_refresh = d.site_xpos[m.site("robot0_eef_site").id].copy()


def run(seed):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for("pear"),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        rec = Rec(env)
        res = ppp.run_pick_and_place(env, "pear", use_oriented_grasp=True, verbose=False,
                                     candidate_selection=None,
                                     wrist_friendly_orientation=True, step_hook=rec)
        return {"success": bool(res.get("success")),
                "pos": np.array(rec.pos), "mat": np.array(rec.mat),
                "refresh": rec.at_refresh}
    finally:
        env.close()


def compare(a, b):
    n = min(len(a["pos"]), len(b["pos"]))
    dp = np.linalg.norm(a["pos"][:n] - b["pos"][:n], axis=1)
    ang = []
    for i in range(n):
        R = a["mat"][i].T @ b["mat"][i]
        c = float(np.clip((np.trace(R) - 1) / 2, -1, 1))
        ang.append(np.degrees(np.arccos(c)))
    ref = (float(np.linalg.norm(a["refresh"] - b["refresh"]))
           if a["refresh"] is not None and b["refresh"] is not None else np.nan)
    return {"max_pos_m": float(dp.max()), "rms_pos_m": float(np.sqrt((dp**2).mean())),
            "max_ori_deg": float(max(ang)), "refresh_pos_m": ref,
            "success_match": a["success"] == b["success"]}


def main():
    rows = []
    for i in range(N_PAIRS):
        seed = 5001 + i
        a, b = run(seed), run(seed)
        c = compare(a, b); c["seed"] = seed
        rows.append(c)
        print(f"  seed={seed}  max_pos={c['max_pos_m']:.3e}  rms={c['rms_pos_m']:.3e}  "
              f"max_ori={c['max_ori_deg']:.4f}deg  refresh={c['refresh_pos_m']:.3e}  "
              f"succ_match={c['success_match']}")

    def agg(k):
        v = [r[k] for r in rows if not np.isnan(r[k])]
        return {"max": float(np.max(v)), "mean": float(np.mean(v)),
                "threshold_1.25x": float(np.max(v) * 1.25)}

    floor = {k: agg(k) for k in ("max_pos_m", "rms_pos_m", "max_ori_deg", "refresh_pos_m")}
    floor["n_pairs"] = N_PAIRS
    floor["success_match_rate"] = f"{sum(r['success_match'] for r in rows)}/{len(rows)}"
    floor["rule"] = "threshold = baseline-baseline observed maximum x 1.25 (frozen pre-treatment)"

    print("\nFROZEN GATE THRESHOLDS (baseline-baseline max x 1.25):")
    for k, v in floor.items():
        if isinstance(v, dict):
            print(f"  {k:16s} max={v['max']:.3e}  ->  threshold={v['threshold_1.25x']:.3e}")
    print(f"  success match: {floor['success_match_rate']}")
    OUT.write_text(json.dumps({"pairs": rows, "floor": floor}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
