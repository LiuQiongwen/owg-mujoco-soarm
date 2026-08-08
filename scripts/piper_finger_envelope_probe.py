"""Where does the object sit along the FINGER's finite contact envelope?

Everything measured so far has been about where the eef/capture point goes.
This measures one level finer: given the capture point, whereabouts along
the finger's usable length does the object actually land, and where does
first contact happen.

Motivation: the finger contact region spans eef-local Z of [-71.5, +5.0]mm
-- strongly asymmetric, ~76mm long, and NOT centred on the eef site. So an
aim shift of +/-15mm can change which part of that envelope the object
occupies without changing any of the quantities already ruled out
(contact-local width, antipodal score, IK residual, joint margin, joint
tracking error, frame semantics).

Recorded per trial, decomposed along the three grasp-frame axes rather than
as a single Euclidean distance -- the longitudinal (finger-length) axis is
the one of interest and would be hidden in a norm:

    obj_rel_finger_closing_mm       (x: jaw closing axis)
    obj_rel_finger_lateral_mm       (y: across the finger face)
    obj_rel_finger_longitudinal_mm  (z: along the finger length)
    obj_envelope_fraction           (0 = finger root end, 1 = tip end)

Plus the contact-sequence terms, which need no swept-volume model to answer
"does the offset change where and when first contact happens":

    first_contact_step_left / _right
    first_contact_longitudinal_left / _right   (eef-local Z of the contact)
    object_z_at_first_contact

Zero production diff -- same descend-only offset injection as P2.

Run:  conda run -n tango python scripts/piper_finger_envelope_probe.py [n_seeds]
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
from scripts.piper_tcp_correction_ab import scene_objects_for
from scripts.piper_cross_section_intervention import AimOffsetArmIK, _ACTIVE
from scripts.piper_execution_trace import ExecutionTracer
from scripts.piper_contact_local_features import finger_contact_zrange

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "piper_finger_envelope.jsonl"

OBJECTS = ["cracker", "mustard", "pear"]
OFFSETS_M = [-0.015, -0.0075, 0.0, 0.0075, 0.015]


class EnvelopeTracer(ExecutionTracer):
    """Tracks first contact per side in the eef's own frame, and the object's
    placement within the finger envelope at close."""

    def __init__(self, env, obj_name, zlo, zhi):
        super().__init__(env, obj_name)
        self.zlo, self.zhi = zlo, zhi
        self.first = {"left": None, "right": None}
        self.at_close = None
        self._step = 0

    def _eef_frame(self):
        m, d = self._env.sim.model._model, self._env.sim.data._data
        sid = m.site("robot0_eef_site").id
        return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()

    def __call__(self, env):
        self._step += 1
        m, d = env.sim.model._model, env.sim.data._data
        if all(v is not None for v in self.first.values()):
            return
        p, R = self._eef_frame()
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if not (pair & self.obj_geoms):
                continue
            side = "left" if (pair & self.left_geoms) else ("right" if (pair & self.right_geoms) else None)
            if side is None or self.first[side] is not None:
                continue
            loc = (np.array(c.pos) - p) @ R
            self.first[side] = {
                "step": self._step,
                "longitudinal_mm": float(loc[2]) * 1000,
                "closing_mm": float(loc[0]) * 1000,
                "obj_z": float(d.xpos[self.body_id][2]),
            }

    def set_phase(self, name):
        super().set_phase(name)
        # Sample BEFORE the jaws close. Sampling at "lift" (post-close) is
        # useless for this question: the closing action itself pulls the
        # object into the jaws, normalising its relative position, so the
        # measurement comes out flat by construction regardless of the aim
        # offset (observed: lateral 1.20mm and longitudinal 20.1 vs 20.0mm
        # at -15 vs +15mm offset, while pre-close first-contact terms varied
        # by 4-6mm over the same trials).
        if name == "descend_refresh" and self.at_close is None:
            m, d = self._env.sim.model._model, self._env.sim.data._data
            p, R = self._eef_frame()
            rel = (d.xpos[self.body_id] - p) @ R
            span = self.zhi - self.zlo
            self.at_close = {
                "closing_mm": float(rel[0]) * 1000,
                "lateral_mm": float(rel[1]) * 1000,
                "longitudinal_mm": float(rel[2]) * 1000,
                "envelope_fraction": float((rel[2] - self.zlo) / span) if span else None,
            }


def run_trial(obj_name, seed, offset, zcache):
    np.random.seed(seed)
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=scene_objects_for(obj_name),
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        if "z" not in zcache:
            zcache["z"] = finger_contact_zrange(env)
        zlo, zhi = zcache["z"]
        original = ppp.ArmIK
        ppp.ArmIK = AimOffsetArmIK
        tracer = EnvelopeTracer(env, obj_name, zlo, zhi)
        _ACTIVE["offset"] = offset
        _ACTIVE["log"] = {}
        real_init = AimOffsetArmIK.__init__

        def _cap(self, e, _o=real_init, _t=tracer):
            _o(self, e)
            self._phase_tracker = _t

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

        rec = {"object": obj_name, "seed": seed, "offset_mm": offset * 1000,
               "success": bool(res.get("success"))}
        if tracer.at_close:
            rec.update({f"obj_rel_finger_{k}": v for k, v in tracer.at_close.items()})
        for side in ("left", "right"):
            f = tracer.first[side]
            if f:
                rec[f"first_contact_step_{side}"] = f["step"]
                rec[f"first_contact_longitudinal_{side}_mm"] = f["longitudinal_mm"]
                rec[f"first_contact_closing_{side}_mm"] = f["closing_mm"]
        s = tracer.snaps.get("lift")
        rec["gripper_q_at_close"] = s["gripper_q"][0] if s else None
        return rec
    finally:
        env.close()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seeds = list(range(5001, 5001 + n))
    zcache = {}
    rows = []
    with OUT.open("w") as fh:
        for obj in OBJECTS:
            for off in OFFSETS_M:
                for s in seeds:
                    try:
                        r = run_trial(obj, s, off, zcache)
                    except Exception as e:
                        print(f"  {obj} {off} {s} ERROR {type(e).__name__}: {e}")
                        continue
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                print(f"[{obj:8s} {off*1000:+6.1f}] done")

    print(f"\nfinger contact envelope, eef-local Z: "
          f"[{zcache['z'][0]*1000:.1f}, {zcache['z'][1]*1000:.1f}] mm")
    keys = ["obj_rel_finger_longitudinal_mm", "obj_rel_finger_envelope_fraction",
            "obj_rel_finger_closing_mm", "first_contact_step_left",
            "first_contact_step_right", "first_contact_longitudinal_left_mm",
            "first_contact_longitudinal_right_mm", "gripper_q_at_close"]
    for obj in OBJECTS:
        print(f"\n=== {obj} ===")
        print(f"{'off':>7s}{'succ':>7s}" + "".join(f"{k[:13]:>15s}" for k in keys))
        for off in OFFSETS_M:
            sub = [r for r in rows if r["object"] == obj
                   and abs(r["offset_mm"] - off * 1000) < 1e-6]
            if not sub:
                continue
            cells = []
            for k in keys:
                v = [r[k] for r in sub if r.get(k) is not None]
                cells.append(f"{np.mean(v):15.2f}" if v else f"{'-':>15s}")
            print(f"{off*1000:+7.1f}{sum(r['success'] for r in sub):4d}/{len(sub):<2d}"
                  + "".join(cells))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
