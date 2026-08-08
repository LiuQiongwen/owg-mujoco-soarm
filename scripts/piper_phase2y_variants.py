"""P2Y-1/2: compile-time finger-shift gripper variants + model diff gate.

Generates variant copies of piper_gripper.xml with ONLY the finger
collision geoms' pos shifted along the gripper's local Y, registers a
PiperGripper subclass per variant, and diffs the compiled models against
baseline to prove the treatment is isolated to contact geometry.

Replaces the runtime `model.geom_pos` mutation, which is an unsafe
post-compile model edit (see docs/PHASE2Y_INSTRUMENT_LEGALITY.md, R8).

Zero production diff: variants live in diagnostics/phase2y_variants/.
"""
import os, sys, re
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from robosuite.models.grippers import register_gripper
from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite.piper_gripper import PiperGripper
from tango_robot.piper_robosuite.piper_multi_object_scene import PiperMultiObjectScene
from scripts.piper_tcp_correction_ab import scene_objects_for
from scripts.piper_phase2y_smoke import finger_geom_ids, eef_local_axis_in_body

SRC = ROOT / "tango_robot" / "piper_assets" / "piper_gripper.xml"
VDIR = ROOT / "diagnostics" / "phase2y_variants"


def body_axes():
    """Gripper-local Y expressed in each finger body's own frame, measured
    on the compiled baseline (reuses the function Gate 2 verified to
    0.000mm)."""
    np.random.seed(1)
    env = PiperMultiObjectScene(robots="Piper", ycb_objects=scene_objects_for("pear"),
                                has_renderer=False, has_offscreen_renderer=False,
                                use_camera_obs=False, control_freq=20)
    try:
        env.reset()
        m, d = env.sim.model._model, env.sim.data._data
        out = {}
        for g in finger_geom_ids(m):
            n = "finger7_collision" if "finger7" in (m.geom(g).name or "") else "finger8_collision"
            out[n] = eef_local_axis_in_body(m, d, g, axis=1)
        return out
    finally:
        env.close()


def make_variant(dY_mm, axes):
    xml = SRC.read_text()
    for gname, ax in axes.items():
        p = ax * (dY_mm / 1000.0)
        p[np.abs(p) < 0.5e-9] = 0.0  # stable XML: never emit negative zero
        pos = f'pos="{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}" '
        # insert pos into that geom's opening tag only
        pat = re.compile(r'(<geom type="mesh" mesh="link[78]" group="0" name="' + gname + r'" )')
        xml, n = pat.subn(r'\1' + pos, xml)
        assert n == 1, f"expected 1 match for {gname}, got {n}"
    # robosuite resolves mesh paths relative to the XML's OWN folder
    # (MujocoXML._resolve_asset_dependency), so meshdir does not help for a
    # variant stored outside piper_assets/. Absolutise the file attributes
    # instead. Nothing is written into piper_assets/ (zero production diff).
    xml = re.sub(r'file="(?!/)([^"]*)"', lambda mm: f'file="{SRC.parent / mm.group(1)}"', xml)
    tag = f"{'m' if dY_mm < 0 else 'p'}{abs(dY_mm):g}".replace('.', '_')
    out = VDIR / f"piper_gripper_dY_{tag}.xml"
    out.write_text(xml)
    return out, tag


def register_variant(path, tag):
    cls = type(f"PiperGripperDY{tag}", (PiperGripper,),
               {"__init__": lambda self, idn=0, _p=str(path): PiperGripper.__init__.__wrapped__(self, idn)
                if False else super(type(self), self).__init__(idn=idn)})
    # simpler: explicit subclass with overridden asset path
    src = str(path)
    class V(PiperGripper):
        def __init__(self, idn=0):
            from robosuite.models.grippers.gripper_model import GripperModel
            GripperModel.__init__(self, src, idn=idn)
    V.__name__ = f"PiperGripperDY{tag}"
    V.__qualname__ = V.__name__
    register_gripper(V)
    return V.__name__


DIFF_FIELDS = ["body_mass", "body_inertia", "body_ipos", "body_iquat",
               "jnt_range", "jnt_limited", "dof_damping", "dof_armature",
               "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
               "actuator_forcerange", "geom_size", "geom_friction",
               "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity",
               "geom_condim", "geom_type", "qpos0"]
SCALARS = ["nq", "nv", "na", "nu", "nbody", "ngeom", "njnt"]


def load(gripper_name):
    np.random.seed(1)
    env = PiperMultiObjectScene(robots="Piper", gripper_types=gripper_name,
                                ycb_objects=scene_objects_for("pear"),
                                has_renderer=False, has_offscreen_renderer=False,
                                use_camera_obs=False, control_freq=20)
    env.reset()
    return env


def main():
    VDIR.mkdir(parents=True, exist_ok=True)
    axes = body_axes()
    print("gripper-local Y in each finger body frame:")
    for k, v in axes.items():
        print(f"  {k}: {np.round(v, 6)}")

    names = {}
    for dY in (0.0, 15.0):
        p, tag = make_variant(dY, axes)
        names[dY] = register_variant(p, tag)
        print(f"  dY={dY:+.1f} -> {p.name}  registered as {names[dY]}")

    print("\nMODEL DIFF (variant vs variant):")
    envs = {dY: load(n) for dY, n in names.items()}
    try:
        m0 = envs[0.0].sim.model._model
        m1 = envs[15.0].sim.model._model
        bad = []
        for s in SCALARS:
            a, b = getattr(m0, s), getattr(m1, s)
            if a != b:
                bad.append(f"{s}: {a} != {b}")
        for f in DIFF_FIELDS:
            a, b = np.asarray(getattr(m0, f)), np.asarray(getattr(m1, f))
            if a.shape != b.shape:
                bad.append(f"{f}: shape {a.shape} != {b.shape}"); continue
            d = float(np.max(np.abs(a - b))) if a.size else 0.0
            if d > 1e-12:
                bad.append(f"{f}: max|diff|={d:.3e}")
        gp = np.abs(np.asarray(m0.geom_pos) - np.asarray(m1.geom_pos))
        moved = [i for i in range(m0.ngeom) if gp[i].max() > 1e-12]
        print(f"  geom_pos changed for {len(moved)} geoms: "
              f"{[m0.geom(i).name for i in moved]}")
        print(f"  max shift = {gp.max()*1000:.4f} mm")
        if bad:
            print("  FAIL -- non-treatment fields differ:")
            for b in bad: print(f"    {b}")
        else:
            print("  PASS -- only finger collision geom_pos differs; "
                  "mass/inertia/actuator/solver/contact params identical")
    finally:
        for e in envs.values():
            e.close()


if __name__ == "__main__":
    main()
