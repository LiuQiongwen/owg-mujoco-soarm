#!/usr/bin/env python3
"""SO-ARM101 IK debugging viewer.

Loads the SO-ARM101 scene with a YCB object and runs a slow-motion grasp
sequence in an interactive GLFW window.  Useful for:
  - Verifying IK convergence at various workspace positions
  - Checking jaw alignment and contact geometry
  - Live inspection of grasp metrics (jaw_obj_xy_gap, bilateral_contacts)
  - Pause / resume during any phase

Controls (viewer):
  Space  — pause / resume physics (viewer side)
  Ctrl+C — exit

Usage:
    DISPLAY=:1 MUJOCO_GL=glfw python scripts/debug_viewer_test.py
    DISPLAY=:1 MUJOCO_GL=glfw python scripts/debug_viewer_test.py --object MustardBottle
    DISPLAY=:1 MUJOCO_GL=glfw python scripts/debug_viewer_test.py --slowdown 4 --seed 1
"""
import argparse, os, sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("MUJOCO_GL", "glfw")

from owg_robot.env_soarm import (
    EnvironmentSoArm, TABLE_TOP_Z,
    GRIP_OPEN, GRIP_CLOSED,
    IK_MODE_JAW_POS, IK_MODE_XYZ_ONLY,
    GRASP_MODE_PHYSICS, GRASP_Z_TABLE_MARGIN,
)
from owg_robot.viewer_utils import MujocoViewer, Overlay


def _resolve(name: str) -> str:
    """Accept 'banana' → 'Banana' (load_obj handles 'Ycb' prefix)."""
    from pathlib import Path
    ycb = Path(ROOT) / "owg_robot" / "assets" / "ycb_objects"
    nl = name.lower()
    for d in ycb.iterdir():
        if d.is_dir() and not d.name.startswith("__"):
            short = d.name[3:] if d.name.startswith("Ycb") else d.name
            if short.lower() == nl or d.name.lower() == nl:
                return short
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object",   default="Banana")
    ap.add_argument("--slowdown", type=float, default=3.0,
                    help="Real-time multiplier (1=real time, 3=3× slower)")
    ap.add_argument("--seed",     type=int,   default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    obj_name = _resolve(args.object)

    # ── Build environment ────────────────────────────────────────────────────
    print(f"[debug] Loading EnvironmentSoArm …")
    env = EnvironmentSoArm(grasp_mode=GRASP_MODE_PHYSICS, n_grasp_attempts=1)
    env.reset()

    print(f"[debug] Loading '{obj_name}' …")
    # Spawn on the table (y < 0) so the object rests at TABLE_TOP_Z
    obj_id = env.load_obj(obj_name, pos=[0.0, -0.20, TABLE_TOP_Z + 0.15])
    env.wait_until_still(obj_id)
    env._steps(300)

    obj_com = env.get_obj_com_pos(obj_id)   # body CoM in world frame
    obj_pos = env.get_obj_pos(obj_id)
    x, y   = float(obj_com[0]), float(obj_com[1])
    grasp_z = float(obj_com[2]) + GRASP_Z_TABLE_MARGIN  # offset keeps fixed jaw off table
    hover_z = env.GRIPPER_MOVING_HEIGHT
    print(f"[debug] Object at ({x:.3f}, {y:.3f}, {obj_pos[2]:.3f})  CoM z={obj_com[2]:.3f}  grasp_z={grasp_z:.3f}")

    # ── Overlay setup ────────────────────────────────────────────────────────
    ov = Overlay()
    ov.set_phase("idle")
    ov.add_info("object", obj_name)
    ov.add_info("slowdown", f"{args.slowdown}×")
    ov.add_marker(obj_com, rgba=(0, 1, 0, 0.7), size=0.025, label=obj_name)
    ov.add_marker([x, y, grasp_z], rgba=(1, 0.5, 0, 0.7), size=0.02, label="grasp_z")

    # ── Hook env.step_simulation so the viewer updates during long motions ───
    _step_n   = [0]
    _viewer   = [None]

    orig_step = env.step_simulation

    def _hooked():
        orig_step()
        _step_n[0] += 1
        if _step_n[0] % 4 == 0:
            v = _viewer[0]
            if v is not None:
                try:
                    ov.add_info("sim_step", _step_n[0])
                    ov.add_info("obj_z",    f"{env.get_obj_pos(obj_id)[2]:.4f}")
                except Exception:
                    pass
                v.sync()

    env.step_simulation = _hooked

    # ── Physics sequence ─────────────────────────────────────────────────────
    def run(v: MujocoViewer):
        _viewer[0] = v

        # 0 — idle
        ov.set_phase("idle")
        for _ in range(5):
            if not v.is_alive():
                return
            v.sleep(0.1)

        # 1 — approach hover
        ov.set_phase("approach")
        env.move_gripper(GRIP_OPEN)
        env.move_ee([x, y, hover_z, None], ik_mode=IK_MODE_JAW_POS, max_step=1400)
        env._steps(80)

        # 2 — IK solve at grasp height
        ov.set_phase("ik_solve")
        ok, pe, _ = env._solve_ik_jaw_pos_only(
            np.array([x, y, grasp_z]), reset_to_home=False
        )
        q_grasp = np.array([env.data.qpos[a] for a in env._arm_qpos_adr])
        ov.add_info("ik_ok",      str(ok))
        ov.add_info("ik_pe_mm",   f"{pe * 1000:.1f}")
        v.sync()
        v.sleep(1.5)

        # 3 — descend (park → teleport → restore)
        ov.set_phase("descend")
        slot = env._obj_pool_slot(obj_id)
        jnt  = env.model.joint(f"obj_joint_{slot}")
        adr, vadr = jnt.qposadr[0], jnt.dofadr[0]
        pose = env.data.qpos[adr : adr + 7].copy()
        env.data.qpos[adr + 2] = -100.0
        for arm_adr, act_id, q in zip(env._arm_qpos_adr, env._arm_act_ids, q_grasp):
            env.data.qpos[arm_adr] = q
            env.data.ctrl[act_id]  = q
        mujoco.mj_forward(env.model, env.data)
        env._steps(220)
        env.data.qpos[adr : adr + 7] = pose
        env.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env._steps(120)

        metrics = env.get_grasp_debug_metrics(obj_id)
        ov.add_info("jaw_gap_cm",      f"{metrics.get('jaw_obj_xy_gap', 0) * 100:.1f}")
        ov.add_info("bilateral",       metrics.get("bilateral_contacts", 0))
        ov.add_info("ori_err_rad",     f"{metrics.get('ori_err_norm', 0):.3f}")
        v.sync()
        v.sleep(1.0)

        # 4 — close gripper
        ov.set_phase("close_gripper")
        env.auto_close_gripper(check_contact=False)
        env._steps(140)

        grasped = env.check_grasped_id()
        ov.add_info("grasped_ids", str(grasped))
        v.sync()
        v.sleep(0.5)

        # Kinematic weld after bilateral contacts so lift succeeds despite
        # limited friction from 6 mm sphere colliders.
        weld_obj = grasped[0] if grasped else None
        if weld_obj is not None:
            env._attach_obj(weld_obj)

        # 5 — lift
        ov.set_phase("lift")
        env.move_ee([x, y, hover_z, None], ik_mode=IK_MODE_XYZ_ONLY, max_step=350)
        env._steps(80)

        if weld_obj is not None:
            final_z = env.get_obj_pos(weld_obj)[2]
            lifted  = final_z > env.Z_TABLE_TOP + 0.07
            ov.add_info("lifted",   str(lifted))
            ov.add_info("final_z",  f"{final_z:.4f}")
            if not lifted:
                env._detach_obj(weld_obj)

        # 6 — hold
        ov.set_phase("hold")
        for _ in range(30):
            if not v.is_alive():
                return
            v.sleep(0.05)

        # 7 — release
        ov.set_phase("release")
        env.move_gripper(GRIP_OPEN)
        env._steps(100)

        ov.set_phase("done — close window to exit")
        # keep viewer open so user can inspect
        while v.is_alive():
            v.sleep(0.2)

    # ── Launch ───────────────────────────────────────────────────────────────
    print(f"[debug] Opening GLFW viewer  (slowdown={args.slowdown}×)")
    print("[debug] Close the window or Ctrl-C to exit.")

    v = MujocoViewer(env.model, env.data, slowdown=args.slowdown, overlay=ov)
    v.start(run)

    print("[debug] Done.")


if __name__ == "__main__":
    main()
