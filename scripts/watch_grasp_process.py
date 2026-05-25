#!/usr/bin/env python3
"""Watch SO-ARM101 grasp process in a live GLFW viewer.

Runs the same phase sequence as scripts/record_grasp_process.py but shows
it in an interactive window instead of saving to MP4.  The viewer stays
open after the sequence completes so you can inspect the final pose.

Phase sequence
--------------
  idle → approach → descend → close_gripper → lift → hold → release → done

Overlay
-------
  Phase label, object z, IK quality, bilateral contacts, grasp result.
  Green sphere  = object centroid
  Orange sphere = jaw-midpoint IK target
  Red sphere    = jaw geom midpoint (post-IK, after solve)

Usage:
    DISPLAY=:1 MUJOCO_GL=glfw python scripts/watch_grasp_process.py
    DISPLAY=:1 MUJOCO_GL=glfw python scripts/watch_grasp_process.py --object MustardBottle
    DISPLAY=:1 MUJOCO_GL=glfw python scripts/watch_grasp_process.py --slowdown 5 --seed 2
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
    from pathlib import Path
    ycb = Path(ROOT) / "owg_robot" / "assets" / "ycb_objects"
    nl  = name.lower()
    for d in ycb.iterdir():
        if d.is_dir() and not d.name.startswith("__"):
            short = d.name[3:] if d.name.startswith("Ycb") else d.name
            if short.lower() == nl or d.name.lower() == nl:
                return short
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object",      default="Banana")
    ap.add_argument("--slowdown",    type=float, default=4.0,
                    help="Real-time multiplier for viewer.sleep()")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--frame-skip",  type=int,   default=4,
                    help="Sync viewer every N sim steps during long motions")
    args = ap.parse_args()

    np.random.seed(args.seed)
    obj_name   = _resolve(args.object)
    frame_skip = max(1, args.frame_skip)

    # ── Build env ────────────────────────────────────────────────────────────
    print(f"[watch] Loading EnvironmentSoArm …")
    env = EnvironmentSoArm(grasp_mode=GRASP_MODE_PHYSICS, n_grasp_attempts=1)
    env.reset()

    print(f"[watch] Loading '{obj_name}' …")
    # Spawn on the table (y < 0) so the object rests at TABLE_TOP_Z
    obj_id = env.load_obj(obj_name, pos=[0.0, -0.20, TABLE_TOP_Z + 0.15])
    env.wait_until_still(obj_id)
    env._steps(300)

    obj_com = env.get_obj_com_pos(obj_id)   # body CoM in world frame
    obj_pos = env.get_obj_pos(obj_id)
    x, y    = float(obj_com[0]), float(obj_com[1])
    grasp_z = float(obj_com[2]) + GRASP_Z_TABLE_MARGIN  # offset keeps fixed jaw off table
    hover_z = env.GRIPPER_MOVING_HEIGHT
    print(f"[watch] {obj_name} at ({x:.3f}, {y:.3f}, {obj_pos[2]:.3f})  CoM z={obj_com[2]:.3f}  grasp_z={grasp_z:.3f}")

    # ── Overlay ──────────────────────────────────────────────────────────────
    ov = Overlay()
    ov.set_phase("idle")
    ov.add_info("object",   obj_name)
    ov.add_info("slowdown", f"{args.slowdown}×")
    ov.add_marker(obj_com,         rgba=(0.0, 1.0, 0.0, 0.7), size=0.025, label=obj_name)
    ov.add_marker([x, y, grasp_z], rgba=(1.0, 0.5, 0.0, 0.6), size=0.018, label="grasp_z")

    # ── Step hook: sync viewer during env.move_ee / _steps ───────────────────
    _step_n  = [0]
    _viewer  = [None]
    orig_step = env.step_simulation

    def _hooked():
        orig_step()
        _step_n[0] += 1
        if _step_n[0] % frame_skip == 0:
            v = _viewer[0]
            if v is not None:
                try:
                    ov.add_info("sim_step", _step_n[0])
                    ov.add_info("obj_z",    f"{env.get_obj_pos(obj_id)[2]:.4f}")
                except Exception:
                    pass
                v.sync()

    env.step_simulation = _hooked

    # ── Grasp sequence ────────────────────────────────────────────────────────
    def run(v: MujocoViewer):
        _viewer[0] = v

        # ── 0: idle ──────────────────────────────────────────────────────────
        ov.set_phase("idle")
        env._steps(80)
        v.sleep(0.5)

        # ── 1: approach ──────────────────────────────────────────────────────
        ov.set_phase("approach")
        env.move_gripper(GRIP_OPEN)
        env.move_ee([x, y, hover_z, None], ik_mode=IK_MODE_JAW_POS, max_step=1400)
        env._steps(80)
        v.sleep(0.3)

        # ── 2: descend (IK solve + arm teleport + object restore) ─────────────
        ov.set_phase("ik_solve")
        ok, pe, _ = env._solve_ik_jaw_pos_only(
            np.array([x, y, grasp_z]), reset_to_home=False
        )
        q_grasp = np.array([env.data.qpos[a] for a in env._arm_qpos_adr])
        ov.add_info("ik_ok",    str(ok))
        ov.add_info("ik_pe_mm", f"{pe * 1000:.1f}")

        # Mark jaw geom midpoint post-IK
        if env._jaw_fixed_geom_id >= 0:
            fg = env.data.geom_xpos[env._jaw_fixed_geom_id].copy()
            mg = env.data.geom_xpos[env._jaw_mv_geom_id].copy()
            jaw_mid = 0.5 * (fg + mg)
            ov.clear_markers()
            ov.add_marker(obj_pos,         rgba=(0, 1, 0, 0.7),    size=0.025, label=obj_name)
            ov.add_marker([x, y, grasp_z], rgba=(1, 0.5, 0, 0.6),  size=0.018, label="grasp_z")
            ov.add_marker(jaw_mid,         rgba=(1, 0.1, 0.1, 0.8), size=0.015, label="jaw_mid")
        v.sync()
        v.sleep(0.8)

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

        m = env.get_grasp_debug_metrics(obj_id)
        ov.add_info("jaw_gap_cm",  f"{(m.get('jaw_obj_xy_gap') or 0) * 100:.1f}")
        ov.add_info("bilateral",   m.get("bilateral_contacts", 0))
        ov.add_info("symmetry",    f"{m.get('symmetry_score', 0):.2f}")
        v.sync()
        v.sleep(0.5)

        # ── 3: close gripper ──────────────────────────────────────────────────
        ov.set_phase("close_gripper")
        env.auto_close_gripper(check_contact=False)
        env._steps(140)

        m2 = env.get_grasp_debug_metrics(obj_id)
        ov.add_info("post_bilateral", m2.get("bilateral_contacts", 0))
        ov.add_info("ncon",           env.data.ncon)
        grasped = env.check_grasped_id()
        ov.add_info("grasped_ids",    str(grasped))
        v.sync()
        v.sleep(0.5)

        # Kinematic weld after bilateral contacts so lift succeeds despite
        # limited friction from 6 mm sphere colliders.
        weld_obj = grasped[0] if grasped else None
        if weld_obj is not None:
            env._attach_obj(weld_obj)

        # ── 4: lift ──────────────────────────────────────────────────────────
        ov.set_phase("lift")
        env.move_ee([x, y, hover_z, None], ik_mode=IK_MODE_XYZ_ONLY, max_step=350)
        env._steps(80)

        if weld_obj is not None:
            final_z = float(env.get_obj_pos(weld_obj)[2])
            lifted  = final_z > env.Z_TABLE_TOP + 0.07
            ov.add_info("final_z", f"{final_z:.4f}")
            ov.add_info("lifted",  str(lifted))
            if not lifted:
                env._detach_obj(weld_obj)
            result = "SUCCESS" if lifted else "CONTACT_NO_LIFT"
        else:
            result = "NO_CONTACT"
            ov.add_info("result", result)

        print(f"[watch] Grasp result: {result}")

        # ── 5: hold ──────────────────────────────────────────────────────────
        ov.set_phase(f"hold ({result})")
        for _ in range(30):
            if not v.is_alive():
                return
            v.sleep(0.05)

        # ── 6: release ───────────────────────────────────────────────────────
        ov.set_phase("release")
        env.move_gripper(GRIP_OPEN)
        env._steps(100)

        # ── done: keep viewer open for inspection ─────────────────────────────
        ov.set_phase(f"done ({result}) — close window to exit")
        while v.is_alive():
            v.sleep(0.3)

    # ── Launch viewer ─────────────────────────────────────────────────────────
    print(f"[watch] Opening GLFW viewer  (slowdown={args.slowdown}×, frame_skip={frame_skip})")
    print("[watch] Close the window or Ctrl-C to exit.")

    viewer = MujocoViewer(env.model, env.data, slowdown=args.slowdown, overlay=ov)
    viewer.start(run)

    print("[watch] Done.")


if __name__ == "__main__":
    main()
