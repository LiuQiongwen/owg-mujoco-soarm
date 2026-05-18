#!/usr/bin/env python3
"""Record SO-ARM101 MuJoCo grasp process to MP4 (offscreen, no viewer).

Usage:
    MUJOCO_GL=egl python scripts/record_grasp_process.py --object banana
    MUJOCO_GL=egl python scripts/record_grasp_process.py --object banana --frame-skip 3
"""
import sys, os, argparse, pathlib

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import cv2

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from owg_robot.env_soarm import (
    EnvironmentSoArm, TABLE_TOP_Z, GRIP_OPEN, GRIP_CLOSED,
    IK_MODE_JAW_POS, IK_MODE_XYZ_ONLY,
    GRASP_MODE_PHYSICS,
)

VIDEO_DIR  = PROJECT_ROOT / "results" / "videos"
FRAME_W, FRAME_H = 640, 480
FPS        = 30
GRASP_Z_OFFSET = 0.06   # jaw-midpoint IK: target jaw this far above table top


# ── Camera ──────────────────────────────────────────────────────────────────

def _make_camera(lookat: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth   = 140.0
    cam.elevation = -22.0
    cam.distance  = 0.80
    cam.lookat[:] = lookat
    return cam


# ── Overlay ──────────────────────────────────────────────────────────────────

def _overlay(frame: np.ndarray, phase: str, obj_z: float) -> np.ndarray:
    frame = frame.copy()
    for text, y in [(f"Phase: {phase}", 36), (f"obj_z: {obj_z:.3f} m", 72)]:
        cv2.putText(frame, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ── Object name resolution ───────────────────────────────────────────────────

_YCB_ROOT = PROJECT_ROOT / "owg_robot" / "assets" / "ycb_objects"

def _resolve_obj_name(name: str) -> str:
    """Accept case-insensitive YCB names like 'banana' → 'Banana' (for load_obj)."""
    nl = name.lower()
    for d in _YCB_ROOT.iterdir():
        if d.is_dir() and not d.name.startswith("__"):
            # "YcbBanana" → strip "Ycb" prefix for comparison
            short = d.name[3:] if d.name.startswith("Ycb") else d.name
            if short.lower() == nl or d.name.lower() == nl:
                return short   # e.g. "Banana" — load_obj will prepend "Ycb"
    return name   # pass through and let load_obj raise if not found


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object",     default="Banana",
                    help="Object name passed to env.load_obj (case-insensitive)")
    ap.add_argument("--frame-skip", type=int, default=5,
                    help="Capture one video frame every N simulation steps")
    ap.add_argument("--yaw",        type=float, default=0.0,
                    help="Approach yaw angle (radians)")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--out",        type=str, default=None,
                    help="Output MP4 path (default: results/videos/grasp_process.mp4)")
    args = ap.parse_args()

    obj_name   = _resolve_obj_name(args.object)
    frame_skip = max(1, args.frame_skip)
    out_path   = pathlib.Path(args.out) if args.out else VIDEO_DIR / "grasp_process.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # ── Build env ────────────────────────────────────────────────────────────
    print(f"[record] Loading EnvironmentSoArm …")
    env = EnvironmentSoArm(grasp_mode=GRASP_MODE_PHYSICS, n_grasp_attempts=1)
    env.reset()

    # ── Load object ──────────────────────────────────────────────────────────
    print(f"[record] Loading object '{obj_name}' …")
    obj_id = env.load_obj(obj_name)
    env.wait_until_still(obj_id)

    obj_pos = env.get_obj_pos(obj_id)
    x_obj, y_obj = float(obj_pos[0]), float(obj_pos[1])
    grasp_z  = TABLE_TOP_Z + GRASP_Z_OFFSET
    hover_z  = env.GRIPPER_MOVING_HEIGHT
    print(f"[record] Object settled at ({x_obj:.3f}, {y_obj:.3f}, {obj_pos[2]:.3f})")
    print(f"[record] Grasp target z={grasp_z:.3f}  hover z={hover_z:.3f}")

    # ── Offscreen renderer ───────────────────────────────────────────────────
    renderer = mujoco.Renderer(env.model, height=FRAME_H, width=FRAME_W)
    cam      = _make_camera(np.array([x_obj, y_obj, 0.90]))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(str(out_path), fourcc, FPS, (FRAME_W, FRAME_H))

    # ── Frame-capture hook ───────────────────────────────────────────────────
    step_count = [0]
    phase      = ["idle"]
    orig_step  = env.step_simulation

    def _hooked_step():
        orig_step()
        step_count[0] += 1
        if step_count[0] % frame_skip == 0:
            renderer.update_scene(env.data, camera=cam)
            rgb = renderer.render()                    # (H, W, 3) uint8 RGB
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            try:
                obj_z = float(env.get_obj_pos(obj_id)[2])
            except Exception:
                obj_z = float(env.data.qpos[
                    env.model.joint(f"obj_joint_{env._obj_pool_slot(obj_id)}").qposadr[0] + 2])
            bgr = _overlay(bgr, phase[0], obj_z)
            vw.write(bgr)

    env.step_simulation = _hooked_step

    def run(name, fn):
        phase[0] = name
        fn()

    # ── Grasp sequence ───────────────────────────────────────────────────────

    # 0. Idle — robot at home, object on table
    run("idle", lambda: env._steps(80))

    # 1. Approach — open gripper, move to hover above object
    env.move_gripper(GRIP_OPEN * 0.9)
    run("approach", lambda: env.move_ee(
        [x_obj, y_obj, hover_z, None],
        ik_mode=IK_MODE_JAW_POS, max_step=1400))
    run("approach", lambda: env._steps(80))

    # 2. Descend — IK to grasp height, park object, teleport arm, restore object
    phase[0] = "descend"
    ok_ik, pe_ik, _ = env._solve_ik_jaw_pos_only(
        np.array([x_obj, y_obj, grasp_z]), reset_to_home=False)
    q_grasp = np.array([env.data.qpos[adr] for adr in env._arm_qpos_adr])
    print(f"[record] IK converged={ok_ik}  pe={pe_ik*100:.1f} cm  q={q_grasp.round(3)}")

    # Park object below floor so arm teleport doesn't collide with it
    slot     = env._obj_pool_slot(obj_id)
    jnt      = env.model.joint(f"obj_joint_{slot}")
    qpos_adr = jnt.qposadr[0]
    dof_adr  = jnt.dofadr[0]
    obj_pose_saved = env.data.qpos[qpos_adr : qpos_adr + 7].copy()
    env.data.qpos[qpos_adr + 2] = -100.0

    # Teleport arm to IK solution, settle without object
    for arm_adr, act_id, q in zip(env._arm_qpos_adr, env._arm_act_ids, q_grasp):
        env.data.qpos[arm_adr] = q
        env.data.ctrl[act_id]  = q
    mujoco.mj_forward(env.model, env.data)
    run("descend", lambda: env._steps(220))

    # Restore object and let it settle into jaw gap
    env.data.qpos[qpos_adr : qpos_adr + 7] = obj_pose_saved
    env.data.qvel[dof_adr  : dof_adr  + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    run("descend", lambda: env._steps(120))

    # 3. Close gripper
    run("close_gripper", lambda: env.auto_close_gripper(check_contact=False))
    run("close_gripper", lambda: env._steps(140))

    grasped = env.check_grasped_id()
    print(f"[record] After close: grasped_ids={grasped}  ncon={env.data.ncon}")

    # 4. Lift
    run("lift", lambda: env.move_ee(
        [x_obj, y_obj, hover_z, None],
        ik_mode=IK_MODE_XYZ_ONLY, max_step=350))
    run("lift", lambda: env._steps(80))

    if grasped:
        final_z = float(env.get_obj_pos(grasped[0])[2])
        lifted  = final_z > env.Z_TABLE_TOP + 0.07
        print(f"[record] Object z after lift: {final_z:.4f}  lifted={lifted}")

    # 5. Hold
    run("hold", lambda: env._steps(100))

    # 6. Release — open gripper
    env.move_gripper(GRIP_OPEN)
    run("release", lambda: env._steps(120))

    # ── Finish ────────────────────────────────────────────────────────────────
    env.step_simulation = orig_step
    vw.release()
    renderer.close()

    n_frames = step_count[0] // frame_skip
    print(f"[record] {step_count[0]} sim steps → {n_frames} video frames")
    print(f"[record] Saved → {out_path}")


if __name__ == "__main__":
    main()
