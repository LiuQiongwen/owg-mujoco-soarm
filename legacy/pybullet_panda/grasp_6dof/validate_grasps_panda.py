# -*- coding: utf-8 -*-
"""Panda 6-DoF grasp validation with structured failure analysis system."""
import pybullet as p
import pybullet_data
import numpy as np
import json
import time
import argparse
import os
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ── Panda joint limits (from URDF) ────────────────────────────
_PANDA_LL   = [-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]
_PANDA_UL   = [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]
_PANDA_JR   = [u - l for l, u in zip(_PANDA_LL, _PANDA_UL)]
_PANDA_HOME = [0.0, -0.5, 0.0, -1.7, 0.0, 1.3, 0.8]

# ── Failure reason taxonomy ───────────────────────────────────
# Priority order: earlier checks preempt later ones
FAILURE_REASONS = [
    "ik_fail",             # IK position residual > 3 cm before descent
    "unreachable",         # target XY clipped → arm can't reach
    "table_collision",     # fingers touch table during approach
    "unstable_object",     # object moved/fell before grasp
    "no_contact",          # descent done, zero finger contact
    "one_side_contact",    # only one finger contacts object
    "orientation_mismatch",# EE yaw error > 30° at grasp pose
    "finger_not_closed",   # gripper gap > 1.5 cm after squeeze
    "slip_after_lift",     # had bilateral contact but object fell
]

# ── Camera (fixed side-view, used for frame capture) ─────────
_CAM_W, _CAM_H = 640, 480
_VIEW_MATRIX = None    # initialised lazily once PyBullet connects
_PROJ_MATRIX = None


def _init_camera():
    global _VIEW_MATRIX, _PROJ_MATRIX
    if _VIEW_MATRIX is not None:
        return
    _VIEW_MATRIX = p.computeViewMatrix(
        cameraEyePosition=[1.0, 0.5, 0.7],
        cameraTargetPosition=[0.38, 0.0, 0.05],
        cameraUpVector=[0, 0, 1],
    )
    _PROJ_MATRIX = p.computeProjectionMatrixFOV(
        fov=60, aspect=_CAM_W / _CAM_H, nearVal=0.01, farVal=5.0,
    )


def capture_frame() -> Optional[np.ndarray]:
    """Return (H, W, 3) uint8 RGB frame using the tiny renderer (GUI + DIRECT)."""
    _init_camera()
    try:
        _, _, rgba, _, _ = p.getCameraImage(
            _CAM_W, _CAM_H, _VIEW_MATRIX, _PROJ_MATRIX,
            renderer=p.ER_TINY_RENDERER,
        )
        arr = np.array(rgba, dtype=np.uint8).reshape(_CAM_H, _CAM_W, 4)
        return arr[:, :, :3]
    except Exception:
        return None


def save_frame(img: Optional[np.ndarray], path: Path) -> None:
    if img is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(str(path))
    except ImportError:
        try:
            import cv2
            cv2.imwrite(str(path), img[:, :, ::-1])
        except ImportError:
            pass


# ── Visualisation helpers ─────────────────────────────────────
def draw_ee_frame(panda_id: int, ee_idx: int, length: float = 0.05) -> None:
    ee  = p.getLinkState(panda_id, ee_idx, computeForwardKinematics=True)
    pos = list(ee[4])
    R   = np.array(p.getMatrixFromQuaternion(ee[5])).reshape(3, 3)
    for i, col in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        end = (np.array(pos) + R[:, i] * length).tolist()
        p.addUserDebugLine(pos, end, col, lineWidth=3, lifeTime=1.0)


def draw_approach_line(pre_pos, grasp_pos, color=(1, 1, 0)) -> None:
    p.addUserDebugLine(list(pre_pos), list(grasp_pos), list(color),
                       lineWidth=2, lifeTime=3.0)


# ── Utilities ─────────────────────────────────────────────────
def set_global_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def save_env_snapshot(out_dir="grasp_6dof/out"):
    os.makedirs(out_dir, exist_ok=True)
    snap = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pybullet_build": p.getAPIVersion()}
    path = os.path.join(out_dir, f"env_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    print(f"[INFO] env snapshot → {path}")


def get_table_top_z(table_id: int) -> float:
    top_z = -1e9
    for ji in range(-1, p.getNumJoints(table_id)):
        aabb = p.getAABB(table_id, ji)
        if aabb:
            top_z = max(top_z, aabb[1][2])
    return top_z


def _rpy_error_deg(q_actual, q_target) -> Tuple[float, float, float]:
    """Per-axis orientation error in degrees (roll, pitch, yaw)."""
    e_actual = p.getEulerFromQuaternion(q_actual)
    e_target = p.getEulerFromQuaternion(q_target)
    roll_err  = float(np.degrees(abs(_angle_diff(e_actual[0], e_target[0]))))
    pitch_err = float(np.degrees(abs(_angle_diff(e_actual[1], e_target[1]))))
    yaw_err   = float(np.degrees(abs(_angle_diff(e_actual[2], e_target[2]))))
    return roll_err, pitch_err, yaw_err


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d >  np.pi: d -= 2 * np.pi
    while d < -np.pi: d += 2 * np.pi
    return d


def _quat_angular_err_deg(q1, q2) -> float:
    dot = float(np.clip(abs(np.dot(list(q1), list(q2))), 0, 1))
    return float(np.degrees(2 * np.arccos(dot)))


# ── Core grasp routine ─────────────────────────────────────────
def grasp_with_panda(
    obj_id,
    grasp_pose:      dict,
    panda_id:        int,
    table_id:        Optional[int]  = None,
    end_effector_index: int         = 11,
    finger_ids:      Optional[List] = None,
    table_top_z:     float          = 0.0,
    init_base_z:     Optional[float]= None,
    open_width_m:    float          = 0.04,
    descent_step:    float          = 0.002,
    descend_clear:   float          = 0.020,
    vel_close:       float          = 0.25,
    pos_close:       float          = 900.0,
    squeeze:         float          = 0.35,
    step_fn                         = None,
    ik_iters:        int            = 400,
    ik_attempts:     int            = 5,
    joint_force:     float          = 900.0,
    frame_dir:       Optional[Path] = None,
    trial_id:        str            = "trial",
) -> Dict[str, Any]:
    """
    Execute one grasp attempt and return a structured result dict.

    Keys: success, failure_reason, ik_converged, ik_pos_err,
          roll_error_deg, pitch_error_deg, yaw_error_deg, orn_error_deg,
          contact_steps, bilateral_contact_steps, total_lift_polls, contact_ratio,
          finger_gap, finger_force_proxy, lift_dz, obj_height
    """
    if finger_ids is None:
        finger_ids = [9, 10]

    MIN_CLEAR      = 0.005
    LIFT_UP        = 0.25
    LIFT_SUCCESS_DZ= 0.05
    JOINT_FORCE    = float(joint_force)
    IK_ITERS       = int(ik_iters)
    POLL_EVERY     = 5       # poll contact every N lift sub-steps
    LIFT_SUBSTEPS  = 60      # subdivide lift into this many move_to calls

    if step_fn is None:
        def step_fn(n):
            for _ in range(int(n)):
                p.stepSimulation()
                time.sleep(1 / 480.0)

    # ── result accumulator ──
    res: Dict[str, Any] = {
        "success": False, "failure_reason": None,
        "ik_converged": False, "ik_pos_err": 999.0,
        "roll_error_deg": 0.0, "pitch_error_deg": 0.0,
        "yaw_error_deg": 0.0, "orn_error_deg": 0.0,
        "contact_steps": 0, "bilateral_contact_steps": 0,
        "total_lift_polls": 0, "contact_ratio": 0.0,
        "finger_gap": 0.08, "finger_force_proxy": 0.0,
        "lift_dz": 0.0, "obj_height": 0.0,
    }

    # ── physics params ──
    p.setPhysicsEngineParameter(numSolverIterations=200)
    p.setTimeStep(1.0 / 480.0)

    # reset arm to home
    for i, q in enumerate(_PANDA_HOME):
        p.resetJointState(panda_id, i, q)
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, q, force=500)
    step_fn(int(0.25 * 480))

    # friction
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, restitution=0.0, rollingFriction=0.05)
    for fid in finger_ids:
        p.changeDynamics(panda_id, fid, lateralFriction=2.5,
                         rollingFriction=0.05, spinningFriction=0.02)

    # ── parse grasp pose ──
    target_pos = np.array(
        grasp_pose.get("position", [0.38, 0.0, table_top_z + 0.12]), dtype=float
    )

    if "world_yaw" in grasp_pose:
        yaw = float(grasp_pose["world_yaw"])
    elif "rpy" in grasp_pose and len(grasp_pose["rpy"]) >= 3:
        yaw = float(grasp_pose["rpy"][2])
    else:
        yaw = float(grasp_pose.get("yaw", 0.0))

    print(f"[DIAG] grasp rpy={grasp_pose.get('rpy')} → world_yaw={np.degrees(yaw):.1f}°")
    quat_target = p.getQuaternionFromEuler([np.pi, 0.0, yaw])

    R_tgt = np.array(p.getMatrixFromQuaternion(quat_target)).reshape(3, 3)
    minus_z = -R_tgt[:, 2] / (np.linalg.norm(R_tgt[:, 2]) + 1e-8)

    # ── workspace clip (detect unreachable) ──
    orig_xy = target_pos[:2].copy()
    target_pos[0] = float(np.clip(target_pos[0], 0.30, 0.60))
    target_pos[1] = float(np.clip(target_pos[1], -0.25, 0.25))
    xy_clipped = float(np.linalg.norm(orig_xy - target_pos[:2])) > 0.005
    if xy_clipped:
        print(f"[WARN] target XY clipped {orig_xy} → {target_pos[:2]}")

    # ── object geometry ──
    obj_pos0, _ = p.getBasePositionAndOrientation(obj_id)
    aabb         = p.getAABB(obj_id, -1)
    cube_top_z   = aabb[1][2]
    cube_bot_z   = aabb[0][2]
    cube_mid_z   = 0.5 * (cube_bot_z + cube_top_z)
    dims         = np.array(aabb[1]) - np.array(aabb[0])
    dx, dy, dz   = float(dims[0]), float(dims[1]), float(dims[2])
    res["obj_height"] = dz

    nearly_sphere = (abs(dx-dy)/max(1e-6,max(dx,dy)) < 0.15 and
                     abs(dx-dz)/max(1e-6,max(dx,dz)) < 0.15 and
                     abs(dy-dz)/max(1e-6,max(dy,dz)) < 0.15)
    if nearly_sphere:
        min_target_z = max(cube_mid_z - 0.35*dz, table_top_z + MIN_CLEAR)
    else:
        min_target_z = max(cube_mid_z, table_top_z + MIN_CLEAR)

    print(f"[DEBUG] shape={'sphere' if nearly_sphere else 'other'} "
          f"top={cube_top_z:.3f} mid={cube_mid_z:.3f} dz={dz:.3f} "
          f"min_z={min_target_z:.3f}")

    # ── check object stability before grasp ──
    vel_lin, vel_ang = p.getBaseVelocity(obj_id)
    obj_speed = float(np.linalg.norm(vel_lin))
    if obj_speed > 0.05:
        print(f"[WARN] Object moving at {obj_speed:.3f} m/s before grasp → unstable_object")
        res["failure_reason"] = "unstable_object"
        return res

    # ── IK move helper ──
    _ik_pos_err_last = [999.0]

    def move_to(pos, orn=None, steps=240, draw_frame=False) -> float:
        """Move EE to pos+orn. Returns position error (m)."""
        if orn is None:
            orn = quat_target
        target = np.array(pos, dtype=float)

        def _solve():
            return p.calculateInverseKinematics(
                panda_id, end_effector_index,
                target.tolist(), orn,
                lowerLimits=_PANDA_LL, upperLimits=_PANDA_UL,
                jointRanges=_PANDA_JR, restPoses=_PANDA_HOME,
                solver=p.IK_DLS,
                maxNumIterations=IK_ITERS,
                residualThreshold=1e-4,
            )

        joints = _solve()
        for j in range(7):
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, joints[j], force=JOINT_FORCE)
        step_fn(steps)

        ee  = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
        cur = np.array(ee[4])
        err = float(np.linalg.norm(cur - target))

        near_table = target[2] < (table_top_z + 0.10)
        tol = 0.012 if near_table else 0.010
        if err > tol:
            for j in range(7):
                p.resetJointState(panda_id, j, _PANDA_HOME[j] + float(np.random.uniform(-0.2, 0.2)))
            joints = _solve()
            for j in range(7):
                p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, joints[j], force=JOINT_FORCE)
            step_fn(max(steps // 2, 120))
            ee  = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
            cur = np.array(ee[4])
            err = float(np.linalg.norm(cur - target))
            if err >= 0.025:
                print(f"[WARN] IK residual={err:.3f}m @ {target.tolist()}")

        _ik_pos_err_last[0] = err
        if draw_frame:
            draw_ee_frame(panda_id, end_effector_index)
        return err

    # ── contact helpers ──
    def _left_right_contact():
        cps = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        left  = any(cp[3] == finger_ids[0] for cp in cps)
        right = any(cp[3] == finger_ids[1] for cp in cps)
        return left, right

    def _table_contact() -> bool:
        if table_id is None:
            return False
        cps = p.getContactPoints(bodyA=panda_id, bodyB=table_id)
        return any(cp[3] in finger_ids for cp in cps)

    # ── FIRST FRAME ──
    first_frame = capture_frame()
    if frame_dir:
        save_frame(first_frame, Path(frame_dir) / f"{trial_id}_first.png")

    # ── open gripper ──
    half = max(0.0, open_width_m * 0.5)
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=half, force=200)
    step_fn(int(0.20 * 480))

    # ── move to pre-grasp ──
    pre_pos = target_pos + minus_z * float(descend_clear)
    draw_approach_line(pre_pos, target_pos)
    pre_err = move_to(pre_pos, orn=quat_target, steps=200, draw_frame=True)
    res["ik_pos_err"] = pre_err

    # ── IK convergence check ──
    if pre_err > 0.030:
        print(f"[FAIL] ik_fail: pos_err={pre_err:.3f}m")
        res["failure_reason"] = "ik_fail"
        res["ik_converged"]   = False
        return res
    res["ik_converged"] = True

    # ── unreachable check ──
    if xy_clipped and pre_err > 0.015:
        res["failure_reason"] = "unreachable"
        return res

    # ── table collision check during approach ──
    move_to(target_pos, orn=quat_target, steps=140, draw_frame=True)
    if _table_contact():
        print("[FAIL] table_collision during approach")
        res["failure_reason"] = "table_collision"
        return res

    # ── orientation error at grasp pose ──
    ee_at_grasp = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    roll_err, pitch_err, yaw_err = _rpy_error_deg(ee_at_grasp[5], quat_target)
    orn_err = _quat_angular_err_deg(ee_at_grasp[5], quat_target)
    res["roll_error_deg"]  = roll_err
    res["pitch_error_deg"] = pitch_err
    res["yaw_error_deg"]   = yaw_err
    res["orn_error_deg"]   = orn_err
    print(f"[DIAG] EE orientation: roll={roll_err:.1f}° pitch={pitch_err:.1f}° yaw={yaw_err:.1f}° (Δq={orn_err:.1f}°)")

    # ── descent loop with contact detection ──
    contact = False
    z       = float(target_pos[2])
    step    = max(0.0015, float(descent_step))
    contact_z = z
    print(f"[DEBUG] descend z: {z:.3f} → {min_target_z:.3f} (step={step:.4f})")
    while z > (min_target_z - 1e-4):
        z -= step
        move_to([target_pos[0], target_pos[1], z], orn=quat_target, steps=60)
        left_c, right_c = _left_right_contact()
        print(f"[TRACE] z={z:.3f}  L={left_c} R={right_c}")
        if left_c or right_c:
            contact_z = z
            contact = True
            break

    # ── CONTACT FRAME ──
    contact_frame = capture_frame()
    if frame_dir:
        save_frame(contact_frame, Path(frame_dir) / f"{trial_id}_contact.png")

    # ── close gripper ──
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.VELOCITY_CONTROL, targetVelocity=-0.2, force=40)
    step_fn(int(float(vel_close) * 480))
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0, force=float(pos_close))
    step_fn(int(0.20 * 480))

    # re-check contact after close
    if not contact:
        left_c, right_c = _left_right_contact()
        contact = left_c or right_c

    # bilateral contact state at squeeze
    left_c_post, right_c_post = _left_right_contact()

    # ── minor lift + re-squeeze ──
    ee_sq = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_sq_pos = np.array(ee_sq[4])
    move_to([ee_sq_pos[0], ee_sq_pos[1], ee_sq_pos[2] + 0.015], orn=quat_target, steps=120)
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0, force=float(pos_close))
    step_fn(int(float(squeeze) * 480))
    probe_z = max(min_target_z, table_top_z + MIN_CLEAR + 0.001)
    move_to([target_pos[0], target_pos[1], probe_z], orn=quat_target, steps=90)

    # finger state after squeeze
    f0_gap   = float(p.getJointState(panda_id, finger_ids[0])[0])
    f1_gap   = float(p.getJointState(panda_id, finger_ids[1])[0])
    f0_force = float(abs(p.getJointState(panda_id, finger_ids[0])[3]))
    f1_force = float(abs(p.getJointState(panda_id, finger_ids[1])[3]))
    finger_gap         = f0_gap + f1_gap
    finger_force_proxy = f0_force + f1_force
    res["finger_gap"]         = finger_gap
    res["finger_force_proxy"] = finger_force_proxy

    # ── LIFT PHASE with contact persistence tracking ──
    ee_lift = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_lift_pos = np.array(ee_lift[4])

    contact_steps    = 0
    bilateral_steps  = 0
    total_lift_polls = 0

    sub_lift  = LIFT_UP / LIFT_SUBSTEPS
    for sub in range(LIFT_SUBSTEPS):
        tgt = [ee_lift_pos[0], ee_lift_pos[1],
               ee_lift_pos[2] + (sub + 1) * sub_lift]
        move_to(tgt, orn=quat_target, steps=max(1, int(360 / LIFT_SUBSTEPS)))

        if sub % POLL_EVERY == 0:
            lc, rc = _left_right_contact()
            total_lift_polls += 1
            if lc or rc:
                contact_steps += 1
            if lc and rc:
                bilateral_steps += 1

    res["contact_steps"]           = contact_steps
    res["bilateral_contact_steps"] = bilateral_steps
    res["total_lift_polls"]        = total_lift_polls
    res["contact_ratio"]           = (bilateral_steps / total_lift_polls
                                      if total_lift_polls > 0 else 0.0)

    # ── FINAL MEASUREMENT ──
    base_z0 = init_base_z if init_base_z is not None else float(obj_pos0[2])
    now_z   = float(p.getBasePositionAndOrientation(obj_id)[0][2])
    dz      = now_z - base_z0
    res["lift_dz"] = dz
    lifted = dz > LIFT_SUCCESS_DZ

    # ── FAILURE FRAME ──
    fail_frame = capture_frame()
    if frame_dir:
        save_frame(fail_frame, Path(frame_dir) / f"{trial_id}_{'success' if lifted else 'fail'}.png")

    # ── STRUCTURED FAILURE REASON ──
    if lifted:
        res["success"]        = True
        res["failure_reason"] = None
    else:
        # walk priority list
        if not contact:
            reason = "no_contact"
        elif not (left_c_post or right_c_post):
            reason = "no_contact"
        elif left_c_post != right_c_post:  # XOR
            reason = "one_side_contact"
        elif orn_err > 30.0:
            reason = "orientation_mismatch"
        elif finger_gap > 0.015:
            reason = "finger_not_closed"
        elif bilateral_steps > 0 and dz < LIFT_SUCCESS_DZ:
            reason = "slip_after_lift"
        else:
            reason = "no_contact"
        res["failure_reason"] = reason

    # ── CONSOLE SUMMARY ──
    ee_final   = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_euler   = p.getEulerFromQuaternion(ee_final[5])
    tgt_euler  = p.getEulerFromQuaternion(quat_target)
    print(f"[GRASP RESULT]")
    print(f"  grasp_yaw       = {np.degrees(yaw):.1f}°")
    print(f"  EE_yaw          = {np.degrees(ee_euler[2]):.1f}°  "
          f"(target={np.degrees(tgt_euler[2]):.1f}°, err={yaw_err:.1f}°)")
    print(f"  contact         = {contact}  bilateral_post={left_c_post and right_c_post}")
    print(f"  fingers         = [{f0_gap:.4f}, {f1_gap:.4f}]  gap={finger_gap:.4f}m")
    print(f"  contact_ratio   = {res['contact_ratio']:.2f}  "
          f"({bilateral_steps}/{total_lift_polls} bilateral polls during lift)")
    print(f"  lift_dz         = {dz:.4f} m  (threshold={LIFT_SUCCESS_DZ})")
    print(f"  result          = {'SUCCESS' if lifted else res['failure_reason']}")

    # release
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.04, force=200)
    step_fn(int(0.20 * 480))

    return res


# ── Object loader ─────────────────────────────────────────────
def load_obj_with_target_height(urdf_path, target_h, table_top_z, xy=(0.38, 0.0)):
    probe = p.loadURDF(urdf_path, basePosition=[0, 0, 1.0], globalScaling=1.0)
    a0    = p.getAABB(probe, -1)
    h0    = max(1e-6, a0[1][2] - a0[0][2])
    p.removeBody(probe)
    sf     = float(target_h) / h0
    base_z = table_top_z + 0.002 + 0.5 * float(target_h)
    obj_id = p.loadURDF(urdf_path, basePosition=[xy[0], xy[1], base_z], globalScaling=sf)
    return obj_id, sf, base_z, float(target_h)


# ── failure_summary.csv writer ────────────────────────────────
_FAILURE_CSV_FIELDS = [
    "trial_id", "success", "failure_reason",
    "ik_converged", "ik_pos_err",
    "yaw_error_deg", "pitch_error_deg", "roll_error_deg", "orn_error_deg",
    "contact_steps", "bilateral_contact_steps", "total_lift_polls", "contact_ratio",
    "finger_gap", "finger_force_proxy",
    "lift_dz", "obj_height",
    "world_yaw_deg", "grasp_score",
    "obj", "seed", "timestamp",
]


def write_failure_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FAILURE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] failure_summary.csv → {path}  ({len(rows)} rows)")


# ── Main ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj",          default="cube.urdf")
    ap.add_argument("--grasps",       default="grasp_6dof/dataset/sample_grasps.json")
    ap.add_argument("--out",          default="grasp_6dof/dataset/validated_grasps_panda.json")
    ap.add_argument("--vis",          type=int, default=1)
    ap.add_argument("--topk",         type=int, default=12)
    ap.add_argument("--fast",         action="store_true")
    ap.add_argument("--fast-scale",   type=float, default=0.9)
    ap.add_argument("--cube-scale",   type=float, default=0.08)
    ap.add_argument("--reset-each-trial", type=int, default=1)
    ap.add_argument("--seed",         type=int, default=123)
    ap.add_argument("--ee-index",     type=int, default=11,
                    help="EE link index (8=hand, 11=gripper_center)")
    ap.add_argument("--ik-iters",     type=int, default=400)
    ap.add_argument("--ik-attempts",  type=int, default=5)
    ap.add_argument("--joint-force",  type=float, default=900.0)
    ap.add_argument("--descent-step", type=float, default=0.002)
    ap.add_argument("--descend-clear",type=float, default=0.020)
    ap.add_argument("--vel-close",    type=float, default=0.25)
    ap.add_argument("--pos-close",    type=float, default=900.0)
    ap.add_argument("--squeeze",      type=float, default=0.35)
    ap.add_argument("--summary-csv",  default="grasp_6dof/out/summary.csv")
    ap.add_argument("--failure-csv",  default="grasp_6dof/out/failure_summary.csv",
                    help="Per-trial failure analysis CSV")
    ap.add_argument("--frame-dir",    default=None,
                    help="Directory to save first/contact/fail PNG frames")
    args = ap.parse_args()

    set_global_seed(args.seed)

    def _append_summary(path, fields, values):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(fields)
            w.writerow(values)

    # ── connect ──
    physicsClient = p.connect(p.GUI if (args.vis and not args.fast) else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    save_env_snapshot()

    def step(n):
        steps = int(max(1, n * (args.fast_scale if args.fast else 1.0)))
        for _ in range(steps):
            p.stepSimulation()
            if not args.fast and args.vis:
                time.sleep(1 / 480.0)

    # ── scene ──
    table_id  = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
    TABLE_TOP_Z = get_table_top_z(table_id)
    print(f"[INFO] table top z = {TABLE_TOP_Z:.3f}")

    CUBE_XY = (0.38, 0.00)
    obj_id, sf, obj_z, TARGET_H = load_obj_with_target_height(
        args.obj, args.cube_scale, TABLE_TOP_Z, xy=CUBE_XY)
    print(f"[INFO] obj: target_h={TARGET_H:.3f}m  scale={sf:.3f}  z={obj_z:.3f}")
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, restitution=0.0, rollingFriction=0.05)
    init_obj_pos, init_obj_orn = p.getBasePositionAndOrientation(obj_id)
    step(int(0.5 * 480))

    # ── load grasps ──
    try:
        with open(args.grasps) as f:
            grasps = json.load(f)
        if not isinstance(grasps, list):
            grasps = []
    except Exception:
        grasps = []

    if args.topk and grasps:
        grasps = sorted(grasps, key=lambda g: g.get("score", 0.0), reverse=True)[:args.topk]

    if not grasps:
        cx, cy, cz = p.getBasePositionAndOrientation(obj_id)[0]
        grasps = [{"position": [float(cx), float(cy), cz + 0.12],
                   "rpy": [float(np.pi), 0.0, float(y)]}
                  for y in np.linspace(-np.pi, np.pi, 12, endpoint=False)]
        print(f"[WARN] No grasps found. Generated {len(grasps)} fallback top-down grasps.")
    print(f"[INFO] {len(grasps)} grasps to validate.")

    # ── Panda ──
    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=True,
    )
    finger_ids        = [9, 10]
    END_EFFECTOR_INDEX = int(args.ee_index)

    _bp, _bo = p.getBasePositionAndOrientation(panda_id)
    _be = p.getEulerFromQuaternion(_bo)
    print(f"[DIAG] Panda base pos={_bp}  "
          f"euler_deg={tuple(round(np.degrees(a),1) for a in _be)}")
    print(f"[DIAG] EE_INDEX={END_EFFECTOR_INDEX}  fingers={finger_ids}")

    frame_dir = Path(args.frame_dir) if args.frame_dir else None

    # ── trial loop ──
    validated      = []
    success_count  = 0
    failure_rows   = []

    for i, grasp in enumerate(grasps):
        if args.reset_each_trial:
            for j in range(7):
                p.resetJointState(panda_id, j, _PANDA_HOME[j])
            for fid in finger_ids:
                p.changeDynamics(panda_id, fid, lateralFriction=2.5,
                                 rollingFriction=0.05, spinningFriction=0.02)
                p.resetJointState(panda_id, fid, 0.04)
            p.resetBasePositionAndOrientation(obj_id, init_obj_pos, init_obj_orn)
            step(int(0.15 * 480))

        init_base_z = float(p.getBasePositionAndOrientation(obj_id)[0][2])

        aabb       = p.getAABB(obj_id, -1)
        need_open  = float(aabb[1][0] - aabb[0][0]) + 0.010
        sug_open   = float(grasp.get("width", 0.04)) + 0.004
        target_open = min(max(need_open, sug_open), 0.080)

        trial_id = f"g{i:03d}_seed{args.seed}"

        result = grasp_with_panda(
            obj_id=obj_id, grasp_pose=grasp, panda_id=panda_id,
            table_id=table_id,
            end_effector_index=END_EFFECTOR_INDEX,
            finger_ids=finger_ids,
            table_top_z=TABLE_TOP_Z, init_base_z=init_base_z,
            open_width_m=target_open,
            descent_step=args.descent_step,
            descend_clear=args.descend_clear,
            vel_close=args.vel_close, pos_close=args.pos_close,
            squeeze=args.squeeze, step_fn=step,
            ik_iters=args.ik_iters, ik_attempts=args.ik_attempts,
            joint_force=args.joint_force,
            frame_dir=frame_dir, trial_id=trial_id,
        )

        ok = result["success"]
        if ok:
            success_count += 1

        out_g = dict(grasp)
        out_g.update(result)
        validated.append(out_g)

        # build failure CSV row
        row = {k: result.get(k) for k in _FAILURE_CSV_FIELDS}
        row["trial_id"]     = trial_id
        row["world_yaw_deg"]= float(np.degrees(
            grasp.get("world_yaw", grasp.get("rpy", [0, 0, 0])[2])))
        row["grasp_score"]  = grasp.get("score", None)
        row["obj"]          = args.obj
        row["seed"]         = args.seed
        row["timestamp"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        failure_rows.append(row)

        print(f"[{i+1}/{len(grasps)}] success={ok}  reason={result['failure_reason']}")

    # ── write outputs ──
    total = max(1, len(grasps))
    rate  = success_count / total
    print(f"\n[INFO] Success rate = {rate:.2f}  ({success_count}/{total})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(validated, f, indent=2)
    print(f"[INFO] validated grasps → {args.out}")

    write_failure_csv(failure_rows, Path(args.failure_csv))

    _append_summary(args.summary_csv, [
        "time","obj","cube_scale","topk","seed",
        "ee_index","ik_iters","joint_force",
        "descent_step","descend_clear","vel_close","pos_close","squeeze",
        "n_trials","success_count","success_rate",
    ], [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.obj, args.cube_scale, args.topk, args.seed,
        args.ee_index, args.ik_iters, args.joint_force,
        args.descent_step, args.descend_clear, args.vel_close,
        args.pos_close, args.squeeze,
        len(grasps), success_count, round(rate, 4),
    ])
    print(f"[INFO] summary → {args.summary_csv}")

    if args.vis and not args.fast:
        input("Press Enter to exit...")
    p.disconnect()


if __name__ == "__main__":
    main()
