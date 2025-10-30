# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np
import json
import time
import argparse
import os
import random
from datetime import datetime

# ---------------------------- Utils ----------------------------

def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)

def save_env_snapshot(out_dir="grasp_6dof/out"):
    os.makedirs(out_dir, exist_ok=True)
    snap = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pybullet_build": p.getAPIVersion(),
    }
    path = os.path.join(out_dir, f"env_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Environment snapshot saved → {path}")

def get_table_top_z(table_id: int) -> float:
    top_z = -1e9
    for ji in range(-1, p.getNumJoints(table_id)):  # -1: base
        aabb = p.getAABB(table_id, ji)
        if aabb:
            top_z = max(top_z, aabb[1][2])
    return top_z

# ------------------------- Grasp Routine -----------------------

def grasp_with_panda(
    obj_id,
    grasp_pose,
    panda_id,
    end_effector_index=8,
    finger_ids=None,
    table_top_z=0.0,
    init_base_z=None,
    *,
    descent_step=0.002,
    descend_clear=0.020,
    vel_close=0.25,
    pos_close=900,
    squeeze=0.35,
    step_fn=None,
    ik_iters=400,
    ik_attempts=5,
    joint_force=900.0,
):

    """
    稳定 + 高速：渐降 -> 速度找物 -> 位置夹紧 -> 轻抬二次挤压 -> 抬升判定
    使用 calculateInverseKinematics2，并只求 7DoF 手臂。
    """
    # 参数
    MIN_CLEAR = 0.005
    GRASP_CLEARANCE = 0.02
    DESCENT_STEP = float(descent_step)
    VEL_CLOSE_TIME = float(vel_close)
    POS_CLOSE_FORCE = float(pos_close)
    SQUEEZE_EXTRA_TIME = float(squeeze)
    LIFT_UP = 0.25
    LIFT_SUCCESS_DZ = 0.05
    JOINT_FORCE = float(joint_force)
    IK_ITERS = int(ik_iters)
    IK_ATTEMPTS = max(1, int(ik_attempts))

    # 如果没传 step_fn，就给一个安全的默认（带 sleep）
    if step_fn is None:
        def step_fn(n):
            for _ in range(int(n)):
                p.stepSimulation()
                time.sleep(1/480.0)

    # 物理设置
    p.setPhysicsEngineParameter(numSolverIterations=200)
    p.setTimeStep(1.0/480.0)

    quat_down = p.getQuaternionFromEuler([np.pi, 0, 0])
    home = [0, -0.5, 0, -1.7, 0, 1.3, 0.8]
    for i in range(7):
        p.resetJointState(panda_id, i, home[i])
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, home[i], force=500)
    step_fn(int(0.25 * 480))

    # 摩擦
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, restitution=0.0, rollingFriction=0.05)
    for fid in finger_ids:
        p.changeDynamics(panda_id, fid, lateralFriction=2.0)

    # 只用 7DoF 手臂做 IK
    ARM_JOINTS = list(range(7))

    def move_to(pos, orn=quat_down, steps=240):
        # —— 用经典 IK（单末端），避免 IK2 的参数/返回兼容性坑 —— 
        target = np.array(pos, dtype=float)

        joints_full = p.calculateInverseKinematics(
            bodyUniqueId=panda_id,
            endEffectorLinkIndex=end_effector_index,
            targetPosition=target.tolist(),
            targetOrientation=orn,
            solver=p.IK_DLS,
            maxNumIterations=int(IK_ITERS),
            residualThreshold=1e-4
            # 不传 jointDamping，避免“维度不匹配”告警
        )

        # 只下发 7DoF 手臂关节
        for j in range(7):
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, joints_full[j],    force=JOINT_FORCE)

        # 自适应步进 + 误差检查
        step_fn(steps)
        ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
        cur = np.array(ee[0])
        err = float(np.linalg.norm(cur - target))
        near_table = target[2] < (table_top_z + 0.10)
        tol = 0.008 if not near_table else 0.012
        if err > tol:
            # 再迭代一轮加密步进兜底
            joints_full = p.calculateInverseKinematics(
                panda_id, end_effector_index, target.tolist(), orn,
                solver=p.IK_DLS, maxNumIterations=max(int(IK_ITERS*2), 400),
                residualThreshold=5e-5
            )
            for j in range(7):
                p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, joints_full[j], force=JOINT_FORCE)
            step_fn(max(steps//2, 120))
            ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
            cur = np.array(ee[0])
            err = float(np.linalg.norm(cur - target))
            if err >= 0.02:
                print(f"[WARN] move_to 未完全到位，残差={err:.3f} m@{target.tolist()}")

    # 目标与高度（AABB 顶面更鲁棒）
    obj_pos0, _ = p.getBasePositionAndOrientation(obj_id)
    obj_aabb = p.getAABB(obj_id, -1)
    cube_top_z = obj_aabb[1][2]
    cube_mid_z = 0.5 * (obj_aabb[0][2] + obj_aabb[1][2])

    cx = float(np.clip(obj_pos0[0], 0.32, 0.55))
    cy = float(np.clip(obj_pos0[1], -0.18, 0.18))
    init_z = float(init_base_z) if init_base_z is not None else float(obj_pos0[2])

    approach_z   = cube_top_z + 0.10
    descend_from = cube_top_z + float(descend_clear) 
    min_target_z = max(cube_mid_z, table_top_z + MIN_CLEAR)

    print(f"[DEBUG] Using table_z={table_top_z:.3f}, cube_top_z={cube_top_z:.3f}, "
          f"approach_z={approach_z:.3f}, descend_from={descend_from:.3f}, min_target_z={min_target_z:.3f}")

    # 张开手指
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.04, force=200)
    step_fn(int(0.25 * 480))

    # 1) 到物体正上方 -> 2) 到顶上 2cm
    move_to([cx, cy, approach_z], steps=200)
    move_to([cx, cy, descend_from], steps=140)

    # 3) 渐降找接触
    contact = False
    z = float(descend_from)
    while z > min_target_z:
        z -= DESCENT_STEP
        move_to([cx, cy, z], steps=60)
        cps = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        # 注意：Panda 侧 linkIndex 在 c[2]
        if any(c[2] in finger_ids for c in cps):
            contact = True
            break

    # 4) 合爪：速度 -> 位置
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.VELOCITY_CONTROL, targetVelocity=-0.2, force=40)
    step_fn(int(VEL_CLOSE_TIME * 480))
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0, force=POS_CLOSE_FORCE)
    step_fn(int(0.25 * 480))

    if not contact:
        cps = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        contact = any(c[2] in finger_ids for c in cps)

    # 5) 轻抬 + 二次挤压
    ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = np.array(ee[0])
    move_to([ee_pos[0], ee_pos[1], ee_pos[2] + 0.015], steps=120)
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0, force=POS_CLOSE_FORCE)
    step_fn(int(SQUEEZE_EXTRA_TIME * 480))

    # 6) 抬升并判定
    ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = np.array(ee[0])
    move_to([ee_pos[0], ee_pos[1], ee_pos[2] + LIFT_UP], steps=360)

    now_z = p.getBasePositionAndOrientation(obj_id)[0][2]
    lifted = (now_z - init_z) > LIFT_SUCCESS_DZ
    print(f"[DEBUG] contact={contact}, Δz={now_z - init_z:.3f}, success={lifted}")

    # 松手
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.04, force=200)
    step_fn(int(0.20 * 480))

    return lifted

# ------------------------------ Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="cube.urdf")
    parser.add_argument("--grasps", type=str, default="grasp_6dof/dataset/sample_grasps.json")
    parser.add_argument("--out", type=str, default="grasp_6dof/dataset/validated_grasps_panda.json")
    parser.add_argument("--vis", type=int, default=1)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--fast", action="store_true", help="关可视化/不sleep/缩短steps")
    parser.add_argument("--fast-scale", type=float, default=0.9, help="steps 缩放系数(0.6~0.95)")
    parser.add_argument("--cube-scale", type=float, default=0.08)
    parser.add_argument("--reset-each-trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)

    # IK/动力学可调参数
    parser.add_argument("--ee-index", type=int, default=11, help="8: hand, 11: gripper center")
    parser.add_argument("--ik-iters", type=int, default=400)
    parser.add_argument("--ik-attempts", type=int, default=5)
    parser.add_argument("--joint-force", type=float, default=900.0)
    parser.add_argument("--descent-step", type=float, default=0.002,
                        dest="descent_step", help="渐降步长(m)，默认 0.002")
    parser.add_argument("--descend-clear", type=float, default=0.020,
                        dest="descend_clear", help="从方块顶面上方多少米开始渐降，默认 0.020")
    parser.add_argument("--vel-close", type=float, default=0.25,
                        dest="vel_close", help="速度合爪阶段时长(s)，默认 0.25")
    parser.add_argument("--pos-close", type=float, default=900,
                        dest="pos_close", help="位置夹紧的力(牛)，默认 900")
    parser.add_argument("--squeeze", type=float, default=0.35,
                        dest="squeeze", help="二次挤压时长(s)，默认 0.35")
    parser.add_argument("--summary-csv", type=str, default="grasp_6dof/out/summary.csv",
                    help="将本次实验的配置与结果附加写入该 CSV")


    args = parser.parse_args()
    set_global_seed(args.seed)
    
    def append_summary_row(path, fields, values):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import csv
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(fields)
            w.writerow(values)
    
    # 连接仿真
    physicsClient = p.connect(p.GUI if (args.vis and not args.fast) else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    save_env_snapshot()

    # 步进函数：fast 模式关闭 sleep 且按 scale 减少步数
    def step(n):
        steps = int(max(1, n * (args.fast_scale if args.fast else 1.0)))
        if args.fast or not args.vis:
            for _ in range(steps):
                p.stepSimulation()
        else:
            for _ in range(steps):
                p.stepSimulation()
                time.sleep(1/480.0)

    # 桌面
    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
    TABLE_TOP_Z = get_table_top_z(table_id)
    print(f"[INFO] Detected table top z = {TABLE_TOP_Z:.3f}")

    # 物体：静置在桌面
    CUBE_SCALE = float(args.cube_scale)
    CUBE_HALF_Z = 0.5 * CUBE_SCALE
    CUBE_XY = (0.38, 0.00)
    obj_z = TABLE_TOP_Z + CUBE_HALF_Z + 0.002
    obj_id = p.loadURDF(args.obj, basePosition=[CUBE_XY[0], CUBE_XY[1], obj_z], globalScaling=CUBE_SCALE)
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, restitution=0.0, rollingFriction=0.05)
    print(f"[INFO] 方块已放在桌面上 (x={CUBE_XY[0]:.2f}, y={CUBE_XY[1]:.2f}, z={obj_z:.3f}, scale={CUBE_SCALE}).")
    
    init_obj_pos, init_obj_orn = p.getBasePositionAndOrientation(obj_id)

    # 先稳定
    step(int(0.5 * 480))

    # 读取 grasps；为空则兜底
    try:
        with open(args.grasps, "r") as f:
            grasps = json.load(f)
            if not isinstance(grasps, list):
                grasps = []
    except Exception:
        grasps = []
    if args.topk is not None and len(grasps) > 0:
        if "score" in grasps[0]:
            grasps = sorted(grasps, key=lambda g: g.get("score", 0.0), reverse=True)
        grasps = grasps[:args.topk]

    if len(grasps) == 0:
        cx, cy, cz = p.getBasePositionAndOrientation(obj_id)[0]
        z_above = cz + 0.12
        yaw_list = np.linspace(-np.pi, np.pi, 12, endpoint=False)
        grasps = [{
            "position": [float(cx), float(cy), float(z_above)],
            "rpy": [float(np.pi), 0.0, float(yaw)]
        } for yaw in yaw_list]
        print(f"[WARN] No grasps in JSON. Generated {len(grasps)} top-down fallback grasps.")
    print(f"[INFO] Loaded {len(grasps)} grasps for validation.")

    # Panda
    panda_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    finger_ids = [9, 10]
    END_EFFECTOR_INDEX = int(args.ee_index)

    # 统计
    validated, success_count = [], 0
    init_base_z = p.getBasePositionAndOrientation(obj_id)[0][2]

    for i, grasp in enumerate(grasps):
        # 每轮重置：臂+手指
        if args.reset_each_trial:
            for j in range(7):
                p.resetJointState(panda_id, j, 0.0)
            for fid in finger_ids:
                p.resetJointState(panda_id, fid, 0.04)

            # ⭐ 复位方块到初始位姿（很关键）
            p.resetBasePositionAndOrientation(obj_id, init_obj_pos, init_obj_orn)
            step(int(0.15 * 480))  # 稍微稳定一下

        # ⭐ 每轮都重新读取“抬升成功”的基准高度
        init_base_z = p.getBasePositionAndOrientation(obj_id)[0][2]

        ok = grasp_with_panda(
            obj_id, grasp, panda_id,
            end_effector_index=END_EFFECTOR_INDEX,
            finger_ids=finger_ids,
            table_top_z=TABLE_TOP_Z,
            init_base_z=init_base_z,    # ← 现在是每轮新读的
            descent_step=args.descent_step,
            descend_clear=args.descend_clear,
            vel_close=args.vel_close,
            pos_close=args.pos_close,
            squeeze=args.squeeze,
        )

        out_g = dict(grasp); out_g["success"] = bool(ok)
        validated.append(out_g)
        if ok: success_count += 1
        print(f"[{i+1}/{len(grasps)}] Grasp success = {ok}")

    # 结果
    total = max(1, len(grasps))
    print(f"[INFO] Success rate = {success_count / total:.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved validated grasps → {args.out}")

    fields = [
        "time","obj","cube_scale","topk","seed",
        "ee_index","ik_iters","ik_attempts","joint_force",
        "descent_step","descend_clear","vel_close","pos_close","squeeze",
        "fast","fast_scale",
        "n_trials","success_count","success_rate"
    ]
    values = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.obj, args.cube_scale, args.topk, args.seed,
        args.ee_index, args.ik_iters, args.ik_attempts, args.joint_force,
        args.descent_step, args.descend_clear, args.vel_close, args.pos_close, args.squeeze,
        int(args.fast), args.fast_scale,
        len(grasps), success_count, round(success_count/max(1,len(grasps)), 4)
    ]
    append_summary_row(args.summary_csv, fields, values)
    print(f"[INFO] Appended summary → {args.summary_csv}")

    if args.vis and not args.fast:
        input("Press Enter to exit simulation...")
    p.disconnect()

if __name__ == "__main__":
    main()

