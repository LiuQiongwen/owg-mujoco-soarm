import os, time, math
import numpy as np
import pybullet as p
import pybullet_data

from .camera import Camera

class EnvironmentPanda:
    def __init__(self, camera: Camera, asset_root="./owg_robot/assets", vis=False, debug=False,
                 finger_length=0.06, n_grasp_attempts=3):
        self.vis = vis
        self.debug = debug
        self.camera = camera
        self.assets_root = asset_root
        self.finger_length = finger_length
        self.N_GRASP_ATTEMPTS = n_grasp_attempts

        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # scene (沿用你现在的桌子/托盘)
        p.loadURDF("plane.urdf")
        self.tableID = p.loadURDF(os.path.join(self.assets_root, "urdf/objects/table.urdf"),
                                  [0.0, -0.65, 0.76], useFixedBase=True)
        # ... target_table / tray 也照搬你现在的

        # ---- Panda robot ----
        self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                  [0,0,0], p.getQuaternionFromEuler([0,0,0]),
                                  useFixedBase=True)

        # 打印关节，手工确认名字与 index（只需要跑一次）
        self.name_to_joint = {}
        self.joint_names = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")
            self.name_to_joint[name] = j
            self.joint_names.append(name)

        # 常见 Panda joint 名字（用 print 确认后再定）
        self.arm_joints = [self.name_to_joint[f"panda_joint{i}"] for i in range(1,8)]
        self.finger_joints = [self.name_to_joint["panda_finger_joint1"],
                              self.name_to_joint["panda_finger_joint2"]]

        # eef link（按你的 URDF 名称确认）
        self.eef_id = self.name_to_joint.get("panda_hand", None)
        if self.eef_id is None:
            raise RuntimeError("Cannot find panda_hand link index; please inspect joint names.")

        self.reset_robot()

        # 你原来的 obj 列表结构可以照搬（obj_ids / obj_grasps / obj_grasp_rects ...）

    def reset_robot(self):
        # Panda 常用 home pose（你可以再微调）
        home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for _ in range(120):
            for jid, q in zip(self.arm_joints, home):
                p.setJointMotorControl2(self.robot_id, jid, p.POSITION_CONTROL, targetPosition=q, force=200)
            self.move_gripper(0.08)  # open
            p.stepSimulation()
            if self.vis:
                time.sleep(1/240)

    def move_gripper(self, opening=0.08, steps=60):
        opening = float(np.clip(opening, 0.0, 0.08))
        for _ in range(steps):
            for jid in self.finger_joints:
                p.setJointMotorControl2(self.robot_id, jid, p.POSITION_CONTROL, targetPosition=opening, force=50)
            p.stepSimulation()
            if self.vis:
                time.sleep(1/240)

    def move_ee(self, action, max_step=300):
        x, y, z, orn = action
        for _ in range(max_step):
            joint_poses = p.calculateInverseKinematics(self.robot_id, self.eef_id, [x,y,z], orn, maxNumIterations=100)

            # joint_poses 返回很多关节，把前 7 个对应 arm_joints 写进去（或按索引映射更稳）
            for k, jid in enumerate(self.arm_joints):
                p.setJointMotorControl2(self.robot_id, jid, p.POSITION_CONTROL,
                                        targetPosition=joint_poses[k], force=200)
            p.stepSimulation()
            if self.vis:
                time.sleep(1/240)

        return True, p.getLinkState(self.robot_id, self.eef_id)[0:2]

