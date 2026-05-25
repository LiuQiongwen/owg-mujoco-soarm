import pybullet as p
from collections import namedtuple

JointInfo = namedtuple(
    "JointInfo",
    ["id", "name", "type", "damping", "friction", "lowerLimit", "upperLimit",
     "maxForce", "maxVelocity", "controllable"]
)

def setup_panda(p_client, robot_id):
    """
    Return:
      joints: dict[name] -> JointInfo
      controlGripper: function(controlMode, targetPosition, force=...)
      controlJoints: list of arm joint names + [mimicParentName] (keep OWG code style)
      mimicParentName: str (we use panda_finger_joint1 as placeholder)
      eef_id: int (end effector link index)
    """
    joints = {}
    nJ = p_client.getNumJoints(robot_id)

    for i in range(nJ):
        info = p_client.getJointInfo(robot_id, i)
        jid = info[0]
        jname = info[1].decode("utf-8")
        jtype = info[2]
        lower = info[8]
        upper = info[9]
        maxF = info[10]
        maxV = info[11]
        controllable = (jtype != p_client.JOINT_FIXED)

        joints[jname] = JointInfo(
            id=jid, name=jname, type=jtype,
            damping=0.0, friction=0.0,
            lowerLimit=lower, upperLimit=upper,
            maxForce=maxF if maxF > 0 else 87.0,
            maxVelocity=maxV if maxV > 0 else 2.0,
            controllable=controllable
        )

    # --- 给 env.py 复用的 “finger pad” 名字做别名（避免 KeyError） ---
    joints["left_inner_finger_pad_joint"]  = joints["panda_finger_joint1"]
    joints["right_inner_finger_pad_joint"] = joints["panda_finger_joint2"]

    # Panda 7-DoF arm joints
    arm = [f"panda_joint{i}" for i in range(1, 8)]

    # end-effector: 用 grasptarget 最稳（你打印过 index=11）
    eef_id = 11

    mimicParentName = "panda_finger_joint1"

    def controlPandaGripper(controlMode=p_client.POSITION_CONTROL,
                            targetPosition=0.08,
                            force=20):
        """
        targetPosition: gripper opening width (m), approx [0, 0.08]
        """
        width = max(0.0, min(float(targetPosition), 0.08))
        each = width / 2.0
        for fn in ["panda_finger_joint1", "panda_finger_joint2"]:
            j = joints[fn]
            p_client.setJointMotorControl2(
                robot_id, j.id,
                controlMode,
                targetPosition=each,
                force=min(j.maxForce, force)
            )

    # 让 env.py 的 for-loop 逻辑保持不改：controlJoints 最后放一个“占位 gripper”
    controlJoints = arm + [mimicParentName]

    return joints, controlPandaGripper, controlJoints, mimicParentName, eef_id

