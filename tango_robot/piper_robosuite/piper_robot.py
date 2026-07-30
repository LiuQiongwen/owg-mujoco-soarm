"""
RoboSuite RobotModel registration for the AgileX Piper arm (6-DoF, arm-only
half of the soulde/Piper_mujoco split -- see tango_robot/piper_assets/
robot_arm.xml). Modelled on robosuite_models' arx5_robot.py template (the
closest existing example: a simple single fixed-base 6-DoF arm).

NOTE on default_base/default_controller_config: copied from the ARX5
template as a reasonable starting point, NOT independently validated for
Piper's actual dynamics/mount -- revisit once real hardware trials are
running and something looks physically wrong (e.g. mount doesn't match the
real desk-clamp mount, controller gains too aggressive/weak for Piper's
actual servo response).
"""
import numpy as np
from pathlib import Path

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import register_robot_class

ASSET_PATH = str(Path(__file__).resolve().parent.parent / "piper_assets" / "robot_arm.xml")


@register_robot_class("FixedBaseRobot")
class Piper(ManipulatorModel):
    """AgileX Piper, single 6-DoF arm."""

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(ASSET_PATH, idn=idn)

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "PiperGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_spot"}

    @property
    def init_qpos(self):
        # Re-tuned (2026-07-13) after moving the base closer to the table
        # (see base_xpos_offset below) -- the previous ready pose was found
        # for the old, much farther-back base position and no longer lands
        # above table centre once the base moved. Grid search over
        # joint2/3/4 for the new base position (table default -> base at
        # x=-0.45): eef ~= (0.007, 0.000, 1.054) in world frame, comfortably
        # above every current YCB object (tallest, PowerDrill, tops out
        # ~0.99m) and centred over the table.
        return np.array([0.0, 0.2, 0.42, 1.6, 0.0, 0.0])

    @property
    def base_xpos_offset(self):
        # RoboSuite's generic "-0.35 - table_length/2" margin is tuned for
        # longer-reach arms (Panda ~855mm). Empirically probed max horizontal
        # reach for this Piper port at table height is only ~0.75m from
        # base_link (see piper_robosuite/piper_multi_object_scene.py's
        # header comment) -- with the generic margin, a 0.8m table's far
        # corners sit >1m from base and are provably unreachable (IK error
        # ~30-40cm, not a local-minimum artifact -- confirmed by sweeping
        # seeds and joint1 pre-rotation). Use a much smaller margin so the
        # whole table stays within reach; pairs with the smaller
        # table_full_size used by PiperMultiObjectScene.
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.05 - table_length / 2, 0.0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
