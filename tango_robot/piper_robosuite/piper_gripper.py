"""
RoboSuite GripperModel registration for the Piper 2-finger parallel gripper
(link7/link8, extracted from soulde/Piper_mujoco's combined robot.xml into
tango_robot/piper_assets/piper_gripper.xml -- see that file's header comment
for the exact split rationale).

Modelled directly on robosuite's built-in PandaGripper (single continuous
open/close action -> maps to both finger slide joints, which move in
opposite directions per the original file's `gripper_slide_joint_eq`
equality constraint).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper

ASSET_PATH = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "piper_assets" / "piper_gripper.xml")


@register_gripper
class PiperGripperBase(GripperModel):
    """Piper's built-in 2-finger parallel gripper (joint7/joint8, slide
    range -0.05 to 0 metres each, coupled via an equality constraint in
    the XML). Range widened from the community-ported -0.038 to match
    AgileX's own official Isaac Lab asset (piper_gripper.urdf), which
    specifies 0.05m of prismatic travel per finger -- see piper_gripper.xml's
    actuator comment. 2026-07-14."""

    def __init__(self, idn=0):
        super().__init__(ASSET_PATH, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([-0.025, -0.025])  # roughly half-open

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger7_collision"],
            "right_finger": ["finger8_collision"],
            "left_fingerpad": ["finger7_collision"],
            "right_fingerpad": ["finger8_collision"],
        }


@register_gripper
class PiperGripper(PiperGripperBase):
    """Single-action wrapper: -1 => open, 1 => closed (same convention as
    robosuite's PandaGripper)."""

    def format_action(self, action):
        assert len(action) == self.dof
        # Sign convention verified empirically (2026-07-14) by measuring
        # link7/link8 body positions at both slide extremes: qpos near the
        # upper/"0" end is CLOSED (small fingertip gap), qpos near the lower
        # end is OPEN (wide gap) -- the opposite of what the joint's own
        # "-0.05 to 0" range name suggests. So action=+1 (close) must move
        # current_action toward -0.004 (i.e. ADD, not subtract, since
        # -0.004 > -0.05).
        #
        # Upper bound -0.004 (not 0.0): fully closed (0) makes the two
        # finger meshes touch with exactly zero clearance, which caused a
        # QACC/NaN instability confirmed via ncon>0 between
        # finger7_collision/finger8_collision at dist~0 (softer
        # solimp/solref alone did not fix it, only leaving a small physical
        # gap did). Lower bound widened -0.038 -> -0.05 (2026-07-14) to
        # match AgileX's official gripper spec (see class docstring) --
        # true opening is closer to 0.10m than the 0.076m previously
        # modeled here, which is directly why several YCB objects
        # (Cracker, Banana, Can) that failed to grasp earlier were
        # re-tested after this change.
        #
        # NOTE (2026-07-15): this value (current_action) is an ABSOLUTE
        # real-units joint position, not a normalized [-1,1] action -- that
        # was, for a long time, silently double-scaled into near-uselessness
        # by robosuite's SimpleGripController on top of this correct value.
        # See piper_controller_config.py's header comment ("ROOT CAUSE
        # FOUND AND FIXED") for the full story and the actual fix
        # (use_action_scaling=False on the gripper controller config) --
        # this function's own logic was correct the whole time.
        self.current_action = np.clip(
            self.current_action + np.array([1.0, 1.0]) * self.speed * np.sign(action),
            -0.05, -0.004,
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
