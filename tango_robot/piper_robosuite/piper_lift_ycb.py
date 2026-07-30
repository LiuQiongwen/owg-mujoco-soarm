"""
Piper + RoboSuite "Lift" task, with RoboSuite's default cube swapped for this
project's actual YCB objects (Pear / TomatoSoupCan) instead of a generic
BoxObject -- keeps the sim scene consistent with every other GeoEBM/consensus
result in the project, which are all evaluated on the real YCB meshes.

Everything else (placement initializer, table, observables, lift-height
success check) is inherited unchanged from robosuite.environments.
manipulation.lift.Lift; only the object class differs.
"""
import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

from tango_robot.piper_robosuite.piper_ycb_objects import (
    YcbPearObject,
    YcbTomatoSoupCanObject,
    YcbBananaObject,
    YcbMustardBottleObject,
    YcbCrackerBoxObject,
    YcbPowerDrillObject,
    YcbMediumClampObject,
)
from tango_robot.piper_robosuite.piper_controller_config import PIPER_JOINT_POSITION_CONFIG

_OBJ_CLASSES = {
    "pear": YcbPearObject,
    "can": YcbTomatoSoupCanObject,
    "banana": YcbBananaObject,
    "mustard": YcbMustardBottleObject,
    "cracker": YcbCrackerBoxObject,
    "drill": YcbPowerDrillObject,
    "clamp": YcbMediumClampObject,
}


class PiperLiftYCB(Lift):
    """Lift task with `ycb_object` in {"pear", "can"} replacing the cube."""

    def __init__(self, *args, ycb_object="pear", **kwargs):
        assert ycb_object in _OBJ_CLASSES, f"ycb_object must be one of {list(_OBJ_CLASSES)}"
        self._ycb_object = ycb_object
        kwargs.setdefault("controller_configs", PIPER_JOINT_POSITION_CONFIG)
        kwargs.setdefault("ignore_done", True)  # see piper_multi_object_scene.py's matching comment
        super().__init__(*args, **kwargs)

    def _load_model(self):
        # Skip Lift._load_model (it hardcodes BoxObject) -- replicate it here
        # with the object class swapped, everything else identical.
        super(Lift, self)._load_model()  # ManipulationEnv._load_model

        self.robots[0].robot_model.set_base_xpos(self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        obj_cls = _OBJ_CLASSES[self._ycb_object]
        self.cube = obj_cls(name=self._ycb_object)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.03,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )
