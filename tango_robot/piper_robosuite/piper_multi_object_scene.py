"""
Piper + RoboSuite: multi-object scene management -- place several of the
project's real YCB objects on the table simultaneously (not the single-cube
Lift task), as the scene-management precursor to the semantic-recognition
grasping step (VLM picks which of several objects on the table to grasp;
that selection logic is a separate, later step -- this class only owns
spawning/placing/settling multiple objects without overlap).

Subclasses Lift (for arena/robot setup, which is generic) but overrides
every cube-specific piece (_load_model, _setup_references,
_setup_observables, reward, _check_success, _visualization) since there is
no longer a single `self.cube`.
"""
import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_body, new_geom
from robosuite.utils.placement_samplers import UniformRandomSampler

# Table size: RoboSuite's Lift default (0.8x0.8) puts the table's far
# corners beyond this Piper port's real reach (empirically probed max
# horizontal reach ~0.75m from base_link at table height -- see
# piper_robot.py's base_xpos_offset comment). Shrunk to 0.6x0.6 so every
# corner, including the placement tray, stays within reach even after
# also moving the base closer (base_xpos_offset margin 0.05 instead of
# 0.35): base lands at x=-0.35, farthest table corner (0.3,0.3) is then
# ~0.72m away -- within the measured envelope with a small margin.
DEFAULT_TABLE_FULL_SIZE = (0.6, 0.6, 0.05)

# Placement tray ("groove"): where the arm puts objects after recognizing
# and grasping them. Sits in a table corner, disjoint from the object
# spawn region (see PLACEMENT_TRAY_CENTER vs. the sampler's x_range/y_range
# below -- kept in one place since they must stay non-overlapping).
# Enlarged 2026-07-14 (0.06 -> 0.09 half-extent, i.e. 12x12cm -> 18x18cm
# interior) for more placement margin -- checked the farthest tray corner
# (0.28,-0.28) is ~0.69m from the base (-0.35,0), still within the ~0.75m
# measured reach envelope with a small buffer, and the tray's outer wall
# (up to x=0.28) stays inside the table's physical edge (table half-size
# 0.3) with a 2cm margin.
PLACEMENT_TRAY_CENTER = (0.19, -0.19)   # (x, y), table-relative
PLACEMENT_TRAY_HALF_EXTENT = 0.09       # groove interior half-width/-depth
PLACEMENT_TRAY_WALL_HEIGHT = 0.03
PLACEMENT_TRAY_WALL_THICKNESS = 0.006


def _add_placement_tray(arena, table_offset, center_xy=PLACEMENT_TRAY_CENTER,
                         half_extent=PLACEMENT_TRAY_HALF_EXTENT,
                         wall_height=PLACEMENT_TRAY_WALL_HEIGHT,
                         wall_thickness=PLACEMENT_TRAY_WALL_THICKNESS):
    """Append a static open-top tray (base plate + 4 walls) to `arena`'s
    worldbody, forming a physical groove objects can be placed into and not
    roll out of. Built from box geoms (MuJoCo has no boolean subtraction),
    same approach as a real tray: a base plus raised rim on all 4 sides."""
    cx, cy = center_xy
    table_top_z = table_offset[2]
    base_z = table_top_z + 0.003
    wall_z = base_z + wall_height / 2

    # NOTE: group=1 (not the new_geom default of 0) -- RoboSuite's rendering
    # convention treats group 0 as "collision-only, hidden" and group 1 as
    # "visible" (this is why the YCB objects use a separate group-0
    # collision mesh alongside a group-1 textured visual mesh). A first
    # version of this tray used the default group=0 and was completely
    # invisible in every render despite being geometrically correct in the
    # compiled model -- confirmed via direct geom_xpos/rgba/group queries
    # and pixel-level search before finding this.
    tray_body = new_body(name="placement_tray", pos=[cx, cy, 0])
    tray_body.append(new_geom(
        name="placement_tray_base", type="box", pos=[0, 0, base_z],
        size=[half_extent, half_extent, 0.003], group=1,
        rgba=[0.75, 0.25, 0.1, 1], contype="1", conaffinity="1",
        friction="1.0 0.05 0.01",
    ))
    wall_specs = [
        ("plus_x",  [half_extent, 0, wall_z], [wall_thickness, half_extent, wall_height / 2]),
        ("minus_x", [-half_extent, 0, wall_z], [wall_thickness, half_extent, wall_height / 2]),
        ("plus_y",  [0, half_extent, wall_z], [half_extent, wall_thickness, wall_height / 2]),
        ("minus_y", [0, -half_extent, wall_z], [half_extent, wall_thickness, wall_height / 2]),
    ]
    for tag, pos, size in wall_specs:
        tray_body.append(new_geom(
            name=f"placement_tray_wall_{tag}", type="box", pos=pos, size=size, group=1,
            rgba=[0.75, 0.25, 0.1, 1], contype="1", conaffinity="1",
        ))
    arena.worldbody.append(tray_body)

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

OBJ_CLASSES = {
    "pear": YcbPearObject,
    "can": YcbTomatoSoupCanObject,
    "banana": YcbBananaObject,
    "mustard": YcbMustardBottleObject,
    "cracker": YcbCrackerBoxObject,
    "drill": YcbPowerDrillObject,
    "clamp": YcbMediumClampObject,
}
ALL_OBJECTS = list(OBJ_CLASSES.keys())

# UniformRandomSampler's collision check uses each object's full
# circumscribed horizontal_radius (e.g. banana=0.122, drill=0.118,
# cracker=0.109, clamp=0.103 -- half their longest in-plane extent), which
# is conservative for elongated/asymmetric objects. Combined with the
# smaller 0.6x0.6 table (needed for reach, see DEFAULT_TABLE_FULL_SIZE),
# the usable spawn region only reliably fits objects with small radii
# together (verified: pear+can+mustard OK; adding banana/clamp/drill/
# cracker to that trio fails the sampler's 5000-retry budget). Default to
# the reliable small-radius trio; pass ycb_objects=[...] explicitly to try
# other combinations (may raise RandomizationError) or use PiperLiftYCB for
# single large objects.
DEFAULT_SCENE_OBJECTS = ["pear", "can", "mustard"]


class PiperMultiObjectScene(Lift):
    """Spawns `ycb_objects` (list of keys into OBJ_CLASSES) simultaneously
    on the table, spread out via UniformRandomSampler with
    ensure_valid_placement=True so they don't overlap each other or the
    robot's resting pose. No reward/success logic -- scene management only."""

    def __init__(self, *args, ycb_objects=None, **kwargs):
        self._ycb_object_names = ycb_objects if ycb_objects is not None else list(DEFAULT_SCENE_OBJECTS)
        for name in self._ycb_object_names:
            assert name in OBJ_CLASSES, f"unknown ycb object '{name}', must be one of {ALL_OBJECTS}"
        kwargs.setdefault("table_full_size", DEFAULT_TABLE_FULL_SIZE)
        kwargs.setdefault("controller_configs", PIPER_JOINT_POSITION_CONFIG)
        # RoboSuite's default horizon=1000 with ignore_done=False was
        # already close to how many steps one full run_pick_and_place trial
        # takes (~1000-1050) before the settle period added 2026-07-14 --
        # that pushed trials over the limit ("executing action in
        # terminated episode"). Not an RL use case here, so just disable
        # episode termination rather than picking a new magic horizon.
        kwargs.setdefault("ignore_done", True)
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super(Lift, self)._load_model()  # ManipulationEnv._load_model (robot placement etc.)

        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        )

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        _add_placement_tray(mujoco_arena, self.table_offset)
        # Table-centred top-down camera -- the generic "birdview" camera is
        # centred on the robot, not the table, and clips the tray corner.
        from robosuite.utils.mjcf_utils import new_element
        mujoco_arena.worldbody.append(new_element(
            tag="camera", name="tablecam", mode="fixed",
            pos=[0, 0, self.table_offset[2] + 1.0], quat=[0.7071, 0, 0, 0.7071],
        ))

        self.objects = [OBJ_CLASSES[name](name=name) for name in self._ycb_object_names]

        # Object spawn region is kept disjoint from the placement tray's
        # corner (PLACEMENT_TRAY_CENTER=(0.19,-0.19), half_extent=0.09, so it
        # occupies roughly x in [0.10,0.28], y in [-0.28,-0.10]) -- capping
        # y_range's lower bound at -0.08 keeps every spawn strictly on the
        # far side of the tray regardless of x (object y always >= -0.08 >
        # tray's y max of -0.10, with a small margin), so objects never
        # spawn inside or overlapping the tray walls.
        #
        # x_range's lower bound is DYNAMIC, based on the largest object's
        # own horizontal_radius in this scene (2026-07-14): the robot's own
        # mount pedestal is a fixed collision cylinder (radius 0.18m,
        # centred at world x=-0.47), reaching to x=-0.29 -- found while
        # investigating why Cracker/Banana (large-radius objects,
        # 0.1085m/0.122m) failed so consistently: their sampled spawn
        # position could place their edge overlapping the STATIONARY mount
        # before the arm ever moved (confirmed via a step-by-step contact
        # trace: 'fixed_mount0_pedestal_col' touching the object from step
        # 0 of "approach"). A single fixed bound tight enough for those
        # large objects (-0.14) was too tight for the small pear/can/mustard
        # trio to reliably co-place (RandomizationError) -- computing the
        # bound per-scene from whichever objects are actually present keeps
        # both cases working instead of picking one fixed compromise value.
        MOUNT_EDGE_X = -0.29
        SAFETY_MARGIN = 0.02
        max_radius = max(obj.horizontal_radius for obj in self.objects)
        x_min = MOUNT_EDGE_X + max_radius + SAFETY_MARGIN
        half_x = self.table_full_size[0] / 2 - 0.05
        x_min = max(x_min, -half_x)  # never exceed the table's own edge either
        # BUG FOUND AND FIXED (2026-07-16): UniformRandomSampler's own
        # default (rng=None -> robosuite's ObjectPositionSampler.__init__
        # does `rng = np.random.default_rng()`) is a FRESH, OS-entropy-
        # seeded generator, completely independent of any prior
        # `np.random.seed(...)` call -- meaning every np.random.seed(trial_id)
        # used throughout this project's Piper work has had ZERO effect on
        # object placement. Confirmed empirically: re-running the identical,
        # unmodified collection script on the same scene_id twice produced
        # three different candidate-success patterns across three runs, and
        # spawn_pos X/Y was found to vary by 7-9cm across the 10 "candidates"
        # within a single supposedly-fixed-pose scene in already-collected
        # data. Fix: explicitly derive a seeded rng from the GLOBAL legacy
        # np.random state at construction time -- np.random.randint DOES
        # respect np.random.seed(), so this restores the reproducibility
        # every caller's `np.random.seed(trial_id)` convention already
        # assumed, with no changes needed at any call site.
        placement_rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        self.placement_initializer = UniformRandomSampler(
            name="MultiObjectSampler",
            mujoco_objects=self.objects,
            x_range=[x_min, 0.08],
            y_range=[-0.08, half_x],
            rotation=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.03,
            rng=placement_rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        # Skip Lift's own _setup_references (sets self.cube_body_id, which
        # doesn't apply here) -- go straight to ManipulationEnv's version.
        super(Lift, self)._setup_references()
        self.object_body_ids = {
            name: self.sim.model.body_name2id(obj.root_body)
            for name, obj in zip(self._ycb_object_names, self.objects)
        }

    def _setup_observables(self):
        return super(Lift, self)._setup_observables()

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def _visualization(self):
        super(Lift, self)._visualization()

    def get_object_positions(self):
        return {name: self.sim.data.body_xpos[bid].copy() for name, bid in self.object_body_ids.items()}
