"""
Full IK-driven pick-and-place on the Piper + RoboSuite scene: pick a named
object off the table and place it into the placement tray/groove, using real
actuator-driven motion (not the kinematic-only teleport check in
piper_robosuite_ik_check.py).

Pipeline per phase (approach -> descend -> close -> lift -> transit ->
lower into tray -> open -> retract):
  1. Solve joint-space IK for the phase's target end-effector POSE (position
     + orientation, not position-only like piper_robosuite_ik_check.py),
     against a saved/restored copy of qpos so the solve itself never
     disturbs the live physics state.
  2. Drive the arm there via the JOINT_POSITION (absolute) controller
     (piper_controller_config.PIPER_JOINT_POSITION_CONFIG) -- repeatedly
     command the same target qpos for several physics steps so the PD
     controller actually converges under real dynamics/contact, instead of
     teleporting.

NOTE on why orientation is now solved too (2026-07-14): a first
position-only version consistently produced an asymmetric one-finger
contact instead of a grip -- measured the two fingers' body positions at
the "descend" pose and found a 1.6cm HEIGHT difference between them (the
gripper was tilted, not level), because position-only IK leaves 3 spare DOF
free to settle into whatever orientation the DLS solver happens to
converge to. Solving position+orientation together keeps the gripper level
(local Z axis pointing straight down) at every waypoint, which is required
for a symmetric two-finger grasp on a roughly-centered object.

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_pick_and_place [pear|can|mustard]
"""
import sys

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa: registers Piper/PiperGripper
from tango_robot.piper_robosuite.piper_multi_object_scene import (
    PiperMultiObjectScene, PLACEMENT_TRAY_CENTER,
)
from tango_robot.piper_robosuite.piper_candidate_selection import (
    sample_candidate_pool, sample_perception_noisy_candidates, select_best, select_consensus,
)
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT

JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
EEF_SITE = "robot0_eef_site"
IK_DAMPING = 1e-3
IK_TOL = 5e-3
READY_QPOS = np.array([0.0, 0.2, 0.42, 1.6, 0.0, 0.0])

# REAL_JOINT_LIMITS (2026-07-21, Gate 3 of the real-hardware readiness plan):
# matches robot_arm.xml's declared ranges = AgileX Piper's real hardware
# wrist/joint limits, NOT just a sim convenience constant.
REAL_JOINT_LIMITS = [(-2.618, 2.618), (0.0, 3.14), (-1.637, 1.33),
                      (-1.832, 1.832), (-1.22, 1.22), (-3.14, 3.14)]


def clip_action_to_real_limits(action):
    """MANDATORY safety layer for any real-hardware deployment (2026-07-21,
    Gate 3). Stage 7 found the CR-CFM model's raw output can diverge
    numerically to ~370 rad for some inputs; Stage 8 added
    `clamp_waypoints_to_limits` to fix this, but found it REDUNDANT in sim
    -- MuJoCo's own physics solver already enforces `jnt_range` regardless
    of the commanded value, so the win-rate-focused default is off. THAT
    REDUNDANCY IS SIM-SPECIFIC. A real motor controller has no such
    guarantee (see ISAACS, "Sim-to-Lab-to-Real": a learned policy must be
    treated as an "untrusted oracle" needing explicit runtime supervision).
    Direct verification (Gate 3): without any clip, trial 1007's actual
    commanded action reached 151.7 rad (48x the real range) with 150/1690
    steps out of range; Stage 8's SCOPED clamp (`clamp_waypoints_to_limits`,
    applied only inside `move_to_cr_cfm_descend`) still left 125/1690
    violations -- ALL from `lower_into_tray`, a completely different code
    path (`move_to_interpolated`) the scoped clamp never covered. This
    UNIVERSAL clip, applied at the lowest common point (immediately before
    any action would reach `env.step`/a real motor interface, regardless of
    which internal function constructed it), was verified to eliminate
    100% of violations (275/1690 before -> 0/1690 after) on the same trial
    with the scoped clamp deliberately disabled. For any real-hardware
    backend, call this on every action immediately before sending it to the
    motors -- this is a safety requirement, independent of win rate."""
    clipped = action.copy()
    for j, (lo, hi) in enumerate(REAL_JOINT_LIMITS):
        clipped[j] = np.clip(action[j], lo, hi)
    return clipped

APPROACH_HEIGHT = 0.10
# Per-object top-surface offset from CoM (z of each object's own asset
# "top_site", tango_robot/piper_robosuite/assets/*.xml). Used to widen
# APPROACH_HEIGHT for tall objects -- found 2026-07-14 while re-diagnosing
# Cracker after the transit-waypoint fix (below) still didn't fix it: the
# waypoint fix stopped the arm from sweeping through the object mid-transit,
# but "approach" (10cm above CoM) turned out to barely clear Cracker's own
# top surface (10.19cm above CoM) at all, so both fingers were already
# grazing the box's top edge during what should have been a clear hover
# phase. A fixed 10cm clearance was tuned around Pear (top 8.05cm) and
# never re-checked against taller objects.
OBJECT_TOP_OFFSET = {
    "banana": 0.0340,
    "cracker": 0.1019,
    "clamp": 0.0987,
    "mustard": 0.1075,
    "pear": 0.0805,
    "drill": 0.1837,
    "can": 0.0437,
}


def approach_height_for(obj_name):
    """APPROACH_HEIGHT, widened with a 3cm safety margin above the
    object's own top surface for objects taller than the original
    Pear-tuned default."""
    top = OBJECT_TOP_OFFSET.get(obj_name, APPROACH_HEIGHT)
    return max(APPROACH_HEIGHT, top + 0.03)


# Safe transit height (2026-07-14, fix for the "arm sweeps through tall
# objects" bug): matches READY_QPOS's own verified-clear eef height, well
# above every tested object's top (Cracker's, the tallest at ~1.02m,
# included). A single direct IK solve from READY_QPOS to the "approach"
# target doesn't guarantee the PD controller traces anything like a
# vertical descent -- it can rotate the base WHILE still low, sweeping the
# arm sideways through a tall object before ever reaching the intended
# hover point above it (confirmed via a step-by-step contact trace: a
# finger touched Cracker mid-"approach", not at the final target).
# Splitting into "rotate at a safe height" then "descend straight down at
# the target XY" as two separately-tracked PD targets forces that ordering
# instead of leaving it to whatever path the controller happens to take.
SAFE_TRANSIT_Z = 1.05
# 0.0 (object's own CoM height), not +0.02 -- grasping above the object's
# centre (e.g. near a pear's narrower "neck") let it slip free during lift
# even though both fingers made contact; centring on the CoM gave a stable
# hold (verified: pear rose from the table height it was closed at, ~0.78m,
# to ~0.86m after the lift phase, tracking the gripper instead of staying
# on the table). Found 2026-07-14.
GRASP_HEIGHT_OFFSET = 0.0
TRAY_DROP_HEIGHT = 0.06
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


# Level, straight-down grasp orientation: local Z axis = world -Z (pointing
# down), local X axis = world +X (arbitrary but fixed in-plane yaw), local Y
# completes a right-handed frame (= world -Y).
DOWN_ORIENTATION = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])
ORI_TOL = 2e-2
ORI_WEIGHT = 0.3  # orientation error is in radians, position in metres -- downweight so position isn't sacrificed

# Diverse fallback seeds for solve_multi_seed: joint1 spread across most of
# its [-2.618, 2.618] range, crossed with a couple of distinct elbow
# postures (joint2, joint3, joint4), so a failed primary-seed solve has a
# real chance of landing in a different basin instead of retrying the same
# one. Kept small (6) since each is a full ~3000-iter DLS solve and this
# only runs on the failure path, not every call.
FALLBACK_SEEDS = [
    np.array([j1, 0.2, 0.42, 1.6, 0.0, 0.0])
    for j1 in [-1.8, -0.9, 0.9, 1.8]
] + [
    np.array([0.0, 1.0, -0.5, 0.5, 0.0, 0.0]),
    np.array([0.0, 0.6, 1.0, -0.6, 0.0, 0.0]),
]

# Local (object-mesh-frame) horizontal axis to align the gripper's closing
# direction with, i.e. the object's narrow side -- computed from each
# object's own bounding box (see piper_ycb_objects.py's asset xml sites /
# the bbox probe run when these objects were first ported): whichever of
# the two horizontal axes (not the vertical/height one) is narrower.
#   pear:    size (x,y,z) = (0.0665, 0.1040, 0.0665) -> x is the narrow one
#   mustard: size (x,y,z) = (0.0964, 0.0581, 0.1915), z=height -> y narrow
#   can:     effectively circular (~8.6cm diameter) -- WIDER than the
#            gripper's ~7.6cm max opening regardless of yaw, so no
#            orientation fixes this; kept here for completeness/documentation.
#   banana:  size (0.1958, 0.0789, 0.0376) -- rests on its SIDE (shortest
#            extent is z), so the two candidate axes are both "horizontal"
#            in the usual sense but only y (0.0789) is narrow enough to grip;
#            x (0.196) is the long axis running along the table.
#   cracker: size (0.0717, 0.1640, 0.2135), z=height (box stands upright) ->
#            x narrow. Rectangular prism, cross-section should be constant
#            through height (unlike mustard's bottle taper), so CoM-height
#            grasping is expected to be more reliable here than for mustard.
#   drill:   size (0.1630, 0.1232, 0.1874) -- irregular L-shape (motor body +
#            handle), z=height. y picked as nominally narrower, but the
#            AXIS-ALIGNED BOUNDING BOX is a poor proxy for a drill's actual
#            graspable region (the handle is much narrower than the full
#            bbox suggests, similar to the mustard neck-vs-body problem) --
#            flagged as likely to underperform without per-height
#            cross-section analysis (not yet done, see mustard's z-slice
#            width scan in this file's git history for the method).
#   clamp:   size (0.1520, 0.1349, 0.0494) scaled -- BOTH horizontal extents
#            substantially exceed the gripper's ~0.076m opening, and this
#            object was already found (piper_multi_object_scene.py) to
#            settle in an unpredictable resting orientation (often tipped
#            onto its side, not a clean upright rest) requiring condim=6 to
#            even stop it sliding off the table. compute_grasp_orientation's
#            "project local axis to horizontal, assume only yaw varies"
#            logic assumes an upright rest and may not hold here. Flagged as
#            the highest-risk object of the four new ones -- expect this one
#            most likely to fail, and to fail in a way that isn't explained
#            by the narrow-axis choice at all.
OBJECT_NARROW_AXIS = {
    "pear": np.array([1.0, 0.0, 0.0]),
    "mustard": np.array([0.0, 1.0, 0.0]),
    "can": np.array([1.0, 0.0, 0.0]),  # arbitrary -- geometrically ungraspable either way
    "banana": np.array([0.0, 1.0, 0.0]),
    "cracker": np.array([1.0, 0.0, 0.0]),
    "drill": np.array([0.0, 1.0, 0.0]),  # bbox-based guess, likely wrong -- see comment above
    "clamp": np.array([1.0, 0.0, 0.0]),  # bbox-based guess, likely wrong -- see comment above
}

# Object-mesh-frame XY offset between the body ORIGIN (what
# env.get_object_positions() returns, and what every grasp target was
# computed around) and the TRUE cross-sectional centroid at grasp height
# (z=0, matching GRASP_HEIGHT_OFFSET) -- computed by slicing each object's
# raw mesh vertices near z=0 and averaging XY. Found 2026-07-14 while
# investigating why grasp generalization was so poor: for most objects this
# offset is small and/or falls mostly along the LONG axis (perpendicular to
# the gripper's closing direction), so it barely matters. For Cracker it is
# +3.06cm along the CLOSING axis on a box only 6.5cm wide -- 47% of the
# object's own narrow-axis width, meaning every single grasp attempt was
# closing around a point nearly at the box's edge, not its centre. This
# alone explains the previously-observed "only one finger contacts the
# object" failure pattern for Cracker. Banana/Mustard/Pear's small offsets
# (all <=7% of their narrow width, and mostly along the long axis) are
# NOT expected to meaningfully change their success rates -- their
# remaining failures have other causes (grip force vs weight, established
# separately). drill/clamp not measured (their narrow-axis choice was
# already flagged as unreliable for other reasons).
#
# 2026-07-15 TRIED TWICE, REVERTED BOTH TIMES: re-measured this offset
# directly from model.mesh_vert on the live compiled model (reading
# cracker_g1's actual 8-vertex collision-hull bbox center: [0.0549,
# -0.0217] vs the [0.0306, -0.0102] below -- roughly double). First attempt
# (before the gripper-controller double-scaling bug was found and fixed,
# see piper_controller_config.py) made things worse, but was confounded --
# with ~0.1mm of real gripper clearance at the time, any target
# imprecision was catastrophic regardless of which offset was more
# correct. Re-tested AFTER the gripper fix (true ~12cm open span) on a
# clean n=20 Cracker batch: 2/20 (10%), far worse than this file's
# original [0.0306, -0.0102] value's 9/20 (45%) under the same fixed
# gripper. So this was not just a confound the first time -- the 8-vertex
# collision-hull bbox center is independently confirmed to be the wrong
# target, twice, under two different gripper-clearance regimes. Likely
# explanation, CONFIRMED 2026-07-15: the collision hull's raw bbox center
# (min+max)/2 over its 8 vertices is NOT the same thing as its true
# volumetric/mass centroid unless the hull happens to be a perfectly
# symmetric box -- and MuJoCo already computes the real answer for us at
# compile time as `model.body_ipos[body_id]` (center of mass in body-local
# frame, used directly by the physics engine, not something to re-derive
# from vertices by hand). Checked cracker: body_ipos = [0.0275, -0.0108],
# almost exactly matching the ORIGINAL [0.0306, -0.0102] below (within
# ~3mm) -- NOT the bbox-center value [0.0549, -0.0217] that was tried and
# reverted twice above. So the original z-slice measurement was basically
# right all along; the bbox-center recomputation was measuring the wrong
# quantity -- a real, useful diagnostic conclusion even though the fix
# attempt below it (re-measuring all four objects from body_ipos and
# swapping them in) did NOT pan out: tested on matched n=20 batches,
# Cracker went 45%->35% (7/20, a mild regression) and Mustard was
# unchanged at 65% (13/20, same as baseline) despite mustard's offset
# changing more than cracker's. No validated improvement, so reverted back
# to the original values below. `model.body_ipos` remains the right thing
# to reach for if this needs revisiting (it IS the authoritative CoM,
# unlike the bbox-center approach) -- it just didn't move the needle here,
# meaning centroid-offset precision likely isn't the dominant remaining
# lever for either object at this point.
OBJECT_CENTROID_OFFSET_LOCAL = {
    "pear": np.array([-0.0014, 0.0155]),
    "mustard": np.array([0.0076, -0.0017]),
    "can": np.array([0.0043, 0.0067]),
    "banana": np.array([0.0225, -0.0018]),
    "cracker": np.array([0.0306, -0.0102]),
}


def _quat_to_mat(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def true_centroid_xy(body_origin_pos, quat, obj_name):
    """The body origin is not generally the object's true cross-sectional
    centre (see OBJECT_CENTROID_OFFSET_LOCAL) -- rotate that mesh-frame
    offset into world frame (by the object's CURRENT orientation) and add
    it to the body origin to get where the gripper should actually centre
    itself, instead of assuming body origin = grasp centre."""
    offset_local = OBJECT_CENTROID_OFFSET_LOCAL.get(obj_name)
    if offset_local is None:
        return body_origin_pos.copy()
    R_obj = _quat_to_mat(quat)
    offset_world = R_obj[:2, :2] @ offset_local
    corrected = body_origin_pos.copy()
    corrected[:2] += offset_world
    return corrected


def grasp_orientation_from_quat(quat, obj_name):
    """Pure-math core of compute_grasp_orientation: build the grasp
    rotation matrix from an explicit (w,x,y,z) object quaternion, instead
    of reading it live from the sim. Factored out so a candidate sampler
    can feed in a NOISY quaternion (simulating an imperfect perception
    estimate of the object's orientation) and get back the grasp target
    that estimate would imply, without needing a live env/body id."""
    R_obj = _quat_to_mat(quat)
    local_axis = OBJECT_NARROW_AXIS.get(obj_name, np.array([1.0, 0.0, 0.0]))
    world_dir = R_obj @ local_axis
    world_dir[2] = 0.0  # project to horizontal -- approach is always straight down
    norm = np.linalg.norm(world_dir)
    if norm < 1e-6:
        world_dir = np.array([1.0, 0.0, 0.0])
    else:
        world_dir /= norm

    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, world_dir)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)  # re-orthogonalize
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def compute_grasp_orientation(env, obj_name):
    """Build a target rotation matrix (gripper pointing straight down, X
    axis aligned with the object's narrow horizontal axis) from the
    object's CURRENT world orientation -- so the grasp works regardless of
    how the placement sampler happened to rotate it, instead of assuming a
    fixed world-X approach direction."""
    bid = env.object_body_ids[obj_name]
    quat = env.sim.data.xquat[bid].copy()  # (w, x, y, z)
    return grasp_orientation_from_quat(quat, obj_name)


def _flip_grasp_orientation(mat):
    """180-degree-around-the-approach-axis equivalent grasp (2026-07-20):
    for a symmetric gripper aligning with an object's narrow horizontal
    axis, negating the x/y axes (keeping the downward z axis unchanged)
    grips the SAME way -- same narrow-axis alignment, same straight-down
    approach -- but reaches the object from the opposite side, which can
    require a very different wrist-roll (joint6) rotation to reach.
    Exploited by pick_wrist_friendly_orientation below."""
    flipped = mat.copy()
    flipped[:, 0] = -mat[:, 0]
    flipped[:, 1] = -mat[:, 1]
    return flipped


def _rotate_grasp_about_approach(mat, angle_rad):
    """Rotate a grasp orientation by `angle_rad` about its OWN approach axis
    (mat's z-column, the straight-down direction) -- for a rectangular
    box's narrow-axis grasp, only 0/180 degrees keep the gripper fingers
    exactly parallel to the object's narrow axis, but a real parallel-jaw
    gripper has SOME finger-width/compliance tolerance before contact
    quality actually degrades. Used by pick_wrist_friendly_orientation's
    optional small-angle sweep (2026-07-21) to search a few degrees either
    side of the two exact candidates for extra joint6 headroom, instead of
    only the exact 0/180 pair."""
    z = mat[:, 2]
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    # Rodrigues' rotation formula for rotating the other two (orthonormal)
    # columns about the z axis by angle_rad; z itself is invariant.
    rotated = mat.copy()
    for col in (0, 1):
        v = mat[:, col]
        rotated[:, col] = v * c + np.cross(z, v) * s + z * np.dot(z, v) * (1 - c)
    return rotated


def pick_wrist_friendly_orientation(ik, target_pos, grasp_mat, seed_qpos, joint6_idx=5, joint6_limit=3.14,
                                     angle_tolerance_deg=0.0, angle_step_deg=5.0):
    """2026-07-20: robot0_joint6 (wrist roll) has a genuine HARDWARE limit
    (+-3.14 rad -- robot_arm.xml's `limited="true" range="-3.14 3.14"`,
    matching AgileX Piper's real +-180-degree wrist-roll range; this is a
    physical constraint, not an arbitrary modeling choice, confirmed before
    implementing this fix). Diagnostic finding (32-trial scan across the
    tuning/held-out/fresh/validation ranges, Fisher's exact p=1.8e-5, odds
    ratio 90): when the IK solution for compute_grasp_orientation's target
    lands joint6 pinned at that exact limit, the trial fails 82% of the
    time (9/11) vs. 4.8% (1/21) when it doesn't -- the single strongest
    predictor found this project, by a wide margin over anything examined
    inside the CR-CFM descend loop (Stages 2, 4, 5, 8, 9, 10, 11 -- all
    seven rejected). The likely mechanism: `grasp_orientation_from_quat`
    picks ONE specific orientation for a given object yaw; when that
    orientation's natural wrist-roll solution would need to go slightly
    past the hardware limit, the DLS solver's per-iteration joint clip
    (ArmIK.solve) pins it exactly at the boundary instead -- a valid,
    converged solution by position/orientation error, but one that leaves
    no headroom, so any further correction (e.g. descend's own re-solve at
    a marginally different height) can be forced into a huge "long way
    around" alternative. Fixes the ROOT CAUSE (a bad target-orientation
    choice, computed once before any of the CR-CFM descend loop even
    starts) instead of any downstream numerical consequence of it, unlike
    every previously-tried mechanism in this plan.

    Solves IK for BOTH `grasp_mat` and its 180-degree-flipped equivalent
    (`_flip_grasp_orientation`, functionally the same grasp from the
    object's other side) from the SAME seed, and keeps whichever leaves
    joint6 further from its hard limit -- a direct, solver-grounded check,
    not a geometric heuristic. Falls back to the original `grasp_mat` if
    IK fails to converge for both (never makes the trial worse than not
    running this check at all)."""
    # angle_tolerance_deg (2026-07-21, opt-in, default 0.0 = exact prior
    # behavior): additionally sweeps small offsets (+-angle_step_deg,
    # +-2*angle_step_deg, ... up to +-angle_tolerance_deg) around EACH of
    # the two exact candidates, in case a few degrees of finger-contact
    # slack buys meaningfully more joint6 headroom. At 0.0 this reduces to
    # exactly the original 2-candidate (0/180) check.
    base_candidates = (grasp_mat, _flip_grasp_orientation(grasp_mat))
    if angle_tolerance_deg > 0:
        offsets_deg = [0.0]
        step = angle_step_deg
        while step <= angle_tolerance_deg + 1e-9:
            offsets_deg += [step, -step]
            step += angle_step_deg
        candidates = [
            _rotate_grasp_about_approach(base, np.radians(off))
            for base in base_candidates for off in offsets_deg
        ]
    else:
        candidates = list(base_candidates)

    best_mat, best_margin, any_converged = grasp_mat, -np.inf, False
    for mat in candidates:
        result, converged, err = ik.solve(target_pos, seed_qpos, target_mat=mat)
        if not converged:
            continue
        margin = joint6_limit - abs(result[joint6_idx])
        if margin > best_margin:
            best_margin, best_mat, any_converged = margin, mat, True
    return best_mat if any_converged else grasp_mat


class ArmIK:
    """DLS position+orientation IK against a live RoboSuite sim, solved on a
    saved qpos snapshot so intermediate iterations never leak into the
    physics state -- only the final result is used as a motion target."""

    def __init__(self, env):
        self.env = env
        self.model = env.sim.model
        self.data = env.sim.data
        self.qpos_adr = [self.model.joint(n).qposadr[0] for n in JOINTS]
        self.dof_adr = [self.model.joint(n).dofadr[0] for n in JOINTS]
        self.jnt_ids = [self.model.joint(n).id for n in JOINTS]
        self.eef_site_id = self.model.site(EEF_SITE).id

    def _get_qpos(self):
        return np.array([self.data.qpos[a] for a in self.qpos_adr])

    def _set_qpos(self, qpos):
        for adr, q in zip(self.qpos_adr, qpos):
            self.data.qpos[adr] = q
        self.env.sim.forward()

    @staticmethod
    def _ori_error(mat_cur, mat_target):
        # Standard per-axis cross-product rotation error (small-angle
        # approximation, exact at zero error): sum over the 3 columns of
        # cur_axis x target_axis. Well-conditioned for the near-vertical
        # target orientation used here.
        e = np.zeros(3)
        for i in range(3):
            e += np.cross(mat_cur[:, i], mat_target[:, i])
        return 0.5 * e

    def solve(self, target_pos, seed_qpos, target_mat=DOWN_ORIENTATION, iters=3000):
        saved = self._get_qpos()
        self._set_qpos(seed_qpos)
        import mujoco
        converged = False
        for _ in range(iters):
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model._model, self.data._data, jacp, jacr, self.eef_site_id)
            Jp = jacp[:, self.dof_adr]
            Jr = jacr[:, self.dof_adr]
            J = np.concatenate([Jp, Jr], axis=0)

            pos_err = target_pos - self.data.site_xpos[self.eef_site_id]
            mat_cur = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
            ori_err = self._ori_error(mat_cur, target_mat)
            err = np.concatenate([pos_err, ORI_WEIGHT * ori_err])

            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), err)
            cur = self._get_qpos()
            for i in range(6):
                lo, hi = self.model.jnt_range[self.jnt_ids[i]]
                cur[i] = np.clip(cur[i] + dq[i], lo, hi)
            self._set_qpos(cur)
            if np.linalg.norm(pos_err) < IK_TOL and np.linalg.norm(ori_err) < ORI_TOL:
                converged = True
                break
        result = self._get_qpos()
        pos_err_final = float(np.linalg.norm(self.data.site_xpos[self.eef_site_id] - target_pos))
        self._set_qpos(saved)  # restore -- solving must not move the live sim
        return result, converged, pos_err_final

    def solve_multi_seed(self, target_pos, primary_seed, target_mat=DOWN_ORIENTATION,
                          iters=3000, fallback_seeds=FALLBACK_SEEDS):
        """Try primary_seed first (cheap, the common case -- warm-started from
        the previous phase's solution, which converges often enough on its
        own). Only if that fails to converge, fall back to a small diverse
        set of seeds (spanning joint1's range x a few elbow postures).
        This is the actual fix for the pilot experiment's convergence gap:
        DLS is a local method with no guarantee the single seed's basin
        contains a valid solution -- porting AgileX's own analytical IK
        was investigated and rejected (their DH parameters do not match
        this ported MJCF model even after searching all 64 per-joint sign
        flips, residual error stayed ~2.3 -- see piper_own_fk.py's
        from-model FK, which DOES match MuJoCo exactly, for the verified
        alternative). Multi-seed restart is the lower-risk fix: it directly
        targets "local method missing an existing solution" without betting
        on a from-scratch closed-form wrist decomposition being bug-free."""
        result, converged, err = self.solve(target_pos, primary_seed, target_mat, iters)
        if converged:
            return result, converged, err, "primary"
        best = (result, converged, err, "primary")
        for seed in fallback_seeds:
            r, c, e = self.solve(target_pos, seed, target_mat, iters)
            if c:
                return r, c, e, "fallback"
            if e < best[2]:
                best = (r, c, e, "fallback")
        return best


def score_candidate_ik(ik, pos, mat, seed_qpos, iters=300):
    """Cheap single-seed, low-iteration IK solve used only to RANK
    candidates for select_best -- not the robust multi-seed solve used to
    actually move the arm. Running full solve_multi_seed (up to 7 x 3000
    iters) for every one of N pool candidates would make each trial
    roughly N times slower for no benefit (only the chosen candidate is
    ever actually used); this quick check is 10x cheaper per candidate and
    only needs to be good enough for relative ranking, not final accuracy."""
    _, _, err = ik.solve(pos, seed_qpos=seed_qpos, target_mat=mat, iters=iters)
    return err


def _set_phase(step_hook, name):
    """step_hook is conventionally a bound method (e.g. recorder.snap), not
    the recorder object itself -- hasattr(step_hook, "set_phase") checks
    the METHOD object, which never has its owner's other attributes, so it
    always silently misses (found 2026-07-18: every phase tag came back ""
    despite this exact check being in place). Resolve the bound method's
    __self__ first, matching how bound methods actually work in Python."""
    owner = getattr(step_hook, "__self__", step_hook)
    set_phase_fn = getattr(owner, "set_phase", None)
    if set_phase_fn is not None:
        set_phase_fn(name)


def move_to(env, target_qpos, gripper_action, steps=80, step_hook=None):
    action = np.concatenate([target_qpos, [gripper_action]])
    for _ in range(steps):
        env.step(action)
        if step_hook is not None:
            step_hook(env)


def move_to_interpolated(env, ik, target_qpos, gripper_action, n_waypoints=6, steps_per_waypoint=25, step_hook=None):
    """Like move_to, but steps through intermediate joint-space waypoints
    instead of commanding the final target in one shot. Added 2026-07-14
    after tracing a held-mustard trial: both fingers had a legitimate,
    decent-force grip right after closing, but the object's height barely
    changed through the "lift" phase (0.886m -> 0.882m) -- it was never
    actually lifted. move_to's single abrupt target commands a fast PD
    yank (kp=50) toward the new pose; for a light object (pear, 0.049kg)
    that's fine, but a heavier, more marginal grip (mustard, 0.603kg) can
    slip from the resulting acceleration even though it easily resists the
    object's static weight. Interpolating gives the grip a gentler ramp
    instead of one sudden jerk."""
    current_qpos = ik._get_qpos()
    for i in range(1, n_waypoints + 1):
        alpha = i / n_waypoints
        waypoint = current_qpos + alpha * (target_qpos - current_qpos)
        action = np.concatenate([waypoint, [gripper_action]])
        for _ in range(steps_per_waypoint):
            env.step(action)
            if step_hook is not None:
                step_hook(env)


def _object_contact_geoms(env, obj_name):
    body_id = env.object_body_ids[obj_name]
    model = env.sim.model._model
    return {i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id}


def _has_object_contact(env, obj_geom_ids):
    """True if ANY geom (finger, arm link, or mount) is currently touching
    the target object -- checked directly from MuJoCo's own contact list,
    privileged sim information a real robot would need tactile/F-T sensors
    for (see piper_robosuite/README.md's 2026-07-16 literature-check entry:
    CoorGrasp uses this same "stop advancing once contact is detected,
    don't blindly drive to a precomputed target" mechanism, but needs real
    tactile hardware this project doesn't have -- MuJoCo gives it for free)."""
    data = env.sim.data
    for k in range(data.ncon):
        c = data.contact[k]
        if c.geom1 in obj_geom_ids or c.geom2 in obj_geom_ids:
            other = c.geom2 if c.geom1 in obj_geom_ids else c.geom1
            other_name = env.sim.model.geom(other).name
            if "table" not in other_name and "floor" not in other_name:
                return True
    return False


def _object_contact_force_magnitude(env, obj_geom_ids):
    """Sum of contact-force magnitudes (normal + 2 tangential components,
    read via MuJoCo's own mj_contactForce -- exact, not estimated) between
    any non-table/floor geom and the target object's geoms. The force-aware
    successor to _has_object_contact's binary check (2026-07-16): v1 (stop
    on any contact) and v2 (stop on sustained contact) of the compliant
    descend pilot were both binary/thresholded hard-stop mechanisms and
    neither showed a real benefit (see README) -- CoorGrasp's own
    mechanism is explicitly force-MAGNITUDE-aware (gentle contact forces,
    wrench-balance criterion), not a binary touching/not-touching signal,
    and needs real tactile force sensors for that on real hardware. MuJoCo
    computes exact contact forces internally for free; this reads them
    directly instead of approximating with a binary proxy."""
    import mujoco
    model = env.sim.model._model
    data = env.sim.data._data
    total_force = 0.0
    for k in range(data.ncon):
        c = data.contact[k]
        if c.geom1 in obj_geom_ids or c.geom2 in obj_geom_ids:
            other = c.geom2 if c.geom1 in obj_geom_ids else c.geom1
            other_name = env.sim.model.geom(other).name
            if "table" in other_name or "floor" in other_name:
                continue
            result = np.zeros(6)
            mujoco.mj_contactForce(model, data, k, result)
            total_force += float(np.linalg.norm(result[:3]))
    return total_force


def move_to_force_compliant_descend(env, ik, target_qpos, gripper_action, obj_name,
                                     total_steps=150, force_gain=2.0, force_threshold=8.0,
                                     step_hook=None):
    """Genuine admittance-style compliant descend (2026-07-16, v3): instead
    of a binary hard-stop (v1/v2, both null results -- see README), the
    commanded advancement toward the target SLOWS DOWN continuously and
    proportionally to detected contact force beyond `force_threshold`,
    rather than freezing outright. This avoids v1/v2's dominant failure
    mode (premature full stop, arm left nowhere near a usable grasp
    position) by construction -- progress never truly halts, it just
    creeps forward more slowly under load, and speeds back up if force
    subsides (e.g. the object settles/stops moving away from the
    approaching finger).

    Per-step: effective progress increment = base_increment / (1 + gain *
    max(0, force - threshold)) -- a standard admittance-control-style
    damping law, computed against MuJoCo's own exact contact force
    (_object_contact_force_magnitude), not a real force sensor.

    Returns (final_progress: float in [0,1], mean_force: float) for
    diagnostics -- final_progress < 1.0 indicates the arm was still being
    actively damped by contact force when the step budget ran out.
    """
    obj_geom_ids = _object_contact_geoms(env, obj_name)
    start_qpos = ik._get_qpos()
    progress = 0.0
    base_increment = 1.0 / total_steps
    forces = []
    for _ in range(total_steps):
        force = _object_contact_force_magnitude(env, obj_geom_ids)
        forces.append(force)
        damping = 1.0 / (1.0 + force_gain * max(0.0, force - force_threshold))
        progress = min(1.0, progress + base_increment * damping)
        waypoint = start_qpos + progress * (target_qpos - start_qpos)
        action = np.concatenate([waypoint, [gripper_action]])
        env.step(action)
        if step_hook is not None:
            step_hook(env)
    return progress, float(np.mean(forces)) if forces else 0.0


def move_to_compliant_descend(env, ik, target_qpos, gripper_action, obj_name,
                               n_waypoints=6, steps_per_waypoint=25, sustain_steps=5,
                               step_hook=None):
    """Compliant variant of move_to_interpolated for the descend phase
    specifically (2026-07-16): stop advancing through the remaining
    waypoints once SUSTAINED contact with the target object is detected
    (finger or otherwise), instead of blindly continuing toward the
    precomputed target regardless of what's physically happening --
    directly informed by CoorGrasp's approach/grasp phase-transition
    mechanism (arXiv:2607.03557), simplified and implemented with
    privileged sim contact data instead of real tactile sensors (which
    that paper requires and this project doesn't have).

    sustain_steps=5 (2026-07-16 correction): the first version triggered on
    a single instantaneous contact detection and stopped the arm cold. A/B
    test (n=20, paired) found this made things WORSE (70%->40%) -- checking
    WHEN it triggered explained why: every trial that stopped early (step
    28-58 out of a 150-step budget) failed 10/10, while every trial that
    "stopped" near step 105-106 (i.e. essentially at the natural end of the
    interpolation anyway) succeeded about as often as the uncompliant
    baseline. A single brief/incidental touch early in the trajectory isn't
    the same signal as genuine sustained grazing contact -- requiring
    `sustain_steps` consecutive contact-positive steps before triggering
    filters out the former while still catching the latter.

    Returns (stopped_early: bool, contact_step: int|None) for diagnostics."""
    obj_geom_ids = _object_contact_geoms(env, obj_name)
    current_qpos = ik._get_qpos()
    stopped_early = False
    contact_step = None
    global_step = 0
    consecutive_contact = 0
    for i in range(1, n_waypoints + 1):
        if stopped_early:
            break
        alpha = i / n_waypoints
        waypoint = current_qpos + alpha * (target_qpos - current_qpos)
        action = np.concatenate([waypoint, [gripper_action]])
        for _ in range(steps_per_waypoint):
            env.step(action)
            if step_hook is not None:
                step_hook(env)
            global_step += 1
            if _has_object_contact(env, obj_geom_ids):
                consecutive_contact += 1
                if consecutive_contact >= sustain_steps:
                    stopped_early = True
                    contact_step = global_step
                    break
            else:
                consecutive_contact = 0
    return stopped_early, contact_step


def move_to_two_stage_align_descend(env, ik, obj_name, grasp_height_offset, use_oriented_grasp,
                                     current_qpos, align_clearance=0.03,
                                     n_waypoints=6, steps_per_waypoint=25, step_hook=None):
    """Two-stage descend (2026-07-17, v5): structurally different from v1-v4
    (compliant hard-stop, sustained-contact stop, force-admittance damping,
    pre-narrowing) -- all four of those modify the SAME single interpolated
    pass from approach height to grasp height, and a Stage-0 re-analysis of
    their own already-collected paired trial data (trial_id 1000-1019) found
    that none of them ever beats baseline on any INDIVIDUAL trial (0
    discordant pairs favoring the intervention across all three tested), so
    there is no per-trial heterogeneity for a learned selector to exploit
    among them -- see RULED_OUT_METHODS.md rows 7-10.

    This instead splits descend into an ALIGN stage (down to a checkpoint
    just above the object's own top surface, `align_clearance` past
    OBJECT_TOP_OFFSET -- low contact risk, the gripper's fingers are not yet
    within the object's own height range) followed by a fresh re-read of the
    object's true position/orientation and a re-solved target, then a COMMIT
    stage for the remaining, highest-risk portion of the descent using that
    corrected target -- catching drift/disturbance that accumulated during
    transit/approach/the align stage itself BEFORE committing to the risky
    final portion, instead of only correcting once at the very end (the
    existing "pre-close refresh" logic a few lines below in
    run_pick_and_place, which by then has already let the full descend
    range's contact risk play out uncorrected).

    Returns (final_qpos, corrected_obj_pos, corrected_grasp_mat, converged,
    err_cm, seed_source) -- the corrected pose is returned so the caller can
    carry it forward instead of recomputing against a now-stale pre-descend
    read."""
    top_offset = OBJECT_TOP_OFFSET.get(obj_name, APPROACH_HEIGHT)
    align_height = top_offset + align_clearance

    body_origin_pos = env.get_object_positions()[obj_name].copy()
    quat = env.sim.data.xquat[env.object_body_ids[obj_name]].copy()
    obj_pos = true_centroid_xy(body_origin_pos, quat, obj_name)
    grasp_mat = grasp_orientation_from_quat(quat, obj_name) if use_oriented_grasp else DOWN_ORIENTATION

    align_target = obj_pos + np.array([0.0, 0.0, align_height])
    align_qpos, _, _, _ = ik.solve_multi_seed(align_target, primary_seed=current_qpos, target_mat=grasp_mat)
    move_to_interpolated(env, ik, align_qpos, GRIPPER_OPEN,
                          n_waypoints=n_waypoints, steps_per_waypoint=steps_per_waypoint, step_hook=step_hook)

    # ALIGN checkpoint: re-read the object's true position/orientation NOW,
    # before committing to the highest-risk final portion of descend --
    # this is the "nudge/re-center before committing" step, done at a
    # low-contact-risk height instead of after the risky pass has already
    # happened.
    body_origin_pos2 = env.get_object_positions()[obj_name].copy()
    quat2 = env.sim.data.xquat[env.object_body_ids[obj_name]].copy()
    corrected_obj_pos = true_centroid_xy(body_origin_pos2, quat2, obj_name)
    corrected_grasp_mat = grasp_orientation_from_quat(quat2, obj_name) if use_oriented_grasp else DOWN_ORIENTATION

    commit_target = corrected_obj_pos + np.array([0.0, 0.0, grasp_height_offset])
    commit_qpos, converged, err, source = ik.solve_multi_seed(
        commit_target, primary_seed=align_qpos, target_mat=corrected_grasp_mat)
    move_to_interpolated(env, ik, commit_qpos, GRIPPER_OPEN,
                          n_waypoints=n_waypoints, steps_per_waypoint=steps_per_waypoint, step_hook=step_hook)

    return commit_qpos, corrected_obj_pos, corrected_grasp_mat, converged, float(err * 100), source


def move_to_cr_cfm_descend(env, ik, target_qpos, gripper_action, current_qpos,
                            cr_cfm_model, cr_cfm_mean_start, cr_cfm_template, horizon=16, num_steps=6,
                            steps_per_waypoint=25, device="cpu", step_hook=None,
                            execute_steps=None, max_iterations=12, converge_tol=0.01,
                            adaptive_execution=False, pace_min_steps=1, pace_max_steps=8,
                            clamp_waypoints_to_limits=False, max_step_per_waypoint=None,
                            adaptive_subdivide=False, velocity_norm_threshold=0.5, max_subdivide=8,
                            difficulty_aware=False, difficulty_extended_max_iterations=24,
                            stall_aware=False, stall_check_iter=8, stall_min_improvement_frac=0.3,
                            stall_extended_max_iterations=20, stall_window=4):
    """CR-CFM descend (2026-07-18, v8 -- receding horizon control): generate
    a corrected descend trajectory via a trained temporal-consistency-
    regularized flow-matching model, instead of blind linear interpolation
    (move_to_interpolated) or any of the six hand-designed rules tried and
    found null/harmful in RULED_OUT_METHODS.md rows 7-13.

    execute_steps controls open-loop (v7) vs. receding-horizon (v8)
    behavior: execute_steps=None (default) executes the FULL generated
    chunk in one shot, matching v7's original behavior, kept as the
    explicit comparison baseline. execute_steps=k<horizon instead executes
    only the first k waypoints, then RE-READS the arm's actual current
    joint_pos and RE-GENERATES a fresh chunk against the same fixed
    target -- standard MPC receding-horizon control. Found necessary
    2026-07-18: v7's open-loop full-chunk execution committed to 16
    waypoints of motion before ever observing real contact dynamics, which
    is structurally incapable of reactive deceleration no matter how the
    loss is weighted (confirmed: zeroing TCR's penalty entirely in the
    final 30% of the horizon did NOT reduce terminal velocity -- it got
    WORSE, ruling out loss-weighting as the mechanism). Under RHC, as
    current->target displacement shrinks near contact, each freshly
    generated chunk represents proportionally smaller motion automatically,
    so executing only its first few steps should produce natural
    deceleration without any explicit contact sensing.

    x0 each iteration is built via build_template_x0 -- the dataset's own
    mean trajectory SHAPE (real PD-convergence pacing), affinely
    re-targeted onto the arm's ACTUAL CURRENT (re-read every iteration
    under RHC) ->target displacement -- NOT a straight-line interpolation
    (build_naive_x0, deprecated 2026-07-18, see that function's docstring).

    Terminates when within converge_tol (radians, summed joint-space norm)
    of target, or after max_iterations RHC steps (safety cap against
    non-convergent oscillation, matching retry_on_drift's max_retries
    precedent) -- whichever comes first."""
    import torch
    from tango_robot.piper_robosuite.cr_cfm.inference import (
        build_template_x0, pace_execute_length, sample_corrected_trajectory,
    )

    if execute_steps is None and not adaptive_execution:
        execute_steps = horizon  # v7 behavior: single shot, no re-planning

    # clamp_waypoints_to_limits (2026-07-20, Stage 8, opt-in): Stage 7's
    # substep-level divergence localization led to inspecting the model's
    # RAW Euler-integrated output directly (not just the post-clip physical
    # outcome every prior diagnostic looked at) -- found that for at least
    # one known-unstable trial (1007), sample_corrected_trajectory's output
    # diverges numerically to values up to ~370 (radians), wildly outside
    # any joint's valid range (checked: all six joints lie within
    # [-3.14, 3.14], most much tighter), while a known-stable trial (1001)
    # produces sane output (inter-waypoint deltas 0.001-0.15 rad) for the
    # same model/checkpoint. robosuite's env.step() silently clips actions
    # to the action_spec bounds before applying them -- so this was never
    # visible in any task-space (Z-trace/terminal-velocity) diagnostic used
    # all session, all of which only see the POST-clip physical outcome.
    # The likely mechanism: trial 1007's (x0, cond) lands in a region far
    # from training data density, where the learned velocity field is
    # poorly conditioned, and repeated Euler steps compound the error into
    # numerical divergence -- the clipped result is then a "slam toward
    # the joint limit" motion, which is exactly the fast/large Z-drop
    # Stage 7 found immediately preceding the measured contact-solver
    # divergence. Clamping the raw waypoints to each joint's real range
    # (not the coarser +-3.14 action_spec) before ever building `chunk`
    # prevents this divergent-but-silently-clipped commanding from
    # happening at all, replacing an implicit, joint-agnostic action clip
    # with an explicit, per-joint-correct one applied to the model's
    # output directly. Off by default so this remains an A/B-testable
    # opt-in exactly like every other Stage in this plan, not a silent
    # behavior change to `v6_narrowed`'s established, already-reported
    # numbers.
    joint_lo = joint_hi = None
    if clamp_waypoints_to_limits:
        joint_ranges = np.array(ik.model.jnt_range[ik.jnt_ids])
        joint_lo, joint_hi = joint_ranges[:, 0], joint_ranges[:, 1]

    qpos = current_qpos.copy()
    all_waypoints = []
    xyz_trace = [ik.data.site_xpos[ik.eef_site_id].copy().tolist()]  # 2026-07-18/19 diagnostic:
    # per-RHC-iteration real eef XYZ. Z alone (via z_trace, now folded into
    # this) distinguishes "just needs more iterations" from "model output
    # collapse" (see README's "10cm Z shortfall" finding). XY is used
    # (2026-07-19) to compute this trial's own approach angle -- the
    # start->end XY displacement direction of the descend phase, using the
    # SAME definition as the training-set approach-angle audit -- so probe
    # failures can be checked against the training data's sparse-angle
    # region instead of assumed to correlate with it.
    n_iter = 0
    effective_max_iterations = max_iterations
    stall_triggered = False  # 2026-07-21, Stage 15 (stall_aware, opt-in): a DIFFERENT
    # signal from Stage 11's difficulty_aware -- that one keys off the model's raw
    # velocity output spiking (numerical divergence), which Stage 12's wrist-fix
    # already resolves as a side effect, so it never fires anymore. Trial-1007-class
    # "compound failure" cases (Stage 13) are now fully STABLE/deterministic but
    # still fail because they converge too slowly to finish within max_iterations --
    # a global budget increase helps such a trial but was shown (Stage 13, full
    # 4-range test) to be net-neutral, breaking a DIFFERENT trial elsewhere that
    # was already converging fine. This tracks per-iteration residual-to-target and
    # only extends the budget for trials that are ACTUALLY stalling (checked at
    # `stall_check_iter`, well before the default budget runs out), leaving
    # already-fast-converging trials at the cheap default untouched.
    residual_norms = []
    difficulty_triggered = False  # 2026-07-20, Stage 11 (difficulty_aware, opt-in): whether
    # ANY RHC iteration so far has seen the model's own velocity output spike
    # past velocity_norm_threshold -- the same free-byproduct signal Stage 10
    # calibrated (known-easy trial max ~0.13, known-hard trial 6.04 at the
    # very first substep). Combines Stage 10's confirmed stabilizer
    # (adaptive_subdivide, which alone removes divergence but can exhaust
    # max_iterations before converging -- Stage 9 found the same budget
    # shortfall with a different stabilizer) with a per-trial EXTENDED
    # budget, so easy trials are untouched (stay at the cheap default
    # max_iterations) and only detected-hard trials pay for -- and receive
    # -- the extra iterations they were shown to need.
    while n_iter < effective_max_iterations:
        n_iter += 1
        x0_np = build_template_x0(qpos, target_qpos, cr_cfm_template)  # (horizon, 6)
        # cond = remaining-distance-to-target ONLY (2026-07-18, v3):
        # concatenating a drift-from-dataset-mean term (v2) was ablated
        # head-to-head against remaining-only and LOST (4/8 vs 5/8 on the
        # same probe set), and remaining-only also tamed trial 1007's
        # Z-residual from a 57cm outlier to a normal ~8.6cm -- absolute-
        # position drift was noise for this 49K model, not useful signal.
        # cr_cfm_mean_start is no longer used here; kept as a parameter for
        # call-site compatibility, not read.
        cond_np = (target_qpos - x0_np[0])[None, :].astype(np.float32)  # (1, 6)
        x0 = torch.from_numpy(x0_np[None, :, :].astype(np.float32))
        cond = torch.from_numpy(cond_np)
        use_subdivide = adaptive_subdivide or difficulty_aware  # difficulty_aware always arms
        # the detector (cheap -- it's the same forward pass, only does extra
        # work when the signal actually fires), so detection and
        # stabilization are literally the same computation, not two passes.
        x1_pred, sample_diag = sample_corrected_trajectory(
            cr_cfm_model, x0, cond, num_steps=num_steps, device=device,
            adaptive_subdivide=use_subdivide, velocity_norm_threshold=velocity_norm_threshold,
            max_subdivide=max_subdivide, return_diagnostics=True)
        if difficulty_aware and sample_diag["triggered_subdivision"] and not difficulty_triggered:
            difficulty_triggered = True
            effective_max_iterations = difficulty_extended_max_iterations
        waypoints = x1_pred[0].detach().cpu().numpy()  # (horizon, 6)
        if clamp_waypoints_to_limits:
            waypoints = np.clip(waypoints, joint_lo[None, :], joint_hi[None, :])

        # adaptive_execution (2026-07-20, PACE-style, arXiv:2606.00537):
        # commit only up to this CHUNK's own first low-speed valley
        # (a natural replanning boundary -- phase transitions/contact
        # preparation) instead of a fixed execute_steps every iteration.
        # Training-free: analyzes the already-generated waypoints only, no
        # model or checkpoint change. Recomputed fresh each RHC iteration
        # since the chunk itself is fresh each iteration.
        this_iter_steps = (pace_execute_length(waypoints, pace_min_steps, pace_max_steps)
                            if adaptive_execution else execute_steps)
        chunk = waypoints[:this_iter_steps].copy()

        # max_step_per_waypoint (2026-07-20, Stage 9, opt-in): Stage 8 clamped
        # the model's raw output VALUE to each joint's range and found it
        # redundant -- MuJoCo's own jnt_range constraint already stops qpos
        # from exceeding physical limits regardless of the commanded target,
        # so capping the value changed nothing about the arm's actual motion.
        # This is a mechanistically different lever: cap how FAR qpos may
        # move PER WAYPOINT relative to the arm's ACTUAL CURRENT position
        # (chain-clamped: waypoint k+1 is capped relative to waypoint k's
        # OWN clamped result, not the raw model target), preserving
        # direction but forcing a gradual, multi-iteration approach toward
        # wherever the model wants to go instead of one maximally fast
        # attempt within a single 25-substep waypoint window -- this is not
        # redundant with any physics-engine constraint, since MuJoCo limits
        # only the final qpos value, never the RATE at which the controller
        # is allowed to drive toward it.
        if max_step_per_waypoint is not None:
            prev = qpos.copy()
            for k in range(len(chunk)):
                delta = chunk[k] - prev
                dist = np.linalg.norm(delta)
                if dist > max_step_per_waypoint:
                    chunk[k] = prev + delta * (max_step_per_waypoint / dist)
                prev = chunk[k]

        for wp in chunk:
            action = np.concatenate([wp, [gripper_action]])
            for _ in range(steps_per_waypoint):
                env.step(action)
                if step_hook is not None:
                    step_hook(env)
        all_waypoints.append(waypoints)
        qpos = ik._get_qpos()  # re-read the ARM's real state (not the plan) for next iteration's x0
        xyz_trace.append(ik.data.site_xpos[ik.eef_site_id].copy().tolist())

        residual_norm = float(np.linalg.norm(qpos - target_qpos))
        residual_norms.append(residual_norm)
        if stall_aware and n_iter == stall_check_iter and not stall_triggered:
            # RECENT-window improvement (iteration stall_check_iter vs. iteration
            # stall_check_iter - stall_window), NOT improvement since iteration 1.
            # A trajectory that drops fast in the first few iterations and then
            # PLATEAUS shows large overall improvement since iteration 1 even
            # though it is actively stuck by the checkpoint -- comparing against
            # a recent window instead of the start catches exactly that pattern
            # (confirmed necessary: an iteration-1-relative version of this check
            # never fired on trial 1007, which Stage 13 already showed plateaus
            # early and then sits flat through iteration 30).
            window_idx = max(0, len(residual_norms) - 1 - stall_window)
            window_residual = residual_norms[window_idx]
            if window_residual > 1e-9:
                improvement_frac = 1.0 - (residual_norm / window_residual)
                if residual_norm >= converge_tol and improvement_frac < stall_min_improvement_frac:
                    stall_triggered = True
                    effective_max_iterations = stall_extended_max_iterations
        if residual_norm < converge_tol:
            break
        if not adaptive_execution and execute_steps >= horizon:
            break  # v7 (open-loop) mode: exactly one iteration by construction

    # final_eef_residual (2026-07-18): task-space (world XYZ) delta between
    # where RHC actually converged and the true IK target -- the direct
    # test for a spatial bias hypothesis (a consistent offset direction
    # across failed trials) vs. no systematic bias. Computed via a safe
    # save/restore FK read (matching ArmIK.solve's own pattern) since
    # target_qpos is hypothetical, not the live sim state.
    saved_qpos = ik._get_qpos()
    actual_eef = ik.data.site_xpos[ik.eef_site_id].copy()
    ik._set_qpos(target_qpos)
    target_eef = ik.data.site_xpos[ik.eef_site_id].copy()
    ik._set_qpos(saved_qpos)
    final_eef_residual = (actual_eef - target_eef).astype(np.float64)

    final_chunk = all_waypoints[-1]
    # terminal_velocity (2026-07-18): mean joint-space speed over the LAST
    # 3 waypoint-to-waypoint steps of the FINAL generated chunk -- the
    # direct diagnostic for "did the model decelerate near contact, or
    # cruise through at constant speed" (see README's TCR-overreach
    # finding and its follow-up RHC fix for trial 1007's table-launch
    # failures).
    terminal_velocity = float(np.linalg.norm(np.diff(final_chunk[-4:], axis=0), axis=1).mean())

    # approach_angle_deg (2026-07-19): start->end XY displacement angle of
    # this trial's own descend phase, SAME definition as the training-set
    # audit (README's "long-tail data sparsity" finding) -- lets probe
    # failures be checked directly against the training data's sparse
    # 60-180/-165deg region instead of assumed to correlate with it.
    xy_disp = np.array(xyz_trace[-1][:2]) - np.array(xyz_trace[0][:2])
    approach_angle_deg = (float(np.degrees(np.arctan2(xy_disp[1], xy_disp[0])))
                           if np.linalg.norm(xy_disp) > 1e-4 else None)
    z_trace = [p[2] for p in xyz_trace]
    return (qpos.astype(np.float64), terminal_velocity, final_eef_residual, z_trace, approach_angle_deg,
            difficulty_triggered, stall_triggered)


def run_pick_and_place(env, obj_name, use_oriented_grasp=True, verbose=False,
                        candidate_selection=None, noise_model="kinematic", n_candidates=10,
                        pos_jitter=0.005, yaw_jitter_rad=np.radians(5.0),
                        angle_jitter_rad=np.radians(10.0), rng=None,
                        compliant_descend=False, force_compliant_descend=False,
                        pre_narrow_descend=False, two_stage_descend=False,
                        compliant_sustain_steps=5, align_clearance=0.03,
                        retry_on_drift=False, drift_threshold_cm=1.0, max_retries=2,
                        cr_cfm_descend=False, cr_cfm_model=None, cr_cfm_mean_start=None, cr_cfm_template=None,
                        cr_cfm_horizon=16, cr_cfm_num_steps=6, cr_cfm_device="cpu",
                        cr_cfm_execute_steps=None, cr_cfm_max_iterations=12,
                        cr_cfm_adaptive_execution=False, cr_cfm_pace_min_steps=1, cr_cfm_pace_max_steps=8,
                        cr_cfm_clamp_waypoints_to_limits=False, cr_cfm_max_step_per_waypoint=None,
                        cr_cfm_adaptive_subdivide=False, cr_cfm_velocity_norm_threshold=0.5,
                        cr_cfm_max_subdivide=8,
                        cr_cfm_difficulty_aware=False, cr_cfm_difficulty_extended_max_iterations=24,
                        cr_cfm_stall_aware=False, cr_cfm_stall_check_iter=8,
                        cr_cfm_stall_min_improvement_frac=0.3, cr_cfm_stall_extended_max_iterations=20,
                        cr_cfm_stall_window=4,
                        wrist_friendly_orientation=False, wrist_friendly_angle_tolerance_deg=0.0,
                        tray_orientation_search=False, tray_orientation_angle_tolerance_deg=30.0,
                        step_hook=None):
    """Run one full pick-and-place trial on an already-reset env. Returns a
    result dict (success flag + per-phase IK diagnostics) instead of
    printing/rendering, so this can be called in a loop by an experiment
    runner. `use_oriented_grasp=False` reproduces the earlier fixed-world-X
    approach (the ablation condition).

    candidate_selection: None (default, exact deterministic nominal target,
    unchanged prior behaviour) | "best" (ikmargin-style: lowest-IK-error
    candidate from a sampled pool) | "consensus" (median-of-pool candidate)
    -- see piper_candidate_selection.py for why this pool exists and how
    the two strategies differ.

    noise_model: "kinematic" (default -- direct Gaussian jitter of the
    already-computed grasp target's position/yaw, sample_candidate_pool) |
    "perception" (perturb the TRUE object pose's position AND full 3D
    orientation, then recompute the grasp target each noisy pose implies,
    sample_perception_noisy_candidates -- a more faithful reproduction of
    where SO-ARM101's candidate diversity actually came from; see that
    function's docstring). Only meaningful when candidate_selection is set.

    rng lets a caller pass a seeded generator so the SAME pool is drawn for
    "best" and "consensus" at a given trial_id (paired comparison, matching
    this project's established statistical convention).

    tray_orientation_search (2026-08-03, opt-in, default False): the
    transit-to-tray phases (transit_above_tray/lower_into_tray/retract)
    normally hold the exact grasp_mat throughout -- see the comment at the
    "close gripper" step for why (switching orientation mid-transit risks
    twisting the object out of the grip). This flag instead searches the
    SAME safe family of orientations pick_wrist_friendly_orientation already
    uses (rotations about the grasp's own approach axis via
    _rotate_grasp_about_approach, which preserve the actual gripping
    geometry -- not an arbitrary reorientation like DOWN_ORIENTATION) but
    evaluated for IK convergence at the TRAY target instead of the grasp
    target, since those are different points in the workspace and a grasp
    orientation chosen for reachability at one is not guaranteed reachable
    at the other. Falls back to grasp_mat if no candidate converges better
    (same safe-fallback contract as pick_wrist_friendly_orientation)."""
    ik = ArmIK(env)
    # Settle before reading the object's pose -- the placement sampler
    # deliberately spawns objects ~3cm above the table (z_offset=0.03, to
    # avoid initial mesh embedding) and reset() alone does not step physics
    # forward, so a pose read immediately after reset() is still ~3cm high
    # in Z (confirmed empirically: XY/orientation are already stable at
    # spawn, only Z drops during the first ~10-20 steps). Every grasp height
    # this session was computed from that stale, pre-settle Z until this
    # fix (2026-07-14) -- found while investigating why generalization
    # across objects was so poor.
    # NOTE: action[:6]=0 would NOT "hold still" here -- the JOINT_POSITION
    # controller is absolute, so a zero action commands qpos=0 for every
    # joint (a very different pose from READY_QPOS), which would yank the
    # arm out of position during this settle period. Hold at READY_QPOS
    # explicitly instead.
    hold_action = np.concatenate([READY_QPOS, [GRIPPER_OPEN]])
    for _ in range(30):
        env.step(hold_action)

    body_origin_pos = env.get_object_positions()[obj_name].copy()
    obj_quat = env.sim.data.xquat[env.object_body_ids[obj_name]].copy()
    obj_pos = true_centroid_xy(body_origin_pos, obj_quat, obj_name)  # NOT the raw body origin --
                                    # see OBJECT_CENTROID_OFFSET_LOCAL's comment for why that matters
    true_obj_pos = obj_pos.copy()  # kept for the result dict -- obj_pos itself gets reassigned to
                                    # the chosen candidate's (jittered) pose below when selection is active
    grasp_mat = compute_grasp_orientation(env, obj_name) if use_oriented_grasp else DOWN_ORIENTATION
    if use_oriented_grasp and wrist_friendly_orientation:
        grasp_mat = pick_wrist_friendly_orientation(
            ik, obj_pos + [0, 0, GRASP_HEIGHT_OFFSET], grasp_mat, READY_QPOS,
            angle_tolerance_deg=wrist_friendly_angle_tolerance_deg)

    candidate_log = None
    candidate_grasp_yaw = None
    if candidate_selection is not None:
        if noise_model == "kinematic":
            candidates = sample_candidate_pool(
                obj_pos, grasp_mat, n=n_candidates,
                pos_jitter=pos_jitter, yaw_jitter_rad=yaw_jitter_rad, rng=rng,
            )
        elif noise_model == "perception":
            true_quat = env.sim.data.xquat[env.object_body_ids[obj_name]].copy()
            candidates = sample_perception_noisy_candidates(
                true_obj_pos, true_quat, obj_name, n=n_candidates,
                pos_jitter=pos_jitter, angle_jitter_rad=angle_jitter_rad, rng=rng,
            )
        else:
            raise ValueError(f"noise_model must be 'kinematic' or 'perception', got {noise_model!r}")

        if candidate_selection == "best":
            scores = [score_candidate_ik(ik, pos + [0, 0, GRASP_HEIGHT_OFFSET], mat, READY_QPOS)
                      for pos, mat in candidates]
            chosen_idx = select_best(candidates, scores)
        elif candidate_selection == "consensus":
            chosen_idx = select_consensus(candidates, pos_jitter=pos_jitter, yaw_jitter_rad=yaw_jitter_rad)
        elif isinstance(candidate_selection, int):
            # Force execution of one specific pool index (2026-07-15) --
            # for collecting genuinely pairwise-labeled data (every
            # candidate in a pool individually executed and labeled, not
            # just whichever one "best"/"consensus" would have picked),
            # needed to train LGGSN's real BPR pairwise objective on Piper
            # data instead of a pointwise proxy. Caller loops this over
            # 0..n_candidates-1 with the SAME `rng` state per trial_id so
            # every candidate in the pool gets executed against the exact
            # same sampled pool/object pose (a fair within-scene comparison).
            chosen_idx = candidate_selection
        else:
            raise ValueError(f"candidate_selection must be None/'best'/'consensus'/int, got {candidate_selection!r}")
        obj_pos, grasp_mat = candidates[chosen_idx]
        # candidate_grasp_yaw (2026-07-16): the TRUE pre-commit candidate
        # orientation, captured here before grasp_mat gets reassigned later
        # at the post-descend "pre-close refresh" step -- unlike the
        # existing `grasp_yaw` field (computed from whatever grasp_mat
        # holds at the end of the function, confirmed execution-derived by
        # auto_tagger.py), this is genuinely pre-execution: derived only
        # from the candidate pool sampled above, before any physical
        # movement toward this candidate has occurred. This is the one
        # per-candidate-varying, causally-valid feature the corrected
        # [z, H] Stage 2 feature set was missing (both z and H are
        # constant/near-constant within a scene -- see
        # AUTO_TAGGER_ALGORITHM.md's diagnostic finding).
        candidate_grasp_yaw = float(np.arctan2(grasp_mat[1, 0], grasp_mat[0, 0]))
        candidate_log = {
            "chosen_idx": chosen_idx,
            "noise_model": noise_model,
            "pool_xy": [c[0][:2].tolist() for c in candidates],
        }
        if verbose:
            print(f"[candidates] strategy={candidate_selection} noise_model={noise_model} "
                  f"chosen_idx={chosen_idx} chosen_xy={obj_pos[:2].round(4)}")

    if verbose:
        print(f"target object '{obj_name}' at {obj_pos.round(3)}")
        print(f"grasp orientation X-axis (world): {grasp_mat[:, 0].round(3)}")

    # CAUSAL VALIDITY COMMIT POINT: the chosen candidate (obj_pos, grasp_mat)
    # is final as of here -- nothing above this line has moved the arm.
    # Every assignment from this point on that reads live env/sim state, or
    # that depends on such a read, reflects information only available AFTER
    # this specific candidate has begun physically executing, and is
    # therefore EXECUTION_DERIVED per CAUSAL_VALIDITY_METHOD.md's Definition
    # 3 -- see causal_validity_audit/auto_tagger.py, which uses this single
    # marker to infer provenance for every field in this function's return
    # dict automatically, instead of requiring per-field manual annotation.
    CAUSAL_VALIDITY_COMMIT_POINT()

    tray_xy = PLACEMENT_TRAY_CENTER
    tray_drop_pos = np.array([tray_xy[0], tray_xy[1], env.table_offset[2] + TRAY_DROP_HEIGHT])
    phase_log = {}

    def solve_and_move(name, target, grip, seed_qpos, interpolated=False, compliant=False, force_compliant=False,
                        target_mat=None):
        _set_phase(step_hook, name)
        qpos, converged, err, source = ik.solve_multi_seed(
            target, primary_seed=seed_qpos, target_mat=grasp_mat if target_mat is None else target_mat)
        phase_log[name] = {"converged": bool(converged), "err_cm": float(err * 100), "seed_source": source}
        if verbose:
            print(f"[{name}] target={target.round(3)} converged={converged} err_cm={err*100:.2f} seed={source}")
        if force_compliant:
            final_progress, mean_force = move_to_force_compliant_descend(
                env, ik, qpos, grip, obj_name, step_hook=step_hook)
            phase_log[name]["force_compliant_final_progress"] = final_progress
            phase_log[name]["force_compliant_mean_force"] = mean_force
            if verbose:
                print(f"[{name}] force_compliant: final_progress={final_progress:.3f} mean_force={mean_force:.3f}")
        elif compliant:
            stopped_early, contact_step = move_to_compliant_descend(
                env, ik, qpos, grip, obj_name, sustain_steps=compliant_sustain_steps, step_hook=step_hook)
            phase_log[name]["compliant_stopped_early"] = stopped_early
            phase_log[name]["compliant_contact_step"] = contact_step
            if verbose:
                print(f"[{name}] compliant: stopped_early={stopped_early} at step={contact_step}")
        elif interpolated:
            move_to_interpolated(env, ik, qpos, grip, step_hook=step_hook)
        else:
            move_to(env, qpos, grip, step_hook=step_hook)
        return qpos

    approach_height = approach_height_for(obj_name)
    transit_high = np.array([obj_pos[0], obj_pos[1], SAFE_TRANSIT_Z])
    qpos_seed = solve_and_move("transit_high", transit_high, GRIPPER_OPEN, READY_QPOS)
    # TRIED AND REVERTED (2026-07-15): a closed-loop move_to_precise() that
    # re-solved/re-moved "approach" up to 3 extra times whenever the REAL
    # (not just IK-reported) eef error exceeded 2mm. Tested in two variants
    # (abrupt re-moves, then interpolated re-moves) -- both plateaued at
    # ~0.4-0.7cm real error after all 4 attempts (never once hit
    # tight_converged=True across two n=5 batches), confirming this is a
    # systematic PD steady-state tracking limit, not solver noise that
    # retrying could fix. Worse, both variants made downstream pre-close
    # drift measurably WORSE than a single plain approach (mean 2.94cm ->
    # 5.81cm abrupt / 6.86cm interpolated, one trial hitting 15.46cm, close
    # to a repeat of the table-launch bug) -- repeatedly re-approaching
    # near the object, even smoothly, cost more than the tighter aim ever
    # gained. See the README's finger-clearance entries for the full trace.
    retry_count = 0
    while True:
        suffix = f"_retry{retry_count}" if retry_count else ""
        qpos_seed = solve_and_move("approach" + suffix, obj_pos + [0, 0, approach_height], GRIPPER_OPEN, qpos_seed)
    # interpolated=True (2026-07-15): a step-by-step trace of "descend" as a
    # single abrupt move_to found it literally launching Cracker off the
    # TABLE onto the floor (obj z 0.908 -> 0.068 within 1.5s) -- the open
    # gripper travels ~13cm straight down starting just above the object's
    # own top surface, so for tall objects the descent's last ~10cm runs
    # alongside the object's own height range; any residual XY error left
    # over from "approach" (typically ~0.5cm, this solver rarely converges
    # to exact 0) is then enough for a fast single-shot PD move to snag a
    # corner/edge with real momentum. Switching to the same waypoint
    # interpolation already used for the post-grasp phases turned that
    # catastrophic launch into a much gentler few-cm nudge that stays on
    # the table (verified via the same trace) -- same fix, applied one
    # phase earlier than it previously was.
    # compliant_descend=True (2026-07-16, opt-in): CoorGrasp-inspired
    # mechanism (see README's literature-check entry) -- stop advancing
    # through the descend waypoints the moment ANY contact with the object
    # is detected, instead of blindly continuing to the precomputed
    # target. Off by default (compliant=False) so existing callers/
    # experiment scripts are unaffected; opt in via run_pick_and_place's
    # compliant_descend=True to A/B against the current interpolated
    # baseline.
    # pre_narrow_descend=True (2026-07-16, opt-in, v4 -- structurally
    # different from v1-v3's "manage contact once it happens" attempts):
    # instead of descending fully open (12cm span) past the object's own
    # height range, narrow the gripper by ~2cm BEFORE descending, reducing
    # the collision cross-section during the risky vertical pass in the
    # first place, rather than reacting to contact after it occurs. One
    # GRIPPER_CLOSE step from full open narrows the span from ~12.0cm to
    # ~10.1cm (calibrated empirically, scratchpad/calibrate_gripper_width*.py
    # -- the action interface only supports coarse per-step increments,
    # finer intermediate widths would need bypassing the controller's
    # normal action path, not done here to stay within the existing,
    # already-validated action interface). gripper_action=0.0 during
    # descend itself HOLDS this narrowed width (format_action's
    # np.sign(0)=0 means no further increment) instead of GRIPPER_OPEN's
    # continued full-open command.
        if pre_narrow_descend:
            move_to(env, qpos_seed, GRIPPER_CLOSE, steps=1, step_hook=step_hook)
            descend_gripper_action = 0.0
        else:
            descend_gripper_action = GRIPPER_OPEN
        if two_stage_descend:
            (qpos_seed, corrected_obj_pos, corrected_grasp_mat,
             converged, err_cm, source) = move_to_two_stage_align_descend(
                env, ik, obj_name, GRASP_HEIGHT_OFFSET, use_oriented_grasp,
                qpos_seed, align_clearance=align_clearance, step_hook=step_hook)
            phase_log["descend" + suffix] = {"converged": bool(converged), "err_cm": err_cm, "seed_source": source,
                                              "two_stage": True}
            # Carry the align-checkpoint-corrected pose forward -- the
            # subsequent pre-close refresh below still re-reads once more right
            # before closing, but starting from an already-corrected target
            # instead of the original pre-descend one.
            obj_pos = corrected_obj_pos
            grasp_mat = corrected_grasp_mat
            if verbose:
                print(f"[descend] two_stage: converged={converged} err_cm={err_cm:.2f} "
                      f"corrected_pos={corrected_obj_pos.round(4)}")
        elif cr_cfm_descend:
            _set_phase(step_hook, "descend" + suffix)
            target_qpos, converged, err, source = ik.solve_multi_seed(
                obj_pos + [0, 0, GRASP_HEIGHT_OFFSET], primary_seed=qpos_seed, target_mat=grasp_mat)
            phase_log["descend" + suffix] = {"converged": bool(converged), "err_cm": float(err * 100),
                                              "seed_source": source, "cr_cfm": True}
            if verbose:
                print(f"[descend] cr_cfm: ik_converged={converged} ik_err_cm={err*100:.2f}")
            (final_wp, terminal_velocity, final_eef_residual, z_trace, approach_angle_deg,
             difficulty_triggered, stall_triggered) = move_to_cr_cfm_descend(
                env, ik, target_qpos, descend_gripper_action, qpos_seed,
                cr_cfm_model, cr_cfm_mean_start, cr_cfm_template, horizon=cr_cfm_horizon,
                num_steps=cr_cfm_num_steps, device=cr_cfm_device, step_hook=step_hook,
                execute_steps=cr_cfm_execute_steps, max_iterations=cr_cfm_max_iterations,
                adaptive_execution=cr_cfm_adaptive_execution, pace_min_steps=cr_cfm_pace_min_steps,
                pace_max_steps=cr_cfm_pace_max_steps,
                clamp_waypoints_to_limits=cr_cfm_clamp_waypoints_to_limits,
                max_step_per_waypoint=cr_cfm_max_step_per_waypoint,
                adaptive_subdivide=cr_cfm_adaptive_subdivide,
                velocity_norm_threshold=cr_cfm_velocity_norm_threshold,
                max_subdivide=cr_cfm_max_subdivide,
                difficulty_aware=cr_cfm_difficulty_aware,
                difficulty_extended_max_iterations=cr_cfm_difficulty_extended_max_iterations,
                stall_aware=cr_cfm_stall_aware, stall_check_iter=cr_cfm_stall_check_iter,
                stall_min_improvement_frac=cr_cfm_stall_min_improvement_frac,
                stall_extended_max_iterations=cr_cfm_stall_extended_max_iterations,
                stall_window=cr_cfm_stall_window)
            phase_log["descend" + suffix]["terminal_velocity"] = terminal_velocity
            phase_log["descend" + suffix]["final_eef_residual"] = final_eef_residual.tolist()
            phase_log["descend" + suffix]["difficulty_triggered"] = difficulty_triggered
            phase_log["descend" + suffix]["stall_triggered"] = stall_triggered
            phase_log["descend" + suffix]["approach_angle_deg"] = approach_angle_deg
            phase_log["descend" + suffix]["z_trace"] = z_trace
            qpos_seed = final_wp
        else:
            qpos_seed = solve_and_move("descend" + suffix, obj_pos + [0, 0, GRASP_HEIGHT_OFFSET], descend_gripper_action,
                                        qpos_seed, interpolated=True, compliant=compliant_descend,
                                        force_compliant=force_compliant_descend)

        # Re-read the object's true pose right before closing (2026-07-15):
        # found via a step-by-step contact trace that one finger was ALREADY
        # touching the object BEFORE the close command was even issued -- the
        # descend target computed at the top of this function (several phases
        # and ~1000+ physics steps earlier) can go stale if the object shifted
        # even a few mm during approach/descend (finger/air disturbance, or
        # accumulated PD-tracking drift over the whole multi-phase trajectory).
        # Re-solving one more short IK hop to the LATEST true position/
        # orientation right here catches that drift instead of closing on a
        # target that may no longer be centered.
        refreshed_body_origin = env.get_object_positions()[obj_name].copy()
        refreshed_quat = env.sim.data.xquat[env.object_body_ids[obj_name]].copy()
        refreshed_obj_pos = true_centroid_xy(refreshed_body_origin, refreshed_quat, obj_name)
        refreshed_grasp_mat = (
            grasp_orientation_from_quat(refreshed_quat, obj_name) if use_oriented_grasp else DOWN_ORIENTATION
        )
        pre_close_drift_cm = float(np.linalg.norm(refreshed_obj_pos[:2] - obj_pos[:2]) * 100)
        phase_log["pre_close_drift_cm" + suffix] = pre_close_drift_cm
        if verbose:
            print(f"[pre-close refresh]{suffix} drift_cm={pre_close_drift_cm:.2f} "
                  f"old_pos={obj_pos.round(4)} new_pos={refreshed_obj_pos.round(4)}")
        obj_pos = refreshed_obj_pos
        grasp_mat = refreshed_grasp_mat

        # DRIFT-GATED RETRY (2026-07-17, v6): pre_close_drift_cm was found,
        # pooled across 60 already-collected trials from 3 independent
        # batches (baseline + two_stage x2), to near-perfectly separate
        # outcome (0/20 success when >1cm, 35/40 when <=1cm, Fisher's exact
        # p=1.3e-11) -- a near-certain-failure signal available BEFORE the
        # close command commits. This retries (retract to the safe transit
        # height, then a fresh approach+descend) instead of blindly closing
        # on what this checkpoint already flags as doomed. Structurally
        # different from the 2026-07-15 reverted move_to_precise() retry
        # (see comment above "approach", ~30 lines up): that one retried
        # blindly on IK-residual precision while still down near the
        # object, and made drift WORSE (2.94cm -> 5.81-6.86cm); this one
        # only retries when the checkpoint already signals a doomed
        # attempt, and fully retracts to transit height first, rather than
        # repeating a fine re-aim in place.
        if not retry_on_drift or pre_close_drift_cm <= drift_threshold_cm or retry_count >= max_retries:
            break
        retry_count += 1
        qpos_seed = solve_and_move(f"retry_lift{retry_count}", transit_high, GRIPPER_OPEN, qpos_seed, interpolated=True)

    phase_log["retry_count"] = retry_count
    # Normalize: always expose the FINAL attempt's drift under the
    # unsuffixed key, for backward compat with analysis code that reads
    # phases['pre_close_drift_cm'] without knowing about retries.
    phase_log["pre_close_drift_cm"] = pre_close_drift_cm
    # TRIED AND REVERTED (2026-07-15): interpolated=True here, on the
    # hypothesis that this abrupt move was reproducing the same
    # catastrophic-launch mechanism "descend" itself had. Tested on a
    # matched n=20 Cracker batch (trial_id 400-419): success dropped from
    # 45% (this abrupt version) to 30% (interpolated) -- worse, not
    # better. Likely explanation: when the real problem is SUSTAINED
    # grazing contact against a target that's still slightly
    # miscentered (not a single high-velocity impact), a slower
    # interpolated path spends MORE total time in contact, accumulating
    # more net displacement than a fast abrupt move that clips past
    # quickly -- the opposite of what fixed "descend" itself, where the
    # problem really was impact velocity/momentum from a large single
    # jump. Reverted; the remaining high-drift Cracker failures need a
    # different fix (better target centering, not motion pacing).
    qpos_seed = solve_and_move("descend_refresh", obj_pos + [0, 0, GRASP_HEIGHT_OFFSET], descend_gripper_action, qpos_seed)

    if verbose:
        print("[close gripper]")
    move_to(env, qpos_seed, GRIPPER_CLOSE, steps=250, step_hook=step_hook)

    # Keep grasp_mat (not the default DOWN_ORIENTATION) for every remaining
    # waypoint -- switching orientation mid-transit would twist the object
    # out of the grip instead of just carrying it. Interpolated (not
    # move_to's single abrupt jump) for every phase from here on, since the
    # object is now held -- see move_to_interpolated's docstring.
    qpos_seed = solve_and_move("lift", obj_pos + [0, 0, approach_height], GRIPPER_CLOSE, qpos_seed, interpolated=True)
    tray_above = tray_drop_pos + [0, 0, 0.08]
    # tray_orientation_search (opt-in, see docstring): grasp_mat was chosen
    # for reachability at the GRASP target, not the tray target -- these are
    # different points in the workspace. Re-search the same safe family of
    # orientations (rotations about the approach axis, which preserve the
    # actual gripping geometry) for reachability at tray_above specifically.
    # A fresh local variable, NOT reassigning grasp_mat itself, so the
    # reported grasp_yaw (computed from grasp_mat further below) still
    # describes the actual grasp, not this transit-specific adjustment.
    tray_transit_mat = grasp_mat
    if tray_orientation_search:
        tray_transit_mat = pick_wrist_friendly_orientation(
            ik, tray_above, grasp_mat, qpos_seed,
            angle_tolerance_deg=tray_orientation_angle_tolerance_deg)
    qpos_seed = solve_and_move("transit_above_tray", tray_above, GRIPPER_CLOSE, qpos_seed, interpolated=True,
                                target_mat=tray_transit_mat)
    qpos_seed = solve_and_move("lower_into_tray", tray_drop_pos, GRIPPER_CLOSE, qpos_seed, interpolated=True,
                                target_mat=tray_transit_mat)

    if verbose:
        print("[open gripper]")
    move_to(env, qpos_seed, GRIPPER_OPEN, steps=40, step_hook=step_hook)
    qpos, converged, err, source = ik.solve_multi_seed(tray_above, primary_seed=qpos_seed, target_mat=tray_transit_mat)
    phase_log["retract"] = {"converged": bool(converged), "err_cm": float(err * 100), "seed_source": source}
    move_to(env, qpos, GRIPPER_OPEN, step_hook=step_hook)

    final_pos = env.get_object_positions()[obj_name]
    dist_to_tray = float(np.linalg.norm(final_pos[:2] - np.array(tray_xy)))
    from tango_robot.piper_robosuite.piper_multi_object_scene import PLACEMENT_TRAY_HALF_EXTENT
    success = dist_to_tray < PLACEMENT_TRAY_HALF_EXTENT
    if verbose:
        print(f"final '{obj_name}' pos: {final_pos.round(3)}  dist_to_tray_center_xy={dist_to_tray:.3f}m  success={success}")

    # grasp_yaw / object_H (2026-07-15): added for the cross-embodiment
    # feature-parity pilot -- roll/pitch are always the DOWN_ORIENTATION
    # constants (not informative per-trial), so only yaw carries real
    # per-trial variation; object_H is the per-object top-surface offset
    # (OBJECT_TOP_OFFSET), a rough analog of LGGSN's "H" feature.
    grasp_yaw = float(np.arctan2(grasp_mat[1, 0], grasp_mat[0, 0]))
    object_H = float(OBJECT_TOP_OFFSET.get(obj_name, APPROACH_HEIGHT))

    return {
        "object": obj_name,
        "strategy": "oriented" if use_oriented_grasp else "fixed",
        "candidate_selection": candidate_selection,
        "noise_model": noise_model if candidate_selection is not None else None,
        "success": success,
        "dist_to_tray": dist_to_tray,
        "spawn_pos": true_obj_pos.tolist(),
        "final_pos": final_pos.tolist(),
        "phases": phase_log,
        "candidates": candidate_log,
        "grasp_yaw": grasp_yaw,
        "candidate_grasp_yaw": candidate_grasp_yaw,
        "object_H": object_H,
    }


def main():
    obj_name = sys.argv[1] if len(sys.argv) > 1 else "pear"
    env = PiperMultiObjectScene(
        robots="Piper", ycb_objects=["pear", "can", "mustard"],
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
        camera_names="tablecam", control_freq=20,
    )
    env.reset()
    result = run_pick_and_place(env, obj_name, use_oriented_grasp=True, verbose=True)

    import imageio
    img = env.sim.render(camera_name="tablecam", width=800, height=800)[::-1]
    out_path = f"/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/piper_pick_place_{obj_name}.png"
    imageio.imwrite(out_path, img)
    print(f"saved -> {out_path}")
    return result


if __name__ == "__main__":
    main()
