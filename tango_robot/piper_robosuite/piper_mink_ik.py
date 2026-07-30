"""
Stage 2: mink-based collision-aware reach, promoted from the Stage 1
prototype (validated in scratchpad/mink_stage1_prototype.py -- IK converged
to 0.0004m error with zero unwanted arm/mount-object contact, versus the
DLS+manual-waypoint approach that still swept through Cracker).

Scope: replaces ONLY the pre-grasp reach (the leg from wherever the arm
currently is straight to the descend/grasp pose) -- the phase where the
custom DLS `ArmIK` had no collision awareness and could sweep the arm
through tall objects. Post-grasp phases (lift/transit/lower into tray) are
NOT touched here: the object is already held by the gripper at that point,
so a collision-avoidance term between the arm and the held object would
fight the grasp itself rather than help it -- those phases keep using the
existing `ArmIK`/`move_to_interpolated` machinery from piper_pick_and_place.py.
"""
import numpy as np
import mink

JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
EEF_SITE = "robot0_eef_site"

# Every body whose geoms should be treated as "the arm" for collision
# avoidance -- link1-6 AND the stationary mount (fixed_mount0_*). Omitting
# the mount was Stage 1's first failure mode (IK never converged, mount
# still touched the object) -- see mink_stage1_prototype.py's "Errors and
# fixes" history for the debugging trace.
ARM_BODY_NAMES = [
    "robot0_base_link", "robot0_link1", "robot0_link2", "robot0_link3",
    "robot0_link4", "robot0_link5", "robot0_link6",
    "fixed_mount0_base", "fixed_mount0_controller_box",
    "fixed_mount0_pedestal_feet", "fixed_mount0_torso", "fixed_mount0_pedestal",
]


def object_geom_ids(env, obj_name):
    model = env.sim.model._model
    bid = env.object_body_ids[obj_name]
    return [i for i in range(model.ngeom) if model.geom_bodyid[i] == bid]


def mink_reach(env, target_pos, target_mat, target_geom_ids, gripper_action,
                n_steps=80, dt=0.05, avoid_body_names=ARM_BODY_NAMES,
                posture_cost=1e-2, min_dist=0.003, verbose=False):
    """Drive the arm from its CURRENT live qpos straight to (target_pos,
    target_mat), staying `min_dist` clear of `target_geom_ids` throughout.

    min_dist=0.003 (2026-07-15, Stage 2): the Stage 1 value of 0.01 (1cm)
    worked cleanly for the single hand-picked trial that prototype used, but
    a Stage 2 batch run on other Cracker spawn poses found the QP getting
    genuinely STUCK -- velocity collapsing to ~0 with position error still
    ~0.1-0.2m, for many consecutive steps, i.e. `CollisionAvoidanceLimit`'s
    hard 1cm-clearance constraint was infeasible (or a hard local minimum)
    given how close the mount/table/object are to each other in this scene.
    Confirmed via a direct A/B (same trial, same everything else): 1cm
    stayed stuck at plan_err=0.11-0.16m with vel_norm=0 from step ~15
    onward; 2mm on the identical trial converged to plan_err=0.0006m,
    real_err=0.0006m, by step 60. A tighter margin does still leave a real
    (if smaller) safety buffer, and this is a reactive/local method either
    way -- it was never going to have global-path-planning guarantees, just
    a different failure mode (getting stuck) than the un-collision-aware
    DLS solver it replaced (sweeping through the object).

    Re-solves mink's velocity-level IK once per env.step() (dt matched
    to the environment's real control-step duration -- see Stage 1's
    "critical timestep-matching lesson", a 10x dt mismatch there caused
    severe plan/execution divergence even though the isolated planning loop
    converged perfectly).

    Returns (final_qpos, pos_err) -- final_qpos is the 6-vector of ARM joint
    values actually reached (not a solve-only result to hand to something
    else), since the loop already drove the real sim there via env.step().
    """
    model = env.sim.model._model
    data = env.sim.data._data

    configuration = mink.Configuration(model, q=data.qpos.copy())
    eef_task = mink.FrameTask(frame_name=EEF_SITE, frame_type="site",
                               position_cost=1.0, orientation_cost=1.0)
    posture_task = mink.PostureTask(model, cost=posture_cost)
    posture_task.set_target(data.qpos.copy())

    avoid_geoms = []
    for name in avoid_body_names:
        bid = model.body(name).id
        avoid_geoms.extend(mink.get_body_geom_ids(model, bid))
    collision_limit = mink.CollisionAvoidanceLimit(
        model, geom_pairs=[(avoid_geoms, target_geom_ids)],
        minimum_distance_from_collisions=min_dist,
    )
    config_limit = mink.ConfigurationLimit(model)

    target_se3 = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(target_mat), target_pos,
    )
    eef_task.set_target(target_se3)

    solver = "quadprog"
    arm_target = None
    for step in range(n_steps):
        vel = mink.solve_ik(configuration, [eef_task, posture_task], dt, solver,
                             limits=[config_limit, collision_limit])
        configuration.integrate_inplace(vel, dt)
        arm_target = np.array([configuration.q[model.joint(n).qposadr[0]] for n in JOINTS])
        action = np.concatenate([arm_target, [gripper_action]])
        env.step(action)
        if verbose and step % 20 == 0:
            eef_pos = env.sim.data.site_xpos[env.sim.model.site(EEF_SITE).id]
            print(f"  [mink_reach] step {step}: eef={eef_pos.round(3)} err={np.linalg.norm(eef_pos - target_pos):.4f}")

    final_eef_pos = env.sim.data.site_xpos[env.sim.model.site(EEF_SITE).id].copy()
    pos_err = float(np.linalg.norm(final_eef_pos - target_pos))
    return arm_target, pos_err
