"""
Stage 2: full pick-and-place trial using mink's collision-aware reach for
the pre-grasp leg (READY_QPOS -> descend pose), replacing the DLS
`ArmIK`-driven transit_high/approach/descend phases that could still sweep
the arm through tall objects even after the manual SAFE_TRANSIT_Z waypoint
hack (see piper_pick_and_place.py's OBJECT_TOP_OFFSET/SAFE_TRANSIT_Z
comments for that history). Everything AFTER the object is grasped (lift,
transit to tray, lower, open, retract) still uses the existing
ArmIK/move_to_interpolated machinery unchanged -- see piper_mink_ik.py's
module docstring for why collision-avoidance against the held object itself
would be counterproductive there.

Usage:
  conda run -n tango python3 -m tango_robot.piper_robosuite.piper_pick_and_place_mink [obj_name] [n_trials] [start_trial]
"""
import sys

import numpy as np

from tango_robot.piper_robosuite import piper_robot, piper_gripper  # noqa
from tango_robot.piper_robosuite.piper_multi_object_scene import (
    PiperMultiObjectScene, PLACEMENT_TRAY_CENTER, PLACEMENT_TRAY_HALF_EXTENT,
)
from tango_robot.piper_robosuite.piper_pick_and_place import (
    ArmIK, READY_QPOS, GRASP_HEIGHT_OFFSET, TRAY_DROP_HEIGHT,
    GRIPPER_OPEN, GRIPPER_CLOSE, approach_height_for,
    compute_grasp_orientation, true_centroid_xy, move_to, move_to_interpolated,
)
from tango_robot.piper_robosuite.piper_mink_ik import mink_reach, object_geom_ids


def run_pick_and_place_mink(env, obj_name, verbose=False):
    ik = ArmIK(env)

    hold_action = np.concatenate([READY_QPOS, [GRIPPER_OPEN]])
    for _ in range(30):
        env.step(hold_action)

    body_origin_pos = env.get_object_positions()[obj_name].copy()
    obj_quat = env.sim.data.xquat[env.object_body_ids[obj_name]].copy()
    obj_pos = true_centroid_xy(body_origin_pos, obj_quat, obj_name)
    grasp_mat = compute_grasp_orientation(env, obj_name)
    approach_height = approach_height_for(obj_name)
    descend_target = obj_pos + np.array([0, 0, GRASP_HEIGHT_OFFSET])

    if verbose:
        print(f"target object '{obj_name}' at {obj_pos.round(3)}, descend target {descend_target.round(3)}")

    target_geoms = object_geom_ids(env, obj_name)
    qpos_seed, pos_err = mink_reach(
        env, descend_target, grasp_mat, target_geoms, GRIPPER_OPEN,
        n_steps=80, dt=0.05, verbose=verbose,
    )
    if verbose:
        print(f"[mink pre-grasp reach] final pos_err={pos_err:.4f}")

    if verbose:
        print("[close gripper]")
    move_to(env, qpos_seed, GRIPPER_CLOSE, steps=250)

    tray_xy = PLACEMENT_TRAY_CENTER
    tray_drop_pos = np.array([tray_xy[0], tray_xy[1], env.table_offset[2] + TRAY_DROP_HEIGHT])
    phase_log = {"pre_grasp_reach": {"pos_err_cm": float(pos_err * 100)}}

    def solve_and_move(name, target, grip, seed_qpos, interpolated=True):
        qpos, converged, err, source = ik.solve_multi_seed(target, primary_seed=seed_qpos, target_mat=grasp_mat)
        phase_log[name] = {"converged": bool(converged), "err_cm": float(err * 100), "seed_source": source}
        if verbose:
            print(f"[{name}] target={target.round(3)} converged={converged} err_cm={err*100:.2f} seed={source}")
        if interpolated:
            move_to_interpolated(env, ik, qpos, grip)
        else:
            move_to(env, qpos, grip)
        return qpos

    qpos_seed = solve_and_move("lift", obj_pos + [0, 0, approach_height], GRIPPER_CLOSE, qpos_seed)
    tray_above = tray_drop_pos + [0, 0, 0.08]
    qpos_seed = solve_and_move("transit_above_tray", tray_above, GRIPPER_CLOSE, qpos_seed)
    qpos_seed = solve_and_move("lower_into_tray", tray_drop_pos, GRIPPER_CLOSE, qpos_seed)

    if verbose:
        print("[open gripper]")
    move_to(env, qpos_seed, GRIPPER_OPEN, steps=40)
    qpos, converged, err, source = ik.solve_multi_seed(tray_above, primary_seed=qpos_seed, target_mat=grasp_mat)
    phase_log["retract"] = {"converged": bool(converged), "err_cm": float(err * 100), "seed_source": source}
    move_to(env, qpos, GRIPPER_OPEN)

    final_pos = env.get_object_positions()[obj_name]
    dist_to_tray = float(np.linalg.norm(final_pos[:2] - np.array(tray_xy)))
    success = dist_to_tray < PLACEMENT_TRAY_HALF_EXTENT
    if verbose:
        print(f"final '{obj_name}' pos: {final_pos.round(3)}  dist_to_tray_center_xy={dist_to_tray:.3f}m  success={success}")

    return {
        "object": obj_name,
        "strategy": "mink",
        "success": success,
        "dist_to_tray": dist_to_tray,
        "spawn_pos": obj_pos.tolist(),
        "final_pos": final_pos.tolist(),
        "phases": phase_log,
    }


def main():
    obj_name = sys.argv[1] if len(sys.argv) > 1 else "cracker"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    start_trial = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    results = []
    for trial_id in range(start_trial, start_trial + n_trials):
        np.random.seed(trial_id)
        env = PiperMultiObjectScene(
            robots="Piper", ycb_objects=[obj_name],
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, control_freq=20,
        )
        env.reset()
        result = run_pick_and_place_mink(env, obj_name, verbose=True)
        result["trial_id"] = trial_id
        results.append(result)
        print(f"=== trial {trial_id}: success={result['success']} dist_to_tray={result['dist_to_tray']:.3f} ===")
        env.close()

    n_success = sum(r["success"] for r in results)
    print(f"\n{obj_name}: {n_success}/{n_trials} success ({100*n_success/n_trials:.0f}%)")

    import json
    out_path = f"/tmp/claude-1000/-lena/7288f7ab-dc84-4b44-a682-e7d1d9c85e05/scratchpad/mink_pipeline_{obj_name}_{start_trial}-{start_trial+n_trials}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
