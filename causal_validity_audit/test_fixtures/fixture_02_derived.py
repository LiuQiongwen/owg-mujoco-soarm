"""Category 2: directly execution-derived (reads env state after env.step() post-marker). Baseline true positive."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def grasp_and_log(env, candidate):
    target_x = candidate[0]
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    post_close_drift = env.data.gripper_pos[0] - target_x
    return {
        "target_x": target_x,
        "post_close_drift": post_close_drift,
    }
