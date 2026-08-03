"""Category 4: variable reassigned post-commit -- mirrors the real `grasp_yaw` bug found in this
project (grasp_mat/grasp_yaw assigned once pre-commit admissibly, then REASSIGNED at a later
"pre-close refresh" step after commit, so the value actually returned reflects post-execution
state even though a naive human reviewer might see only the first, clean assignment and stop
looking). Ground truth: EXECUTION_DERIVED (the reassigned, post-commit value)."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def grasp_with_refresh(env, candidate):
    grasp_yaw = candidate[3]  # clean, pre-commit
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    # "pre-close refresh": re-reads the live orientation right before closing,
    # silently overwriting the pre-execution value with a post-execution one.
    grasp_yaw = env.data.gripper_orientation[2]
    env.step(candidate)
    return {
        "grasp_yaw": grasp_yaw,
    }
