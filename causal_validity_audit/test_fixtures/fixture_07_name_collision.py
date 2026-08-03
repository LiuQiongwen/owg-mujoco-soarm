"""Category 7: field-name collision -- mirrors the real "yaw" vs "grasp_yaw" bug in this project's
own retrospective_audit.py, where a generic field-name lookup for the Piper side silently resolved
against the SO-ARM101 side's similarly-named (but semantically different, and correctly admissible)
"yaw" entry instead of the actual Piper field "grasp_yaw", masking a real violation. This fixture
has two similarly-named fields with DIFFERENT ground truth, to test exact-name disambiguation."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def grasp_with_two_yaw_fields(env, candidate):
    yaw = candidate[3]  # clean, pre-execution candidate-pose component
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    grasp_yaw = env.data.gripper_orientation[2]  # execution-derived, different field, similar name
    return {
        "yaw": yaw,
        "grasp_yaw": grasp_yaw,
    }
