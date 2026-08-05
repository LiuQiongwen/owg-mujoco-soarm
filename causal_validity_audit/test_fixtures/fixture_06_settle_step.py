"""Category 6: a physics "settle" step calling the SAME method name as a real execution step
(env.step), but occurring BEFORE the marker -- part of establishing the initial pre-execution
scene observation, not a genuine execution commitment. Tests whether the tagger respects marker
placement rather than flagging any env.step call anywhere in the function. Ground truth:
PRE_EXECUTION for a field read after the settle step but before the marker."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def observe_scene_then_grasp(env, candidate):
    for _ in range(5):
        env.step(None)  # let the object settle onto the table before observing it -- pre-execution
    object_height = env.data.object_pos[2]  # read AFTER settle, but still BEFORE the marker
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    return {
        "object_height": object_height,
    }
