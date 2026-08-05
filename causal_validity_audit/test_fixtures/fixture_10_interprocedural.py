"""Category 10: interprocedural taint propagation. `_execute_and_measure` is a module-level helper
that itself calls `env.step(...)` (a KNOWN execution entry method, unlike fixture 8) and returns a
live measurement. `grasp_via_helper` calls this helper post-marker and uses its result in a
returned field. Tests whether `_find_execution_touching`'s fixed-point analysis over module-level
defs correctly propagates execution-touching status through the helper call, not just direct
`env.step` calls in the analyzed function itself. Ground truth: EXECUTION_DERIVED, and -- unlike
category 8 -- the tool IS expected to catch this correctly, since the helper's own entry method
(`step`) is in `DEFAULT_EXECUTION_ENTRY_METHODS`."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def _execute_and_measure(env, candidate):
    env.step(candidate)
    return env.data.contact_force


def grasp_via_helper(env, candidate):
    approach_yaw = candidate[3]
    CAUSAL_VALIDITY_COMMIT_POINT()
    measured_force = _execute_and_measure(env, candidate)
    return {
        "approach_yaw": approach_yaw,
        "measured_force": measured_force,
    }
