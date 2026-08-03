"""Category 9: false-positive stress test. `env.config.gripper_width` is a STATIC configuration
value fixed at environment construction time, not live physical state -- reading it after the
marker is not actually a causal-validity violation. But the tool's heuristic ("any attribute chain
rooted at a variable named `env` is a live-state read") cannot distinguish this from a genuine
`env.data.qpos`-style live read. Ground truth: PRE_EXECUTION. This case is EXPECTED to be
mis-tagged EXECUTION_DERIVED by the tool -- record it as a documented false positive, a known
over-approximation of the "any env.* attribute chain" heuristic, not a fixture bug."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def grasp_with_static_config_read(env, candidate):
    approach_z = candidate[2]
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    gripper_width = env.config.gripper_width  # static config, fixed at construction -- not live state
    return {
        "approach_z": approach_z,
        "gripper_width": gripper_width,
    }
