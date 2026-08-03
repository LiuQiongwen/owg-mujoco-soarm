"""Category 3: multi-hop taint (feature computed via 2-3 intermediate variables chained from a
tainted value). Tests transitive taint propagation, not just direct one-hop tainting."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def grasp_and_score(env, candidate):
    approach_yaw = candidate[3]
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    contact_force = env.data.sensor_force  # hop 1: tainted
    normalized_force = contact_force / env.data.max_force  # hop 2: derived from hop 1
    quality_score = normalized_force * 0.5 + 0.25  # hop 3: derived from hop 2
    return {
        "approach_yaw": approach_yaw,
        "quality_score": quality_score,
    }
