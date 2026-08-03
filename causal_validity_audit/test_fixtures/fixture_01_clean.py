"""Category 1: clean pre-execution feature, direct assignment. Baseline true negative."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def select_candidate(env, candidate_pool):
    x, y, z = candidate_pool[0]
    yaw = candidate_pool[0][3]
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate_pool[0])
    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
    }
