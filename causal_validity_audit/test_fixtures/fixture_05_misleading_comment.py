"""Category 5: dead/constant feature with a misleading docstring -- mirrors the real
dz/dz_lift/need_dz case in this project, where a comment (describing a different, inactive
dataset) implied these fields were execution-derived, but the LIVE code path actually hardcodes
them to a constant. Ground truth: PRE_EXECUTION despite the misleading comment -- the tagger must
trust the actual data flow, not prose."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def grasp_and_log_legacy_fields(env, candidate):
    x = candidate[0]
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    # NOTE (misleading, describes an older dataset variant, not this live path):
    # "dz is the measured post-close vertical drift, execution-derived."
    dz = 0.0  # actually a hardcoded dead constant in this live code path, never updated
    return {
        "x": x,
        "dz": dz,
    }
