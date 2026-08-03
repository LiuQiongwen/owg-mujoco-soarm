"""Category 11 (padding): additional straightforward true-positive/true-negative cases, to bring
the total labeled-field count to a reasonable sample size for precision/recall to mean something
(the 10 core categories alone label ~17 fields; this file adds several more of each class)."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def grasp_object_id_and_width(env, candidate, object_class):
    width = candidate[6]
    object_id = object_class
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    return {
        "width": width,
        "object_id": object_id,
    }


def grasp_and_measure_lift(env, candidate):
    roll = candidate[3]
    pitch = candidate[4]
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidate)
    lift_height = env.data.object_pos[2]
    settled_ok = env.data.contact_flags[0]
    return {
        "roll": roll,
        "pitch": pitch,
        "lift_height": lift_height,
        "settled_ok": settled_ok,
    }


def grasp_with_static_geometry(env, candidate, point_cloud_stats):
    dist_to_centroid = point_cloud_stats["dist_to_centroid"]
    bbox_extent = point_cloud_stats["bbox_extent"]
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.put_obj_in_tray(candidate)
    return {
        "dist_to_centroid": dist_to_centroid,
        "bbox_extent": bbox_extent,
    }
