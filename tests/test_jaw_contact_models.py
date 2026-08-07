"""Tests for the measured-pad jaw contact model (added 2026-08-07, step 3).

The value of the A/B in scripts/compare_jaw_contact_models.py rests entirely on
the two modes differing in contact geometry and NOTHING else. These tests pin
that: same control constants, same approach kinematics, same success rule, with
only the collidable geometry swapped.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
import pytest

from tango_robot.env_soarm import (
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS_WELD,
    GRIP_CLOSED,
    GRIP_OPEN,
    JAW_CONTACT_MEASURED_PADS,
    JAW_CONTACT_MEASURED_PADS_AIMED,
    JAW_CONTACT_PROXY_SPHERES,
    TABLE_TOP_Z,
)
from tango_robot.jaw_pads import derive, pad_geom_xml


@pytest.fixture(scope="module")
def pads():
    return derive()


# ── the derived pads are real gripping faces ─────────────────────────────────

def test_pad_normals_oppose_and_point_at_each_other(pads):
    c = pads["_check"]
    assert c["normals_dot"] < -0.8          # the two faces look at each other
    assert c["fixed_normal_dot_sep"] > 0.8  # ...and not away


def test_fitted_faces_are_planar_enough_to_be_a_box(pads):
    # rms deviation from the fitted plane; the finger surface is textured and
    # slightly tapered, so this is small but not zero.
    for tag in ("fixed", "moving"):
        assert pads[tag]["flatness"] < 0.004, (tag, pads[tag]["flatness"])


def test_pad_sizes_are_finger_scale(pads):
    for tag in ("fixed", "moving"):
        half = pads[tag]["size"]
        assert 0.008 < half[0] < 0.035, (tag, half)   # length along the finger
        assert 0.004 < half[1] < 0.020, (tag, half)   # width across it
        assert half[2] == pytest.approx(0.0015)


def test_pad_xml_targets_both_finger_bodies(pads):
    xml = pad_geom_xml(pads)
    assert set(xml) == {"gripper", "moving_jaw_so101_v1"}
    for g in xml.values():
        assert 'type="box"' in g


# ── the A/B differs in contact geometry only ─────────────────────────────────

@pytest.fixture(scope="module")
def env_pair():
    envs = {m: EnvironmentSoArm(obj_names=["BananaC"], vis=False,
                               grasp_mode=GRASP_MODE_PHYSICS_WELD,
                               enable_jaw_metrology=True,
                               jaw_contact_model=m)
            for m in (JAW_CONTACT_PROXY_SPHERES, JAW_CONTACT_MEASURED_PADS,
                      JAW_CONTACT_MEASURED_PADS_AIMED)}
    yield envs
    for e in envs.values():
        e.close()


def _collidable(env, body_name):
    bid = env.model.body(body_name).id
    return [env.model.geom(gi).name or f"geom{gi}"
            for gi in range(env.model.ngeom)
            if env.model.geom_bodyid[gi] == bid and env.model.geom_contype[gi] != 0]


def test_pad_mode_puts_all_jaw_contact_on_the_pads(env_pair):
    env = env_pair[JAW_CONTACT_MEASURED_PADS]
    assert _collidable(env, "gripper") == ["jaw_pad_fixed"]
    assert _collidable(env, "moving_jaw_so101_v1") == ["jaw_pad_moving"]


def test_legacy_mode_still_collides_only_the_proxy_spheres(env_pair):
    env = env_pair[JAW_CONTACT_PROXY_SPHERES]
    for body in ("gripper", "moving_jaw_so101_v1"):
        names = _collidable(env, body)
        assert len(names) == 1
    assert env._jaw_pad_geom_ids == []


def test_approach_kinematics_are_bit_identical_across_modes(env_pair):
    """The jaw-midpoint IK reads geom_xpos of the proxy spheres. Pad mode leaves
    those geoms exactly where they are and only stops them colliding, so the arm
    solves to the same configuration and any label flip is attributable to
    contact rather than to a different approach."""
    a = env_pair[JAW_CONTACT_PROXY_SPHERES]
    b = env_pair[JAW_CONTACT_MEASURED_PADS]
    for e in (a, b):
        e.reset_robot()
    for q in (GRIP_CLOSED, 0.3, 0.7, GRIP_OPEN):
        for e in (a, b):
            e.data.qpos[e._grip_qpos_adr] = q
            mujoco.mj_forward(e.model, e.data)
        assert np.array_equal(a._get_jaw_geom_midpoint(), b._get_jaw_geom_midpoint())


def test_control_constants_are_untouched_by_step_3():
    """Opening calibration is deliberately deferred to step 1; if these move,
    the A/B stops isolating the collider."""
    assert GRIP_CLOSED == 0.05
    assert GRIP_OPEN == 1.0


def test_pad_gap_tracks_the_real_fingertips_unlike_the_proxy(env_pair):
    a = env_pair[JAW_CONTACT_PROXY_SPHERES]
    b = env_pair[JAW_CONTACT_MEASURED_PADS]
    pf = b.model.geom("jaw_pad_fixed").id
    pm = b.model.geom("jaw_pad_moving").id

    def spans(lo, hi):
        vals = []
        for e, f in ((b, lambda: mujoco.mj_geomDistance(b.model, b.data, pf, pm,
                                                        2.0, np.zeros(6))),
                     (a, lambda: a._jaw_metrology.proxy_gap_m(a.data))):
            out = []
            for q in (lo, hi):
                e.data.qpos[e._grip_qpos_adr] = q
                mujoco.mj_forward(e.model, e.data)
                out.append(f())
            vals.append(out[1] - out[0])
        return vals

    pad_span, proxy_span = spans(GRIP_CLOSED, GRIP_OPEN)
    # measured: pads travel ~59 mm over the commanded window, proxies ~15 mm
    assert pad_span > 0.045
    assert pad_span > 2.5 * proxy_span


def test_invalid_contact_model_is_rejected():
    with pytest.raises(ValueError, match="jaw_contact_model"):
        EnvironmentSoArm(vis=False, grasp_mode=GRASP_MODE_PHYSICS_WELD,
                         jaw_contact_model="spheres_please")


def test_pad_mode_stamps_the_model_and_legacy_does_not(env_pair):
    """The stamp must be absent off the default, or every pre-existing caller's
    metrics dict changes shape."""
    a = env_pair[JAW_CONTACT_PROXY_SPHERES]
    b = env_pair[JAW_CONTACT_MEASURED_PADS]
    assert "jaw_contact_model" not in a.get_grasp_debug_metrics()
    assert b.get_grasp_debug_metrics()["jaw_contact_model"] == JAW_CONTACT_MEASURED_PADS


# ── the IK target is a separate defect from the collider ─────────────────────

def test_legacy_ik_target_is_far_from_the_gripping_faces(env_pair):
    """The legacy jaw-midpoint IK aims the midpoint of the finger meshes' frame
    ORIGINS at the grasp point.  Measured, that is 52-57 mm from the pads -- so
    aiming it parks the finger roots on the object while the fingers extend
    past it.  This is why swapping only the collider makes things worse: the
    contact moves to a surface the arm was never aiming at."""
    b = env_pair[JAW_CONTACT_MEASURED_PADS]
    b.reset_robot()
    pf, pm = b._jaw_pad_geom_ids
    for q in (GRIP_CLOSED, 0.3, GRIP_OPEN):
        b.data.qpos[b._grip_qpos_adr] = q
        mujoco.mj_forward(b.model, b.data)
        legacy = b._get_jaw_geom_midpoint()
        pad_mid = 0.5 * (b.data.geom_xpos[pf] + b.data.geom_xpos[pm])
        assert 0.045 < float(np.linalg.norm(legacy - pad_mid)) < 0.070


def test_aimed_mode_targets_the_pads_and_others_do_not(env_pair):
    aimed = env_pair[JAW_CONTACT_MEASURED_PADS_AIMED]
    aimed.reset_robot()
    pf, pm = aimed._jaw_pad_geom_ids
    assert np.allclose(aimed._get_jaw_geom_midpoint(),
                       0.5 * (aimed.data.geom_xpos[pf] + aimed.data.geom_xpos[pm]))
    for m in (JAW_CONTACT_PROXY_SPHERES, JAW_CONTACT_MEASURED_PADS):
        e = env_pair[m]
        e.reset_robot()
        assert np.allclose(e._get_jaw_geom_midpoint(),
                           0.5 * (e.data.geom_xpos[e._jaw_fixed_geom_id]
                                  + e.data.geom_xpos[e._jaw_mv_geom_id]))


def test_aimed_mode_still_uses_the_pads_for_contact(env_pair):
    env = env_pair[JAW_CONTACT_MEASURED_PADS_AIMED]
    assert _collidable(env, "gripper") == ["jaw_pad_fixed"]
    assert _collidable(env, "moving_jaw_so101_v1") == ["jaw_pad_moving"]
