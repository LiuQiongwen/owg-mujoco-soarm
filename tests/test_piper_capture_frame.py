"""Regression tests for Piper's capture-frame (T_eef_capture) targeting.

The central claim these lock in: introducing capture-frame grasp targeting
is EXACTLY behaviour-preserving for level (DOWN_ORIENTATION) grasps, and
differs only for tilted grasp orientations -- where the capture-frame
version is the correct one. If someone later changes T_EEF_CAPTURE_LOCAL or
GRASP_CAPTURE_HEIGHT_OFFSET independently, the equivalence test below
fails loudly rather than silently shifting every Piper grasp by centimetres.

Pure numpy against module-level constants/helpers -- no MuJoCo, no env
construction, so these run fast and without a GPU/EGL context.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tango_robot.piper_robosuite.piper_pick_and_place import (  # noqa: E402
    DOWN_ORIENTATION,
    GRASP_CAPTURE_HEIGHT_OFFSET,
    GRASP_HEIGHT_OFFSET,
    T_EEF_CAPTURE_LOCAL,
    capture_pose_from_eef,
    eef_target_for_capture_target,
    grasp_capture_target,
)


def test_capture_offset_is_pure_local_z_translation():
    """T_eef_capture is a translation along the eef frame's own local Z,
    with no component on the jaw-closing (local X) or local Y axes. This is
    what makes it composable with any orientation without leaking sideways."""
    assert T_EEF_CAPTURE_LOCAL[0] == 0.0
    assert T_EEF_CAPTURE_LOCAL[1] == 0.0
    assert T_EEF_CAPTURE_LOCAL[2] == pytest.approx(-0.0656)


def test_capture_and_eef_conversions_are_inverses():
    """Round-tripping through both helpers must be identity, for arbitrary
    (including tilted) orientations."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        # Random proper rotation via QR of a random matrix.
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        capture_pos = rng.normal(size=3)

        eef_pos = eef_target_for_capture_target(capture_pos, q)
        back = capture_pose_from_eef(eef_pos, q)
        np.testing.assert_allclose(back, capture_pos, atol=1e-12)


def test_level_grasp_is_exactly_equivalent_to_legacy():
    """THE key regression: for a level (DOWN_ORIENTATION) grasp, the new
    capture-frame target must reduce to the legacy eef-site target exactly.
    The +0.0656 capture height and the -0.0656 local-Z offset cancel.

    If this fails, every level Piper grasp has silently moved vertically --
    which is exactly the 65.6mm regression this refactor exists to avoid.
    """
    obj_pos = np.array([0.35, -0.12, 0.83])
    legacy_target = obj_pos + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    new_target = grasp_capture_target(obj_pos, DOWN_ORIENTATION)
    np.testing.assert_allclose(new_target, legacy_target, atol=1e-12)


def test_capture_height_offset_matches_the_validated_physical_height():
    """GRASP_CAPTURE_HEIGHT_OFFSET must equal -T_EEF_CAPTURE_LOCAL[2] for
    the level-grasp equivalence above to hold. Encoded separately so the
    intent ('this is the historically validated fingertip height, restated
    in the correct frame') is checked, not just the arithmetic."""
    assert GRASP_CAPTURE_HEIGHT_OFFSET == pytest.approx(-T_EEF_CAPTURE_LOCAL[2])


def test_tilted_grasp_differs_from_legacy_and_is_capture_correct():
    """For a tilted grasp the new target deliberately differs from legacy.
    Verify the difference is real AND that the fingertip midpoint genuinely
    lands at the intended capture point (which legacy would have missed)."""
    # 30 degrees about world X.
    a = np.radians(30.0)
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a), np.cos(a)]])
    tilted = rot_x @ DOWN_ORIENTATION

    obj_pos = np.array([0.35, -0.12, 0.83])
    intended_capture = obj_pos + np.array([0.0, 0.0, GRASP_CAPTURE_HEIGHT_OFFSET])

    new_target = grasp_capture_target(obj_pos, tilted)
    legacy_target = obj_pos + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])

    # They must actually differ for a tilted grasp.
    assert np.linalg.norm(new_target - legacy_target) > 1e-3

    # The new target puts the FINGERTIPS at the intended capture point.
    np.testing.assert_allclose(
        capture_pose_from_eef(new_target, tilted), intended_capture, atol=1e-12)

    # Legacy would have missed it. Both displacements have the same
    # magnitude (0.0656) but point `a` apart, so the miss is the chord
    # length 2*L*sin(a/2) -- 34.0mm at 30 degrees of tilt, growing to the
    # full 65.6mm only at 60 degrees and beyond.
    legacy_capture = capture_pose_from_eef(legacy_target, tilted)
    expected_miss = 2 * 0.0656 * np.sin(a / 2)
    assert np.linalg.norm(legacy_capture - intended_capture) == pytest.approx(expected_miss, abs=1e-9)
    assert expected_miss == pytest.approx(0.0340, abs=1e-4)


def test_explicit_height_offset_argument_is_respected():
    """move_to_two_stage_align_descend passes its own height offset through;
    make sure the parameter isn't silently ignored in favour of the default."""
    obj_pos = np.array([0.3, 0.0, 0.8])
    t = grasp_capture_target(obj_pos, DOWN_ORIENTATION, height_offset=0.10)
    capture = capture_pose_from_eef(t, DOWN_ORIENTATION)
    np.testing.assert_allclose(capture, obj_pos + np.array([0, 0, 0.10]), atol=1e-12)
