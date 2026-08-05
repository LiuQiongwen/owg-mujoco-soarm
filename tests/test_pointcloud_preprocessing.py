"""Tests for tango_robot.pointcloud's preprocessing functions (added 2026-08-05
alongside the functions themselves -- see pointcloud.py's module docstring for
why these were consolidated from scattered ad hoc implementations)."""
import numpy as np
import pytest

from tango_robot.pointcloud import (
    crop_to_workspace,
    voxel_downsample,
    remove_statistical_outliers,
    segment_and_remove_plane,
    preprocess_pointcloud,
)


def test_crop_to_workspace_keeps_only_inside_points():
    points = np.array([
        [0.0, 0.0, 0.0],    # inside
        [10.0, 0.0, 0.0],   # outside x
        [0.0, -10.0, 0.0],  # outside y
        [0.0, 0.0, 10.0],   # outside z
        [0.1, 0.1, 0.1],    # inside
    ])
    bounds = ((-1, 1), (-1, 1), (-1, 1))
    out = crop_to_workspace(points, bounds)
    assert len(out) == 2
    assert np.allclose(sorted(out.tolist()), sorted([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]))


def test_crop_to_workspace_empty_input():
    out = crop_to_workspace(np.zeros((0, 3)), ((-1, 1), (-1, 1), (-1, 1)))
    assert len(out) == 0


def test_crop_to_workspace_boundary_inclusive():
    points = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    out = crop_to_workspace(points, ((-1, 1), (-1, 1), (-1, 1)))
    assert len(out) == 2  # boundary points are kept (>=, <=)


def test_voxel_downsample_reduces_dense_cluster():
    o3d = pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    # 1000 points crammed into a 1mm cube -> a single 5mm voxel should
    # collapse them to (at most) a handful of points.
    dense = rng.uniform(-0.0005, 0.0005, size=(1000, 3))
    out = voxel_downsample(dense, voxel_size=0.005)
    assert 0 < len(out) < 50


def test_voxel_downsample_noop_when_disabled():
    points = np.random.default_rng(0).normal(size=(50, 3))
    assert np.array_equal(voxel_downsample(points, voxel_size=None), points)
    assert np.array_equal(voxel_downsample(points, voxel_size=0), points)


def test_remove_statistical_outliers_drops_far_point():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(200, 3))
    far_outlier = np.array([[5.0, 5.0, 5.0]])
    points = np.vstack([cluster, far_outlier])
    inliers, mask = remove_statistical_outliers(points, nb_neighbors=10, std_ratio=2.0)
    assert mask[-1] == False  # the far point must be flagged as an outlier
    assert len(inliers) == mask.sum()
    assert not np.any(np.all(np.isclose(inliers, far_outlier), axis=1))


def test_remove_statistical_outliers_too_few_points_is_noop():
    points = np.random.default_rng(0).normal(size=(5, 3))
    out, mask = remove_statistical_outliers(points, nb_neighbors=20)
    assert np.array_equal(out, points)
    assert mask.all()


def test_remove_statistical_outliers_empty_input():
    out, mask = remove_statistical_outliers(np.zeros((0, 3)))
    assert len(out) == 0 and len(mask) == 0


def test_segment_and_remove_plane_removes_flat_table():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    xs = rng.uniform(-0.5, 0.5, size=500)
    ys = rng.uniform(-0.5, 0.5, size=500)
    table = np.stack([xs, ys, np.zeros(500)], axis=1)  # z=0 plane
    object_pts = rng.normal(loc=[0, 0, 0.1], scale=0.02, size=(50, 3))  # above table
    points = np.vstack([table, object_pts])

    above, plane_model = segment_and_remove_plane(points, distance_threshold=0.01)
    assert plane_model is not None
    # nearly all surviving points should be the object cluster, not the table
    assert len(above) < len(points)
    assert len(above) >= 40  # most of the 50 object points should survive


def test_segment_and_remove_plane_too_few_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out, plane_model = segment_and_remove_plane(points, ransac_n=3)
    assert np.array_equal(out, points)
    assert plane_model is None


def test_preprocess_pointcloud_chains_only_requested_steps():
    points = np.random.default_rng(0).normal(size=(30, 3))
    out, stats = preprocess_pointcloud(points)
    assert stats == {"n_raw": 30}
    assert np.array_equal(out, points)


def test_preprocess_pointcloud_records_stats_per_step():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    xs = rng.uniform(-0.5, 0.5, size=500)
    ys = rng.uniform(-0.5, 0.5, size=500)
    table = np.stack([xs, ys, np.zeros(500)], axis=1)
    object_pts = rng.normal(loc=[0, 0, 0.1], scale=0.02, size=(50, 3))
    points = np.vstack([table, object_pts])

    out, stats = preprocess_pointcloud(
        points,
        workspace_bounds=((-1, 1), (-1, 1), (-1, 1)),
        voxel_size=0.005,
        outlier_nb_neighbors=15,
        plane_distance_threshold=0.01,
    )
    assert list(stats.keys()) == [
        "n_raw", "n_after_crop", "n_after_voxel",
        "n_after_outlier_removal", "n_after_plane_removal",
    ]
    assert stats["n_raw"] == 550
    # each step should be non-increasing in point count
    values = list(stats.values())
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    assert len(out) == stats["n_after_plane_removal"]


def test_preprocess_pointcloud_empty_input_does_not_crash():
    out, stats = preprocess_pointcloud(
        np.zeros((0, 3)),
        workspace_bounds=((-1, 1), (-1, 1), (-1, 1)),
        voxel_size=0.005,
        outlier_nb_neighbors=15,
        plane_distance_threshold=0.01,
    )
    assert len(out) == 0
    assert stats["n_raw"] == 0
