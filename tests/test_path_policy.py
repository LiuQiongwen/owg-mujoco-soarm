import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_agent.models import ExperimentSpec
from research_agent.policies.paths import (
    PathPolicyViolation,
    assert_paths_allowed,
    assert_within_worktree,
    is_path_allowed,
    matches_any,
)


def _spec(**overrides):
    base = dict(
        task_id="demo",
        goal="demo",
        allowed_paths=["experiments", "research_agent_runs"],
        forbidden_paths=["research_agent", "tests"],
        smoke_command=["python", "-c", "print('ok')"],
    )
    base.update(overrides)
    return ExperimentSpec(**base)


def test_exact_allowed_path_is_allowed():
    spec = _spec()
    assert is_path_allowed("experiments", spec)


def test_subpath_of_allowed_dir_is_allowed():
    spec = _spec()
    assert is_path_allowed("experiments/example_smoke.yaml", spec)
    assert is_path_allowed("research_agent_runs/run1/artifacts/metrics.json", spec)


def test_unrelated_path_is_not_allowed():
    spec = _spec()
    assert not is_path_allowed("paper_final.tex", spec)
    assert not is_path_allowed("data/secret.npz", spec)


def test_forbidden_subpath_wins_over_a_broader_allowed_parent():
    spec = _spec(allowed_paths=["experiments"], forbidden_paths=["experiments/secret"])
    assert is_path_allowed("experiments/example_smoke.yaml", spec)
    assert not is_path_allowed("experiments/secret/leak.yaml", spec)


def test_forbidden_takes_precedence_over_allowed():
    spec = _spec()
    assert not is_path_allowed("research_agent/flow.py", spec)
    assert not is_path_allowed("tests/test_path_policy.py", spec)


def test_assert_paths_allowed_raises_and_lists_all_violations():
    spec = _spec()
    with pytest.raises(PathPolicyViolation) as exc_info:
        assert_paths_allowed(
            ["experiments/example_smoke.yaml", "research_agent/flow.py", "paper_final.tex"], spec
        )
    message = str(exc_info.value)
    assert "research_agent/flow.py" in message
    assert "paper_final.tex" in message
    assert "experiments/example_smoke.yaml" not in message


def test_assert_paths_allowed_passes_for_clean_set():
    spec = _spec()
    assert_paths_allowed(["experiments/example_smoke.yaml"], spec)  # does not raise


def test_matches_any_glob_pattern():
    assert matches_any("experiments/example_smoke.yaml", ["experiments/*.yaml"])
    # fnmatch's "*" is not filesystem-aware and does cross "/", unlike a shell glob.
    assert matches_any("experiments/sub/other.yaml", ["experiments/*.yaml"])
    assert not matches_any("other_dir/example_smoke.yaml", ["experiments/*.yaml"])


def test_assert_within_worktree_passes_for_inner_path(tmp_path):
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()
    inner = worktree_root / "artifacts" / "metrics.json"
    inner.parent.mkdir(parents=True)
    inner.write_text("{}")
    assert_within_worktree(inner, worktree_root)  # does not raise
    assert_within_worktree(worktree_root, worktree_root)  # root itself is fine


def test_assert_within_worktree_rejects_escape(tmp_path):
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()
    outside = tmp_path / "elsewhere" / "file.txt"
    outside.parent.mkdir(parents=True)
    outside.write_text("x")
    with pytest.raises(PathPolicyViolation):
        assert_within_worktree(outside, worktree_root)


def test_assert_within_worktree_rejects_parent_traversal(tmp_path):
    worktree_root = tmp_path / "sub" / "worktree"
    worktree_root.mkdir(parents=True)
    parent_escape = worktree_root / ".." / ".." / "outside.txt"
    with pytest.raises(PathPolicyViolation):
        assert_within_worktree(parent_escape, worktree_root)
