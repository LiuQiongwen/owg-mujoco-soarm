"""Path allowlist policy.

Constrains where the Claude Code executor may write: only inside the paths
declared as `allowed_paths` in the experiment specification, and never
inside `forbidden_paths`.
"""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from research_agent.models import ExperimentSpec


class PathPolicyViolation(RuntimeError):
    pass


def _normalize(path: str) -> str:
    return PurePosixPath(path).as_posix()


def matches_any(path: str, patterns: Sequence[str]) -> bool:
    norm = _normalize(path)
    for pattern in patterns:
        pat = _normalize(pattern)
        if norm == pat or norm.startswith(pat.rstrip("/") + "/"):
            return True
        if fnmatch(norm, pat):
            return True
    return False


def is_path_allowed(path: str, spec: "ExperimentSpec") -> bool:
    if matches_any(path, spec.forbidden_paths):
        return False
    return matches_any(path, spec.allowed_paths)


def assert_paths_allowed(paths: Iterable[str], spec: "ExperimentSpec") -> None:
    violations = sorted(p for p in paths if not is_path_allowed(p, spec))
    if violations:
        raise PathPolicyViolation(
            f"path(s) outside the allowed scope for task '{spec.task_id}': {violations}"
        )


def assert_within_worktree(path: Path, worktree_root: Path) -> None:
    """Defense in depth: confirm a filesystem path genuinely resolves inside
    the isolated worktree root, not the main checkout or elsewhere."""
    resolved = path.resolve()
    root = worktree_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise PathPolicyViolation(f"path {resolved} escapes isolated worktree root {root}")
