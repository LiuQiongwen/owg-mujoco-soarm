"""MVP2 execute-flow policy: path allowlist/denylist and command allowlist
for the real (or fake) Claude executor running inside an isolated execution
worktree.

Two independent, deterministic gates, both re-checked from scratch against
what Git actually reports happened -- never merely trusted from Claude's own
structured response (research_agent.models.ExecutorImplementationResult is
advisory only):

  * Path policy (this module's `validate_changed_paths`): every changed path
    (tracked or untracked) must resolve (after symlinks) inside the
    execution worktree, inside the spec's effective modify-path allowlist,
    and outside a fixed set of high-risk directories that are blocked no
    matter what a spec's allowed_paths/allowed_modify_paths say. Also
    enforces max_changed_files / max_changed_bytes and rejects nested Git
    repositories/submodules and any touch of `.git` itself.

  * Command policy (this module's `assert_executor_command_allowed`): a
    command may run only if it matches the small fixed MVP2 executor
    allowlist (python -m compileall / python -m pytest <path> / git status
    --short / git diff --check / git diff --name-status) or an exact entry
    in spec.allowed_executor_commands, and does not contain any forbidden
    token. This module never spawns anything -- it only classifies argv
    arrays supplied by the caller (used both to gate the harness's own
    static-check commands and to audit whatever Claude's structured
    response claims it ran; see execute_flow.py).
"""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from research_agent.models import ExecutorImplementationResult, ExperimentSpec


class ExecutionPolicyViolation(RuntimeError):
    """Carries a specific deterministic terminal-state code (see the MVP2
    task contract's "Failure states" list / models.EXECUTE_FAILURE_STATES)
    so callers never have to string-sniff a message to decide which code to
    persist."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ── high-risk directories/files, blocked unconditionally ───────────────────
# Defense in depth: blocked even if a spec's allowed_paths/allowed_modify_paths
# somehow included one of these (which a well-formed MVP2 spec never should).
HIGH_RISK_PREFIXES = (
    "data/",
    "datasets/",
    "results/",
    "checkpoints/",
    "paperA_data/",
    ".agents/locks/",
    ".git/",
)
HIGH_RISK_EXACT = (".git", ".gitmodules")
# Top-level paper artifacts in this repository are all named paper*.{tex,pdf,md,...}
# (paper_final.tex, paper_tro.tex, paper_draft.md, ...) -- block the whole family.
HIGH_RISK_BASENAME_PREFIX = "paper"
# Robot calibration scripts/data are scattered by filename, not by a single
# directory (scripts/calibrate_*.py, paperA_data/scripts/*calibrat*.py, ...).
HIGH_RISK_FILENAME_SUBSTRINGS = ("calibrat",)

# Repo-root configuration files that must never be touched by the executor,
# even though they are not under any of the directories above.
PROTECTED_ROOT_CONFIG_FILES = (".gitignore", "pyproject.toml")

MAX_HASH_BYTES = 2_000_000  # do not hash large files; record metadata only


def _normalize(path: str) -> str:
    return PurePosixPath(path).as_posix()


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    norm = _normalize(path)
    for pattern in patterns:
        pat = _normalize(pattern)
        if norm == pat or norm.startswith(pat.rstrip("/") + "/"):
            return True
        if fnmatch(norm, pat):
            return True
    return False


def effective_modify_paths(spec: "ExperimentSpec") -> list[str]:
    """MVP2 falls back to spec.allowed_paths when allowed_modify_paths is
    not explicitly set, so a spec written before MVP2 existed still has a
    sane (if unchanged) modification scope."""
    return list(spec.allowed_modify_paths) if spec.allowed_modify_paths else list(spec.allowed_paths)


def is_high_risk_path(path: str) -> bool:
    norm = _normalize(path)
    parts = PurePosixPath(norm).parts
    basename = parts[-1].lower() if parts else ""
    if norm in HIGH_RISK_EXACT or norm.startswith(".git/"):
        return True
    if any(norm == p.rstrip("/") or norm.startswith(p) for p in HIGH_RISK_PREFIXES):
        return True
    # Every top-level paper artifact in this repository is named paper*
    # (paper_final.tex, paper_tro.tex, paper_draft.md, paperA_data/, ...).
    if parts and parts[0].lower().startswith(HIGH_RISK_BASENAME_PREFIX):
        return True
    if any(sub in basename for sub in HIGH_RISK_FILENAME_SUBSTRINGS):
        return True
    if norm in PROTECTED_ROOT_CONFIG_FILES:
        return True
    return False


def is_modify_path_allowed(path: str, spec: "ExperimentSpec") -> bool:
    if is_high_risk_path(path):
        return False
    if _matches_any(path, spec.forbidden_paths):
        return False
    return _matches_any(path, effective_modify_paths(spec))


def assert_within_worktree_no_symlink_escape(worktree_root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` (relative to worktree_root) through any symlinks
    and confirm the effective real path still lands inside worktree_root.
    Also rejects a raw path-traversal attempt (a literal ".." component or
    an absolute path) before ever touching the filesystem."""
    if not rel_path or rel_path.startswith("/") or ".." in PurePosixPath(rel_path).parts:
        raise ExecutionPolicyViolation(
            "CLAUDE_SYMLINK_ESCAPE_DETECTED", f"path traversal or absolute path rejected: {rel_path!r}"
        )
    root = worktree_root.resolve()
    candidate = (worktree_root / rel_path)
    # Resolve symlinks in the parent chain even if the final component does
    # not exist (e.g. a deleted file) -- resolve the deepest existing
    # ancestor, then re-append the remaining components.
    existing = candidate
    remainder: list[str] = []
    while not existing.exists() and existing != existing.parent:
        remainder.insert(0, existing.name)
        existing = existing.parent
    resolved = existing.resolve().joinpath(*remainder) if remainder else existing.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ExecutionPolicyViolation(
            "CLAUDE_SYMLINK_ESCAPE_DETECTED",
            f"path {rel_path!r} resolves to {resolved}, which escapes the execution worktree root {root}",
        ) from None
    return resolved


def detect_nested_git_repo(worktree_root: Path, changed_paths: Iterable[str]) -> list[str]:
    """A changed path that is itself a `.git` entry anywhere other than the
    worktree root's own `.git`, or a directory containing a nested `.git`
    (a submodule-style nested repository), is never permitted."""
    offenders: list[str] = []
    for rel_path in changed_paths:
        parts = PurePosixPath(_normalize(rel_path)).parts
        if ".git" in parts[:-1] or (parts and parts[-1] == ".git" and len(parts) > 1):
            offenders.append(rel_path)
            continue
        candidate = worktree_root / rel_path
        if candidate.is_dir() and (candidate / ".git").exists():
            offenders.append(rel_path)
    return offenders


def validate_changed_paths(
    worktree_root: Path,
    changed_paths: Sequence[str],
    spec: "ExperimentSpec",
) -> None:
    """Raise ExecutionPolicyViolation on the first category of violation
    found, in a fixed priority order (symlink escape / traversal first,
    since that is a filesystem-resolution concern, then nested-repo, then
    plain allowlist membership). Does not check count/byte limits -- see
    assert_change_limits, called separately once sizes are known."""
    for rel_path in changed_paths:
        assert_within_worktree_no_symlink_escape(worktree_root, rel_path)

    nested = detect_nested_git_repo(worktree_root, changed_paths)
    if nested:
        raise ExecutionPolicyViolation(
            "CLAUDE_NESTED_GIT_DETECTED", f"nested Git repository / submodule change detected: {sorted(nested)}"
        )

    violations = sorted(p for p in changed_paths if not is_modify_path_allowed(p, spec))
    if violations:
        raise ExecutionPolicyViolation(
            "CLAUDE_PATH_POLICY_FAILED", f"path(s) outside the allowed modification scope: {violations}"
        )


def assert_change_limits(*, file_count: int, byte_count: int, spec: "ExperimentSpec") -> None:
    if file_count > spec.max_changed_files:
        raise ExecutionPolicyViolation(
            "CLAUDE_CHANGED_FILE_LIMIT_EXCEEDED",
            f"{file_count} changed files exceeds max_changed_files={spec.max_changed_files}",
        )
    if byte_count > spec.max_changed_bytes:
        raise ExecutionPolicyViolation(
            "CLAUDE_CHANGED_BYTES_LIMIT_EXCEEDED",
            f"{byte_count} changed bytes exceeds max_changed_bytes={spec.max_changed_bytes}",
        )


# ── command policy ──────────────────────────────────────────────────────────

_FIXED_ALLOWLIST_EXACT: tuple[tuple[str, ...], ...] = (
    ("git", "status", "--short"),
    ("git", "diff", "--check"),
    ("git", "diff", "--name-status"),
)
_FIXED_ALLOWLIST_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "compileall"),
)
# python -m pytest must always be given at least one explicit test path/target
# -- a bare "python -m pytest" (whole-suite run) is never in the fixed allowlist.
_PYTEST_PREFIX = ("python", "-m", "pytest")

_FORBIDDEN_EXECUTABLES = {"bash", "sh", "sudo", "docker", "docker-compose", "curl", "wget", "rm"}
_FORBIDDEN_SOLO_TOKENS = {"--gpu", "gpu", "cuda", "nvidia-smi", "-c"}
_FORBIDDEN_SUBSTRINGS = (
    "cuda",
    "docker",
    "real_hw",
    "soarm_real_backend",
    "/dev/ttyusb",
    "lerobot_train.py",
    "train_lggsn_pairwise.py",
)
_FORBIDDEN_GIT_SUBCOMMANDS = {"push", "commit", "reset", "clean", "worktree"}
_FORBIDDEN_PACKAGE_MANAGER_PAIRS = (("pip", "install"), ("conda", "install"))


def find_forbidden_executor_token(command: Sequence[str]) -> str | None:
    lowered = [str(tok).lower() for tok in command]
    if lowered and lowered[0] in _FORBIDDEN_EXECUTABLES:
        return lowered[0]
    for tok in lowered:
        if tok in _FORBIDDEN_SOLO_TOKENS:
            return tok
        for sub in _FORBIDDEN_SUBSTRINGS:
            if sub in tok:
                return sub
    if "git" in lowered:
        for bad in _FORBIDDEN_GIT_SUBCOMMANDS:
            if bad in lowered:
                return f"git {bad}"
    for first, second in _FORBIDDEN_PACKAGE_MANAGER_PAIRS:
        if first in lowered and second in lowered:
            return f"{first} {second}"
    return None


def is_executor_command_allowed(command: Sequence[str], spec: "ExperimentSpec") -> bool:
    command = [str(tok) for tok in command]
    if tuple(command) in _FIXED_ALLOWLIST_EXACT:
        return True
    for prefix in _FIXED_ALLOWLIST_PREFIXES:
        if tuple(command[: len(prefix)]) == prefix:
            return True
    if tuple(command[: len(_PYTEST_PREFIX)]) == _PYTEST_PREFIX and len(command) > len(_PYTEST_PREFIX):
        return True
    return any(command == approved for approved in spec.allowed_executor_commands)


def assert_executor_command_allowed(command: Sequence[str], spec: "ExperimentSpec") -> None:
    if isinstance(command, (str, bytes)):
        raise ExecutionPolicyViolation(
            "CLAUDE_COMMAND_POLICY_FAILED", f"command must be an argv array, never a shell string: {command!r}"
        )
    command = [str(tok) for tok in command]
    forbidden = find_forbidden_executor_token(command)
    if forbidden is not None:
        raise ExecutionPolicyViolation(
            "CLAUDE_COMMAND_POLICY_FAILED", f"command contains forbidden token {forbidden!r}: {command}"
        )
    if not is_executor_command_allowed(command, spec):
        raise ExecutionPolicyViolation(
            "CLAUDE_COMMAND_POLICY_FAILED", f"command is not in the MVP2 executor command policy: {command}"
        )


def validate_claimed_commands(
    implementation: "ExecutorImplementationResult", spec: "ExperimentSpec"
) -> list[str]:
    """Audit whatever Claude's structured response claims it ran (advisory
    only -- see module docstring) against the same command policy. Returns
    human-readable violation messages; empty means every claimed command
    passes the policy. The caller must never treat an empty result as proof
    those commands actually ran -- only the deterministic Git diff and the
    harness's own static-check commands are authoritative."""
    violations: list[str] = []
    for command in implementation.commands_run:
        try:
            assert_executor_command_allowed(command, spec)
        except ExecutionPolicyViolation as e:
            violations.append(str(e))
    return violations
