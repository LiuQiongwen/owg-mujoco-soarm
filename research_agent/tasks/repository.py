"""Repository snapshotting and isolated-worktree management.

The snapshot step records the repository's pre-existing state (Git SHA,
diff, status, and a redacted environment variable dump) before any agent
runs -- this is the user's existing state, never agent-generated work.

The worktree functions create/remove a throwaway `git worktree` per run, so
the Claude Code executor can write freely without ever touching the main
checkout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from research_agent import subprocess_runner
from research_agent.models import RunPaths


def snapshot_repository(repo_root: Path, run_paths: RunPaths, *, timeout: float = 30.0) -> dict:
    run_paths.commands_dir.mkdir(parents=True, exist_ok=True)

    sha_result = subprocess_runner.run_command(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        cwd=repo_root,
        run_dir=run_paths.commands_dir,
        name="git_rev_parse",
        timeout=timeout,
    )
    git_sha = Path(sha_result.stdout_path).read_text().strip()
    run_paths.git_sha_path.write_text(git_sha + "\n")

    diff_result = subprocess_runner.run_command(
        ["git", "-C", str(repo_root), "diff", "--binary"],
        cwd=repo_root,
        run_dir=run_paths.commands_dir,
        name="git_diff",
        timeout=timeout,
    )
    run_paths.git_diff_path.write_text(Path(diff_result.stdout_path).read_text())

    status_result = subprocess_runner.run_command(
        ["git", "-C", str(repo_root), "status", "--porcelain=v1"],
        cwd=repo_root,
        run_dir=run_paths.commands_dir,
        name="git_status",
        timeout=timeout,
    )

    env_snapshot = subprocess_runner.redact_environment(dict(os.environ))
    run_paths.environment_path.write_text(json.dumps(env_snapshot, indent=2, sort_keys=True) + "\n")

    return {
        "git_sha": git_sha,
        "git_status_stdout_path": status_result.stdout_path,
        "git_diff_path": str(run_paths.git_diff_path),
        "environment_path": str(run_paths.environment_path),
    }


def isolated_branch_name(run_id: str) -> str:
    return f"research-agent-run/{run_id}"


def create_isolated_worktree(
    repo_root: Path, run_paths: RunPaths, run_id: str, *, base_ref: str = "HEAD", timeout: float = 60.0
) -> Path:
    worktree_dir = run_paths.worktree_dir
    if worktree_dir.exists():
        raise FileExistsError(f"worktree directory already exists: {worktree_dir}")
    branch = isolated_branch_name(run_id)
    subprocess_runner.run_command(
        ["git", "-C", str(repo_root), "worktree", "add", "-b", branch, str(worktree_dir), base_ref],
        cwd=repo_root,
        run_dir=run_paths.commands_dir,
        name="git_worktree_add",
        timeout=timeout,
    )
    return worktree_dir


def remove_isolated_worktree(repo_root: Path, run_paths: RunPaths, run_id: str, *, timeout: float = 60.0) -> None:
    """Best-effort cleanup: called from a `finally` block, so failures here
    are recorded but never allowed to mask the pipeline's real outcome."""
    worktree_dir = run_paths.worktree_dir
    branch = isolated_branch_name(run_id)
    if worktree_dir.exists():
        subprocess_runner.run_command(
            ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_dir)],
            cwd=repo_root,
            run_dir=run_paths.commands_dir,
            name="git_worktree_remove",
            timeout=timeout,
        )
    subprocess_runner.run_command(
        ["git", "-C", str(repo_root), "branch", "-D", branch],
        cwd=repo_root,
        run_dir=run_paths.commands_dir,
        name="git_branch_delete",
        timeout=timeout,
    )


_IGNORED_STATUS_PREFIXES = (".prefect", "prefect.db")


def capture_repo_fingerprint(repo_root: Path, run_paths: RunPaths, *, label: str, timeout: float = 30.0) -> dict:
    """Cheap, Git-plumbing-based fingerprint of the repository's current
    tracked/staged/untracked state and HEAD -- used to independently verify
    (never merely trust Codex's own claim) that a read-only planning
    subprocess did not modify anything outside its assigned run directory.
    Called once before and once after the Codex planner runs; see
    diff_fingerprints below for the comparison."""
    run_paths.commands_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"codex_repo_check_{label}"

    def _git(args: list, name: str) -> str:
        result = subprocess_runner.run_command(
            ["git", "-C", str(repo_root), *args],
            cwd=repo_root,
            run_dir=run_paths.commands_dir,
            name=name,
            timeout=timeout,
        )
        return Path(result.stdout_path).read_text()

    return {
        "head": _git(["rev-parse", "HEAD"], f"{prefix}_head").strip(),
        "status": _git(["status", "--porcelain=v1", "--ignored=matching"], f"{prefix}_status"),
        "diff": _git(["diff", "--binary"], f"{prefix}_diff"),
        "diff_cached": _git(["diff", "--cached", "--binary"], f"{prefix}_diff_cached"),
    }


def _relevant_status_lines(raw: str, *, repo_root: Path, run_dir: Path) -> set:
    run_dir_resolved = run_dir.resolve()
    lines: set = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        path_part = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        path_part = path_part.strip('"')
        abs_path = (repo_root / path_part).resolve()
        try:
            abs_path.relative_to(run_dir_resolved)
            continue  # inside the immutable run directory -- expected, not a violation
        except ValueError:
            pass
        if any(path_part == p or path_part.startswith(p + "/") for p in _IGNORED_STATUS_PREFIXES):
            continue  # expected Prefect local-mode runtime files
        lines.add(line)
    return lines


def diff_fingerprints(before: dict, after: dict, *, run_dir: Path, repo_root: Path) -> list[str]:
    """Compare two capture_repo_fingerprint() snapshots, ignoring anything
    that changed only inside the immutable run directory or known Prefect
    runtime files. Returns a list of human-readable change descriptions;
    empty means the repository was untouched (aside from the run
    directory) between the two snapshots."""
    changes: list[str] = []
    if before["head"] != after["head"]:
        changes.append(f"HEAD moved from {before['head']} to {after['head']}")

    before_status = _relevant_status_lines(before["status"], repo_root=repo_root, run_dir=run_dir)
    after_status = _relevant_status_lines(after["status"], repo_root=repo_root, run_dir=run_dir)
    if before_status != after_status:
        added = sorted(after_status - before_status)
        removed = sorted(before_status - after_status)
        changes.append(f"git status changed outside the run directory: added={added} removed={removed}")

    if before["diff"] != after["diff"]:
        changes.append("tracked working-tree diff changed outside the run directory")
    if before["diff_cached"] != after["diff_cached"]:
        changes.append("staged diff changed")

    return changes


def worktree_changed_paths(worktree_dir: Path, run_paths: RunPaths, *, timeout: float = 30.0) -> list[str]:
    """Paths (tracked or untracked) changed inside the isolated worktree,
    relative to the worktree root -- used to enforce the path allowlist
    against whatever the executor actually did."""
    result = subprocess_runner.run_command(
        ["git", "-C", str(worktree_dir), "status", "--porcelain=v1"],
        cwd=worktree_dir,
        run_dir=run_paths.commands_dir,
        name="git_worktree_status",
        timeout=timeout,
    )
    paths: list[str] = []
    for line in Path(result.stdout_path).read_text().splitlines():
        if not line.strip():
            continue
        rest = line[3:] if len(line) > 3 else line.strip()
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        paths.append(rest.strip())
    return paths
