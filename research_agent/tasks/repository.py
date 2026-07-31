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
