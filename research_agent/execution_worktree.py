"""MVP2 execution-worktree lifecycle: isolated Git worktree the real Claude
executor runs inside, plus the stronger before/after ignored-and-untracked
file manifest that MVP1 was missing (a plain repository fingerprint does not
detect an in-place edit to a file Git already ignores).

Every run gets its own throwaway branch and worktree directory, created
outside the main repository's own working-tree path (never a subdirectory a
normal `git status` on the main checkout would ever see) so real Claude can
never physically touch the main worktree even if every policy layer above
this module somehow failed. Nothing in this module ever merges, commits, or
pushes; nothing here removes a worktree automatically -- see
`remove_execution_worktree`, called only from the explicit
`worktree-cleanup` CLI command (research_agent.cli), never from the
execute flow itself (research_agent.execute_flow).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from research_agent import subprocess_runner
from research_agent.models import ChangedFileRecord, ExecutionWorktreeRecord, ManifestEntry, RunPaths
from research_agent.policies import execution_policy


class ExecutionWorktreeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def execution_branch_name(run_id: str) -> str:
    return f"research-agent-execute/{run_id}"


def default_execution_worktrees_root(repo_root: Path) -> Path:
    """Outside the main repository worktree by construction: a sibling of
    repo_root, never a path under it."""
    return repo_root.resolve().parent / "research_agent_execution_worktrees"


def resolve_execution_worktree_path(repo_root: Path, execution_worktrees_root: Path, run_id: str) -> Path:
    repo_root = repo_root.resolve()
    path = (execution_worktrees_root / run_id).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise ExecutionWorktreeError(
            "CLAUDE_EXECUTION_WORKTREE_INVALID",
            f"execution worktree path {path} is inside the main repository worktree {repo_root}",
        )
    if path.exists():
        raise ExecutionWorktreeError(
            "CLAUDE_EXECUTION_WORKTREE_INVALID", f"execution worktree path already exists: {path}"
        )
    return path


def _git(repo_root: Path, args: list, *, run_paths: RunPaths, name: str, timeout: float = 60.0) -> str:
    result = subprocess_runner.run_command(
        ["git", "-C", str(repo_root), *args], cwd=repo_root, run_dir=run_paths.commands_dir, name=name, timeout=timeout,
    )
    return Path(result.stdout_path).read_text()


def create_execution_worktree(
    repo_root: Path,
    run_paths: RunPaths,
    run_id: str,
    *,
    execution_worktrees_root: Optional[Path] = None,
    base_ref: str = "HEAD",
    timeout: float = 60.0,
) -> ExecutionWorktreeRecord:
    repo_root = Path(repo_root).resolve()
    execution_worktrees_root = (
        Path(execution_worktrees_root).resolve()
        if execution_worktrees_root is not None
        else default_execution_worktrees_root(repo_root)
    )
    worktree_path = resolve_execution_worktree_path(repo_root, execution_worktrees_root, run_id)

    branch = execution_branch_name(run_id)
    existing_branches = _git(repo_root, ["branch", "--list", branch], run_paths=run_paths, name="git_branch_list_precheck")
    if existing_branches.strip():
        raise ExecutionWorktreeError("CLAUDE_EXECUTION_WORKTREE_INVALID", f"branch already exists: {branch}")

    base_commit = _git(repo_root, ["rev-parse", base_ref], run_paths=run_paths, name="git_rev_parse_base").strip()

    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess_runner.run_command(
        ["git", "-C", str(repo_root), "worktree", "add", "-b", branch, str(worktree_path), base_commit],
        cwd=repo_root, run_dir=run_paths.commands_dir, name="git_worktree_add_execute", timeout=timeout,
    )

    initial_status = _git(
        worktree_path, ["status", "--porcelain=v1", "--untracked-files=all"],
        run_paths=run_paths, name="git_status_initial",
    )
    initial_diff = _git(worktree_path, ["diff", "--binary"], run_paths=run_paths, name="git_diff_initial")
    initial_untracked = [
        line[3:].strip() for line in initial_status.splitlines() if line.startswith("??")
    ]

    record = ExecutionWorktreeRecord(
        run_id=run_id,
        base_commit=base_commit,
        branch_name=branch,
        worktree_path=str(worktree_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        initial_status=initial_status,
        initial_tracked_diff_empty=(initial_diff.strip() == ""),
        initial_untracked_files=initial_untracked,
        preserved=True,
    )
    run_paths.execution_worktree_record_path.write_text(record.model_dump_json(indent=2) + "\n")
    return record


def collect_worktree_status(worktree_path: Path, run_paths: RunPaths) -> list[tuple[str, str]]:
    """Return (status_code, relative_path) pairs from `git status
    --porcelain=v1 --untracked-files=all`, covering both tracked and
    untracked changes. `--untracked-files=all` is required so a brand-new
    untracked directory is expanded into its individual files rather than
    reported as a single directory line -- otherwise per-file path/size
    accounting below would silently skip everything inside it."""
    raw = _git(
        worktree_path, ["status", "--porcelain=v1", "--untracked-files=all"],
        run_paths=run_paths, name="git_status_final",
    )
    entries: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        code = line[:2]
        rest = line[3:] if len(line) > 3 else line.strip()
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        entries.append((code.strip() or code, rest.strip().strip('"')))
    return entries


def _sha256_if_small(path: Path, *, max_bytes: int) -> Optional[str]:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_changed_file_records(worktree_path: Path, status_entries: list[tuple[str, str]]) -> list[ChangedFileRecord]:
    records: list[ChangedFileRecord] = []
    for code, rel_path in status_entries:
        abs_path = worktree_path / rel_path
        size: Optional[int] = None
        sha: Optional[str] = None
        if abs_path.exists() and abs_path.is_file():
            try:
                size = abs_path.stat().st_size
            except OSError:
                size = None
            sha = _sha256_if_small(abs_path, max_bytes=execution_policy.MAX_HASH_BYTES)
        records.append(ChangedFileRecord(path=rel_path, status=code, size_bytes=size, sha256=sha))
    return records


def total_changed_bytes(records: list[ChangedFileRecord]) -> int:
    return sum(r.size_bytes for r in records if r.size_bytes is not None)


# ── ignored/untracked-file manifest (before/after) ──────────────────────────

_MANIFEST_MAX_ENTRIES_PER_ROOT = 500


def _manifest_roots(worktree_path: Path, spec) -> list[str]:
    roots: set[str] = set()
    roots.update(execution_policy.effective_modify_paths(spec))
    roots.update(spec.allowed_paths)
    roots.update(spec.forbidden_paths)
    roots.update(p.rstrip("/") for p in execution_policy.HIGH_RISK_PREFIXES if p != ".git/")
    roots.update(execution_policy.PROTECTED_ROOT_CONFIG_FILES)
    for pattern in ("paper*", "scripts/*calibrat*", "paperA_data/scripts/*calibrat*"):
        for entry in sorted(worktree_path.glob(pattern)):
            try:
                roots.add(entry.relative_to(worktree_path).as_posix())
            except ValueError:
                continue
    return sorted(r for r in roots if r)


def _entry_for(worktree_path: Path, rel_path: str) -> ManifestEntry:
    abs_path = worktree_path / rel_path
    if abs_path.is_symlink():
        try:
            target = os.readlink(abs_path)
        except OSError:
            target = None
        return ManifestEntry(path=rel_path, file_type="symlink", symlink_target=target)
    if not abs_path.exists():
        return ManifestEntry(path=rel_path, file_type="missing")
    if abs_path.is_dir():
        return ManifestEntry(path=rel_path, file_type="dir")
    try:
        stat = abs_path.stat()
    except OSError:
        return ManifestEntry(path=rel_path, file_type="missing")
    sha = _sha256_if_small(abs_path, max_bytes=execution_policy.MAX_HASH_BYTES)
    return ManifestEntry(path=rel_path, file_type="file", size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha)


def build_ignored_and_untracked_manifest(worktree_path: Path, spec) -> list[ManifestEntry]:
    """Stronger before/after manifest than a plain Git status fingerprint:
    walks every configured root (not just what Git already tracks or
    reports as changed) so an in-place edit to a file Git ignores is still
    detected by comparing two of these snapshots. Bounded per root so this
    never turns into an exhaustive hash of a large dataset/checkpoint
    directory -- see _MANIFEST_MAX_ENTRIES_PER_ROOT and
    execution_policy.MAX_HASH_BYTES (large files get metadata only, no
    hash)."""
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    # The worktree's own top-level `.git` is always tracked as exactly one
    # non-recursive entry (never walked): for a linked worktree (the only
    # kind execution_worktree.py ever creates) it is a small pointer file,
    # so hashing it is cheap and stable -- but a mutation (e.g. replacing it
    # to redirect Git elsewhere) is otherwise completely invisible to `git
    # status`, since `.git` itself is never a tracked working-tree path.
    entries.append(_entry_for(worktree_path, ".git"))
    seen.add(".git")
    for root in _manifest_roots(worktree_path, spec):
        abs_root = worktree_path / root
        if not abs_root.exists() and not abs_root.is_symlink():
            continue
        if abs_root.is_file() or abs_root.is_symlink():
            if root not in seen:
                entries.append(_entry_for(worktree_path, root))
                seen.add(root)
            continue
        count = 0
        for dirpath, dirnames, filenames in os.walk(abs_root):
            dirnames[:] = [d for d in sorted(dirnames) if d != ".git"]
            for fname in sorted(filenames):
                if count >= _MANIFEST_MAX_ENTRIES_PER_ROOT:
                    break
                abs_file = Path(dirpath) / fname
                rel = abs_file.relative_to(worktree_path).as_posix()
                if rel in seen:
                    continue
                entries.append(_entry_for(worktree_path, rel))
                seen.add(rel)
                count += 1
            if count >= _MANIFEST_MAX_ENTRIES_PER_ROOT:
                break
    return sorted(entries, key=lambda e: e.path)


def save_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    path.write_text(json.dumps([e.model_dump() for e in entries], indent=2) + "\n")


def diff_manifests(before: list[ManifestEntry], after: list[ManifestEntry]) -> list[str]:
    """Compare two build_ignored_and_untracked_manifest() snapshots. Returns
    human-readable change descriptions for entries whose type/size/mtime/
    hash/symlink-target changed, including entries that appeared or
    disappeared between snapshots."""
    before_map = {e.path: e for e in before}
    after_map = {e.path: e for e in after}
    changes: list[str] = []
    for path in sorted(set(before_map) | set(after_map)):
        b, a = before_map.get(path), after_map.get(path)
        if b is None:
            changes.append(f"{path}: appeared ({a.file_type})")
        elif a is None:
            changes.append(f"{path}: disappeared (was {b.file_type})")
        elif (b.file_type, b.size_bytes, b.sha256, b.symlink_target) != (a.file_type, a.size_bytes, a.sha256, a.symlink_target):
            changes.append(f"{path}: changed ({b.file_type}/{b.size_bytes}/{b.sha256} -> {a.file_type}/{a.size_bytes}/{a.sha256})")
    return changes


def remove_execution_worktree(
    repo_root: Path, worktree_path: Path, branch: str, *, timeout: float = 60.0, delete_branch: bool = False
) -> None:
    """Only ever called from the explicit worktree-cleanup CLI path -- never
    from execute_flow itself. Never uses `git clean -fdx`. Deleting the
    branch is opt-in (delete_branch) and separate from removing the worktree
    directory itself."""
    repo_root = Path(repo_root).resolve()
    if Path(worktree_path).exists():
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_path)],
            shell=False, check=True, capture_output=True, text=True, timeout=timeout,
        )
    if delete_branch:
        subprocess.run(
            ["git", "-C", str(repo_root), "branch", "-D", branch],
            shell=False, check=True, capture_output=True, text=True, timeout=timeout,
        )
