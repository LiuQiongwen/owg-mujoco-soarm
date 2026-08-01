"""MVP4 restricted-execution artifact policy.

Scans the run's assigned artifacts directory AFTER an approved command has
run and builds an immutable, forensic artifact_manifest.json -- never
trusting a command's own claim about what it wrote. Every entry's effective
real path is validated to still resolve inside the artifacts directory (a
symlink pointing outside it is a violation, not silently followed), FIFOs/
sockets/device files are rejected outright, a nested `.git` directory is
rejected, and file-count/total-bytes/per-file-bytes limits are enforced.

The artifacts directory itself is always a fresh, empty, run-scoped
directory created by research_agent.execution_flow (never data/, datasets/,
results/, checkpoints/, a paper directory, calibration file, the main
repository, or a home config path) -- so "no modification of existing
research outputs" holds structurally, by construction of where this
directory lives, not merely by a check in this module.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Sequence

from research_agent.policies import execution_policy

if TYPE_CHECKING:
    from research_agent.models import ArtifactRecord, ExperimentSpec


class ArtifactPolicyViolation(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _normalize(path: str) -> str:
    return PurePosixPath(path).as_posix()


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    from fnmatch import fnmatch

    norm = _normalize(path)
    for pattern in patterns:
        pat = _normalize(pattern)
        if norm == pat or norm.startswith(pat.rstrip("/") + "/"):
            return True
        if fnmatch(norm, pat):
            return True
    return False


def _sha256_if_small(path: Path, *, max_bytes: int) -> str | None:
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


def scan_artifacts(artifacts_dir: Path, spec: "ExperimentSpec") -> tuple[list["ArtifactRecord"], list[str]]:
    """Walk `artifacts_dir` (a fresh, run-scoped directory) and return
    (records, violations). `records` covers every entry found, including
    ones a violation was raised for, so the manifest remains a complete
    forensic record even on failure. `violations` is empty iff the artifact
    set is fully policy-compliant."""
    from research_agent.models import ArtifactRecord

    artifacts_dir = Path(artifacts_dir).resolve()
    execution = spec.execution
    allowed_output_paths = list(execution.allowed_output_paths) if execution else []
    max_files = execution.limits.max_artifact_files if execution else 10
    max_total_bytes = execution.limits.max_artifact_bytes if execution else 2_000_000
    max_file_bytes = execution.limits.max_artifact_file_bytes if execution else 1_000_000

    records: list[ArtifactRecord] = []
    violations: list[str] = []
    total_bytes = 0

    if not artifacts_dir.exists():
        return records, violations

    for dirpath, dirnames, filenames in os.walk(artifacts_dir, followlinks=False):
        dirnames.sort()
        current = Path(dirpath)
        if ".git" in dirnames:
            rel_git = (current / ".git").relative_to(artifacts_dir).as_posix()
            violations.append(f"nested Git repository detected under artifacts directory: {rel_git}")
            dirnames.remove(".git")

        for name in sorted(dirnames) + sorted(filenames):
            abs_path = current / name
            try:
                rel_path = abs_path.relative_to(artifacts_dir).as_posix()
            except ValueError:
                continue

            try:
                st = os.lstat(abs_path)
            except OSError as e:
                violations.append(f"could not stat artifact {rel_path}: {e}")
                continue

            if stat.S_ISLNK(st.st_mode):
                try:
                    target = os.readlink(abs_path)
                except OSError:
                    target = None
                records.append(ArtifactRecord(relative_path=rel_path, artifact_type="symlink", symlink_target=target))
                try:
                    resolved = abs_path.resolve()
                    resolved.relative_to(artifacts_dir)
                except ValueError:
                    violations.append(f"symlink escapes artifacts directory: {rel_path} -> {target}")
                continue

            if stat.S_ISFIFO(st.st_mode):
                violations.append(f"FIFO (named pipe) is not permitted as an artifact: {rel_path}")
                continue
            if stat.S_ISSOCK(st.st_mode):
                violations.append(f"Unix domain socket is not permitted as an artifact: {rel_path}")
                continue
            if stat.S_ISBLK(st.st_mode) or stat.S_ISCHR(st.st_mode):
                violations.append(f"device file is not permitted as an artifact: {rel_path}")
                continue

            if stat.S_ISDIR(st.st_mode):
                records.append(ArtifactRecord(relative_path=rel_path, artifact_type="dir"))
                continue

            if not stat.S_ISREG(st.st_mode):
                violations.append(f"artifact is not a regular file, directory, or symlink: {rel_path}")
                continue

            size = st.st_size
            total_bytes += size
            if size > max_file_bytes:
                violations.append(f"artifact {rel_path} is {size} bytes, exceeding max_artifact_file_bytes={max_file_bytes}")
            if allowed_output_paths and not _matches_any(rel_path, allowed_output_paths):
                violations.append(f"artifact {rel_path} is outside allowed_output_paths={allowed_output_paths}")
            if os.access(abs_path, os.X_OK):
                violations.append(f"artifact {rel_path} is an executable file, which is not expected in this MVP4 build")

            sha = _sha256_if_small(abs_path, max_bytes=execution_policy.MAX_HASH_BYTES)
            records.append(ArtifactRecord(relative_path=rel_path, artifact_type="file", size_bytes=size, mtime_ns=st.st_mtime_ns, sha256=sha))

    file_count = sum(1 for r in records if r.artifact_type == "file")
    if file_count > max_files:
        violations.append(f"{file_count} artifact files exceeds max_artifact_files={max_files}")
    if total_bytes > max_total_bytes:
        violations.append(f"{total_bytes} total artifact bytes exceeds max_artifact_bytes={max_total_bytes}")

    return records, violations
