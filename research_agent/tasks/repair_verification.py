"""MVP3 deterministic content verifier: the RETRIABLE half of the repair
flow's verification step.

Deliberately narrow scope -- checks only expected_file_contents,
required_artifacts, and static/targeted-test checks (research_agent.models
.ExperimentSpec.expected_file_contents / .required_artifacts /
.required_executor_checks). Every failure this module can produce maps to a
RETRIABLE research_agent.failure_taxonomy failure_class (ARTIFACT_MISSING,
EXPECTED_CONTENT_MISMATCH, STATIC_CHECK_FAILURE, TEST_FAILURE) -- see
failure_taxonomy.RETRIABLE_FAILURE_CLASSES.

The NON-retriable policy gates (path/symlink/nested-git escape, command
policy, main-worktree/`.git`-pointer tampering, implementation-claim vs.
actual-diff consistency, changed-file/byte limits) are deliberately NOT
checked here: they are checked earlier and separately by
research_agent.repair_flow, directly against
research_agent.policies.execution_policy / research_agent.execution_worktree
/ research_agent.tasks.repository -- exactly mirroring
research_agent.execute_flow's structure -- because those must always
short-circuit the loop immediately, before this function is ever called.
Never trusts Claude's or Codex's own claims -- only actual file content on
disk and actual subprocess exit codes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

from research_agent import subprocess_runner
from research_agent.models import CommandResult, ExperimentSpec, VerifierResult
from research_agent.policies import execution_policy


class ContentFailureBreakdown(NamedTuple):
    """Categorized failed-check names, in the fixed priority order
    repair_flow.py uses to pick the single primary failure_class for a
    verifier FAIL (ARTIFACT_MISSING first, since nothing else can be
    meaningfully checked about a file that was never even created)."""

    artifact_missing: list[str]
    content_mismatch: list[str]
    static_check_failed: list[str]
    test_failed: list[str]

    def is_empty(self) -> bool:
        return not (self.artifact_missing or self.content_mismatch or self.static_check_failed or self.test_failed)


def run_content_verifier(
    *,
    spec: ExperimentSpec,
    worktree_dir: Path,
    attempt_dir: Path,
    task_id: str,
    run_id: str,
    attempt_index: int,
    python_executable: str = sys.executable,
) -> tuple[VerifierResult, ContentFailureBreakdown, list[CommandResult]]:
    worktree_dir = Path(worktree_dir)
    command_results_dir = attempt_dir / "command_results"
    command_results_dir.mkdir(parents=True, exist_ok=True)

    checks_passed: list[str] = []
    checks_failed: list[str] = []
    details: list[str] = []
    artifact_missing: list[str] = []
    content_mismatch: list[str] = []
    static_check_failed: list[str] = []
    test_failed: list[str] = []
    static_results: list[CommandResult] = []

    # ── expected_file_contents ──────────────────────────────────────────
    for rel_path in sorted(spec.expected_file_contents):
        expected = spec.expected_file_contents[rel_path]
        check_name = f"expected_content:{rel_path}"
        abs_path = worktree_dir / rel_path
        if not abs_path.exists() or not abs_path.is_file():
            checks_failed.append(check_name)
            artifact_missing.append(check_name)
            details.append(f"required file missing: {rel_path}")
            continue
        try:
            actual = abs_path.read_text()
        except (OSError, UnicodeDecodeError) as e:
            checks_failed.append(check_name)
            content_mismatch.append(check_name)
            details.append(f"could not read {rel_path} as text: {e}")
            continue
        if actual.rstrip("\n") != expected.rstrip("\n"):
            checks_failed.append(check_name)
            content_mismatch.append(check_name)
            details.append(f"content mismatch for {rel_path}: expected {expected!r}, got {actual!r}")
        else:
            checks_passed.append(check_name)

    # ── required_artifacts (existence only) ─────────────────────────────
    for rel_path in spec.required_artifacts:
        check_name = f"required_artifact:{rel_path}"
        if rel_path in spec.expected_file_contents:
            continue  # already covered above, more strictly
        abs_path = worktree_dir / rel_path
        if not abs_path.exists():
            checks_failed.append(check_name)
            artifact_missing.append(check_name)
            details.append(f"required artifact missing: {rel_path}")
        else:
            checks_passed.append(check_name)

    # ── static checks: compileall + required_executor_checks pytest targets ─
    targets_spec = spec.required_executor_checks or ["compileall"]
    for i, check in enumerate(targets_spec):
        check_name = f"static_check:{check}"
        try:
            if check == "compileall":
                targets = [str(worktree_dir / p) for p in spec.allowed_paths if (worktree_dir / p).exists()]
                if not targets:
                    targets = [str(worktree_dir)]
                command = [python_executable, "-m", "compileall", "-q", *targets]
                result = subprocess_runner.run_command(
                    command, cwd=worktree_dir, run_dir=command_results_dir,
                    name=f"attempt{attempt_index:02d}_static_check_{i}", timeout=spec.timeouts.verifier_seconds,
                )
                bucket = static_check_failed
            elif check.startswith("pytest:"):
                target = check.split(":", 1)[1]
                command = [python_executable, "-m", "pytest", "-q", target]
                execution_policy.assert_executor_command_allowed(command, spec)
                result = subprocess_runner.run_command(
                    command, cwd=worktree_dir, run_dir=command_results_dir,
                    name=f"attempt{attempt_index:02d}_static_check_pytest_{i}", timeout=spec.timeouts.verifier_seconds,
                )
                bucket = test_failed
            else:
                checks_failed.append(check_name)
                static_check_failed.append(check_name)
                details.append(f"unknown required_executor_checks entry: {check!r}")
                continue
        except subprocess_runner.InfrastructureError as e:
            checks_failed.append(check_name)
            static_check_failed.append(check_name)
            details.append(f"{check_name} raised an infrastructure error: {e}")
            continue
        except execution_policy.ExecutionPolicyViolation as e:
            checks_failed.append(check_name)
            static_check_failed.append(check_name)
            details.append(f"{check_name} is not an approved command: {e}")
            continue

        static_results.append(result)
        if result.returncode != 0 or result.timed_out:
            checks_failed.append(check_name)
            bucket.append(check_name)
            details.append(f"{check_name} failed: returncode={result.returncode} timed_out={result.timed_out}")
        else:
            checks_passed.append(check_name)

    breakdown = ContentFailureBreakdown(
        artifact_missing=artifact_missing, content_mismatch=content_mismatch,
        static_check_failed=static_check_failed, test_failed=test_failed,
    )
    verdict = "FAIL" if checks_failed else "PASS"
    verifier_result = VerifierResult(
        task_id=task_id, run_id=run_id, attempt_index=attempt_index, verdict=verdict,
        checks_passed=checks_passed, checks_failed=checks_failed, details=details,
    )
    return verifier_result, breakdown, static_results


def primary_failure_class(breakdown: ContentFailureBreakdown) -> str:
    """Fixed priority: a missing artifact/file is reported before a content
    mismatch (nothing else is meaningful to check about a file that does
    not exist), then static checks, then targeted tests."""
    if breakdown.artifact_missing:
        return "ARTIFACT_MISSING"
    if breakdown.content_mismatch:
        return "EXPECTED_CONTENT_MISMATCH"
    if breakdown.static_check_failed:
        return "STATIC_CHECK_FAILURE"
    if breakdown.test_failed:
        return "TEST_FAILURE"
    return "UNKNOWN_FAILURE"
