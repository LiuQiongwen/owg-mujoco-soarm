"""MVP3 failure taxonomy: which research_agent.models.FailureClass values are
retriable, and which research_agent.models.RepairState a repair run must
terminate in when a given failure_class is the reason the loop stops.

Both mappings live in exactly one place so "unknown/unclassified defaults to
non-retriable" (the task contract's explicit rule) and "every failure class
maps to exactly one terminal state" cannot drift apart across callers.
"""
from __future__ import annotations

from research_agent.models import FailureRecord

# Failure classes that MAY be retried (i.e., research_agent.repair_flow is
# permitted to proceed to DIAGNOSING/REPAIRING for these, budget permitting).
# Every class not in this set is non-retriable by construction -- including
# UNKNOWN_FAILURE, per the task contract's "Unknown or unclassified failures
# must default to non-retriable unless a safe explicit rule says otherwise."
RETRIABLE_FAILURE_CLASSES = frozenset({
    "TEST_FAILURE",
    "STATIC_CHECK_FAILURE",
    "ARTIFACT_MISSING",
    "ARTIFACT_MALFORMED",
    "EXPECTED_CONTENT_MISMATCH",
})

# The failure classes explicitly called out in the task contract's "Do not
# automatically retry" list, restated here for documentation/traceability;
# is_retriable() below is the actual gate (derived from the allowlist above,
# not this list) so there is exactly one source of truth.
_NEVER_RETRY_DOCUMENTED = frozenset({
    "PATH_POLICY_FAILURE",
    "COMMAND_POLICY_FAILURE",
    "MAIN_WORKTREE_CHANGED",
    "SYMLINK_ESCAPE",
    "NESTED_GIT",
    "GIT_METADATA_TAMPERING",
    "DIFF_MISMATCH",
    "IMPLEMENTATION_SCHEMA_FAILURE",
    "PLAN_SCHEMA_FAILURE",
    "CODEX_INFRASTRUCTURE_FAILURE",
    "CLAUDE_INFRASTRUCTURE_FAILURE",
    "WORKTREE_INVALID",
    "RETRY_LIMIT_REACHED",
    "UNKNOWN_FAILURE",
})
assert _NEVER_RETRY_DOCUMENTED.isdisjoint(RETRIABLE_FAILURE_CLASSES)


def is_retriable(failure_class: str) -> bool:
    return failure_class in RETRIABLE_FAILURE_CLASSES


# failure_class -> the RepairState a run must terminate in when the loop
# stops because of a failure of this class (either because it is
# non-retriable, or because the budget ran out while retrying a retriable
# one -- see repair_flow.py, which always maps budget exhaustion to
# RETRY_LIMIT_REACHED before calling this function).
_TERMINAL_STATE_FOR_FAILURE_CLASS: dict[str, str] = {
    "TEST_FAILURE": "BLOCKED",
    "STATIC_CHECK_FAILURE": "BLOCKED",
    "ARTIFACT_MISSING": "BLOCKED",
    "ARTIFACT_MALFORMED": "BLOCKED",
    "EXPECTED_CONTENT_MISMATCH": "BLOCKED",
    "DIFF_MISMATCH": "POLICY_FAILURE",
    "IMPLEMENTATION_SCHEMA_FAILURE": "BLOCKED",
    "PLAN_SCHEMA_FAILURE": "BLOCKED",
    "CODEX_INFRASTRUCTURE_FAILURE": "INFRASTRUCTURE_FAILURE",
    "CLAUDE_INFRASTRUCTURE_FAILURE": "INFRASTRUCTURE_FAILURE",
    "COMMAND_POLICY_FAILURE": "POLICY_FAILURE",
    "PATH_POLICY_FAILURE": "POLICY_FAILURE",
    "MAIN_WORKTREE_CHANGED": "POLICY_FAILURE",
    "WORKTREE_INVALID": "POLICY_FAILURE",
    "SYMLINK_ESCAPE": "POLICY_FAILURE",
    "NESTED_GIT": "POLICY_FAILURE",
    "GIT_METADATA_TAMPERING": "POLICY_FAILURE",
    "RETRY_LIMIT_REACHED": "RETRY_EXHAUSTED",
    "UNKNOWN_FAILURE": "BLOCKED",
}


def terminal_state_for(failure_class: str) -> str:
    """Never raises: an unrecognized failure_class (should be impossible --
    FailureRecord.failure_class is a Literal validated by Pydantic) still
    falls back to the safest terminal state, BLOCKED, rather than crashing
    the flow mid-shutdown."""
    return _TERMINAL_STATE_FOR_FAILURE_CLASS.get(failure_class, "BLOCKED")


def build_failure_record(
    *,
    task_id: str,
    run_id: str,
    attempt_index: int,
    failure_class: str,
    summary: str,
    evidence: list[str] | None = None,
    failed_checks: list[str] | None = None,
    recommended_action: str,
) -> FailureRecord:
    """The single place a FailureRecord is constructed: `retriable` is always
    derived from `failure_class` via is_retriable(), never passed in by the
    caller, so it can never drift from the taxonomy above."""
    return FailureRecord(
        task_id=task_id,
        run_id=run_id,
        attempt_index=attempt_index,
        failure_class=failure_class,
        summary=summary,
        evidence=evidence or [],
        failed_checks=failed_checks or [],
        retriable=is_retriable(failure_class),
        recommended_action=recommended_action,
    )
