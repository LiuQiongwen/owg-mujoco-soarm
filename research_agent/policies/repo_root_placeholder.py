"""MVP4 repository-relative command portability: a single, narrowly-scoped
`${REPO_ROOT}` placeholder that trusted Python code (research_agent
.execution_flow) expands into an absolute path BEFORE any command policy gate
or the exact-match authorization check ever runs -- never a shell, never
os.path.expandvars, never os.environ.

Why this exists: an ExperimentSpec's `execution.approved_commands` is
committed to the repository and must work unchanged from the main repository
checkout, from this development worktree, and from any future Git worktree --
none of which share a path. A literal absolute path baked into the YAML (e.g.
`/lena/projects/OWG-agent-pilot1/research_agent_pilots/.../run_pilot.py`)
breaks the moment that specific worktree is removed. `${REPO_ROOT}` defers
that one directory component to whatever `--repo-root` (default: cwd) the
`run-experiment` CLI is actually invoked with -- see research_agent.cli's
`--repo-root` default of "." and research_agent.execution_flow.run_experiment_flow's
`repo_root` parameter, both already fully caller-supplied and worktree-agnostic.

Two distinct forms of every approved command exist from here on:
  declared  -- exactly what's committed in the YAML (may contain ${REPO_ROOT})
  resolved  -- declared with ${REPO_ROOT} substituted for the real, current
               repo_root, verified to still resolve inside it

Every one of the existing command-policy gates (research_agent.policies
.experiment_commands: shape / shell-metacharacter / executable-allowlist /
forbidden-token / network-indicator / exact-match) runs ONLY against the
resolved form -- by the time authorize_execution ever sees a command, no
`${...}` syntax remains in it. That keeps this module the single place that
understands placeholder syntax at all, and keeps every existing regression
test for those gates (written against plain literal commands) unaffected.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

# The ONLY placeholder this build understands. Adding another requires a
# deliberate code change here -- there is no generic/extensible mechanism,
# by design (see class docstring: "narrowly scoped").
KNOWN_PLACEHOLDERS = frozenset({"REPO_ROOT"})

# ${NAME} -- braces required (never bare $NAME/$(...)), so this can never be
# confused with, or accidentally satisfy, shell parameter/command-substitution
# syntax. NAME is restricted to the same identifier shape env-var names use.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PlaceholderPolicyViolation(RuntimeError):
    """Carries a deterministic `.code`, mirroring
    research_agent.policies.experiment_commands.ExperimentCommandPolicyViolation
    and research_agent.policies.execution_policy.ExecutionPolicyViolation, so
    callers never have to string-sniff a message to decide which terminal
    code this maps to."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _reject_unknown_placeholders(token: str) -> None:
    for match in _PLACEHOLDER_RE.finditer(token):
        name = match.group(1)
        if name not in KNOWN_PLACEHOLDERS:
            raise PlaceholderPolicyViolation(
                "UNKNOWN_PLACEHOLDER",
                f"token {token!r} references undeclared placeholder \"${{{name}}}\"; "
                f"only {sorted(KNOWN_PLACEHOLDERS)} are recognized",
            )


def _reject_traversal(token: str) -> None:
    """Reject a literal '..' path component anywhere in the DECLARED token
    (before substitution) -- a spec author must never be able to write
    `${REPO_ROOT}/../../etc/passwd` and have it silently normalized away.
    Checked on the raw string, not a parsed Path, so this also catches a
    traversal component that appears before the placeholder itself."""
    for part in token.replace("\\", "/").split("/"):
        if part == "..":
            raise PlaceholderPolicyViolation(
                "PLACEHOLDER_PATH_TRAVERSAL_REJECTED",
                f"token {token!r} contains a '..' path component, which is never permitted",
            )


def resolve_token(token: str, repo_root: Path) -> str:
    """Expand ${REPO_ROOT} in a single argv token via plain Python string
    substitution -- never a shell, never an environment-variable lookup.
    Tokens with no placeholder pass through completely unchanged (including
    any literal, unrelated '$' -- those are left for the ordinary
    shell-metacharacter gate to reject downstream, exactly as for a command
    that never mentions ${REPO_ROOT} at all).

    Deliberately narrow: ${REPO_ROOT} is only accepted as the TOKEN'S OWN
    PREFIX (index 0) -- i.e. the token is ${REPO_ROOT} itself or
    ${REPO_ROOT}/some/relative/path. This keeps the whole substituted token
    unambiguously a single filesystem path, so the containment check below
    (does it resolve inside repo_root?) is exact rather than a guess about
    which part of a mixed token (e.g. `--out=${REPO_ROOT}/out`) is a path.
    A spec never needs anything richer than this for an argv script-path
    token, which is the only use case this mechanism exists for."""
    if "${" not in token:
        return token
    _reject_unknown_placeholders(token)
    if not token.startswith("${REPO_ROOT}"):
        raise PlaceholderPolicyViolation(
            "PLACEHOLDER_MUST_BE_TOKEN_PREFIX",
            f"token {token!r}: \"${{REPO_ROOT}}\" must be the start of the token "
            f"(e.g. \"${{REPO_ROOT}}/path/to/script.py\"), not embedded partway through it",
        )
    _reject_traversal(token)
    resolved = _PLACEHOLDER_RE.sub(lambda m: str(repo_root), token)

    # Containment check -- mirrors execution_policy
    # .assert_within_worktree_no_symlink_escape's resolve-then-relative_to
    # pattern: resolve symlinks against the deepest existing ancestor (the
    # target file may not exist yet, e.g. at spec-validation time before a
    # worktree checkout), then confirm the result is still inside repo_root.
    root = repo_root.resolve()
    candidate = Path(resolved)
    existing = candidate
    remainder: list[str] = []
    while not existing.exists() and existing != existing.parent:
        remainder.insert(0, existing.name)
        existing = existing.parent
    real = existing.resolve().joinpath(*remainder) if remainder else existing.resolve()
    try:
        real.relative_to(root)
    except ValueError:
        raise PlaceholderPolicyViolation(
            "PLACEHOLDER_ESCAPES_REPO_ROOT",
            f"token {token!r} resolves to {real}, which escapes repo_root {root}",
        ) from None
    return resolved


def resolve_command(command: Sequence[str], repo_root: Path) -> list[str]:
    """Resolve every token of one argv command. repo_root must already be an
    absolute path (research_agent.execution_flow.run_experiment_flow always
    calls Path(repo_root).resolve() before this is ever invoked)."""
    return [resolve_token(tok, repo_root) for tok in command]
