# Task: TANGO Experiment Agent MVP 1 -- real Codex read-only planning

## Status

Complete. Base commit `d367edc` on `feat/tango-agent-real-codex`. Builds directly on
`.agents/tasks/active/tango_agent_mvp0.md` (Prefect architecture, mock adapters, policy
engines) -- see that file for the pre-existing MVP0 scope; this file only tracks what
MVP1 added on top of it.

## Goal

Replace only the mock Codex **planner** with an optional real Codex read-only planner.
The mock Codex reviewer and the mock Claude executor are unchanged and remain the only
adapters used for those two roles in this MVP -- "replace only the mock Codex planner"
is this task's explicit scope; the real Claude adapter stays disconnected entirely.

## What changed

- `research_agent/agents/codex.py`: `RealCodexAgent.plan()` rewritten to actually invoke
  `codex exec --sandbox read-only --ephemeral --output-schema <path> --output-last-message
  <path> -` (shell=False, argv array, prompt via stdin), persist the full artifact set
  (`prompts/codex_planner_prompt.md`, `commands/codex_planner.{command.json,stdout,stderr,
  exit_code,result.json}`, `plan.raw.json`, `plan.schema.json`), and classify every failure
  mode into one of the CODEX_* terminal codes below. `CodexPlannerError` added for
  non-infrastructure failures (never retried). `_strict_output_schema()` forces `required`
  to list every property (the real Codex CLI forwards `--output-schema` as an OpenAI
  structured-output `response_format`, which -- unlike Pydantic's own schema -- requires
  every property to be listed as required; confirmed against a live invocation).
- `research_agent/subprocess_runner.py`: `ExecutableNotFoundError` / `CommandTimeoutError`
  added as `InfrastructureError` subclasses (additive; existing `isinstance`/`pytest.raises`
  checks and the flow's retry_condition_fn are unaffected).
- `research_agent/models.py`: `PlanResult` gained `expected_artifacts`, `expected_metrics`,
  `risks`, `assumptions` (all default `[]`, so MVP0 plan construction is unaffected).
  `RunPaths` gained `prompts_dir`, `plan_raw_path`, `plan_schema_path`.
- `research_agent/policies/commands.py`: `MAX_PROPOSED_COMMANDS = 1` (command-count-limit
  gate) and duplicate-proposed-command detection added to `validate_plan_commands`; `bash`
  and `sh` added to `FORBIDDEN_EXACT_TOKENS`.
- `research_agent/tasks/repository.py`: `capture_repo_fingerprint()` /
  `diff_fingerprints()` -- an independent (never merely Codex's own claim) before/after
  check of HEAD, `git status --porcelain=v1 --ignored=matching`, tracked diff, and staged
  diff, excluding only the immutable run directory and known Prefect runtime files.
- `research_agent/tasks/reporting.py`: `save_json_artifact()` now writes via
  temp-file-plus-`os.replace` (atomic), per the plan.json requirement.
- `research_agent/flow.py`: planner prompt expanded to carry every required field (task
  ID, run ID, repo root, allowed/forbidden paths, approved command arrays, required
  artifacts/metrics, command-count limit, mode, explicit safety restrictions);
  `codex_plan_task` now wraps the Codex call with the before/after repository-fingerprint
  check; `_run_codex_planner` / `_synthetic_blocked_plan` turn every possible Codex failure
  (infrastructure or research/policy) into a terminal `PLAN_BLOCKED` PlanResult, never an
  uncaught exception; `smoke_flow` gained a `codex_reviewer_agent` parameter that defaults
  to a fresh `MockCodexAgent()` whenever `codex_agent` is a `RealCodexAgent` (so the
  reviewer role stays mock even when `--codex real` is selected, matching this task's
  scope), while still reusing a single passed-in `MockCodexAgent` for both roles when both
  are mock (preserving exact MVP0 behavior for that path).
- `research_agent/cli.py`: `--agents {mock,real}` replaced by independent `--codex
  {mock,real}` (both `plan` and `smoke`) and `--claude {mock,real}` (`smoke` only).
  `--claude real` is rejected deterministically with `REAL_CLAUDE_DISABLED_IN_MVP1` before
  anything runs (no run directory is even created).
- `tests/test_real_codex_planner.py` (new): a single fake `codex` executable
  (`FAKE_CODEX_SCENARIO`-driven) covering all 20 required scenarios from the task
  contract -- PLAN_PASS, executable-unavailable, timeout, nonzero exit, auth failure,
  missing/malformed output, unknown field, invalid verdict, task/run ID mismatch,
  forbidden/unapproved proposed command, shell-string proposed command,
  repository-unchanged / repository-mutation-detected, mock-Claude-stays-active,
  sentinel-`claude`-never-invoked, and CLI-level `--claude real` rejection (including
  combined with `--codex real`).
- `tests/test_command_policy.py`, `tests/test_smoke_flow.py`: updated for the new
  MAX_PROPOSED_COMMANDS/duplicate gates and the `--codex`/`--claude` CLI flags
  respectively; no MVP0 assertion was weakened, only adapted to the (explicitly
  requested) new CLI shape and stricter command-count policy.

## Terminal codes (never leaves a run without a persisted, terminal plan.json)

`CODEX_EXECUTABLE_UNAVAILABLE`, `CODEX_TIMEOUT`, `CODEX_AUTHENTICATION_FAILED`,
`CODEX_NONZERO_EXIT`, `CODEX_OUTPUT_MISSING`, `CODEX_OUTPUT_MALFORMED`,
`CODEX_SCHEMA_INVALID`, `CODEX_TASK_ID_MISMATCH`, `CODEX_RUN_ID_MISMATCH`,
`CODEX_PLAN_REVISE`, `CODEX_PLAN_BLOCKED`, `CODEX_COMMAND_POLICY_FAILED`,
`REPOSITORY_CHANGED_DURING_CODEX_PLANNING`, `REAL_CLAUDE_DISABLED_IN_MVP1`.

## Verification performed

- `python -m compileall -q research_agent tests` -- clean.
- `pytest tests/{test_smoke_flow,test_subprocess_runner,test_command_policy,
  test_path_policy,test_experiment_spec,test_real_codex_planner}.py` -- 92 passed
  (73 MVP0-lineage + 19 new MVP1).
- `git diff --check` -- clean.
- Default `python -m research_agent.cli smoke experiments/example_smoke.yaml` run against
  the real repo -- PASS, worktree/branch cleaned up, no leaked state.
- Live `python -m research_agent.cli plan experiments/example_smoke.yaml --codex real`
  against the real, authenticated `codex` CLI -- PLAN_PASS, full artifact set persisted,
  repository confirmed unchanged, no smoke command run, no Claude invoked.

## Limitations / MVP2 scope

- The real Claude adapter (`RealClaudeAgent`) remains implemented but entirely
  disconnected; MVP2 is the natural place to connect it (still worktree-isolated,
  still path-policy-checked after every call) behind its own explicit, narrow opt-in.
- `RealCodexAgent.review()` exists (mirrors the planner's command shape) but is never
  wired to a real invocation by the CLI in this MVP -- only the planner role is real by
  design; a future task could extend the same fingerprinting/validation pattern to the
  reviewer if that's ever brought in scope.
- The repository-fingerprint check is Git-plumbing-based (HEAD, status, diff, staged
  diff); it does not currently snapshot ignored-file *contents*, only `git status
  --ignored=matching` listing -- sufficient to catch new/removed ignored files but not a
  silent in-place edit of an already-ignored file's contents.

Do not commit or push.
