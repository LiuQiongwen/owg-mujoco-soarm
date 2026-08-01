# Task: TANGO Experiment Agent MVP 4 -- restricted, pre-approved experiment execution

## Status

Complete. Base commit `7f4032f` on `feat/tango-agent-restricted-execution`. Builds directly
on MVP0-3 (`.agents/tasks/active/tango_agent_mvp0.md`, `tango_agent_mvp1.md`, and the
MVP2/MVP3 work already merged into this base commit: `research_agent/execute_flow.py`,
`research_agent/repair_flow.py`) -- this file only tracks what MVP4 added on top of them.

## Goal

Add a restricted experiment runner that can safely execute one or more pre-approved,
harmless, CPU-only, short-running, non-confirmatory experiment commands after planning,
policy validation, and an explicit `--execute` gate. Confirmatory execution stays out of
scope entirely (`CONFIRMATORY_EXECUTION_REQUIRES_MVP5_APPROVAL`).

## What changed

- `research_agent/models.py`: `MetricCheck`, `ExecutionLimits`, `RetryPolicy`,
  `ExecutionSpec` (embedded as `ExperimentSpec.execution: Optional[ExecutionSpec] = None`,
  additive -- every MVP0-3 spec keeps validating unchanged). `ExecutionSpec`'s own
  Pydantic validators enforce the MVP4 safety invariants (`cpu_only=true`,
  `network_allowed=gpu_allowed=robot_allowed=training_allowed=false`,
  `execution_mode != "confirmatory"`, `confirmatory != true`, `approved_commands` argv-array
  shape and within `limits.max_commands`) so an unsafe spec never even loads. Also added:
  `ExecutionCommandResult`, `ArtifactRecord`, `ExecutionVerifierResult`, `ExecutionAttempt`,
  `ExecutionState`/`EXECUTION_TERMINAL_STATES`/`EXECUTION_ACTIVE_STATES`,
  `ExecutionRunStateRecord`, `ExecutionFinalReport`, `ExecutionRunPaths` (extends
  `RepairRunPaths`, reusing its attempts/diagnoses/repairs/state.json/final_report.json
  layout unchanged, adding `execution/`, `execution_cwd/`, `artifact_manifest.json`,
  `metrics.json`, `verifier.json`, `failure.json`). `FailureClass` gained
  `EXECUTION_NONZERO_EXIT` (retriable), `EXECUTION_TIMEOUT`, `ARTIFACT_POLICY_FAILURE`
  (both non-retriable).
- `research_agent/failure_taxonomy.py`: `EXECUTION_NONZERO_EXIT` added to
  `RETRIABLE_FAILURE_CLASSES`; `EXECUTION_TIMEOUT`/`ARTIFACT_POLICY_FAILURE` documented as
  non-retriable; all three given an explicit `terminal_state_for` fallback entry.
- `research_agent/policies/experiment_commands.py` (new): layered command authorization --
  argv shape, shell-metacharacter scan (with an explicit, narrow exemption for the single
  argv element following `-c`, since a `python -c <script>` script body legitimately
  contains semicolons/braces/etc.), a two-item `ALLOWED_EXECUTABLE_BASENAMES =
  {"python", "python3"}` allowlist, a forbidden-executable/token/substring scan covering
  every category the task listed (shells, sudo, docker/podman, curl/wget/nc, pip/conda/apt,
  git, scp/ssh, GPU/CUDA/nvidia-smi, robot/serial/ROS, training entry points), a
  network-indicator (URL scheme / proxy-flag) scan, and byte-for-byte exact-match against
  `spec.execution.approved_commands`. `authorize_execution()` is the single entry point
  `execution_flow.py` calls before spawning anything; `validate_approved_commands()` runs
  the same gates (minus exact-match) against every entry in the spec itself, at plan-policy
  time, so an unsafe "approved" command is caught before execution is ever attempted.
- `research_agent/policies/environment_policy.py` (new): builds the restricted subprocess's
  entire environment from a small fixed allowlist (`PATH`, `PYTHONPATH`,
  `PYTHONUNBUFFERED`, locale vars) plus `spec.execution.environment_allowlist` -- never a
  copy of `os.environ`. Always strips proxy variables and anything matching a
  sensitive-name marker (KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL/AUTH/WANDB/SSH/PROXY/...),
  always forces `CUDA_VISIBLE_DEVICES=""`, `NVIDIA_VISIBLE_DEVICES=""`,
  `WANDB_MODE=disabled`, and always injects `RESEARCH_AGENT_RUN_DIR`/
  `RESEARCH_AGENT_ARTIFACTS_DIR`. `environment_allowlist`/`environment_overrides` both
  independently reject a sensitive-looking name at spec-validation time too (belt and
  suspenders). Explicitly documented as policy-based, not kernel-enforced, network
  isolation.
- `research_agent/policies/artifact_policy.py` (new): fresh filesystem walk of the run's
  assigned artifacts directory after execution -- never trusts a command's own claim.
  Resolves every entry's real path and rejects a symlink that escapes the artifacts
  directory, rejects FIFOs/sockets/device files, rejects a nested `.git` directory, and
  enforces `max_artifact_files`/`max_artifact_bytes`/`max_artifact_file_bytes` and
  `allowed_output_paths`. Produces the immutable `artifact_manifest.json` (type, size,
  mtime_ns, sha256, symlink target per entry).
- `research_agent/restricted_subprocess.py` (new): the ONLY place an approved experiment
  command is spawned. `shell=False`, its own process group (`start_new_session=True`) so a
  timeout/terminate/kill always reaches every child, `stdin=DEVNULL`, and on Linux a
  `preexec_fn` applying `RLIMIT_CPU`/`RLIMIT_AS`/`RLIMIT_FSIZE`/`RLIMIT_NPROC`/
  `RLIMIT_NOFILE` (best-effort process-level limits -- explicitly NOT container isolation).
  Terminate-then-kill escalation always targets the whole process group via `os.killpg`.
  Captured stdout/stderr truncated at `max_output_bytes`, with the truncation flag
  recorded. Never raises for a nonzero exit or timeout -- both are recorded on the returned
  `ExecutionCommandResult`; only a genuine launch failure raises. Persists
  `command.json`/`environment.json` (names only, never values)/`limits.json`/
  `stdout`/`stderr`/`exit_code`/`result.json` per command.
- `research_agent/tasks/metric_verifier.py` (new): deterministic verification of
  `spec.execution.required_artifacts`/`.required_metrics` against actual files under the
  run's artifacts directory -- `exists`/`bool_equals`/`str_equals`/`int_equals`/
  `int_range`/`float_equals` (with explicit tolerance)/`float_range`/`type_is` checks.
  Command-execution and artifact-policy outcomes are passed in by the caller as
  `command_issues`/`policy_issues`, so this module's `issues` list (and hence `verdict`) is
  the single place a run's pass/fail is decided.
- `research_agent/tasks/experiment_execution.py` (new): `init_execution_run`/
  `persist_execution_state` -- MVP4's run-directory bookkeeping, mirroring
  `tasks/repair.py`'s `init_repair_run`/`persist_state` exactly, extended with the
  `execution/`/`execution_cwd/` directories.
- `research_agent/execution_flow.py` (new, ~600 lines): the MVP4 orchestrator --
  `run_experiment_flow()`. PLANNING (Codex, mock by default) -> plan-policy validation ->
  PLAN_VALIDATED -> a shared, isolated execution worktree is always created (for
  isolation, even when no implementation phase runs) -> an OPTIONAL single pre-execution
  implementation attempt (real/mock Claude executor + the MVP3 deterministic content
  verifier, reusing `execution_worktree.py`/`execution_policy.py`/
  `tasks/repair_verification.py` unchanged; skipped entirely when the spec configures no
  `expected_file_contents`/`required_artifacts`/`allowed_executor_commands`, which is the
  common case and what the live-validation spec uses) -> READY_FOR_EXECUTION /
  EXECUTION_NOT_REQUESTED gate (nothing below this line runs without `--execute`) ->
  EXECUTING (every command independently re-authorized via
  `experiment_commands.authorize_execution` immediately before
  `restricted_subprocess.run_restricted_command` spawns it; cwd is either the isolated
  `execution_cwd/` directory or the execution worktree, per
  `working_directory_policy`) -> COLLECTING_ARTIFACTS -> VERIFYING_RESULTS -> on a
  retriable execution/verification failure (`EXECUTION_NONZERO_EXIT` or a metric/artifact
  mismatch) AND a pre-execution implementation scope configured, a bounded
  (Codex diagnosis -> Claude repair -> re-verify implementation ->
  RETRYING_EXECUTION, re-running the exact same approved command) loop follows, governed
  by `execution.limits.max_repair_rounds`/`max_total_codex_invocations`/
  `max_total_claude_invocations`/`max_execution_attempts`/`max_wall_clock_seconds`/
  `max_total_command_runtime_seconds`. A confirmatory-looking spec (task_id/goal/approved
  command naming `confirmatory`/`paper_final`/`final_result`/a final-results directory) is
  rejected via `ConfirmatoryRejected` as defense in depth beyond `ExecutionSpec`'s own
  validators. `state.json` is rewritten atomically after every transition; the outer
  `try/except` guarantees a terminal state is always persisted before the function returns,
  even on an unexpected internal exception.
- `research_agent/cli.py`: added `run-experiment` (`--codex {mock,real}`,
  `--claude {mock,real}`, both defaulting to mock; `--execute`, defaulting to False, the
  only way to run anything; no generic `--agents` flag), `experiment-status`, and
  `experiment-cleanup` (a thin alias for `worktree-cleanup`). `_load_recorded_report`
  extended to distinguish `run-experiment` (`ExecutionFinalReport`) from `repair`
  (`RepairFinalReport`) even though both write to `final_report.json` -- both models are
  `extra="forbid"`, so the two schemas are mutually exclusive and the distinction is never
  ambiguous for a well-formed report (an ambiguous/malformed report defaults to the
  pre-existing `"repair"` label purely for backward-compatible metadata; every caller
  already refuses whenever `report_data is None`, regardless of that label).
  `_recorded_changed_fingerprints_for_run` extended with a `run-experiment` branch reusing
  the same `attempts/attempt_NN/changed_file_manifest.json` layout `repair` uses. This
  means the EXISTING `worktree-status`/`worktree-cleanup` commands now also transparently
  understand MVP4 runs, per the task's "reuse existing worktree cleanup" instruction.
- `experiments/example_mvp4_restricted.yaml` (new): the designated single harmless live
  validation spec -- one approved `python -c <fixed script>` command, no pre-execution
  implementation scope, writing `metrics.json = {"restricted_execution_ok": true, "value":
  1.0}` under the run's artifacts directory.
- Tests (new, 122 tests, all offline/deterministic, no GPU/robot/network/research code):
  `tests/test_experiment_commands_policy.py` (32), `tests/test_environment_policy.py` (9),
  `tests/test_artifact_policy.py` (10), `tests/test_restricted_subprocess.py` (11),
  `tests/test_metric_verifier.py` (14), `tests/test_execution_flow.py` (36).

## Bugs found and fixed during live validation

1. `research_agent/cli.py::_load_recorded_report`: `except ValidationError as repair_err:
   pass` followed by a later reference to `repair_err` -- Python deletes an `except ... as
   name` binding at the end of its own except block (PEP 3110), so the later reference
   raised `UnboundLocalError` on the "malformed final_report.json" path. Fixed by capturing
   `str(repair_err)` into a plain variable before the block exits.
2. `execution_flow.py`'s plan-policy-validation step originally checked every entry in
   `plan.proposed_commands` only against the MVP2/3 executor-command allowlist (which
   explicitly forbids the `-c` token, since Claude's IMPLEMENTATION role is restricted to
   `compileall`/`pytest`/`git status`). A real Codex planner run correctly echoed the
   approved `python -c <script>` MVP4 experiment command into `proposed_commands` (a
   reasonable reading of the prompt), which then failed that unrelated MVP2/3 gate with
   `CLAUDE_COMMAND_POLICY_FAILED: forbidden token '-c'`. Fixed by also accepting a proposed
   command that exactly matches an entry in `spec.execution.approved_commands` -- this does
   not change what is ever authorized to execute (that is decided independently, later, by
   `experiment_commands.authorize_execution` at EXECUTING time), only what the advisory
   plan-cleanliness check accepts.

## Live validation

`python -m research_agent.cli run-experiment experiments/example_mvp4_restricted.yaml
--codex real --claude mock --execute` -> `final_state=PASS`, `overall_status=PASS`,
one command executed (`python -c <script>`, resolved to the project's venv Python,
duration 0.024s), one artifact (`metrics.json`, sha256-recorded, matching
`{"restricted_execution_ok": true, "value": 1.0}` exactly), main worktree unchanged,
execution worktree left at its base commit (no writes, no commits -- the live spec used
`working_directory_policy: isolated_run_directory`). Run id `live_mvp4_restricted_2`
(the first attempt, `live_mvp4_restricted_1`, hit bug #2 above and was removed after the
fix).

## Verification

- `python -m compileall -q research_agent tests` -- clean.
- Full research-agent suite (MVP0-3 regression + all new MVP4 tests): 292 passed.
- `git diff --check` -- clean.
- Changed files confined to `research_agent/`, `tests/`, `experiments/`,
  `.agents/tasks/active/` -- no forbidden path touched.
