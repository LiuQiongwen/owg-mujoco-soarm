# Local Research Agent Workflow

This directory orchestrates Codex planning, Claude Code implementation, independent
Shell/Python verification, and Codex review. It does not deploy services or run
research experiments by itself.

Each run has a unique `run_id` and stores machine-readable verdicts, command output,
exit codes, Git snapshots, and protected-data manifests. Existing user changes are
snapshotted before an agent is invoked and are never counted as agent changes.

The default workflow is exactly:

1. Codex planner (once).
2. Claude Code executor.
3. Independent verifier.
4. Codex reviewer.

Later rounds only repair `REVISE` findings. A planner is rerun only when the reviewer
emits `REPLAN_REQUIRED`. Confirmatory jobs require a lock under `.agents/locks/`.

Use the dry-run task under `tasks/active/` to test orchestration without changing
research code or running experiments.

Canonical dry-run command (the run ID is passed only through the `RUN_ID` environment
variable; there is no second positional run-ID syntax):

```bash
RUN_ID=dryrun_001 bash .agents/scripts/orchestrate.sh \
  .agents/tasks/active/dry_run_workflow.md --dry-run
```

Acceptance tests are listed in `TESTING.md`. The current read-only environment is
`BLOCKED_BY_READ_ONLY_ENVIRONMENT`; no PASS result should be inferred until those
tests run in a writable workspace.

## Fixed prompt composition

The orchestrator creates combined prompts inside the run directory only:

```text
project_context.md + planner.md + task.md
    -> combined_planner_prompt.md -> plan.json

project_context.md + executor.md + task.md + plan.json
    -> combined_executor_prompt.md -> implementation.json

task.md + run artifacts
    -> verify.py -> verifier.json

project_context.md + reviewer.md + task.md + plan.json
  + implementation.json + verifier.json + git diff
    -> combined_reviewer_prompt.md -> review.json
```

The current scaffold only creates the planner composition and performs CLI discovery;
it does not invoke Codex or Claude. CLI help is saved as separate stdout, stderr, and
exit-code files and summarized in `cli_capabilities.json`.
