# Agent Workflow Acceptance Tests

当前工作流状态：`BLOCKED_BY_READ_ONLY_ENVIRONMENT`。

以下测试必须在可写工作区执行。当前只读环境中未执行的项目必须标记为
`NOT_RUN`，不得伪造 PASS 输出。

## Test matrix

### 1. Normal dry-run

```bash
RUN_ID=dryrun_001 bash .agents/scripts/orchestrate.sh \
  .agents/tasks/active/dry_run_workflow.md --dry-run
```

Expected: exit code `0`, a new `.agents/runs/dryrun_001/` directory, and
`state.json.status == "PASS"` with no `*.tmp.*` files left behind.

### 2. JSON validity

```bash
python3 -m json.tool .agents/runs/dryrun_001/state.json
python3 -m json.tool .agents/runs/dryrun_001/cli_verdict.json
python3 -m json.tool .agents/runs/dryrun_001/protected_before.json
python3 -m json.tool .agents/runs/dryrun_001/protected_after.json
```

Expected: every command exits `0`.

### 3. Existing run ID is rejected

Run the normal dry-run twice with `RUN_ID=dryrun_001`.

Expected: second invocation exits non-zero and does not overwrite the first run.

### 4. Verifier failure blocks completion

Use a controlled protected-file mutation or deliberately failing verification command.
Expected: `verifier.json.verdict != PASS` and the task is not complete.

### 5. Reviewer PASS cannot override verifier FAIL

Create machine-readable fixtures with reviewer `PASS` and verifier `FAIL`.
Expected: final verdict is not `PASS`; `safe_to_stop` is false.

### 6. Timeout terminates a child process

```bash
timeout 1s sh -c 'sleep 5'
```

Expected: non-zero exit code, with stdout, stderr, and exit-code files saved.

### 7. Non-dry-run is refused

```bash
RUN_ID=non_dry_run_refusal bash .agents/scripts/orchestrate.sh \
  .agents/tasks/active/dry_run_workflow.md
```

Expected: non-zero exit code and explicit refusal; no experiment command runs.

### 8. Confirmatory lock prevents repeat execution

```bash
RUN_ID=confirmatory_lock_001 CONFIRMATORY=1 bash .agents/scripts/orchestrate.sh \
  .agents/tasks/active/dry_run_workflow.md --dry-run
RUN_ID=confirmatory_lock_002 CONFIRMATORY=1 bash .agents/scripts/orchestrate.sh \
  .agents/tasks/active/dry_run_workflow.md --dry-run
```

Expected: second invocation refuses the existing
`.agents/locks/dry_run_workflow.lock`.

### 9. Separate command streams

Every run must contain separate files:

```text
<step>.stdout
<step>.stderr
<step>.exit_code
```

### 10. User Git diff is preserved

Compare the initial `git status --short` and binary diff with:

```text
.agents/runs/<run_id>/git_before.txt
.agents/runs/<run_id>/user_changes_before.patch
```

Expected: pre-existing user changes are not altered or reclassified as agent changes.

## Reporting rule

Each test must be reported as exactly one of:

```text
PASS
FAIL
NOT_RUN
```

The current environment requires filesystem-dependent tests to be reported as
`NOT_RUN` until `.agents/runs/` and `.agents/locks/` are writable.
