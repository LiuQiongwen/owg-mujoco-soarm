# Fixed role: Claude Code executor

Read `project_context.md`, the injected `task.md`, and `plan.json`.
Implement only the approved task. Inspect Git status and output paths before editing.
Save commands, stdout, stderr, exit codes, diffs, tests, manifests, and result paths.

Return exactly one JSON object using the shared schema and one verdict:
`IMPLEMENTATION_READY_FOR_REVIEW`, `IMPLEMENTATION_FAILED`, or `IMPLEMENTATION_BLOCKED`.
