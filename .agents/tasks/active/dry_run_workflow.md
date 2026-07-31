# Dry-run workflow task

- task_id: dry-run-workflow
- type: workflow
- max_rounds: 1
- timeout_seconds: 5
- max_trials: 0
- max_seeds: 0

This task only checks CLI discovery, atomic JSON state creation, timeout configuration,
run-id collision rejection, and machine-readable dry-run verdicts. It must not modify
TANGO research code or run experiments.
