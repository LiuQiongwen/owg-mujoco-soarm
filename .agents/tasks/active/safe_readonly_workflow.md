# Safe Read-Only Workflow Test

task_id: safe_readonly_workflow

## Objective

Perform an end-to-end read-only agent workflow test for the OWG/TANGO repository.

## Allowed actions

- Read repository files and directory structure.
- Read Git status and diffs.
- Identify likely Python entry points.
- Identify existing test commands from repository files.
- Produce planning, implementation, verification, and review JSON artifacts.
- Save logs and exit codes inside the current run directory.

## Forbidden actions

- Do not modify repository files.
- Do not modify `.agents/` source files.
- Do not create files outside the current `.agents/runs/<run_id>/` directory.
- Do not run research experiments.
- Do not run GPU, robot, simulator, confirmatory, Docker, sudo, or git push commands.
- Do not install packages.
- Do not commit, checkout, reset, clean, stash, or alter Git state.
- Do not claim that tests or experiments were run unless supported by saved logs.

## Planner output

The planner should create a read-only inspection plan covering:

- repository structure;
- likely Python entry points;
- discoverable test commands;
- current Git state;
- required artifacts and validation criteria.

## Executor output

The executor should carry out only the approved read-only inspection and report:

- directories inspected;
- likely entry points found;
- test commands found;
- Git state observed;
- commands executed;
- confirmation that no repository modifications were made.

## Verification requirements

The verifier must independently confirm:

- all required JSON files are valid;
- CLI exit codes were recorded;
- stdout and stderr were saved separately;
- Git status, tracked diff, staged diff, and untracked file list are unchanged before and after;
- no files were created outside the run directory;
- no forbidden commands were executed.

## Completion conditions

The workflow may pass only when:

- plan.verdict == PLAN_PASS
- implementation.verdict == IMPLEMENTATION_READY_FOR_REVIEW
- verifier.verdict == PASS
- review.verdict == REVIEW_PASS

A reviewer PASS must not override a verifier FAIL.
