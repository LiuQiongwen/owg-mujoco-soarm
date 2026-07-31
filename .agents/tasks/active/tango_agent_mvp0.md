# Task: TANGO Experiment Agent MVP 0

## Reconciliation note (current status — supersedes the "Prefect workflow"
## line in Goal below and the former "install Prefect" forbidden-action item)

This MVP0 draft originally scoped a Prefect-free implementation. That scope
was superseded in a later, explicit instruction: **Prefect 3.8.1 is now the
accepted deterministic workflow runtime** for the TANGO Experiment Agent,
implemented in `research_agent/flow.py` (`@flow`/`@task`, with retries wired
only to `subprocess_runner.InfrastructureError`). This is the current,
accepted architecture — do not revert to a non-Prefect implementation.

Status of the pieces this file originally scoped:

- **Codex and Claude adapters**: both a mock adapter pair (`MockCodexAgent`,
  `MockClaudeAgent` — offline, deterministic, the default and the only
  adapters any automated test uses) and a real adapter pair
  (`RealCodexAgent`, `RealClaudeAgent`, invoking the actual `codex`/`claude`
  CLIs as subprocesses) are implemented. **The real adapters are implemented
  but unverified**: no `codex` or `claude` binary has been available in any
  environment this code has run in, so `RealCodexAgent`/`RealClaudeAgent`
  have never actually been exercised end-to-end. Mock agents remain the
  default in the CLI; real adapters require an explicit `--agents real`
  flag, which also prints an experimental-and-unverified warning to stderr.
- **Confirmatory execution remains disabled**, unchanged from this file's
  original scope: `policies/confirmatory.py`'s `CONFIRMATORY_EXECUTION_ENABLED`
  is a hardcoded `False` with no caller anywhere in `flow.py`.

## Goal

Implement a deterministic Python MVP for a research experiment agent.

This stage must use mock Codex and mock Claude adapters by default. It must
not invoke any real LLM CLI, GPU job, robot, research experiment, Docker,
sudo, network download, or git push. (The "or Prefect workflow" restriction
in this line is the specific item reconciled above — Prefect is now the
accepted runtime.)

## Architecture

experiment YAML
→ Pydantic validation
→ immutable run directory
→ mock Codex plan
→ mock Claude proposal
→ Python command and path policy validation
→ deterministic harmless smoke command
→ artifact verification
→ final report

## Required files

Create:

- research_agent/__init__.py
- research_agent/cli.py
- research_agent/models.py
- research_agent/state.py
- research_agent/subprocess_runner.py

- research_agent/agents/__init__.py
- research_agent/agents/mock_codex.py
- research_agent/agents/mock_claude.py

- research_agent/policies/__init__.py
- research_agent/policies/command_policy.py
- research_agent/policies/path_policy.py

- research_agent/verification/__init__.py
- research_agent/verification/repository.py
- research_agent/verification/artifacts.py

- research_agent/reporting/__init__.py
- research_agent/reporting/report.py

- experiments/example_smoke.yaml

- tests/test_experiment_spec.py
- tests/test_command_policy.py
- tests/test_path_policy.py
- tests/test_subprocess_runner.py
- tests/test_state.py
- tests/test_mock_flow.py

- docs/tango_experiment_agent.md

## Experiment specification

Use Pydantic models with unknown fields rejected.

Required concepts:

- schema_version
- task_id
- goal
- mode
- repository root
- allowed paths
- forbidden paths
- approved commands
- timeout
- required artifacts
- required metrics
- maximum command count
- human gates

Only `development` mode is supported in MVP 0.

Reject:

- confirmatory mode
- robot mode
- missing required fields
- unknown fields
- empty command arrays
- invalid timeout
- duplicate or conflicting allowed/forbidden paths

## Command policy

Commands must:

- be represented as `list[str]`;
- use `shell=False`;
- exactly match a command declared in the experiment YAML;
- use an allowed executable;
- contain no forbidden token;
- remain within the maximum command count.

Allowed executable examples:

- python
- python3
- pytest
- git

Forbidden commands or tokens include:

- sudo
- docker
- git push
- git reset --hard
- rm -rf
- chmod 777
- curl
- wget
- bash -c
- sh -c

Do not use substring-only security checks where token-aware checks are needed.

## Path policy

The system must obtain changed files from Git, not from mock Claude claims.

It must:

- permit only paths listed under allowed paths;
- reject files under forbidden paths;
- reject path traversal;
- resolve symlinks safely;
- reject absolute paths outside the worktree;
- report every disallowed changed path.

## Subprocess runner

Implement a runner using:

- `subprocess.run`;
- `shell=False`;
- command arrays;
- explicit cwd;
- explicit timeout;
- captured stdout and stderr;
- persisted exit code;
- duration measurement.

Every execution must write:

- command.json
- stdout.log
- stderr.log
- exit_code.txt
- result.json

Timeout must produce exit code 124 or an equivalent explicit timeout status.

## Run directory

Every run must have a unique immutable directory:

agent_runs/<run_id>/

It must contain:

- spec.yaml
- state.json
- plan.json
- proposal.json
- final_report.md
- logs/
- artifacts/
- provenance/

Existing run IDs must never be overwritten.

State updates must use temporary-file plus atomic replacement.

A failed run must never remain in RUNNING state.

## Mock Codex

Mock Codex returns a deterministic plan containing:

- task ID;
- approved smoke command;
- expected artifact;
- expected metric;
- decision `APPROVE_PLAN`.

It must not execute commands.

## Mock Claude

Mock Claude returns a deterministic proposal containing:

- summary;
- requested approved command;
- expected artifact;
- no changed files;
- status `READY_FOR_VALIDATION`.

It must not edit repository files.

## Harmless smoke experiment

The example YAML must approve exactly one harmless Python command.

The command must write a valid JSON file to the assigned artifact directory, for example:

{
  "schema_version": "1.0",
  "valid": true,
  "number_of_episodes": 2,
  "number_of_frames": 20
}

The command must not write outside the assigned run artifact directory.

Do not use shell interpolation. Resolve placeholders before execution and validate the resulting command exactly against a safe command template.

## Artifact verification

Verify:

- required artifact exists;
- artifact is valid JSON;
- required metrics exist;
- metric types are correct;
- command exit code is zero;
- forbidden paths were not modified;
- no unexpected artifact was produced outside the run directory.

## CLI

Implement:

python -m research_agent.cli validate experiments/example_smoke.yaml

python -m research_agent.cli smoke experiments/example_smoke.yaml --mock-agents

python -m research_agent.cli status <run_id>

The successful smoke command must print:

MOCK_DEVELOPMENT_PASS
Commands executed: 1
Forbidden paths modified: 0
Required artifacts present: yes
Codex decisions: mock
Claude rounds: mock

## Failure cases to test

Tests must cover:

- unknown YAML field;
- forbidden command;
- unapproved command;
- shell command string instead of list;
- command timeout;
- command nonzero exit;
- forbidden changed path;
- path traversal;
- existing run ID;
- missing artifact;
- malformed metrics JSON;
- missing required metric;
- failed run state is not left as RUNNING;
- valid mock smoke flow.

## Allowed paths

Modify or add only:

- research_agent/
- experiments/example_smoke.yaml
- tests/test_experiment_spec.py
- tests/test_command_policy.py
- tests/test_path_policy.py
- tests/test_subprocess_runner.py
- tests/test_state.py
- tests/test_mock_flow.py
- docs/tango_experiment_agent.md
- .agents/tasks/active/tango_agent_mvp0.md
- .gitignore, only if needed to ignore agent_runs/

## Forbidden actions

Do not:

- invoke real Codex (as the default; see the reconciliation note above —
  `--agents real` is an explicit, non-default opt-in, still unverified);
- invoke real Claude (same caveat as above);
- run a GPU process;
- run a robot;
- run training;
- access the network;
- use Docker;
- use sudo;
- commit;
- push;
- modify research data;
- modify paper files;
- modify existing experiment results.

## Acceptance criteria

- Python compile checks pass;
- targeted pytest tests pass;
- git diff --check passes;
- changed paths are all allowed;
- mock smoke flow completes;
- exactly one command is executed;
- run artifacts and logs are complete;
- forbidden paths modified equals zero;
- required metrics are verified;
- failed tests never leave state RUNNING;
- no real agent is invoked.

## Required final report

Report:

- files added;
- architecture implemented;
- exact commands run;
- exact test results;
- mock smoke run ID;
- run directory contents;
- limitations;
- next step for MVP 1.

Do not commit.
