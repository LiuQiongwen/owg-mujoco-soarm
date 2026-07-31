# TANGO Project Context

TANGO is a robotic grasp-generation and manipulation research project involving MuJoCo/PyBullet simulation, VLA-related policies, executable action verification, and sim-to-real trajectory tooling.

Shared long-term rules:

- Research correctness has priority over execution speed.
- Never fabricate experiment results, statistics, citations, or hardware outcomes.
- Never overwrite historical experiment results or frozen baselines.
- Confirmatory experiments require a lock under `.agents/locks/` and must not be rerun automatically.
- Every new output uses a unique `run_id`.
- Preserve the user's pre-existing Git diff and do not classify it as agent work.
- Do not modify research code, data, or paper conclusions unless the task explicitly allows it.
- Do not expand task scope.
- Every statistical or paper claim must be supported by an actual saved artifact.
- A verifier failure cannot be overridden by a reviewer pass.
