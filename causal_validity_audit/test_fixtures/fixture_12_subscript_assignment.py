"""Category 12 (added 2026-08-02): subscript-assignment dict mutation --
`container[...]["field"] = value` -- the pattern found in the wild while auditing Sim-Grasp
(github.com/junchengli1/Sim-Grasp, grasp_simulation.py::main_simulation_loop writes
`new_candidates[obj]["grasp_samples"][idx]["simulation_quality"] = 1` this exact way). Neither
`return {...}` nor `dict(k=v)` -- `analyze_function` could not see this pattern at all before the
2026-08-02 extension (`_subscript_field_name`). This fixture uses a clean `env`-named handle so it
tests ONLY the new pattern-recognition mechanism in isolation, not conflated with category 8's
aliasing issue (the real Sim-Grasp bug happens to combine both -- `world.step(...)` AND
subscript-assignment -- which is why it was doubly invisible before today)."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def label_candidate_pool(env, candidates, idx):
    approach_yaw = candidates[idx]["yaw"]
    candidates[idx]["approach_yaw"] = approach_yaw  # pre-execution, echoed back before commit
    CAUSAL_VALIDITY_COMMIT_POINT()
    env.step(candidates[idx])
    outcome_success = env.data.contact_flags[0]
    candidates[idx]["simulation_quality"] = outcome_success  # execution-derived, subscript-assigned
    return candidates
