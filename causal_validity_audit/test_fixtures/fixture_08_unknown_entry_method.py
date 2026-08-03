"""Category 8, REDESIGNED after two failed attempts revealed the tool's actual mechanism
empirically (see Stage-1 run log / VALIDATION_RESULTS.md for the full story -- this docstring
records the corrected understanding, not the original guess).

Original hypothesis: an unrecognized METHOD NAME (e.g. `env.advance_physics` instead of
`env.step`) on the DEFAULT_EXECUTION_ENTRY_METHODS list would cause a miss. WRONG, verified by
running the suite: `_reads_env_state()` matches ANY attribute chain rooted at a variable literally
named `env`, regardless of the specific method/attribute name -- so `env.advance_physics(...)` is
already caught, same as `env.step(...)`. Two fixture revisions (adding then removing a direct
`env.data.*` read inside the helper) both still passed, because the call expression
`env.advance_physics(...)` ITSELF is an `env.*` attribute chain and `_reads_env_state` walks the
whole call node, not just its result.

Actual mechanism, confirmed by reading `_reads_env_state`/`_is_env_step_call` directly: BOTH checks
require the physical-actuation object be a variable literally named `env` (`base.id == "env"` /
`f.value.id == "env"`, hardcoded). The real gap is not the METHOD NAME -- it is any ALIASED or
DIFFERENTLY-NAMED reference to the same physical handle (e.g. `self.env`, `sim`, a renamed
parameter). This fixture now tests that: `_step_via_sim_alias` receives the handle as `sim`, not
`env`, and calls `sim.step(...)` -- the exact, canonical entry method name, just under the wrong
variable name. Ground truth: EXECUTION_DERIVED (sim_x genuinely depends on physical execution).
Expected tool outcome: miss (PRE_EXECUTION), because neither `_reads_env_state` nor
`_is_env_step_call` fires on a base named `sim`, and the plain (non-attribute) local read
`buffer[0]` in the caller gives no further signal either. This is a real, non-hypothetical gap --
`piper_pick_and_place.py`/`batch_s3s4.py`-style code that assigns the environment handle to
anything other than a variable named exactly `env` (e.g. inside a class as `self.env`, or a
renamed local) would silently defeat this tool today."""
from causal_validity_audit.commit_marker import CAUSAL_VALIDITY_COMMIT_POINT


def _step_via_sim_alias(sim, action, output_buffer):
    sim.step(action)  # canonical entry method name, but on a variable NOT named `env`
    output_buffer[0] = sim.data.contact_x


def grasp_with_aliased_handle(env, candidate):
    approach_x = candidate[0]
    CAUSAL_VALIDITY_COMMIT_POINT()
    buffer = [0.0]
    _step_via_sim_alias(env, candidate, buffer)  # `env` is passed in, but bound to param name `sim` inside
    sim_x = buffer[0]
    return {
        "approach_x": approach_x,
        "sim_x": sim_x,
    }
