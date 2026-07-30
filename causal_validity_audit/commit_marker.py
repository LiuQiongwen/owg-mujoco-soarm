"""
CAUSAL_VALIDITY_COMMIT_POINT() is a no-op at runtime. Its only purpose is to
be a syntactic marker that auto_tagger.py's static analysis can find: a
single, human-placed statement in a function's source marking the line after
which that function's candidate has begun physically executing.

This is the one piece of domain knowledge a purely syntactic analyzer cannot
recover on its own -- see CAUSAL_VALIDITY_METHOD.md's discussion of why full
automation from unannotated code isn't possible (settle/observation steps
and genuine execution steps are both just `env.step()` calls; only a human
who knows which candidate, if any, has been committed to can tell them
apart). Placing this marker reduces the manual annotation burden from O(number
of logged fields) to O(number of physical-commit points in the codebase) --
here, one call site.
"""


def CAUSAL_VALIDITY_COMMIT_POINT():
    return None
