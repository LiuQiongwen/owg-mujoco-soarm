"""
Automated causal-validity tagger: infers PRE_EXECUTION / EXECUTION_DERIVED
provenance for every field in a function's return dict via static dataflow
analysis, instead of requiring a human to hand-register each field (as
provenance.py's registry does).

Design (see CAUSAL_VALIDITY_METHOD.md for the full writeup): full automation
from unannotated code is not possible -- a syntactic analyzer cannot tell a
physics "settle" step (part of establishing the initial, pre-execution scene
observation) apart from a genuine "this candidate is now executing" step;
both are just calls to `env.step(...)`. This tool instead requires ONE
human-placed marker per analyzed function (see commit_marker.py) -- a single
semantic judgment call, not one per field -- and then automatically
propagates provenance to every downstream field via forward taint analysis
from that point. This cuts the manual annotation burden from O(number of
logged fields) to O(number of commit points in the codebase).

Algorithm:
  1. Build the set of "execution-touching" functions in the module: any
     function whose body directly calls `env.step(...)`, or transitively
     calls another execution-touching function (fixed point over the
     module's top-level defs, then extended to any function nested inside
     the target function being analyzed).
  2. Walk the target function's body in program order. Before the marker
     statement, nothing is tainted (everything is admissible pre-execution
     setup/candidate-selection code, however complex). After the marker:
       - An assignment is tainted if its RHS references an already-tainted
         name, calls an execution-touching function, or reads through an
         attribute chain rooted at a variable named `env` (a live physical-
         state read).
       - If/For bodies are handled conservatively: taint is unioned across
         branches/one loop pass (fail toward EXECUTION_DERIVED, not away
         from it, on any ambiguity).
  3. At the function's `return {...}` (a literal dict), each key's
     provenance is EXECUTION_DERIVED if its value expression is tainted per
     the above, else PRE_EXECUTION.

This is intentionally scoped to the patterns this project's own pipeline
functions actually use (straight-line code with occasional if/for), not a
general-purpose Python analyzer.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Optional

MARKER_NAME = "CAUSAL_VALIDITY_COMMIT_POINT"

# Confidence flagging (2026-07-17): a call to a function this module cannot
# resolve -- not env.step-touching, not a builtin/stdlib pure function --
# is currently invisible to expr_is_tainted, which silently treats it as
# untainted. That is fine within this project's own two validated target
# functions (every call they make resolves to something in this allowlist,
# a same-module function, or env state), but is a real gap for a
# general-purpose version of this tool pointed at a codebase that spreads
# execution-touching logic across many files -- an unresolved call there
# could secretly touch physical state and this tool would have no way to
# know. KNOWN_PURE_CALLS is a manually curated allowlist of calls verified
# to never touch execution state, analogous to
# DEFAULT_EXECUTION_ENTRY_METHODS above -- not automatically inferred, an
# honest limitation, not a hidden one.
KNOWN_PURE_CALLS = {
    "float", "int", "str", "bool", "len", "round", "abs", "min", "max", "sum",
    "list", "tuple", "dict", "set", "range", "enumerate", "zip", "sorted",
    "isinstance", "hasattr", "getattr",
    # numpy pure numeric/array-construction calls used throughout this
    # codebase's pre-marker candidate-pool and post-marker arithmetic --
    # none of these read live simulator state.
    "array", "arctan2", "concatenate", "clip", "copy", "asarray", "radians",
    "degrees", "sqrt", "norm", "cos", "sin", "mean", "std", "get",
    "tolist", "reshape", "flatten", "item", "round",
}


@dataclass
class TagResult:
    field_provenance: dict          # field_name -> "PRE_EXECUTION" | "EXECUTION_DERIVED"
    tainted_vars: set               # variable names tainted by the time of `return`
    marker_found: bool
    field_confidence: dict = None   # field_name -> (certain: bool, unresolved_calls: list[str])


def _call_name(func_node) -> Optional[str]:
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return func_node.attr
    return None


# Physical-actuation entry points, by method name, on a variable literally
# named `env`. Not automatically inferred -- a manually curated list, and an
# honest limitation of this tool: different simulator/hardware APIs use
# different method names for "actually move something in the physical
# world." `step` covers the MuJoCo/RoboSuite Piper codebase
# (piper_pick_and_place.py); `put_obj_in_tray` and `step_simulation` cover
# the PyBullet-based SO-ARM101 codebase (batch_s3s4.py) -- found necessary
# when extending interprocedural analysis to that second codebase, where
# `env.put_obj_in_tray(...)` is the actual grasp-execution call and a
# hardcoded `.step`-only check would have silently failed to recognize it
# as execution-touching (failing open, the wrong direction).
DEFAULT_EXECUTION_ENTRY_METHODS = {"step", "put_obj_in_tray", "step_simulation"}


def _is_env_step_call(call: ast.Call, entry_methods=None) -> bool:
    entry_methods = entry_methods if entry_methods is not None else DEFAULT_EXECUTION_ENTRY_METHODS
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr in entry_methods
        and isinstance(f.value, ast.Name)
        and f.value.id == "env"
    )


def _calls_in(node) -> list:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _all_module_function_names(module: ast.Module) -> set:
    """Every function name defined anywhere in this module (any nesting
    depth). Used for confidence flagging: a call to a name in this set that
    is NOT in execution_touching has been proven pure by
    _find_execution_touching's own fixed-point analysis (it looked at
    every one of these and didn't find an execution-touching path) --
    resolved, not unknown. A call to a name NOT in this set at all is
    genuinely unresolved (defined elsewhere, imported, or a typo)."""
    return {n.name for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}


def _find_execution_touching(module: ast.Module) -> set:
    """Fixed-point: a function is execution-touching if it calls env.step
    directly, or calls another function already known to be."""
    funcs = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    touching = set()
    changed = True
    while changed:
        changed = False
        for name, node in funcs.items():
            if name in touching:
                continue
            for call in _calls_in(node):
                if _is_env_step_call(call):
                    touching.add(name)
                    changed = True
                    break
                cname = _call_name(call.func)
                if cname in touching:
                    touching.add(name)
                    changed = True
                    break
    return touching


def _names_in(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _reads_env_state(node) -> bool:
    for a in ast.walk(node):
        if isinstance(a, ast.Attribute):
            base = a
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id == "env":
                return True
    return False


def _assigned_names(target) -> list:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for elt in target.elts:
            out.extend(_assigned_names(elt))
        return out
    return []


def _mutated_base_name(target) -> Optional[str]:
    """For `x[i] = ...` or `x.attr = ...`, return 'x' -- the container
    being mutated in place, as opposed to rebound. Simple Name rebinding
    (`x = ...`) is handled separately by _assigned_names; this catches the
    case a naive Name-only tracker misses (found during development: the
    first version of this tool silently under-flagged `phase_log[name] = ...`
    because Subscript/Attribute targets returned no names at all, so a
    post-commit mutation into an already-PRE_EXECUTION container went
    untracked -- a false PRE_EXECUTION verdict, i.e. failing OPEN instead of
    closed, the wrong direction for a validity-safety tool. Fixed by
    unconditionally tainting the base name on any post-commit
    subscript/attribute mutation, since we cannot in general prove such a
    mutation is safe with this level of static analysis."""
    base = target
    if not isinstance(base, (ast.Subscript, ast.Attribute)):
        return None
    while isinstance(base, (ast.Subscript, ast.Attribute)):
        base = base.value
    return base.id if isinstance(base, ast.Name) else None


class _FunctionTagger:
    def __init__(self, execution_touching: set, resolved_pure_functions: Optional[set] = None):
        self.execution_touching = execution_touching
        # Module-level functions proven pure by _find_execution_touching's
        # own fixed-point analysis (defined somewhere in this module, not
        # in execution_touching) -- resolved, not unknown, for confidence
        # purposes. Defaults to empty for callers that only care about
        # provenance, not confidence (e.g. resolve_parameter_provenance).
        self.resolved_pure_functions = resolved_pure_functions or set()
        self.tainted = set()
        self.committed = False
        self.local_functions = set()  # all nested defs seen (pure or not) -- resolved, not unknown
        self.uncertain = set()        # variable names whose value came from an unresolved call

    def expr_is_tainted(self, expr) -> bool:
        if not self.committed:
            return False
        if _names_in(expr) & self.tainted:
            return True
        if _reads_env_state(expr):
            return True
        for call in _calls_in(expr):
            cname = _call_name(call.func)
            if cname in self.execution_touching:
                return True
        return False

    def expr_confidence(self, expr) -> tuple:
        """Returns (certain: bool, unresolved_calls: list[str]). A call is
        unresolved if its name is not env.step itself, not in
        execution_touching, not in KNOWN_PURE_CALLS, and not a nested
        function defined in this same analyzed function -- i.e. this
        module cannot determine, from this file alone, whether it touches
        execution state. Fails toward UNCERTAIN, not toward assuming safe.

        Also propagates through variables: an expression that references a
        name whose value was itself derived from an unresolved call (found
        during development -- a first version only checked the return
        expression's OWN calls, missing the common case where the
        unresolved call happens in an earlier assignment and only the
        resulting variable appears in the field expression) is uncertain
        too, even if the field expression itself makes no calls at all."""
        unresolved = []
        for name in _names_in(expr) & self.uncertain:
            unresolved.append(f"<propagated from {name}>")
        for call in _calls_in(expr):
            if _is_env_step_call(call):
                continue
            # env.<anything>(...) is a live physical-state method call --
            # already a fully understood, resolved case via
            # _reads_env_state's own root-finding logic (which is what
            # actually taints these calls), not an unknown one. Excluding
            # it here specifically (rather than broadly skipping all
            # attribute-call resolution) keeps calls on OTHER objects
            # (e.g. an unresolved third-party helper) correctly flagged.
            f = call.func
            if isinstance(f, ast.Attribute):
                base = f
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name) and base.id == "env":
                    continue
            cname = _call_name(call.func)
            if cname is None:
                continue
            if cname in self.execution_touching:
                continue
            if cname in KNOWN_PURE_CALLS:
                continue
            if cname in self.local_functions:
                continue
            if cname in self.resolved_pure_functions:
                continue
            unresolved.append(cname)
        return (len(unresolved) == 0, unresolved)

    def _is_marker(self, stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and _call_name(stmt.value.func) == MARKER_NAME
        )

    def _resolve_arg_expr(self, target_call: ast.Call, arg_spec):
        keyword_name, positional_index = arg_spec
        for kw in target_call.keywords:
            if kw.arg == keyword_name:
                return kw.value
        if positional_index is not None and positional_index < len(target_call.args):
            return target_call.args[positional_index]
        return None

    def evaluate_arg_at_call(self, body: list, target_call: ast.Call, arg_spec):
        """Walk body in program order, same taint rules as visit_body, and
        the moment the statement containing `target_call` (identity match)
        is reached, evaluate the taint of the argument bound to `arg_spec`
        -- a (keyword_name, positional_index) pair, keyword tried first --
        using whatever tainted/committed state has accumulated up to that
        exact point. Raises _FoundCall to unwind immediately once found;
        recurses into If/For bodies (unlike plain visit_body's statement
        dispatch) so a call nested inside a conditional is still found with
        the correct accumulated state, not treated as opaque."""
        for stmt in body:
            if isinstance(stmt, ast.If) and self._contains_call(stmt, target_call):
                pre_tainted = set(self.tainted)
                if self._contains_call_in_body(stmt.body, target_call):
                    self.evaluate_arg_at_call(stmt.body, target_call, arg_spec)
                else:
                    self.tainted = pre_tainted
                    self.evaluate_arg_at_call(stmt.orelse, target_call, arg_spec)
                return  # unreachable if _FoundCall was raised, but keeps flow explicit
            if isinstance(stmt, (ast.For, ast.While)) and self._contains_call(stmt, target_call):
                self.evaluate_arg_at_call(stmt.body, target_call, arg_spec)
                return
            if self._contains_call(stmt, target_call) and not isinstance(stmt, (ast.If, ast.For, ast.While)):
                arg_expr = self._resolve_arg_expr(target_call, arg_spec)
                if arg_expr is None:
                    raise _FoundCall("UNKNOWN (argument not found at call site)")
                raise _FoundCall("EXECUTION_DERIVED" if self.expr_is_tainted(arg_expr) else "PRE_EXECUTION")
            self._visit_one(stmt)

    @staticmethod
    def _contains_call(stmt, target_call) -> bool:
        return any(n is target_call for n in ast.walk(stmt))

    @staticmethod
    def _contains_call_in_body(body, target_call) -> bool:
        return any(any(n is target_call for n in ast.walk(s)) for s in body)

    def visit_body(self, body: list):
        for stmt in body:
            self._visit_one(stmt)

    def _visit_one(self, stmt):
        if self._is_marker(stmt):
            self.committed = True
            return
        if isinstance(stmt, ast.FunctionDef):
            # Nested def (e.g. solve_and_move): if it calls an
            # execution-touching function or env.step directly, it is
            # itself execution-touching for the rest of this analysis.
            self.local_functions.add(stmt.name)
            calls_exec = any(_is_env_step_call(c) for c in _calls_in(stmt)) or any(
                _call_name(c.func) in self.execution_touching for c in _calls_in(stmt)
            )
            if calls_exec:
                self.execution_touching.add(stmt.name)
            return
        if isinstance(stmt, ast.Assign):
            tainted = self.expr_is_tainted(stmt.value)
            _, unresolved = self.expr_confidence(stmt.value)
            uncertain = len(unresolved) > 0
            for target in stmt.targets:
                for name in _assigned_names(target):
                    if tainted:
                        self.tainted.add(name)
                    else:
                        self.tainted.discard(name)
                    if uncertain:
                        self.uncertain.add(name)
                    else:
                        self.uncertain.discard(name)
                mutated = _mutated_base_name(target)
                if mutated is not None and self.committed:
                    # in-place mutation of a container after commit:
                    # fail closed regardless of the RHS's own taint
                    self.tainted.add(mutated)
                    if uncertain:
                        self.uncertain.add(mutated)
        elif isinstance(stmt, ast.AugAssign):
            tainted = self.expr_is_tainted(stmt.value) or (
                isinstance(stmt.target, ast.Name) and stmt.target.id in self.tainted
            )
            for name in _assigned_names(stmt.target):
                if tainted:
                    self.tainted.add(name)
        elif isinstance(stmt, ast.If):
            pre_tainted = set(self.tainted)
            self.visit_body(stmt.body)
            after_if = set(self.tainted)
            self.tainted = set(pre_tainted)
            self.visit_body(stmt.orelse)
            self.tainted |= after_if  # conservative union across branches
        elif isinstance(stmt, (ast.For, ast.While)):
            self.visit_body(stmt.body)
        elif isinstance(stmt, ast.Return):
            pass  # handled by caller
        elif isinstance(stmt, ast.Expr):
            pass  # bare call statement, e.g. env.step(...) itself: nothing to taint (no assignment)
        # other statement kinds (Import, Assert, ...) intentionally ignored


class _FoundCall(Exception):
    def __init__(self, provenance: str):
        super().__init__(provenance)
        self.provenance = provenance


def analyze_function(func_def: ast.FunctionDef, module: ast.Module) -> TagResult:
    """Two field-defining patterns are recognized: `return {...}` (a dict
    literal) and `x = dict(k=v, ...)` (a constructor call with keyword args,
    e.g. a row about to be written to a log) -- the latter added after
    validating against a second real function (batch_s3s4.py's
    _emit_lggsn_candidates) that uses this pattern instead of the first.
    For the dict()-constructor pattern, provenance is read off at the point
    the dict is BUILT, not at a return (this function-under-analysis has no
    return value at all; it writes rows out as a side effect)."""
    execution_touching = _find_execution_touching(module)
    resolved_pure = _all_module_function_names(module) - execution_touching
    tagger = _FunctionTagger(execution_touching, resolved_pure_functions=resolved_pure)
    tagger.visit_body(func_def.body)

    field_provenance = {}
    field_confidence = {}
    for stmt in ast.walk(func_def):
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
            for k, v in zip(stmt.value.keys, stmt.value.values):
                if isinstance(k, ast.Constant):
                    field_provenance[k.value] = (
                        "EXECUTION_DERIVED" if tagger.expr_is_tainted(v) else "PRE_EXECUTION"
                    )
                    field_confidence[k.value] = tagger.expr_confidence(v)
        elif (
            isinstance(stmt, ast.Call)
            and isinstance(stmt.func, ast.Name)
            and stmt.func.id == "dict"
        ):
            for kw in stmt.keywords:
                if kw.arg is not None:
                    field_provenance[kw.arg] = (
                        "EXECUTION_DERIVED" if tagger.expr_is_tainted(kw.value) else "PRE_EXECUTION"
                    )
                    field_confidence[kw.arg] = tagger.expr_confidence(kw.value)

    return TagResult(
        field_provenance=field_provenance, tainted_vars=tagger.tainted,
        marker_found=tagger.committed, field_confidence=field_confidence,
    )


def tag_file(path: str, function_name: str) -> TagResult:
    with open(path) as f:
        source = f.read()
    module = ast.parse(source, filename=path)
    target = None
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target = node
            break
    if target is None:
        raise ValueError(f"function {function_name!r} not found in {path}")
    return analyze_function(target, module)


def _find_call_sites(module: ast.Module, function_name: str):
    """Every (enclosing_function, call_node) pair for a call to
    function_name found anywhere inside a top-level function body in this
    module. Calls at module scope (outside any function) are not covered --
    out of scope for this project's pipeline-function-shaped codebases."""
    sites = []
    for func in [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and _call_name(node.func) == function_name:
                sites.append((func, node))
    return sites


def resolve_parameter_provenance(path: str, function_name: str, param_name: str, param_index=None):
    """Interprocedural extension of analyze_function: instead of assuming a
    function's PARAMETER is safe (analyze_function's single-function scope
    cannot see past the function boundary -- see AUTO_TAGGER_ALGORITHM.md's
    documented limitation, found via _emit_lggsn_candidates), find every
    real call site of `function_name` in this module and determine whether
    the argument bound to `param_name` (keyword) / `param_index`
    (positional fallback) is tainted AT THAT CALL SITE, using the same
    taint rules applied to the CALLING function's own body up to the call.

    Fails closed: if ANY call site passes a tainted argument, the parameter
    is EXECUTION_DERIVED overall (a function is only as safe as its least
    safe caller). If no call sites are found in this module, returns
    UNKNOWN rather than assuming safety -- calls from other files are
    invisible to this single-module analysis (see AUTO_TAGGER_ALGORITHM.md's
    remaining-work list)."""
    with open(path) as f:
        module = ast.parse(f.read(), filename=path)
    execution_touching = _find_execution_touching(module)
    sites = _find_call_sites(module, function_name)
    if not sites:
        return "UNKNOWN (no call sites found in this module)", []

    per_site = []
    any_tainted = False
    for enclosing_func, call_node in sites:
        tagger = _FunctionTagger(execution_touching)
        try:
            tagger.evaluate_arg_at_call(enclosing_func.body, call_node, (param_name, param_index))
            verdict = "UNKNOWN (call site not reached during traversal)"
        except _FoundCall as fc:
            verdict = fc.provenance
        per_site.append((enclosing_func.name, call_node.lineno, verdict))
        if verdict.startswith("EXECUTION_DERIVED"):
            any_tainted = True

    overall = "EXECUTION_DERIVED" if any_tainted else "PRE_EXECUTION"
    return overall, per_site
