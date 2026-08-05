"""
Causal-validity provenance registry for grasp-candidate feature fields.

Core criterion: a feature is PRE_EXECUTION-admissible for live candidate
selection if and only if its value can be computed from (a) the candidate
pose itself, (b) static, execution-independent object/scene geometry,
(c) an ISOLATED kinematic/collision check performed against that pose
alone, or (d) the poses of OTHER not-yet-executed candidates in the same
selection pool (e.g. distance to the pool's centroid, normalized height
within the pool's range) -- (d) matters because "depends on more than one
candidate" is not the same thing as "depends on execution," and conflating
the two is itself a mistake this registry made and corrected once already
(see the dist_to_centroid/z_rel entries below). What actually disqualifies
a feature is a dependency on having run the arm through some or all of the
physical trajectory (drift, contact force, a re-solved IK residual computed
from a post-motion pose, the success label itself, or any diagnostic
computed FROM those) -- that is EXECUTION_DERIVED: valid for offline
evaluation of already-labeled historical data, but NOT valid as an input to
a scorer used to pick among not-yet-executed candidates, because at
selection time those values do not exist yet.

This distinction is easy to violate silently: every offline AUC/pairwise-
accuracy number computed on historical logs looks identical whether the
features used are admissible or not -- the violation only becomes visible
the moment the same feature set is reused for live selection. See
IDEA_REPORT.md's "Direction 2" correction for the concrete case this
registry was built to prevent: a 5-times-replicated, p<0.0001 "pooling
beats zero-shot" cross-embodiment reranking result that vanished entirely
(p=1.0000) once execution-derived features were removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provenance(Enum):
    PRE_EXECUTION = "pre_execution"
    EXECUTION_DERIVED = "execution_derived"


@dataclass(frozen=True)
class FieldSpec:
    provenance: Provenance
    reason: str


# ── SO-ARM101 / LGGSN 14-dim raw log fields (grasp_6dof pipeline) ──────────
SOARM_FIELDS = {
    "x": FieldSpec(Provenance.PRE_EXECUTION, "candidate pose component"),
    "y": FieldSpec(Provenance.PRE_EXECUTION, "candidate pose component"),
    "z": FieldSpec(Provenance.PRE_EXECUTION, "candidate pose component"),
    "roll": FieldSpec(Provenance.PRE_EXECUTION, "candidate pose component"),
    "pitch": FieldSpec(Provenance.PRE_EXECUTION, "candidate pose component"),
    "yaw": FieldSpec(Provenance.PRE_EXECUTION, "candidate pose component"),
    "width": FieldSpec(Provenance.PRE_EXECUTION, "candidate gripper-width parameter"),
    "H": FieldSpec(Provenance.PRE_EXECUTION, "static per-object geometry constant"),
    "score_candidate_ik": FieldSpec(
        Provenance.PRE_EXECUTION,
        "isolated kinematic IK check against the candidate pose alone, same "
        "function select_best() legitimately uses pre-execution",
    ),
    # CORRECTED (2026-07-16): an earlier version of this registry labeled
    # score/dz/dz_lift/need_dz EXECUTION_DERIVED by quoting
    # train_geo_ebm_grasp.py's header comment without checking whether that
    # comment describes the currently-active live pipeline. Traced directly
    # against the actual deployed code (tango_robot/grasp_ranker_lggsn.py's
    # _featurize_one, tango/policy.py's live rank() call site, and
    # batch_s3s4.py's _emit_lggsn_candidates -- the sole writer of
    # logs/lggsn_live_candidates.jsonl that train_lggsn_pairwise.py reads)
    # and independently verified against the on-disk log (4288/4288 rows
    # have dz=dz_lift=need_dz=0.0): the comment describes a DIFFERENT,
    # unused-in-production legacy dataset (grasp_6dof/dataset/all_lggsn.csv).
    # In the live pipeline actually feeding the trained/deployed checkpoint:
    "score": FieldSpec(
        Provenance.PRE_EXECUTION,
        "verified (2026-07-16) against grasp_ranker_lggsn.py:395 and "
        "batch_s3s4.py:86-101, both training-log-write and live-inference "
        "call sites: this is the GR-ConvNet quality prediction, a genuine "
        "pre-execution proxy, computed identically in both places -- NOT "
        "the execution-derived quantity train_geo_ebm_grasp.py's comment "
        "describes for a separate, inactive legacy dataset",
    ),
    "dz": FieldSpec(
        Provenance.PRE_EXECUTION,
        "verified (2026-07-16): hardcoded constant 0.0 in both the live "
        "inference path and the training-log writer -- technically does "
        "not leak future information (it never varies with execution "
        "outcome at all), but is a DEAD/uninformative feature, a distinct "
        "problem from causal-validity leakage -- flag separately, do not "
        "conflate with genuine leakage",
    ),
    "dz_lift": FieldSpec(Provenance.PRE_EXECUTION, "same as dz -- verified constant 0.0, dead feature not leakage"),
    "need_dz": FieldSpec(Provenance.PRE_EXECUTION, "same as dz -- verified constant 0.0, dead feature not leakage"),
    # CORRECTED (2026-07-16, second correction to this registry): originally
    # labeled EXECUTION_DERIVED under the assumption "episode-level" implied
    # "requires already-executed, already-labeled candidates." Traced against
    # grasp_ranker_lggsn.py::_featurize (live inference) and
    # train_lggsn_pairwise.py's context-feature block (lines ~97-111): both
    # compute these from the (x, y, z) POSES of every candidate in the
    # current pool, via the pool's own centroid/z-range -- never touching
    # execution outcome, contact, drift, or the success label. A pool-
    # relative feature (depends on sibling not-yet-executed candidates'
    # poses) is a DIFFERENT thing from an execution-derived feature (depends
    # on having run the trial) -- conflating the two was the registry's own
    # error here. Both are legitimately PRE_EXECUTION.
    "dist_to_centroid": FieldSpec(
        Provenance.PRE_EXECUTION,
        "pool-relative: distance to the (x,y) centroid of the current "
        "candidate pool, computed from poses alone -- verified identical "
        "logic in grasp_ranker_lggsn.py (live) and train_lggsn_pairwise.py "
        "(training log), no execution outcome involved",
    ),
    "z_rel": FieldSpec(
        Provenance.PRE_EXECUTION,
        "pool-relative: min-max normalized z within the current candidate "
        "pool's poses -- same verification as dist_to_centroid",
    ),
    # ADDED (Tier-3, LGGSN per-candidate IK reachability): unlike the
    # existing "score_candidate_ik" entry above (a single scalar reused
    # across a whole selection pool) or "pe_ik" (episode-level, one CoM
    # target for every candidate — see compute_ik_reachability), these three
    # are solved independently per candidate via
    # EnvironmentSoArm.compute_ik_reachability_per_candidate, which re-solves
    # IK from HOME_QPOS against that candidate's own (x, y, z, yaw) and
    # restores qpos/qvel/ctrl afterward. Admissible for the same reason
    # score_candidate_ik is: an isolated kinematic check against the
    # candidate pose alone, never a value read back from the actual
    # grasp-execution trajectory (see the Piper grasp_yaw entry below for
    # the failure mode this registry exists to catch: a field that looks
    # pre-execution but is silently reassigned post-motion).
    "ik_converged": FieldSpec(
        Provenance.PRE_EXECUTION,
        "isolated kinematic IK re-solve against the candidate's own "
        "(x, y, z, yaw), from HOME_QPOS, via "
        "compute_ik_reachability_per_candidate -- same admissible pattern "
        "as score_candidate_ik, never read back from the actual "
        "grasp-execution trajectory",
    ),
    "ik_residual": FieldSpec(
        Provenance.PRE_EXECUTION,
        "same isolated per-candidate re-solve as ik_converged",
    ),
    "max_joint_delta": FieldSpec(
        Provenance.PRE_EXECUTION,
        "same isolated per-candidate re-solve as ik_converged: max absolute "
        "arm-joint displacement from HOME_QPOS after that re-solve",
    ),
}

# ── Piper raw log fields (piper_pick_and_place.py run_pick_and_place) ─────
PIPER_FIELDS = {
    "spawn_pos": FieldSpec(Provenance.PRE_EXECUTION, "object spawn pose, known before any motion"),
    # CORRECTED (2026-07-16, third correction, this one caught by
    # auto_tagger.py's static analysis rather than by hand): grasp_mat is
    # reassigned TWICE in run_pick_and_place -- once pre-commit (candidate
    # selection, fine) and once post-commit at the "pre-close refresh" step
    # (piper_pick_and_place.py, `grasp_mat = refreshed_grasp_mat`, computed
    # from env.sim.data.xquat read AFTER the descend phase already
    # physically executed). grasp_yaw is computed from whichever value
    # grasp_mat holds by the time the return dict is built -- the SECOND
    # (post-descend, execution-derived) one. The hand-built registry missed
    # this because reading the code once, top-to-bottom, sees
    # `grasp_mat = compute_grasp_orientation(...)` early and stops tracking
    # -- it takes deliberate dataflow tracing to notice the later
    # reassignment. The auto-tagger caught it because it tracks every
    # reassignment mechanically, not just the first one a human happens to
    # notice.
    "grasp_yaw": FieldSpec(
        Provenance.EXECUTION_DERIVED,
        "verified via causal_validity_audit/auto_tagger.py against the live "
        "code (2026-07-16): grasp_mat is reassigned post-commit at the "
        "pre-close-refresh step from a post-descend env.sim.data.xquat read "
        "-- grasp_yaw reflects that reassigned value, not the original "
        "pre-execution candidate orientation",
    ),
    "candidate_grasp_yaw": FieldSpec(
        Provenance.PRE_EXECUTION,
        "added 2026-07-16, verified via auto_tagger.py: captured immediately "
        "after candidate selection (obj_pos, grasp_mat = candidates[chosen_idx]), "
        "before the marker's later post-descend grasp_mat reassignment that "
        "contaminates grasp_yaw -- the genuinely per-candidate-varying, "
        "causally valid feature the [z, H] Stage 2 set was missing",
    ),
    "object_H": FieldSpec(Provenance.PRE_EXECUTION, "static per-object top-surface offset constant"),
    "score_candidate_ik": FieldSpec(
        Provenance.PRE_EXECUTION,
        "isolated kinematic IK check, recomputed fresh from READY_QPOS against "
        "the candidate pose alone -- verified correlation with success is only "
        "0.06, i.e. admissible but not very informative for Cracker",
    ),
    "quality_score": FieldSpec(
        Provenance.EXECUTION_DERIVED,
        "negative descend-phase IK error LOGGED AFTER prior phases' physical "
        "trajectory already ran -- not an isolated pre-execution check",
    ),
    "correction_proxy": FieldSpec(
        Provenance.EXECUTION_DERIVED,
        "negative pre_close_drift_cm -- by definition only knowable after the "
        "arm has already moved through descend",
    ),
    "dist_to_tray": FieldSpec(Provenance.EXECUTION_DERIVED, "final placement outcome, only known post-execution"),
    "success": FieldSpec(Provenance.EXECUTION_DERIVED, "the label itself"),
}

# ── world_model/ MLP critic fields (Risk-Gated VLA Phase 1/2 study) ───────
# Registered 2026-07-30 during the risk_gated_vla audit (results/risk_gated_vla/audit.md
# Section 3). This pipeline predates causal_validity_audit/ entirely (mlp_predictor.pkl is
# dated 2026-05-15) and had never been run through this gate before. Traced against the
# actual call sites: data/transition_logger.py::build_feature (feature assembly) and
# scripts/collect_mujoco_transitions.py::run_collection (capture timing).
WORLD_MODEL_FIELDS = {
    "grasp_pose": FieldSpec(
        Provenance.PRE_EXECUTION,
        "sampled by sample_grasp()/_sample_grasp() from obj_pos + RNG, before "
        "execute_grasp()/_execute() is ever called -- the proposed action itself",
    ),
    "obj_pos_before": FieldSpec(
        Provenance.PRE_EXECUTION,
        "env.get_obj_pos(oid) captured after settle steps, strictly before grasp "
        "execution -- static pre-grasp scene state",
    ),
    "obj_quat_before": FieldSpec(
        Provenance.PRE_EXECUTION,
        "env.get_obj_pose(oid)['quaternion'] captured at the same pre-grasp point as "
        "obj_pos_before",
    ),
    "pc_stats_before": FieldSpec(
        Provenance.PRE_EXECUTION,
        "compute_pc_stats(obs_before, oid) -- derived from the pre-grasp point cloud "
        "capture only, no execution state involved",
    ),
    "pc_stats_local": FieldSpec(
        Provenance.PRE_EXECUTION,
        "compute_pc_stats_local(obs_before, oid, candidate_pose[:3]) -- same pre-grasp "
        "point cloud capture as pc_stats_before, cropped around the CANDIDATE's own "
        "generator-time pose rather than the whole object. The crop center is the "
        "candidate's proposed position (known before it executes, same provenance as "
        "grasp_pose itself), not anything derived from that candidate's own outcome or "
        "any other candidate's outcome -- pool-relative-but-not-execution-derived, "
        "the same category grasp_pose already occupies.",
    ),
    "success": FieldSpec(Provenance.EXECUTION_DERIVED, "the label itself -- valid as a training target, not as model input"),
    "dz": FieldSpec(Provenance.EXECUTION_DERIVED, "obj_pos_after[2] - obj_pos_before[2] -- requires execution to have happened"),
    "fell_off": FieldSpec(Provenance.EXECUTION_DERIVED, "thresholded on obj_pos_after -- post-execution only"),
}

ALL_FIELDS = {**SOARM_FIELDS, **PIPER_FIELDS, **WORLD_MODEL_FIELDS}


class CausalValidityViolation(Exception):
    """Raised when a feature set includes fields not admissible for live,
    pre-execution candidate selection."""


def audit_feature_set(feature_names, *, context: str = "live candidate selection", registry=None):
    """Check every name in feature_names against the provenance registry.

    Raises CausalValidityViolation listing every EXECUTION_DERIVED or
    unregistered field if any are found. Returns True if all admissible.
    Unregistered fields are treated as violations (fail closed, not open) --
    an unaudited field must not be assumed safe.
    """
    registry = registry if registry is not None else ALL_FIELDS
    violations = []
    for name in feature_names:
        spec = registry.get(name)
        if spec is None:
            violations.append((name, "UNREGISTERED -- provenance not declared, cannot certify admissibility"))
        elif spec.provenance is Provenance.EXECUTION_DERIVED:
            violations.append((name, spec.reason))
    if violations:
        detail = "; ".join(f"{n} ({r})" for n, r in violations)
        raise CausalValidityViolation(
            f"{len(violations)} feature(s) not admissible for {context}: {detail}"
        )
    return True
