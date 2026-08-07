"""Read-only pad-contact fidelity diagnostic.

Classifies, from geometry alone, whether the legacy `bilateral_contact` /
`weld_triggered` signal is geometrically plausible, absent, or resting on
excessive (non-physical) interpenetration.

This module does not read or write MuJoCo state itself -- it operates on
plain per-timestep records the caller collects (see `PadFidelitySample`) and
is fully unit-testable without a simulator. `env_soarm.py`'s integration
(`enable_pad_fidelity_trace`) is a thin, strictly additive recording hook: it
appends samples during the existing close+settle window and never feeds
anything back into control, weld triggering, or the success rule. Off by
default, and its presence changes no key in `last_grasp_metrics` for any
caller that does not opt in. See
docs/JAW_METROLOGY_FINDINGS_20260807.md and
docs/JAW_CONTACT_MODEL_AB_20260807.md for why this exists: the legacy
success rule is built entirely on `bilateral_contact`, which step 3's A/B
showed can be driven by interpenetration a real pad could not produce
(scripts/compare_jaw_contact_models.py). This module gives that signal an
independent, purely geometric second opinion without touching it.

Explicitly NOT modified by this module or its integration: GRIP_CLOSED,
GRIP_OPEN, move_gripper(), weld triggering, contact solver parameters
(solref/solimp/friction), or any recorded success label. It reports; it does
not decide.

Threshold policy
-----------------
`PadFidelityConfig`'s defaults are derived from PAD GEOMETRY
(`tango_robot/jaw_pads.py`'s PAD_HALF_THICK / PAD_PROUD), not fit to any
trial's recorded success or failure. Fitting a "is this contact plausible"
threshold to observed outcomes would make the diagnostic circular -- its
whole purpose is to check outcomes independently of them. If you have a
better geometric or solver-derived basis for a threshold, override it
explicitly; do not tune these to move a particular trial's verdict.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class PadState(Enum):
    """Per-timestep classification from pad-to-object signed distances alone."""
    NO_BILATERAL = "NO_BILATERAL"
    PLAUSIBLE_BILATERAL = "PLAUSIBLE_BILATERAL"
    EXCESSIVE_PENETRATION = "EXCESSIVE_PENETRATION"
    AMBIGUOUS = "AMBIGUOUS"


class GeometricVerdict(Enum):
    """Trial-level reclassification -- a NEW label, never written over
    `success`/`bilateral_contact`/`weld_triggered`. Reported alongside them."""
    NO_ENGAGEMENT = "NO_ENGAGEMENT"
    PLAUSIBLE_ENGAGEMENT = "PLAUSIBLE_ENGAGEMENT"
    EXCESSIVE_PENETRATION_DOMINANT = "EXCESSIVE_PENETRATION_DOMINANT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class PadFidelityConfig:
    """Thresholds a caller may override. See module docstring: derived from
    pad geometry, not from any trial's outcome."""

    # A pad is "clear" of the object once its surface distance exceeds this.
    # ~1mm: comparable to PAD_PROUD (0.5mm, tango_robot/jaw_pads.py) plus a
    # little slack for solver/measurement noise.
    contact_tol_m: float = 0.001

    # Interpenetration up to this depth is treated as ordinary MuJoCo contact
    # softness rather than a geometry failure. 6mm = 2x the derived pad's own
    # full thickness (PAD_HALF_THICK=1.5mm -> 3mm thick box,
    # tango_robot/jaw_pads.py PAD_HALF_THICK). A rigid box sinking more than
    # twice its own thickness into an object is no longer explainable as pad
    # compression under a compliant contact model.
    plausible_penetration_max_m: float = 0.006

    # A per-step state must repeat for at least this many consecutive
    # (sub-sampled) samples to count in the persistence-based aggregates.
    # Guards single-sample solver noise from being read as sustained
    # engagement. Matches the 4x physics-step sub-sampling already used by
    # the existing close-window trace in env_soarm.py, so 8 samples ~= 32
    # physics steps ~= 64ms at the model's 2ms timestep.
    persistence_steps: int = 8


def classify_step(dist_fixed: Optional[float], dist_moving: Optional[float],
                  cfg: PadFidelityConfig = PadFidelityConfig()) -> PadState:
    """Classify one timestep from the two pads' signed surface distances.

    Positive = clear of the object. Negative = interpenetrating.
    EXCESSIVE_PENETRATION takes priority over every other state: a single pad
    buried past `plausible_penetration_max_m` marks the step excessive
    regardless of what the other pad reads, so a deep, unphysical penetration
    can never be absorbed into a "plausible" verdict by a good-looking
    opposite side.
    """
    if dist_fixed is None or dist_moving is None:
        return PadState.AMBIGUOUS

    def side(d: float) -> str:
        if d > cfg.contact_tol_m:
            return "clear"
        if d < -cfg.plausible_penetration_max_m:
            return "excessive"
        return "touching"

    sf, sm = side(dist_fixed), side(dist_moving)
    if sf == "excessive" or sm == "excessive":
        return PadState.EXCESSIVE_PENETRATION
    if sf == "clear" and sm == "clear":
        return PadState.NO_BILATERAL
    if sf == "touching" and sm == "touching":
        return PadState.PLAUSIBLE_BILATERAL
    return PadState.AMBIGUOUS   # unilateral: one clear, one touching


def find_runs(states: Sequence[PadState], min_len: int) -> Dict[PadState, List[Tuple[int, int]]]:
    """Runs of `min_len`+ consecutive identical states, as [start, end) index pairs."""
    runs: Dict[PadState, List[Tuple[int, int]]] = {s: [] for s in PadState}
    if not states:
        return runs
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            if i - start >= min_len:
                runs[states[start]].append((start, i))
            start = i
    return runs


@dataclass
class PadFidelitySample:
    """One timestep's raw observation. Every field the spec asked recorded.

    `weld_triggered`, `lifted`, `retained` are trial-level decisions made
    once, after the closing window this sample lives in has already ended --
    they are not physically meaningful per-timestep, so they default to
    None here and the run's actual values live on
    `PadFidelityTrial.final_weld_triggered` / `final_lifted` /
    `final_retained` instead. The fields stay on the sample for schema
    completeness (a future closing protocol might have a real per-step
    estimate) and so a caller who does have one can fill it in.
    """
    step: int
    t: Optional[float] = None
    pad_obj_dist_fixed_m: Optional[float] = None
    pad_obj_dist_moving_m: Optional[float] = None
    true_opening_m: Optional[float] = None
    grip_qpos_rad: Optional[float] = None
    bilateral_contact: Optional[bool] = None   # legacy signal, recorded not trusted
    weld_triggered: Optional[bool] = None
    obj_pos_rel_gripper_m: Optional[Tuple[float, float, float]] = None
    lifted: Optional[bool] = None
    retained: Optional[bool] = None


@dataclass
class PadFidelityTrial:
    """One grasp attempt's full sample sequence plus the legacy trial-level
    labels it is being cross-checked against."""
    object_name: Optional[str] = None
    seed: Optional[int] = None
    samples: List[PadFidelitySample] = field(default_factory=list)
    cfg: PadFidelityConfig = field(default_factory=PadFidelityConfig)
    final_bilateral_contact: Optional[bool] = None
    final_weld_triggered: Optional[bool] = None
    final_lifted: Optional[bool] = None
    final_retained: Optional[bool] = None
    final_success: Optional[bool] = None

    def states(self) -> List[PadState]:
        return [classify_step(s.pad_obj_dist_fixed_m, s.pad_obj_dist_moving_m, self.cfg)
                for s in self.samples]

    def persistent_runs(self) -> Dict[PadState, List[Tuple[int, int]]]:
        return find_runs(self.states(), self.cfg.persistence_steps)

    def geometric_verdict(self) -> GeometricVerdict:
        """Trial-level reclassification. Computed ONLY from the geometric
        per-step states -- takes no legacy field as input -- so it cannot be
        made to agree with `success` by construction.

        EXCESSIVE_PENETRATION_DOMINANT is checked first and wins outright:
        a trial with a persistent excessive-penetration run is never reported
        as PLAUSIBLE_ENGAGEMENT, even if a persistent plausible run also
        occurred elsewhere in the same trial.
        """
        runs = self.persistent_runs()
        if runs[PadState.EXCESSIVE_PENETRATION]:
            return GeometricVerdict.EXCESSIVE_PENETRATION_DOMINANT
        if runs[PadState.PLAUSIBLE_BILATERAL]:
            return GeometricVerdict.PLAUSIBLE_ENGAGEMENT
        if runs[PadState.NO_BILATERAL] and not runs[PadState.AMBIGUOUS]:
            return GeometricVerdict.NO_ENGAGEMENT
        return GeometricVerdict.AMBIGUOUS

    def _dist_series(self) -> Tuple[List[float], List[float]]:
        fs = [s.pad_obj_dist_fixed_m for s in self.samples if s.pad_obj_dist_fixed_m is not None]
        ms = [s.pad_obj_dist_moving_m for s in self.samples if s.pad_obj_dist_moving_m is not None]
        return fs, ms

    def confusion_row(self) -> Counter:
        """Per-step (legacy bilateral_contact, geometric PadState) pair counts.

        This is the fine-grained version of item 4 -- whether MuJoCo's own
        live contact list agrees with a purely distance-based read of the
        same geometry at the same instant.
        """
        c: Counter = Counter()
        for s, st in zip(self.samples, self.states()):
            c[(s.bilateral_contact, st.value)] += 1
        return c

    def summary(self) -> dict:
        """Items 1-3 and 6 for this one trial: distances, engagement/excessive
        durations (in samples), and the reclassification verdict, reported
        alongside -- never in place of -- the legacy labels."""
        fs, ms = self._dist_series()
        runs = self.persistent_runs()

        def stat(vals, fn):
            return fn(vals) if vals else None

        def run_len(state):
            return sum(e - s for s, e in runs[state])

        return {
            "object": self.object_name,
            "seed": self.seed,
            "n_samples": len(self.samples),
            "min_pad_dist_fixed_m": stat(fs, min),
            "median_pad_dist_fixed_m": stat(fs, statistics.median),
            "final_pad_dist_fixed_m": fs[-1] if fs else None,
            "min_pad_dist_moving_m": stat(ms, min),
            "median_pad_dist_moving_m": stat(ms, statistics.median),
            "final_pad_dist_moving_m": ms[-1] if ms else None,
            "bilateral_engagement_samples": run_len(PadState.PLAUSIBLE_BILATERAL),
            "excessive_penetration_samples": run_len(PadState.EXCESSIVE_PENETRATION),
            "no_bilateral_samples": run_len(PadState.NO_BILATERAL),
            "ambiguous_samples": run_len(PadState.AMBIGUOUS),
            "geometric_verdict": self.geometric_verdict().value,
            "legacy_bilateral_contact": self.final_bilateral_contact,
            "legacy_weld_triggered": self.final_weld_triggered,
            "legacy_lifted": self.final_lifted,
            "legacy_retained": self.final_retained,
            "legacy_success": self.final_success,
        }
