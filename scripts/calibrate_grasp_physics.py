#!/usr/bin/env python3
"""
SO-ARM101 physics grasp calibration — MuJoCo backend.

Systematically sweeps grasp parameters on primitive shapes (cube/cylinder/
sphere) to find configurations that produce stable lift-off.  Banana is held
out as final validation.

Usage
-----
# Single trial, interactive viewer
MUJOCO_GL=glfw python scripts/calibrate_grasp_physics.py \\
    --shape cube --vis

# Full sweep on all primitives (headless, ~5–10 min)
MUJOCO_GL=egl python scripts/calibrate_grasp_physics.py \\
    --sweep --shapes cube,cylinder,sphere

# Validate best params on banana
MUJOCO_GL=egl python scripts/calibrate_grasp_physics.py \\
    --validate

# Quick smoke test (2 seeds, narrow sweep)
MUJOCO_GL=egl python scripts/calibrate_grasp_physics.py \\
    --shape cube --quick

Output
------
  calib_logs/
    trials.csv               — all trial metrics
    best_params.json         — best parameter combo per shape
    timeline_<id>.png        — per-trial time-series plots
    heatmap_<shape>.png      — success-rate heatmap
    replay_<id>/             — rendered frames (if --save-frames)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    GRASP_MODE_PHYSICS,
    GRIP_CLOSED,
    GRIP_OPEN,
    TABLE_TOP_Z,
)

# GRIP_REDUCTION is a class attribute on EnvironmentSoArm
GRIP_REDUCTION = EnvironmentSoArm.GRIP_REDUCTION

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("calib_logs")

# ── primitive shape definitions (size = MuJoCo half-extents) ─────────────────
SHAPES: Dict[str, dict] = {
    "cube": dict(
        shape="box",
        size=(0.025, 0.025, 0.025),   # 5 × 5 × 5 cm
        mass=0.150,
        friction=(1.5, 0.05, 0.01),
        rgba=(0.2, 0.5, 0.8, 1.0),
    ),
    "cylinder": dict(
        shape="cylinder",
        size=(0.025, 0.030),           # r=2.5 cm, h=6 cm
        mass=0.120,
        friction=(1.5, 0.05, 0.01),
        rgba=(0.2, 0.7, 0.3, 1.0),
    ),
    "sphere": dict(
        shape="sphere",
        size=(0.028,),                 # r=2.8 cm
        mass=0.100,
        friction=(1.5, 0.05, 0.01),
        rgba=(0.8, 0.6, 0.1, 1.0),
    ),
}

# ── default sweep grid ────────────────────────────────────────────────────────
# NOTE: SO-ARM101 jaw bodies are 4-9cm in +Y of the EEF due to oblique IK approach.
# y_approach_offsets < 0 moves the arm "past" the object to bring jaws into contact.
# The arm's physical Y limit caps overshoot at ~−0.025m from nominal.
DEFAULT_DESCEND_OFFSETS   = [-0.04, -0.02, 0.00, 0.02, 0.04]   # m relative to obj CoM z
DEFAULT_OPEN_LENGTHS      = [0.04, 0.06, 0.08, 0.09]            # m (pre-grasp opening)
DEFAULT_CLOSE_FRACTIONS   = [0.80, 0.90, 1.00]                  # 1.0 = fully closed
DEFAULT_Y_APPROACH_OFFSETS = [0.00, -0.02, -0.04]               # m; negative = overshoot


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibParams:
    """All per-trial configurable parameters."""
    # --- approach ---
    descend_z_offset:  float = 0.00    # m relative to object CoM z
    y_approach_offset: float = 0.00    # m added to approach y (negative = past object).
                                       # SO-ARM101 jaw bodies are ~4-9cm behind EEF in Y
                                       # due to oblique approach; use ≈−0.04 to compensate.
    open_length:       float = 0.07    # m — pre-grasp gripper opening
    close_fraction:    float = 1.00    # fraction of full close (1.0 = GRIP_CLOSED)
    close_steps:       int   = 100     # physics steps while closing
    hold_steps:        int   = 80      # steps to hold after closing before lift
    lift_height:       float = TABLE_TOP_Z + 0.20  # m absolute z
    lift_max_steps:    int   = 300     # max steps for lift move_ee call
    settle_steps:      int   = 300     # steps after load_obj before grasp
    # --- dynamic friction overrides (applied to model in-place) ---
    finger_friction:   Optional[float] = None   # None → keep model default
    obj_friction:      Optional[float] = None   # None → keep object default


@dataclass
class StepSnapshot:
    """One timestep of per-step recording."""
    t:              int
    phase:          str        # "close" | "lift"
    grip_angle:     float
    eef_z:          float
    obj_z:          float
    n_contacts:     int
    max_force:      float
    fixed_jaw_z:    float
    mv_jaw_z:       float
    finger_gap:     float      # Euclidean distance between jaw body origins


@dataclass
class TrialMetrics:
    """Full record for one calibration trial."""
    trial_id:        int
    shape_name:      str
    shape_type:      str
    seed:            int

    # params
    descend_z_offset:  float
    y_approach_offset: float = 0.0
    open_length:       float = 0.07
    close_fraction:    float = 1.0
    close_steps:       int   = 100
    hold_steps:        int   = 80
    finger_friction:   float = -1.0
    obj_friction:      float = -1.0

    # geometry at grasp
    obj_com_z:         float = TABLE_TOP_Z
    descend_z_actual:  float = TABLE_TOP_Z   # clipped EEF target z
    eef_z_reached:     float = TABLE_TOP_Z   # actual EEF z after IK
    fixed_jaw_pos:     Tuple[float, float, float] = field(default_factory=lambda: (0., 0., 0.))
    mv_jaw_pos:        Tuple[float, float, float] = field(default_factory=lambda: (0., 0., 0.))
    finger_gap_open:   float = 0.
    finger_gap_closed: float = 0.

    # contacts & forces at close
    n_contacts_after_close:  int   = 0
    max_force_after_close:   float = 0.
    contact_bool:            bool  = False

    # object dynamics
    obj_vel_after_close:     float = 0.
    obj_z_after_close:       float = 0.
    obj_orient_change:       float = 0.   # quaternion distance from initial

    # outcome
    grasped:         bool  = False
    obj_z_before:    float = TABLE_TOP_Z
    obj_z_after:     float = TABLE_TOP_Z
    dz:              float = 0.
    lifted:          bool  = False
    success:         bool  = False

    # time-series (not written to CSV)
    timeline: List[StepSnapshot] = field(default_factory=list, repr=False)

    def to_csv_row(self) -> dict:
        d = asdict(self)
        d.pop("timeline", None)
        # flatten tuples
        for k in ("fixed_jaw_pos", "mv_jaw_pos"):
            if isinstance(d[k], (list, tuple)):
                x, y, z = d[k]
                d.pop(k)
                d[f"{k}_x"] = x
                d[f"{k}_y"] = y
                d[f"{k}_z"] = z
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Contact / physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_contact_info(model, data, jaw_ids: List[int], obj_body_ids: set
                      ) -> Tuple[int, float]:
    """Return (n_contacts_with_object, max_normal_force)."""
    n = 0
    max_f = 0.0
    force_buf = np.zeros(6)
    for i in range(data.ncon):
        c  = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        jaw_side = b1 in jaw_ids or b2 in jaw_ids
        obj_side = b1 in obj_body_ids or b2 in obj_body_ids
        if jaw_side and obj_side:
            n += 1
            mujoco.mj_contactForce(model, data, i, force_buf)
            max_f = max(max_f, abs(float(force_buf[0])))
    return n, max_f


def _finger_gap(data, fixed_jaw_id: int, mv_jaw_id: int) -> float:
    """Euclidean distance between fixed-jaw and moving-jaw body origins."""
    p1 = data.xpos[fixed_jaw_id]
    p2 = data.xpos[mv_jaw_id]
    return float(np.linalg.norm(p1 - p2))


def _obj_lin_vel(model, data, pool_slot: int) -> float:
    """Sum of |linear velocity| for the object at pool_slot."""
    jnt  = model.joint(f"obj_joint_{pool_slot}")
    vadr = jnt.dofadr[0]
    return float(np.abs(data.qvel[vadr:vadr + 3]).sum())


def _obj_quat(model, data, pool_slot: int) -> np.ndarray:
    jnt = model.joint(f"obj_joint_{pool_slot}")
    adr = jnt.qposadr[0]
    return data.qpos[adr + 3: adr + 7].copy()


def _quat_dist(q1: np.ndarray, q2: np.ndarray) -> float:
    """Quaternion angular distance in radians."""
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)


def _set_finger_friction(model, jaw_ids: List[int], friction: float):
    """In-place update of gripper geom friction (no model rebuild needed)."""
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] in jaw_ids:
            model.geom_friction[geom_id, 0] = friction


def _set_obj_friction(model, obj_body_id: int, friction: float):
    """In-place update of object geom friction."""
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == obj_body_id:
            model.geom_friction[geom_id, 0] = friction


# ─────────────────────────────────────────────────────────────────────────────
# Core calibration runner
# ─────────────────────────────────────────────────────────────────────────────

class GraspCalibrator:
    """
    Runs parametric physics-grasp trials on a single primitive shape.

    Parameters
    ----------
    shape_name : str        key in SHAPES dict
    vis        : bool       open MuJoCo viewer
    seed       : int        RNG seed for spawn jitter
    save_frames: bool       render and save frames for replay video
    record_steps: bool      capture per-step snapshots in TrialMetrics.timeline
    out_dir    : Path       output root
    """

    def __init__(
        self,
        shape_name:   str  = "cube",
        vis:          bool = False,
        seed:         int  = 0,
        save_frames:  bool = False,
        record_steps: bool = True,
        out_dir:      Path = OUT_DIR,
    ):
        self.shape_name   = shape_name
        self.shape_cfg    = SHAPES[shape_name]
        self.vis          = vis
        self.seed         = seed
        self.save_frames  = save_frames
        self.record_steps = record_steps
        self.out_dir      = Path(out_dir)
        self._trial_id    = 0

        self.env: Optional[EnvironmentSoArm] = None
        self._obj_id:     Optional[int]  = None
        self._pool_slot:  Optional[int]  = None
        self._obj_com_z:  float          = TABLE_TOP_Z + 0.05

        self._orig_finger_friction: Optional[np.ndarray] = None
        self._orig_obj_friction:    Optional[np.ndarray] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def setup(self):
        """Create environment and load the primitive object once."""
        self.env = EnvironmentSoArm(vis=self.vis, grasp_mode=GRASP_MODE_PHYSICS)
        self._reload_obj()

    def _reload_obj(self, rng_seed: Optional[int] = None):
        """(Re-)spawn the object with a small random XY jitter."""
        rng = np.random.default_rng(rng_seed if rng_seed is not None else self.seed)
        jitter_x = float(rng.uniform(-0.02, 0.02))
        jitter_y = float(rng.uniform(-0.02, 0.02))

        env = self.env
        env.remove_all_obj()
        env.reset_robot()

        self._obj_id = env.load_primitive(
            **self.shape_cfg,
            pos=[jitter_x, -0.40 + jitter_y, TABLE_TOP_Z + max(self.shape_cfg["size"]) + 0.06],
        )
        self._pool_slot = env._pool_slots[0]

        # Save baseline frictions for restoration
        jaw_ids  = [env._jaw_body_id, env._jaw_mv_body_id]
        obj_body = env.model.body(f"obj_{self._pool_slot}").id
        self._orig_finger_friction = env.model.geom_friction[
            [g for g in range(env.model.ngeom)
             if env.model.geom_bodyid[g] in jaw_ids], 0
        ].copy() if any(env.model.geom_bodyid[g] in jaw_ids
                        for g in range(env.model.ngeom)) else None
        self._orig_obj_friction = env.model.geom_friction[
            [g for g in range(env.model.ngeom)
             if env.model.geom_bodyid[g] == obj_body], 0
        ].copy() if any(env.model.geom_bodyid[g] == obj_body
                        for g in range(env.model.ngeom)) else None

    def close(self):
        if self.env:
            self.env.close()
            self.env = None

    # ── single trial ──────────────────────────────────────────────────────────

    def run_trial(self, params: CalibParams, trial_seed: int = 0) -> TrialMetrics:
        """Execute one grasp attempt with the given params; return all metrics."""
        env        = self.env
        tid        = self._trial_id
        self._trial_id += 1

        # ── respawn object ────────────────────────────────────────────────────
        self._reload_obj(rng_seed=trial_seed)

        # settle physics
        env._steps(params.settle_steps)
        env.wait_until_all_still(max_wait_epochs=100)

        obj_com_z  = float(env.get_obj_pos(self._obj_id)[2])
        obj_quat0  = _obj_quat(env.model, env.data, self._pool_slot)

        m = TrialMetrics(
            trial_id       = tid,
            shape_name     = self.shape_name,
            shape_type     = self.shape_cfg["shape"],
            seed           = trial_seed,
            descend_z_offset  = params.descend_z_offset,
            y_approach_offset = params.y_approach_offset,
            open_length    = params.open_length,
            close_fraction = params.close_fraction,
            close_steps    = params.close_steps,
            hold_steps     = params.hold_steps,
            finger_friction = params.finger_friction or -1.0,
            obj_friction   = params.obj_friction    or -1.0,
            obj_com_z      = obj_com_z,
            descend_z_actual = 0.0,
            eef_z_reached  = 0.0,
            obj_z_before   = obj_com_z,
        )

        # ── apply friction overrides ──────────────────────────────────────────
        jaw_ids  = [env._jaw_body_id, env._jaw_mv_body_id]
        obj_body = env.model.body(f"obj_{self._pool_slot}").id

        if params.finger_friction is not None:
            _set_finger_friction(env.model, jaw_ids, params.finger_friction)
        if params.obj_friction is not None:
            _set_obj_friction(env.model, obj_body, params.obj_friction)

        # ── reset arm + open gripper ──────────────────────────────────────────
        env.reset_robot()
        env.move_gripper(params.open_length)

        # record open finger gap
        m.finger_gap_open = _finger_gap(env.data, env._jaw_body_id, env._jaw_mv_body_id)

        # ── approach ──────────────────────────────────────────────────────────
        x = float(env.get_obj_pos(self._obj_id)[0])
        y = float(env.get_obj_pos(self._obj_id)[1])
        # y_approach_offset compensates for SO-ARM101 jaw Y-offset: jaws are 4-9cm
        # behind EEF in Y due to oblique IK approach.  Negative = overshoot past obj.
        y_approach = y + params.y_approach_offset

        target_z = float(np.clip(
            obj_com_z + params.descend_z_offset,
            TABLE_TOP_Z,
            TABLE_TOP_Z + 0.40,
        ))
        m.descend_z_actual = target_z

        # move to above-object position first
        env.move_ee([x, y_approach, env.GRIPPER_MOVING_HEIGHT, None])
        # descend to grasp height
        env.move_ee([x, y_approach, target_z, None])

        m.eef_z_reached = float(env._get_eef_pos()[2])
        m.fixed_jaw_pos = tuple(float(v) for v in env.data.xpos[env._jaw_body_id])
        m.mv_jaw_pos    = tuple(float(v) for v in env.data.xpos[env._jaw_mv_body_id])

        # ── close gripper ─────────────────────────────────────────────────────
        close_target_angle = (
            GRIP_CLOSED
            + (1.0 - params.close_fraction) * (GRIP_OPEN - GRIP_CLOSED)
        )
        timeline: List[StepSnapshot] = []

        for step_i in range(params.close_steps):
            t = 1.0 - step_i / params.close_steps
            blend = params.close_fraction * (1.0 - t)    # ramp from 0 → close_fraction
            angle = GRIP_OPEN + blend * (GRIP_CLOSED - GRIP_OPEN)
            env.data.ctrl[env._grip_act_id] = angle
            env.step_simulation()

            if self.record_steps:
                nc, mf = _get_contact_info(
                    env.model, env.data, jaw_ids, {obj_body})
                snap = StepSnapshot(
                    t           = step_i,
                    phase       = "close",
                    grip_angle  = float(env.data.qpos[env._grip_qpos_adr]),
                    eef_z       = float(env._get_eef_pos()[2]),
                    obj_z       = float(env.get_obj_pos(self._obj_id)[2]),
                    n_contacts  = nc,
                    max_force   = mf,
                    fixed_jaw_z = float(env.data.xpos[env._jaw_body_id][2]),
                    mv_jaw_z    = float(env.data.xpos[env._jaw_mv_body_id][2]),
                    finger_gap  = _finger_gap(env.data, env._jaw_body_id, env._jaw_mv_body_id),
                )
                timeline.append(snap)

        # hold at final close angle
        env.data.ctrl[env._grip_act_id] = close_target_angle
        env._steps(params.hold_steps)

        # ── measure at-close state ────────────────────────────────────────────
        m.finger_gap_closed = _finger_gap(env.data, env._jaw_body_id, env._jaw_mv_body_id)
        nc, mf = _get_contact_info(env.model, env.data, jaw_ids, {obj_body})
        m.n_contacts_after_close = nc
        m.max_force_after_close  = mf
        m.contact_bool           = len(env.check_grasped_id()) > 0
        m.obj_z_after_close      = float(env.get_obj_pos(self._obj_id)[2])
        m.obj_vel_after_close    = _obj_lin_vel(env.model, env.data, self._pool_slot)
        quat_now = _obj_quat(env.model, env.data, self._pool_slot)
        m.obj_orient_change      = float(_quat_dist(obj_quat0, quat_now))

        # ── lift ──────────────────────────────────────────────────────────────
        env.move_ee([x, y_approach, params.lift_height, None], max_step=params.lift_max_steps)
        env._steps(100)

        grasped_ids = env.check_grasped_id()

        if self.record_steps:
            for step_i in range(50):
                env.step_simulation()
                nc2, mf2 = _get_contact_info(env.model, env.data, jaw_ids, {obj_body})
                snap = StepSnapshot(
                    t           = params.close_steps + step_i,
                    phase       = "lift",
                    grip_angle  = float(env.data.qpos[env._grip_qpos_adr]),
                    eef_z       = float(env._get_eef_pos()[2]),
                    obj_z       = float(env.get_obj_pos(self._obj_id)[2]),
                    n_contacts  = nc2,
                    max_force   = mf2,
                    fixed_jaw_z = float(env.data.xpos[env._jaw_body_id][2]),
                    mv_jaw_z    = float(env.data.xpos[env._jaw_mv_body_id][2]),
                    finger_gap  = _finger_gap(env.data, env._jaw_body_id,
                                             env._jaw_mv_body_id),
                )
                timeline.append(snap)

        # ── final metrics ─────────────────────────────────────────────────────
        m.grasped     = len(grasped_ids) > 0
        m.obj_z_after = float(env.get_obj_pos(self._obj_id)[2])
        m.dz          = m.obj_z_after - m.obj_z_before
        m.lifted      = m.obj_z_after > TABLE_TOP_Z + 0.07
        m.success     = m.grasped and m.lifted
        m.timeline    = timeline

        # ── restore frictions ─────────────────────────────────────────────────
        self._restore_frictions(env, jaw_ids, obj_body)

        return m

    def _restore_frictions(self, env, jaw_ids, obj_body):
        if self._orig_finger_friction is not None:
            geom_ids = [g for g in range(env.model.ngeom)
                        if env.model.geom_bodyid[g] in jaw_ids]
            for i, gid in enumerate(geom_ids):
                if i < len(self._orig_finger_friction):
                    env.model.geom_friction[gid, 0] = self._orig_finger_friction[i]
        if self._orig_obj_friction is not None:
            geom_ids = [g for g in range(env.model.ngeom)
                        if env.model.geom_bodyid[g] == obj_body]
            for i, gid in enumerate(geom_ids):
                if i < len(self._orig_obj_friction):
                    env.model.geom_friction[gid, 0] = self._orig_obj_friction[i]

    # ── sweep ─────────────────────────────────────────────────────────────────

    def sweep(
        self,
        descend_offsets:   List[float] = DEFAULT_DESCEND_OFFSETS,
        open_lengths:      List[float] = DEFAULT_OPEN_LENGTHS,
        close_fractions:   List[float] = DEFAULT_CLOSE_FRACTIONS,
        y_approach_offsets: List[float] = DEFAULT_Y_APPROACH_OFFSETS,
        finger_frictions:  List[Optional[float]] = [None],
        obj_frictions:     List[Optional[float]] = [None],
        n_seeds:           int = 3,
    ) -> List[TrialMetrics]:
        """Grid sweep over all parameter combinations."""
        grid = list(product(
            descend_offsets, open_lengths, close_fractions,
            y_approach_offsets, finger_frictions, obj_frictions,
            range(n_seeds),
        ))
        total = len(grid)
        print(f"\n[calib] sweep: {total} trials on '{self.shape_name}'")
        results: List[TrialMetrics] = []

        for i, (dz_off, ol, cf, y_off, ff, of_, seed) in enumerate(grid):
            p = CalibParams(
                descend_z_offset  = dz_off,
                y_approach_offset = y_off,
                open_length       = ol,
                close_fraction    = cf,
                finger_friction   = ff,
                obj_friction      = of_,
            )
            t0 = time.time()
            m  = self.run_trial(p, trial_seed=seed)
            dt = time.time() - t0
            sym = "✓" if m.success else ("~" if m.contact_bool else "✗")
            print(
                f"  [{sym}] trial {i+1:3d}/{total}"
                f"  dz_off={dz_off:+.3f}  y_off={y_off:+.3f}  open={ol:.2f}  cf={cf:.2f}"
                f"  seed={seed}"
                f"  contact={m.contact_bool}  lifted={m.lifted}"
                f"  dz={m.dz:+.3f}m  ({dt:.1f}s)"
            )
            results.append(m)

        return results

    # ── diagnostics ───────────────────────────────────────────────────────────

    def run_diagnostic(self, params: Optional[CalibParams] = None) -> TrialMetrics:
        """Run one verbose trial and print detailed debug info at each phase."""
        if params is None:
            params = CalibParams()
        env = self.env

        print("\n" + "=" * 60)
        print(f"DIAGNOSTIC GRASP — shape={self.shape_name}")
        print("=" * 60)

        self._reload_obj(rng_seed=0)
        env._steps(params.settle_steps)
        env.wait_until_all_still(max_wait_epochs=100)

        obj_pos = env.get_obj_pos(self._obj_id)
        pool_slot = self._pool_slot
        jaw_ids   = [env._jaw_body_id, env._jaw_mv_body_id]
        obj_body  = env.model.body(f"obj_{pool_slot}").id

        print(f"\n[0] Object spawned:")
        print(f"    CoM pos     : {obj_pos}")
        print(f"    pool_slot   : {pool_slot}")
        print(f"    jaw body IDs: fixed={env._jaw_body_id}  moving={env._jaw_mv_body_id}")

        # gripper model info
        grip_jnt  = env.model.joint("gripper")
        print(f"\n[0] Gripper joint:")
        print(f"    range       : {np.degrees(grip_jnt.range)} deg")
        print(f"    GRIP_OPEN   : {GRIP_OPEN:.3f} rad ({np.degrees(GRIP_OPEN):.1f}°)")
        print(f"    GRIP_CLOSED : {GRIP_CLOSED:.3f} rad ({np.degrees(GRIP_CLOSED):.1f}°)")
        print(f"    GRIP_REDUC  : {GRIP_REDUCTION}")

        # finger geometry at home
        env.reset_robot()
        env.move_gripper(params.open_length)
        fixed_pos = env.data.xpos[env._jaw_body_id].copy()
        mv_pos    = env.data.xpos[env._jaw_mv_body_id].copy()
        eef_pos   = env._get_eef_pos()
        print(f"\n[1] After open gripper (opening={params.open_length:.3f}m):")
        print(f"    EEF (gripperframe): {eef_pos}")
        print(f"    fixed_jaw   : {fixed_pos}")
        print(f"    moving_jaw  : {mv_pos}")
        print(f"    finger gap  : {_finger_gap(env.data, env._jaw_body_id, env._jaw_mv_body_id):.4f}m")
        print(f"    grip angle  : {env.data.qpos[env._grip_qpos_adr]:.4f} rad")

        # approach
        x, y = float(obj_pos[0]), float(obj_pos[1])
        target_z = float(np.clip(
            float(obj_pos[2]) + params.descend_z_offset,
            TABLE_TOP_Z, TABLE_TOP_Z + 0.40,
        ))
        print(f"\n[2] Approach:")
        print(f"    obj CoM z        : {obj_pos[2]:.4f}m")
        print(f"    descend_z_offset : {params.descend_z_offset:+.4f}m")
        print(f"    target EEF z     : {target_z:.4f}m")
        print(f"    ee_pos_limit z   : {env.ee_position_limit[2]}")

        env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])
        env.move_ee([x, y, target_z, None])

        eef_reached = env._get_eef_pos()
        fixed_pos2  = env.data.xpos[env._jaw_body_id].copy()
        mv_pos2     = env.data.xpos[env._jaw_mv_body_id].copy()
        print(f"    EEF reached      : {eef_reached}  (err z={abs(eef_reached[2]-target_z)*1000:.1f}mm)")
        print(f"    fixed_jaw @ approach: {fixed_pos2}")
        print(f"    moving_jaw @ approach: {mv_pos2}")
        print(f"    finger gap @ approach: {_finger_gap(env.data, env._jaw_body_id, env._jaw_mv_body_id):.4f}m")

        # is obj between fingers?
        obj_x, obj_y, obj_z = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])
        fj_x, fj_y, fj_z = fixed_pos2
        mj_x, mj_y, mj_z = mv_pos2
        print(f"\n[2] Object-finger geometry:")
        print(f"    obj CoM x={obj_x:.4f}  y={obj_y:.4f}  z={obj_z:.4f}")
        print(f"    fixed jaw z offset from obj CoM : {fj_z - obj_z:+.4f}m")
        print(f"    moving jaw z offset from obj CoM: {mj_z - obj_z:+.4f}m")
        in_z = min(fj_z, mj_z) <= obj_z <= max(fj_z, mj_z)
        print(f"    object CoM z IS between jaws    : {in_z}")
        # Key Y-gap diagnostic: jaw bodies are always ~4-9cm in +Y from EEF
        # due to SO-ARM101's oblique approach.  Negative jaw_y_gap = jaws behind object.
        fj_y_gap = float(env.data.xpos[env._jaw_body_id][1]) - obj_y
        mj_y_gap = float(env.data.xpos[env._jaw_mv_body_id][1]) - obj_y
        eef_y = float(env._get_eef_pos()[1])
        print(f"    fixed jaw Y offset from obj CoM : {fj_y_gap:+.4f}m  (jaw is {'behind' if fj_y_gap>0 else 'past'} obj)")
        print(f"    moving jaw Y offset from obj CoM: {mj_y_gap:+.4f}m  (jaw is {'behind' if mj_y_gap>0 else 'past'} obj)")
        print(f"    EEF y={eef_y:.4f}  jaw_body Y-offset from EEF: fixed={float(env.data.xpos[env._jaw_body_id][1])-eef_y:+.4f}  mv={float(env.data.xpos[env._jaw_mv_body_id][1])-eef_y:+.4f}")
        print(f"    NOTE: jaw mesh tips extend ~6-7cm further in -Y than jaw body origin.")

        # geom friction info
        print(f"\n[2] Friction (gripper geoms):")
        for g in range(env.model.ngeom):
            if env.model.geom_bodyid[g] in jaw_ids:
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
                print(f"    {name}: friction={env.model.geom_friction[g]}")
        print(f"[2] Friction (object geom):")
        for g in range(env.model.ngeom):
            if env.model.geom_bodyid[g] == obj_body:
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
                print(f"    {name}: friction={env.model.geom_friction[g]}")

        # close
        print(f"\n[3] Closing gripper (close_fraction={params.close_fraction:.2f}, steps={params.close_steps})...")
        for step_i in range(params.close_steps):
            t = 1.0 - step_i / params.close_steps
            angle = GRIP_OPEN + params.close_fraction * (1.0 - t) * (GRIP_CLOSED - GRIP_OPEN)
            env.data.ctrl[env._grip_act_id] = angle
            env.step_simulation()

        close_target = GRIP_CLOSED + (1.0 - params.close_fraction) * (GRIP_OPEN - GRIP_CLOSED)
        env.data.ctrl[env._grip_act_id] = close_target
        env._steps(params.hold_steps)

        nc, mf = _get_contact_info(env.model, env.data, jaw_ids, {obj_body})
        gap_closed = _finger_gap(env.data, env._jaw_body_id, env._jaw_mv_body_id)
        vel_after  = _obj_lin_vel(env.model, env.data, pool_slot)
        grasped_pre = env.check_grasped_id()

        print(f"    grip angle now    : {env.data.qpos[env._grip_qpos_adr]:.4f} rad")
        print(f"    finger gap closed : {gap_closed:.4f}m")
        print(f"    contacts with obj : {nc}")
        print(f"    max contact force : {mf:.3f}N")
        print(f"    object lin vel    : {vel_after:.6f}m/s")
        print(f"    check_grasped_id  : {grasped_pre}")
        print(f"    contact=          : {len(grasped_pre)>0}")

        # lift
        print(f"\n[4] Lifting to z={params.lift_height:.3f}m ...")
        env.move_ee([x, y, params.lift_height, None], max_step=params.lift_max_steps)
        env._steps(100)

        grasped_post = env.check_grasped_id()
        obj_z_after  = float(env.get_obj_pos(self._obj_id)[2])
        dz           = obj_z_after - float(obj_pos[2])
        lifted       = obj_z_after > TABLE_TOP_Z + 0.07
        success      = len(grasped_post) > 0 and lifted

        print(f"    check_grasped_id  : {grasped_post}")
        print(f"    obj_z before      : {obj_pos[2]:.4f}m")
        print(f"    obj_z after       : {obj_z_after:.4f}m")
        print(f"    dz                : {dz:+.4f}m")
        print(f"    lifted (>7cm)     : {lifted}")
        print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}")
        print("=" * 60 + "\n")

        return TrialMetrics(
            trial_id=self._trial_id - 1,
            shape_name=self.shape_name,
            shape_type=self.shape_cfg["shape"],
            seed=0,
            descend_z_offset=params.descend_z_offset,
            open_length=params.open_length,
            close_fraction=params.close_fraction,
            close_steps=params.close_steps,
            hold_steps=params.hold_steps,
            finger_friction=params.finger_friction or -1.,
            obj_friction=params.obj_friction or -1.,
            obj_com_z=float(obj_pos[2]),
            descend_z_actual=target_z,
            eef_z_reached=float(eef_reached[2]),
            n_contacts_after_close=nc,
            max_force_after_close=mf,
            contact_bool=len(grasped_pre) > 0,
            obj_z_before=float(obj_pos[2]),
            obj_z_after=obj_z_after,
            dz=dz,
            lifted=lifted,
            success=success,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Banana validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_banana(params: CalibParams, vis: bool = False, n_seeds: int = 5
                    ) -> List[TrialMetrics]:
    """Run banana holdout validation with the given params."""
    print(f"\n[validate] Banana holdout ({n_seeds} seeds):")
    env = EnvironmentSoArm(vis=vis, grasp_mode=GRASP_MODE_PHYSICS)
    results: List[TrialMetrics] = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        jx  = float(rng.uniform(-0.02, 0.02))
        jy  = float(rng.uniform(-0.02, 0.02))
        env.remove_all_obj()
        env.reset_robot()
        obj_id = env.load_obj("YcbBanana", pos=[jx, -0.40 + jy, TABLE_TOP_Z + 0.12])
        env._steps(300)
        env.wait_until_all_still(max_wait_epochs=150)

        obj_pos   = env.get_obj_pos(obj_id)
        pool_slot = env._pool_slots[0]
        jaw_ids   = [env._jaw_body_id, env._jaw_mv_body_id]
        obj_body  = env.model.body(f"obj_{pool_slot}").id

        if params.finger_friction is not None:
            _set_finger_friction(env.model, jaw_ids, params.finger_friction)
        if params.obj_friction is not None:
            _set_obj_friction(env.model, obj_body, params.obj_friction)

        env.reset_robot()
        env.move_gripper(params.open_length)

        x, y = float(obj_pos[0]), float(obj_pos[1])
        target_z = float(np.clip(
            float(obj_pos[2]) + params.descend_z_offset,
            TABLE_TOP_Z, TABLE_TOP_Z + 0.40,
        ))

        env.move_ee([x, y, env.GRIPPER_MOVING_HEIGHT, None])
        env.move_ee([x, y, target_z, None])

        close_target = GRIP_CLOSED + (1.0 - params.close_fraction) * (GRIP_OPEN - GRIP_CLOSED)
        for step_i in range(params.close_steps):
            t = 1.0 - step_i / params.close_steps
            angle = GRIP_OPEN + params.close_fraction * (1.0 - t) * (GRIP_CLOSED - GRIP_OPEN)
            env.data.ctrl[env._grip_act_id] = angle
            env.step_simulation()
        env.data.ctrl[env._grip_act_id] = close_target
        env._steps(params.hold_steps)

        contact_bool = len(env.check_grasped_id()) > 0
        env.move_ee([x, y, params.lift_height, None], max_step=params.lift_max_steps)
        env._steps(100)

        grasped   = len(env.check_grasped_id()) > 0
        obj_z_aft = float(env.get_obj_pos(obj_id)[2])
        dz        = obj_z_aft - float(obj_pos[2])
        lifted    = obj_z_aft > TABLE_TOP_Z + 0.07
        sym       = "✓" if (grasped and lifted) else "✗"
        print(f"  [{sym}] seed={seed}  contact={contact_bool}  dz={dz:+.4f}m  lifted={lifted}")

        results.append(TrialMetrics(
            trial_id=seed, shape_name="banana", shape_type="ycb",
            seed=seed,
            descend_z_offset=params.descend_z_offset,
            open_length=params.open_length,
            close_fraction=params.close_fraction,
            close_steps=params.close_steps, hold_steps=params.hold_steps,
            finger_friction=params.finger_friction or -1.,
            obj_friction=params.obj_friction or -1.,
            obj_com_z=float(obj_pos[2]),
            descend_z_actual=target_z,
            eef_z_reached=float(env._get_eef_pos()[2]),
            contact_bool=contact_bool,
            grasped=grasped,
            obj_z_before=float(obj_pos[2]),
            obj_z_after=obj_z_aft,
            dz=dz, lifted=lifted, success=grasped and lifted,
        ))

    env.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results: List[TrialMetrics], path: Path):
    if not results:
        return
    rows   = [r.to_csv_row() for r in results]
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[calib] saved CSV → {path}  ({len(rows)} rows)")


def save_best_params(results: List[TrialMetrics], path: Path):
    """Aggregate best params per shape by success rate, then dz."""
    from collections import defaultdict
    buckets: Dict[str, List[TrialMetrics]] = defaultdict(list)
    for r in results:
        key = (r.descend_z_offset, r.open_length, r.close_fraction,
               r.finger_friction, r.obj_friction)
        buckets[f"{r.shape_name}|{key}"].append(r)

    # group by shape first
    by_shape: Dict[str, Dict] = defaultdict(list)
    for _, group in buckets.items():
        if not group:
            continue
        r0   = group[0]
        srate = sum(1 for r in group if r.success) / len(group)
        dz_mean = np.mean([r.dz for r in group]) if group else 0.
        by_shape[r0.shape_name].append({
            "descend_z_offset": r0.descend_z_offset,
            "open_length":      r0.open_length,
            "close_fraction":   r0.close_fraction,
            "finger_friction":  r0.finger_friction,
            "obj_friction":     r0.obj_friction,
            "success_rate":     srate,
            "dz_mean":          float(dz_mean),
            "n_trials":         len(group),
        })

    best: Dict[str, dict] = {}
    for shape, combos in by_shape.items():
        combos.sort(key=lambda c: (-c["success_rate"], -c["dz_mean"]))
        best[shape] = combos[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[calib] best params → {path}")
    for shape, b in best.items():
        print(f"  {shape}: success={b['success_rate']:.0%}  dz={b['dz_mean']:.3f}m"
              f"  dz_off={b['descend_z_offset']:+.3f}"
              f"  open={b['open_length']:.2f}"
              f"  cf={b['close_fraction']:.2f}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _plt_available() -> bool:
    try:
        import matplotlib  # noqa
        return True
    except ImportError:
        return False


def plot_trial_timeline(m: TrialMetrics, out_dir: Path):
    """Save 4-panel time-series plot for one trial."""
    if not _plt_available() or not m.timeline:
        return
    import matplotlib.pyplot as plt

    snaps = m.timeline
    ts    = [s.t for s in snaps]
    phases = [s.phase for s in snaps]
    phase_change = next((i for i, p in enumerate(phases) if p == "lift"), len(ts))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        f"Trial {m.trial_id}  shape={m.shape_name}  "
        f"dz_off={m.descend_z_offset:+.3f}  open={m.open_length:.2f}  "
        f"cf={m.close_fraction:.2f}  "
        f"{'SUCCESS' if m.success else 'FAILED'}",
        fontsize=10,
    )

    # panel 0: gripper angle
    ax = axes[0]
    ax.plot(ts, [s.grip_angle for s in snaps], color="tab:purple")
    ax.axhline(GRIP_CLOSED, ls="--", color="tab:red", alpha=0.5, label=f"CLOSED={GRIP_CLOSED:.2f}")
    ax.axhline(GRIP_OPEN,   ls="--", color="tab:blue", alpha=0.5, label=f"OPEN={GRIP_OPEN:.2f}")
    ax.set_ylabel("Gripper angle (rad)")
    ax.legend(fontsize=7)

    # panel 1: object z vs EEF z vs jaw z
    ax = axes[1]
    ax.plot(ts, [s.obj_z       for s in snaps], label="obj CoM z",    color="tab:orange")
    ax.plot(ts, [s.eef_z       for s in snaps], label="EEF z",         color="tab:blue",  ls="--")
    ax.plot(ts, [s.fixed_jaw_z for s in snaps], label="fixed jaw z",   color="tab:green", lw=0.8)
    ax.plot(ts, [s.mv_jaw_z    for s in snaps], label="moving jaw z",  color="tab:red",   lw=0.8)
    ax.axhline(TABLE_TOP_Z,        ls=":", color="k", alpha=0.4, label="table")
    ax.axhline(TABLE_TOP_Z + 0.07, ls=":", color="g", alpha=0.4, label="lift thresh")
    ax.set_ylabel("Z position (m)")
    ax.legend(fontsize=7, ncol=3)

    # panel 2: finger gap
    ax = axes[2]
    ax.plot(ts, [s.finger_gap for s in snaps], color="tab:brown")
    ax.set_ylabel("Finger gap (m)")

    # panel 3: contact count and force
    ax = axes[3]
    ax2 = ax.twinx()
    ax.bar(ts, [s.n_contacts for s in snaps], color="tab:blue", alpha=0.4, label="n contacts")
    ax2.plot(ts, [s.max_force for s in snaps], color="tab:red", lw=0.8, label="max force (N)")
    ax.set_ylabel("N contacts", color="tab:blue")
    ax2.set_ylabel("Force (N)", color="tab:red")
    ax.set_xlabel("Step")

    # shade lift phase
    for ax_ in axes:
        if phase_change < len(ts):
            ax_.axvspan(ts[phase_change], ts[-1], alpha=0.08, color="tab:green")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"timeline_{m.trial_id:04d}.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def plot_success_heatmap(
    results: List[TrialMetrics],
    shape_name: str,
    x_param: str = "descend_z_offset",
    y_param: str = "open_length",
    out_dir: Path = OUT_DIR,
):
    """2-D success-rate heatmap over two sweep parameters."""
    if not _plt_available():
        return
    import matplotlib.pyplot as plt
    from collections import defaultdict

    sub = [r for r in results if r.shape_name == shape_name]
    if not sub:
        return

    xs   = sorted(set(getattr(r, x_param) for r in sub))
    ys   = sorted(set(getattr(r, y_param) for r in sub))
    grid = defaultdict(list)
    for r in sub:
        grid[(getattr(r, x_param), getattr(r, y_param))].append(r.success)

    data = np.zeros((len(ys), len(xs)))
    for xi, xv in enumerate(xs):
        for yi, yv in enumerate(ys):
            trials = grid[(xv, yv)]
            data[yi, xi] = np.mean(trials) if trials else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(xs)));  ax.set_xticklabels([f"{v:+.3f}" for v in xs], fontsize=8)
    ax.set_yticks(range(len(ys)));  ax.set_yticklabels([f"{v:.2f}"  for v in ys],  fontsize=8)
    ax.set_xlabel(x_param);  ax.set_ylabel(y_param)
    ax.set_title(f"Success rate — {shape_name}  ({len(sub)} trials)")
    plt.colorbar(im, ax=ax, label="success rate")

    # annotate cells
    for xi in range(len(xs)):
        for yi in range(len(ys)):
            v = data[yi, xi]
            ax.text(xi, yi, f"{v:.0%}", ha="center", va="center",
                    fontsize=8, color="black" if 0.2 < v < 0.8 else "white")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"heatmap_{shape_name}_{x_param}_vs_{y_param}.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"[calib] heatmap → {fname}")


def save_replay_frames(env: EnvironmentSoArm, n_frames: int, out_dir: Path):
    """Render N frames from the current env state to PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(n_frames):
        env._renderer.update_scene(env.data)
        frame = env._renderer.render()          # (H, W, 3) uint8
        path  = out_dir / f"frame_{fi:04d}.png"
        try:
            import imageio
            imageio.imwrite(str(path), frame)
        except ImportError:
            # Fall back to numpy raw save if imageio unavailable
            np.save(str(path.with_suffix(".npy")), frame)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # mode
    p.add_argument("--diagnostic", action="store_true",
                   help="Single diagnostic trial with verbose per-phase output")
    p.add_argument("--sweep",      action="store_true",
                   help="Full parameter sweep on --shapes")
    p.add_argument("--validate",   action="store_true",
                   help="Banana holdout validation (uses best_params.json if found)")
    p.add_argument("--quick",      action="store_true",
                   help="Narrow sweep for smoke-test (2 seeds, 3×2 grid)")

    # shape
    p.add_argument("--shape",  default="cube",
                   choices=list(SHAPES.keys()),
                   help="Single shape to test (default: cube)")
    p.add_argument("--shapes", default="",
                   help="Comma-separated shapes for --sweep (default: all)")

    # params (override defaults for single trial / diagnostic)
    p.add_argument("--descend-z-offset", type=float, default=None,
                   help="EEF descend offset relative to object CoM z (m)")
    p.add_argument("--open-length",      type=float, default=None,
                   help="Pre-grasp gripper opening length (m)")
    p.add_argument("--close-fraction",   type=float, default=None,
                   help="Gripper close fraction 0–1 (1.0 = fully closed)")
    p.add_argument("--finger-friction",  type=float, default=None,
                   help="Override sliding friction for gripper geoms")
    p.add_argument("--obj-friction",     type=float, default=None,
                   help="Override sliding friction for object geom")

    # sweep grid overrides
    p.add_argument("--n-seeds",  type=int, default=3,
                   help="Seeds per parameter combo (default: 3)")

    # output
    p.add_argument("--out-dir",     default="calib_logs",
                   help="Output directory (default: calib_logs/)")
    p.add_argument("--save-frames", action="store_true",
                   help="Render and save frames for replay video")
    p.add_argument("--no-timeline", action="store_true",
                   help="Skip per-trial timeline plots")

    # vis
    p.add_argument("--vis", action="store_true",
                   help="Open interactive MuJoCo viewer")

    return p


def main():
    args   = _parser().parse_args()
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shapes = (
        [s.strip() for s in args.shapes.split(",") if s.strip()]
        if args.shapes else list(SHAPES.keys())
    )
    if not args.sweep and not args.validate:
        shapes = [args.shape]

    # build CalibParams from CLI overrides
    base_params = CalibParams(
        descend_z_offset = args.descend_z_offset if args.descend_z_offset is not None else 0.00,
        open_length      = args.open_length      if args.open_length      is not None else 0.07,
        close_fraction   = args.close_fraction   if args.close_fraction   is not None else 1.00,
        finger_friction  = args.finger_friction,
        obj_friction     = args.obj_friction,
    )

    all_results: List[TrialMetrics] = []

    # ── diagnostic mode ───────────────────────────────────────────────────────
    if args.diagnostic:
        for shape in shapes:
            cal = GraspCalibrator(
                shape_name   = shape,
                vis          = args.vis,
                save_frames  = args.save_frames,
                record_steps = True,
                out_dir      = out,
            )
            cal.setup()
            m = cal.run_diagnostic(base_params)
            if not args.no_timeline:
                plot_trial_timeline(m, out)
            all_results.append(m)
            cal.close()

    # ── sweep mode ────────────────────────────────────────────────────────────
    elif args.sweep or args.quick:
        if args.quick:
            dz_vals  = [-0.02, 0.00, 0.02]
            ol_vals  = [0.06,  0.08]
            cf_vals  = [1.00]
            n_seeds  = 2
        else:
            dz_vals  = DEFAULT_DESCEND_OFFSETS
            ol_vals  = DEFAULT_OPEN_LENGTHS
            cf_vals  = DEFAULT_CLOSE_FRACTIONS
            n_seeds  = args.n_seeds

        ff_vals = ([args.finger_friction] if args.finger_friction is not None
                   else [None])
        of_vals = ([args.obj_friction]    if args.obj_friction    is not None
                   else [None])

        for shape in shapes:
            print(f"\n{'='*60}")
            print(f"Sweeping shape: {shape}")
            print(f"{'='*60}")
            cal = GraspCalibrator(
                shape_name   = shape,
                vis          = args.vis,
                save_frames  = args.save_frames,
                record_steps = not args.no_timeline,
                out_dir      = out,
            )
            cal.setup()
            results = cal.sweep(
                descend_offsets  = dz_vals,
                open_lengths     = ol_vals,
                close_fractions  = cf_vals,
                finger_frictions = ff_vals,
                obj_frictions    = of_vals,
                n_seeds          = n_seeds,
            )
            cal.close()

            all_results.extend(results)

            # per-shape plots
            plot_success_heatmap(results, shape, "descend_z_offset", "open_length", out)
            plot_success_heatmap(results, shape, "descend_z_offset", "close_fraction", out)

            if not args.no_timeline:
                successes = [r for r in results if r.success]
                for r in successes[:3]:          # plot up to 3 success timelines
                    plot_trial_timeline(r, out)
                failures = [r for r in results if not r.success and r.contact_bool]
                for r in failures[:2]:           # and 2 contact-but-no-lift cases
                    plot_trial_timeline(r, out)

    # ── validate mode ─────────────────────────────────────────────────────────
    elif args.validate:
        # try to load best params from prior sweep
        best_path = out / "best_params.json"
        if best_path.exists():
            with open(best_path) as f:
                best = json.load(f)
            if "cube" in best:
                b = best["cube"]
                base_params = CalibParams(
                    descend_z_offset = b.get("descend_z_offset", 0.00),
                    open_length      = b.get("open_length",      0.07),
                    close_fraction   = b.get("close_fraction",   1.00),
                    finger_friction  = b.get("finger_friction")  if b.get("finger_friction", -1) > 0 else None,
                    obj_friction     = b.get("obj_friction")     if b.get("obj_friction",    -1) > 0 else None,
                )
                print(f"[validate] Loaded best params from {best_path}:")
                print(f"  dz_off={base_params.descend_z_offset:+.3f}  "
                      f"open={base_params.open_length:.2f}  "
                      f"cf={base_params.close_fraction:.2f}")
        results = validate_banana(base_params, vis=args.vis, n_seeds=5)
        all_results.extend(results)
        n_ok = sum(r.success for r in results)
        print(f"\n[validate] Banana: {n_ok}/{len(results)} success "
              f"({100*n_ok/max(1,len(results)):.0f}%)")

    # ── default: single diagnostic trial ─────────────────────────────────────
    else:
        cal = GraspCalibrator(
            shape_name   = args.shape,
            vis          = args.vis,
            save_frames  = args.save_frames,
            record_steps = True,
            out_dir      = out,
        )
        cal.setup()
        m = cal.run_diagnostic(base_params)
        if not args.no_timeline:
            plot_trial_timeline(m, out)
        all_results.append(m)
        cal.close()

    # ── save outputs ──────────────────────────────────────────────────────────
    if all_results:
        save_csv(all_results, out / "trials.csv")
        best = save_best_params(all_results, out / "best_params.json")

        n_ok    = sum(r.success for r in all_results)
        n_total = len(all_results)
        print(f"\n[calib] Overall: {n_ok}/{n_total} "
              f"({100*n_ok/max(1,n_total):.1f}%) success")

        if not _plt_available():
            print("[calib] matplotlib not available — skipping plots (pip install matplotlib)")


if __name__ == "__main__":
    main()
