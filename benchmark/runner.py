"""BenchmarkRunner — core grasp benchmark execution loop.

Inspired by boschresearch/mj-grasp-sim evaluation style:
  - fixed scene per (object, seed) so all methods see identical conditions
  - stability check before any grasp attempt
  - per-trial metadata logged to JSONL
  - resumable: already-logged (obj, seed, method) triples are skipped

Supports physics and demo_attach execution modes; benchmark results
must always use physics mode (demo_attach contaminates success labels).

Usage
-----
    from benchmark import BenchmarkRunner, BenchmarkConfig, build_method, TrialLogger

    cfg     = BenchmarkConfig.from_yaml("configs/benchmark/default.yaml")
    methods = [build_method(m) for m in cfg.methods]
    logger  = TrialLogger(run_dir=cfg.results_dir / cfg.run_id)
    runner  = BenchmarkRunner(cfg, methods, logger)
    runner.run()
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z,
    GRASP_MODE_PHYSICS,
    GRASP_MODE_PHYSICS_WELD,
    GRASP_MODE_DEMO_ATTACH,
)
from benchmark.stability import StabilityChecker
from benchmark.logger import TrialLogger, TrialRecord
from benchmark.methods import MethodBase


# ── object registry ───────────────────────────────────────────────────────────
# Maps benchmark short names → YCB pool names (must match manifest + ycb_objects/)

OBJECT_REGISTRY: Dict[str, str] = {
    "banana":   "YcbBanana",
    "pear":     "YcbPear",
    "mustard":  "YcbMustardBottle",
    "cracker":  "YcbCrackerBox",
    "drill":    "YcbPowerDrill",
    "can":      "YcbTomatoSoupCan",
    "cylinder": "YcbMediumClamp",
}


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class SpawnConfig:
    centre_y:   float = -0.40
    spread_xy:  float = 0.06
    drop_z_off: float = 0.12   # above TABLE_TOP_Z


@dataclass
class SamplingConfig:
    spread_xy:    float = 0.04
    z_offset:     float = 0.025
    yaw_lo:       float = -math.pi / 2
    yaw_hi:       float =  math.pi / 2
    opening_lo:   float = 0.04
    opening_hi:   float = 0.09


@dataclass
class StabilityConfig:
    max_pos_z:     float = TABLE_TOP_Z + 0.40
    min_pos_z:     float = TABLE_TOP_Z - 0.05
    max_xy_radius: float = 0.44
    max_velocity:  float = 0.05
    table_centre:  Tuple[float, float] = (0.0, -0.45)


@dataclass
class BenchmarkConfig:
    run_id:          str
    objects:         List[str]
    seeds:           List[int]
    methods:         List[str]
    execution_mode:  str           = GRASP_MODE_PHYSICS_WELD
    n_grasp_candidates: int        = 10
    n_grasp_attempts:   int        = 3
    settle_steps:    int           = 300
    stability_check: bool          = True
    verbose:         bool          = True

    spawn:           SpawnConfig    = field(default_factory=SpawnConfig)
    sampling:        SamplingConfig = field(default_factory=SamplingConfig)
    stability:       StabilityConfig = field(default_factory=StabilityConfig)

    results_dir:     Path           = field(default_factory=lambda: Path("results"))
    plots_dir:       Path           = field(default_factory=lambda: Path("plots"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)

        spawn   = SpawnConfig(**{k: v for k, v in d.pop("spawn",    {}).items()
                                  if k in SpawnConfig.__dataclass_fields__})
        samp    = SamplingConfig(**{k: v for k, v in d.pop("sampling", {}).items()
                                     if k in SamplingConfig.__dataclass_fields__})
        stab    = StabilityConfig(**{k: v for k, v in d.pop("stability", {}).items()
                                      if k in StabilityConfig.__dataclass_fields__})
        d.pop("lggsn_ckpt", None)
        d.pop("wm_ckpt",    None)
        d["results_dir"] = Path(d.pop("results_dir", "results"))
        d["plots_dir"]   = Path(d.pop("plots_dir",   "plots"))

        valid = {k for k in cls.__dataclass_fields__ if k not in ("spawn","sampling","stability")}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d, spawn=spawn, sampling=samp, stability=stab)


# ── trial result ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Outcome of one benchmark trial."""
    object:           str
    seed:             int
    method:           str
    execution_mode:   str
    stability_valid:  bool
    skip_reason:      Optional[str]
    n_candidates:     Optional[int]
    grasp_rank_order: Optional[List[int]]
    grasp_used_rank:  Optional[int]
    grasp_index:      Optional[int]
    contact_count:    Optional[int]
    bilateral_contact: Optional[bool]
    weld_triggered:   Optional[bool]    # True iff kinematic weld was activated
    table_contact:    Optional[bool]    # True iff fixed jaw sphere contacted table
    final_z:          Optional[float]   # object z after lift attempt
    lifted:           Optional[bool]    # obj_z > TABLE_TOP_Z + 0.07
    success:          Optional[bool]
    dz:               Optional[float]
    slip:             Optional[float]
    fell_off:         Optional[bool]
    failure_reason:   Optional[str]
    scene_file:       Optional[str]


# ── runner ────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Execute the full benchmark for all (object × seed × method) combinations.

    The scene for a given (object, seed) pair is identical for every method
    — spawn position is derived from the seed alone.

    Parameters
    ----------
    config  : BenchmarkConfig
    methods : list[MethodBase]   — ranking method instances
    logger  : TrialLogger
    resume  : bool               — skip already-logged trials
    """

    def __init__(
        self,
        config:  BenchmarkConfig,
        methods: List[MethodBase],
        logger:  TrialLogger,
        resume:  bool = False,
    ):
        self.cfg      = config
        self.methods  = methods
        self.logger   = logger
        self.resume   = resume
        self._checker = StabilityChecker(
            max_pos_z     = config.stability.max_pos_z,
            min_pos_z     = config.stability.min_pos_z,
            max_xy_radius = config.stability.max_xy_radius,
            max_velocity  = config.stability.max_velocity,
            table_centre  = config.stability.table_centre,
        ) if config.stability_check else None

        self._env: Optional[EnvironmentSoArm] = None
        self._trial_id = 0

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Run all trials.  Builds the MuJoCo env once, then loops."""
        all_obj_ycb = [OBJECT_REGISTRY[o] for o in self.cfg.objects]

        self._env = EnvironmentSoArm(
            obj_names  = all_obj_ycb,
            vis        = False,
            grasp_mode = self.cfg.execution_mode,
        )

        n_total  = len(self.cfg.objects) * len(self.cfg.seeds) * len(self.methods)
        n_done   = 0
        t_start  = time.time()

        try:
            for obj_name in self.cfg.objects:
                ycb_name = OBJECT_REGISTRY[obj_name]
                for seed in self.cfg.seeds:
                    # Fixed spawn position for this (obj, seed)
                    spawn_pos = self._fixed_spawn(seed)

                    for method in self.methods:
                        if self.resume and self.logger.already_done(
                                obj_name, seed, method.name):
                            n_done += 1
                            continue

                        result = self._run_trial(
                            obj_name, ycb_name, spawn_pos, seed, method
                        )
                        self.logger.log(TrialRecord(
                            trial_id          = self._trial_id,
                            object            = result.object,
                            seed              = result.seed,
                            method            = result.method,
                            execution_mode    = result.execution_mode,
                            stability_valid   = result.stability_valid,
                            skip_reason       = result.skip_reason,
                            n_candidates      = result.n_candidates,
                            grasp_rank_order  = result.grasp_rank_order,
                            grasp_used_rank   = result.grasp_used_rank,
                            grasp_index       = result.grasp_index,
                            contact_count     = result.contact_count,
                            bilateral_contact = result.bilateral_contact,
                            success           = result.success,
                            dz                = result.dz,
                            slip              = result.slip,
                            fell_off          = result.fell_off,
                            failure_reason    = result.failure_reason,
                            scene_file        = result.scene_file,
                        ))
                        self._trial_id += 1
                        n_done += 1

                        if self.cfg.verbose:
                            elapsed = time.time() - t_start
                            self._print_progress(result, n_done, n_total, elapsed)

        finally:
            if self._env:
                self._env.close()

        self.logger.write_summary()
        if self.cfg.verbose:
            print(f"\n[benchmark] done — {n_done} trials in "
                  f"{(time.time()-t_start)/60:.1f} min")
            print(f"  results: {self.logger.run_dir / 'trials.jsonl'}")
            print(f"  summary: {self.logger.run_dir / 'summary.csv'}")

    # ── trial execution ───────────────────────────────────────────────────────

    def _run_trial(
        self,
        obj_name:  str,
        ycb_name:  str,
        spawn_pos: List[float],
        seed:      int,
        method:    MethodBase,
    ) -> TrialResult:
        env = self._env

        # ── spawn ─────────────────────────────────────────────────────────────
        env.reset_robot()
        env.remove_all_obj()
        obj_id = env.load_obj(ycb_name, name=obj_name, pos=spawn_pos)
        env._steps(self.cfg.settle_steps)
        env.wait_until_all_still(max_wait_epochs=200)

        # ── stability check ───────────────────────────────────────────────────
        if self._checker is not None:
            stable = self._checker.check(env, obj_id)
            if not stable:
                return TrialResult(
                    object=obj_name, seed=seed, method=method.name,
                    execution_mode=self.cfg.execution_mode,
                    stability_valid=False, skip_reason=stable.reason,
                    n_candidates=None, grasp_rank_order=None,
                    grasp_used_rank=None, grasp_index=None,
                    contact_count=None, bilateral_contact=None,
                    weld_triggered=None, table_contact=None,
                    final_z=None, lifted=None,
                    success=None, dz=None, slip=None, fell_off=None,
                    failure_reason=stable.reason, scene_file=None,
                )

        # ── observation + state ───────────────────────────────────────────────
        obs     = env.get_obs(pointcloud=True)
        obj_pos = env.get_obj_pos(obj_id)

        # ── sample grasp candidates ───────────────────────────────────────────
        grasp_rng  = np.random.default_rng(seed + 9999)
        candidates = _sample_candidates(obj_pos, grasp_rng, self.cfg)
        n_cands    = len(candidates)

        # ── rank ──────────────────────────────────────────────────────────────
        rank_rng   = np.random.default_rng(seed + 7777)
        try:
            ranked = method.rank(candidates, obj_pos, obs, obj_id, rank_rng)
        except Exception as e:
            ranked = np.arange(n_cands)

        # ── save pre-grasp scene state ────────────────────────────────────────
        scene_file = self._save_scene_state(obj_name, seed, method.name, env, obj_id)

        # ── execute top-K attempts ────────────────────────────────────────────
        pos_before     = env.get_obj_pos(obj_id).copy()
        tried_indices  = []
        success           = False
        used_rank         = None
        grasp_index       = None
        contact_count     = None
        bilateral_contact = None
        weld_triggered    = None
        table_contact     = None
        final_z           = None
        lifted            = None
        failure_reason    = "all_attempts_failed"

        for rank_pos, cand_idx in enumerate(ranked[: self.cfg.n_grasp_attempts]):
            g = candidates[int(cand_idx)]
            cand_idx_int = int(cand_idx)
            tried_indices.append(cand_idx_int)

            ok, _grasped = env._execute_grasp(
                pos   = (float(g[0]), float(g[1]), float(g[2])),
                roll  = float(g[3]),
                gripper_opening_length = float(g[4]),
                obj_height             = float(g[5]),
            )

            # capture contact and result metrics from the post-execution snapshot
            if env.last_grasp_metrics is not None:
                m = env.last_grasp_metrics
                contact_count     = int(m.get("left_contacts", 0)) + int(m.get("right_contacts", 0))
                bilateral_contact = bool(m.get("bilateral_contact", False))
                weld_triggered    = (bool(m["weld_triggered"]) if "weld_triggered" in m else None)
                table_contact     = (bool(m["table_contact"])  if "table_contact"  in m else None)
                final_z           = (float(m["final_z"])       if "final_z"        in m else None)
                lifted            = (bool(m["lifted"])         if "lifted"         in m else None)

            grasp_index = cand_idx_int

            if ok:
                success        = True
                used_rank      = rank_pos
                failure_reason = None
                break
            else:
                failure_reason = "no_lift" if rank_pos == self.cfg.n_grasp_attempts - 1 \
                                 else "attempt_failed"

        # ── outcome ───────────────────────────────────────────────────────────
        pos_after = env.get_obj_pos(obj_id)
        dz        = float(pos_after[2] - pos_before[2])
        slip      = float(np.linalg.norm(pos_after[:2] - pos_before[:2]))
        fell_off  = bool(pos_after[2] < TABLE_TOP_Z - 0.10)

        return TrialResult(
            object=obj_name, seed=seed, method=method.name,
            execution_mode=self.cfg.execution_mode,
            stability_valid=True, skip_reason=None,
            n_candidates=n_cands,
            grasp_rank_order=[int(i) for i in tried_indices],
            grasp_used_rank=used_rank,
            grasp_index=grasp_index,
            contact_count=contact_count,
            bilateral_contact=bilateral_contact,
            weld_triggered=weld_triggered,
            table_contact=table_contact,
            final_z=final_z,
            lifted=lifted,
            success=success,
            dz=dz,
            slip=slip,
            fell_off=fell_off,
            failure_reason=failure_reason,
            scene_file=str(scene_file) if scene_file else None,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save_scene_state(
        self, obj_name: str, seed: int, method: str, env, obj_id: int
    ) -> Optional[Path]:
        """Save pre-grasp MuJoCo state to scenes/ for later replay."""
        try:
            scenes_dir = self.logger.run_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            fname = scenes_dir / f"{obj_name}_seed{seed:04d}_{method}.json"
            obj_pos = env.get_obj_pos(obj_id).tolist()
            state = {
                "version":  1,
                "object":   obj_name,
                "seed":     seed,
                "method":   method,
                "obj_pos":  obj_pos,
                "qpos":     env.data.qpos.tolist(),
                "qvel":     env.data.qvel.tolist(),
            }
            with open(fname, "w") as f:
                json.dump(state, f)
            return fname
        except Exception:
            return None

    def _fixed_spawn(self, seed: int) -> List[float]:
        """Deterministic spawn position for a given seed (independent of method)."""
        rng = np.random.default_rng(seed)
        cx  = float(rng.uniform(-self.cfg.spawn.spread_xy, self.cfg.spawn.spread_xy))
        cy  = self.cfg.spawn.centre_y + float(rng.uniform(-0.04, 0.04))
        cz  = TABLE_TOP_Z + self.cfg.spawn.drop_z_off
        return [cx, cy, cz]

    @staticmethod
    def _print_progress(r: TrialResult, done: int, total: int, elapsed: float):
        pct   = done / total * 100
        eta_s = (elapsed / done) * (total - done) if done else 0
        sym   = "✓" if r.success else ("~" if not r.stability_valid else "✗")
        print(
            f"  [{sym}] {r.object:12s} seed={r.seed:3d}  {r.method:12s}  "
            f"dz={r.dz:+.3f}  "
            f"{done}/{total} ({pct:.0f}%)  "
            f"ETA {eta_s/60:.1f}min"
            if r.dz is not None else
            f"  [{sym}] {r.object:12s} seed={r.seed:3d}  {r.method:12s}  "
            f"SKIPPED ({r.skip_reason})  "
            f"{done}/{total} ({pct:.0f}%)"
        )


# ── grasp candidate sampling ──────────────────────────────────────────────────

def _sample_candidates(
    obj_pos: np.ndarray,
    rng:     np.random.Generator,
    cfg:     BenchmarkConfig,
) -> np.ndarray:
    """Sample N grasp candidates near the object centroid.

    Returns (N, 6) array: [x, y, z, yaw, opening, obj_height]
    """
    N  = cfg.n_grasp_candidates
    sc = cfg.sampling

    xs   = obj_pos[0] + rng.uniform(-sc.spread_xy, sc.spread_xy, N)
    ys   = obj_pos[1] + rng.uniform(-sc.spread_xy, sc.spread_xy, N)
    yaws = rng.uniform(sc.yaw_lo, sc.yaw_hi, N)
    ops  = rng.uniform(sc.opening_lo, sc.opening_hi, N)
    H_val = max(0.05, obj_pos[2] - TABLE_TOP_Z)
    # Ensure grasp_z stays >= TABLE_TOP_Z + H + 3cm after descent
    z_val = max(obj_pos[2] + sc.z_offset, TABLE_TOP_Z + H_val + 0.03)
    zs   = np.full(N, z_val)
    Hs   = np.full(N, H_val)

    return np.column_stack([xs, ys, zs, yaws, ops, Hs]).astype(np.float32)
