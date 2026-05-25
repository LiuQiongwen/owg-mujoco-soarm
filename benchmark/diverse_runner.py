"""DiverseBenchmarkRunner — diversity-aware grasp benchmark for SO-ARM101.

Extends the core benchmark loop with:
  - Difficulty modes (easy / medium / hard)
  - Random object yaw at spawn (via free-joint qpos)
  - Clutter objects for hard difficulty
  - Extra TrialRecord fields: difficulty, spawn_x/y/yaw, clutter_count, grasp_yaw

The interface is identical to BenchmarkRunner; only the scene generation and
logging differ.

Usage
-----
    from benchmark.diverse_runner import DiverseBenchmarkRunner, DiverseBenchmarkConfig
    from benchmark.methods import build_method
    from benchmark.logger import TrialLogger

    cfg     = DiverseBenchmarkConfig.from_yaml("configs/benchmark/diverse_medium.yaml")
    methods = [build_method(m) for m in cfg.methods]
    logger  = TrialLogger(cfg.results_dir / cfg.run_id)
    runner  = DiverseBenchmarkRunner(cfg, methods, logger)
    runner.run()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from owg_robot.env_soarm import (
    EnvironmentSoArm,
    TABLE_TOP_Z,
    GRASP_MODE_PHYSICS_WELD,
)
from benchmark.runner import (
    BenchmarkConfig,
    SpawnConfig,
    SamplingConfig,
    StabilityConfig,
    TrialResult,
    OBJECT_REGISTRY,
    _sample_candidates,
)
from benchmark.stability import StabilityChecker
from benchmark.logger import TrialLogger, TrialRecord
from benchmark.methods import MethodBase
from benchmark.scene_generator import (
    DifficultyConfig,
    SceneConfig,
    DIFFICULTY_PRESETS,
    generate_scene,
)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class DiverseBenchmarkConfig(BenchmarkConfig):
    """BenchmarkConfig extended with a difficulty level selector."""
    difficulty: str = "medium"   # "easy" | "medium" | "hard"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DiverseBenchmarkConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)

        spawn = SpawnConfig(**{k: v for k, v in d.pop("spawn", {}).items()
                               if k in SpawnConfig.__dataclass_fields__})
        samp  = SamplingConfig(**{k: v for k, v in d.pop("sampling", {}).items()
                                   if k in SamplingConfig.__dataclass_fields__})
        stab  = StabilityConfig(**{k: v for k, v in d.pop("stability", {}).items()
                                    if k in StabilityConfig.__dataclass_fields__})
        d.pop("lggsn_ckpt", None)
        d.pop("wm_ckpt",    None)
        d["results_dir"] = Path(d.pop("results_dir", "results"))
        d["plots_dir"]   = Path(d.pop("plots_dir",   "plots"))

        valid = {k for k in cls.__dataclass_fields__
                 if k not in ("spawn", "sampling", "stability")}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d, spawn=spawn, sampling=samp, stability=stab)


# ── runner ────────────────────────────────────────────────────────────────────

class DiverseBenchmarkRunner:
    """Execute the diverse benchmark for all (object × seed × method) triples.

    Scenes are generated deterministically from (difficulty, obj_name, seed)
    so all methods see identical physics conditions for a given triple.

    Parameters
    ----------
    config  : DiverseBenchmarkConfig
    methods : list[MethodBase]
    logger  : TrialLogger
    resume  : bool   — skip (object, seed, method) triples already in the log
    """

    def __init__(
        self,
        config:  DiverseBenchmarkConfig,
        methods: List[MethodBase],
        logger:  TrialLogger,
        resume:  bool = False,
    ):
        self.cfg         = config
        self.methods     = methods
        self.logger      = logger
        self.resume      = resume
        self._difficulty = DIFFICULTY_PRESETS[config.difficulty]
        self._checker    = StabilityChecker(
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
        """Build the MuJoCo environment once and loop over all trials."""
        all_obj_ycb = [OBJECT_REGISTRY[o] for o in self.cfg.objects]

        self._env = EnvironmentSoArm(
            obj_names  = all_obj_ycb,
            vis        = False,
            grasp_mode = self.cfg.execution_mode,
        )

        n_total = len(self.cfg.objects) * len(self.cfg.seeds) * len(self.methods)
        n_done  = 0
        t_start = time.time()

        try:
            for obj_name in self.cfg.objects:
                ycb_name = OBJECT_REGISTRY[obj_name]
                for seed in self.cfg.seeds:
                    # one scene per (obj, seed) — shared across all methods
                    scene = generate_scene(
                        self._difficulty, obj_name, ycb_name, seed, all_obj_ycb
                    )
                    for method in self.methods:
                        if self.resume and self.logger.already_done(
                                obj_name, seed, method.name):
                            n_done += 1
                            continue

                        result, extra = self._run_trial(scene, method)
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
                            weld_triggered    = result.weld_triggered,
                            table_contact     = result.table_contact,
                            final_z           = result.final_z,
                            lifted            = result.lifted,
                            success           = result.success,
                            dz                = result.dz,
                            slip              = result.slip,
                            fell_off          = result.fell_off,
                            failure_reason    = result.failure_reason,
                            scene_file        = result.scene_file,
                            difficulty        = extra["difficulty"],
                            spawn_x           = extra["spawn_x"],
                            spawn_y           = extra["spawn_y"],
                            spawn_yaw         = extra["spawn_yaw"],
                            clutter_count     = extra["clutter_count"],
                            grasp_yaw         = extra["grasp_yaw"],
                        ))
                        self._trial_id += 1
                        n_done += 1

                        if self.cfg.verbose:
                            elapsed = time.time() - t_start
                            self._print_progress(result, extra, n_done, n_total, elapsed)

        finally:
            if self._env:
                self._env.close()

        self.logger.write_summary()
        if self.cfg.verbose:
            elapsed = time.time() - t_start
            print(f"\n[diverse_bench] done — {n_done} trials in {elapsed/60:.1f} min")
            print(f"  results: {self.logger.run_dir / 'trials.jsonl'}")
            print(f"  summary: {self.logger.run_dir / 'summary.csv'}")

    # ── trial execution ───────────────────────────────────────────────────────

    def _run_trial(
        self,
        scene:  SceneConfig,
        method: MethodBase,
    ) -> Tuple[TrialResult, dict]:
        """Execute one trial and return (TrialResult, extra_diversity_fields)."""
        env = self._env

        # ── spawn target object ───────────────────────────────────────────────
        env.reset_robot()
        env.remove_all_obj()

        obj_id = env.load_obj(
            scene.ycb_name,
            name = scene.obj_name,
            pos  = [scene.spawn_x, scene.spawn_y, scene.spawn_z],
            yaw  = scene.spawn_yaw,   # applied as Z-axis quaternion in load_obj
        )

        # ── spawn clutter objects ─────────────────────────────────────────────
        clutter_ids: List[int] = []
        for c_name, c_pos in zip(scene.clutter_ycb_names, scene.clutter_positions):
            try:
                cid = env.load_obj(c_name, pos=c_pos)
                clutter_ids.append(cid)
            except Exception:
                pass  # pool exhausted or name mismatch — skip silently

        env._steps(self.cfg.settle_steps)
        env.wait_until_all_still(max_wait_epochs=200)

        extra: dict = {
            "difficulty":    scene.difficulty,
            "spawn_x":       round(scene.spawn_x, 4),
            "spawn_y":       round(scene.spawn_y, 4),
            "spawn_yaw":     round(scene.spawn_yaw, 4),
            "clutter_count": len(clutter_ids),
            "grasp_yaw":     None,
        }

        # ── stability check ───────────────────────────────────────────────────
        if self._checker is not None:
            stable = self._checker.check(env, obj_id)
            if not stable:
                return TrialResult(
                    object=scene.obj_name, seed=scene.seed, method=method.name,
                    execution_mode=self.cfg.execution_mode,
                    stability_valid=False, skip_reason=stable.reason,
                    n_candidates=None, grasp_rank_order=None,
                    grasp_used_rank=None, grasp_index=None,
                    contact_count=None, bilateral_contact=None,
                    weld_triggered=None, table_contact=None,
                    final_z=None, lifted=None,
                    success=None, dz=None, slip=None, fell_off=None,
                    failure_reason=stable.reason, scene_file=None,
                ), extra

        # ── observe + sample candidates ───────────────────────────────────────
        obs     = env.get_obs(pointcloud=True)
        obj_pos = env.get_obj_pos(obj_id)

        grasp_rng  = np.random.default_rng(scene.seed + 9999)
        candidates = _sample_candidates(obj_pos, grasp_rng, self.cfg)
        n_cands    = len(candidates)

        # ── rank ──────────────────────────────────────────────────────────────
        rank_rng = np.random.default_rng(scene.seed + 7777)
        try:
            ranked = method.rank(candidates, obj_pos, obs, obj_id, rank_rng)
        except Exception:
            ranked = np.arange(n_cands)

        # ── execute top-K attempts ────────────────────────────────────────────
        pos_before        = env.get_obj_pos(obj_id).copy()
        tried_indices:    List[int] = []
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
        chosen_yaw        = None

        for rank_pos, cand_idx in enumerate(ranked[: self.cfg.n_grasp_attempts]):
            g           = candidates[int(cand_idx)]
            cand_idx_i  = int(cand_idx)
            tried_indices.append(cand_idx_i)
            chosen_yaw  = float(g[3])

            ok, _grasped = env._execute_grasp(
                pos                    = (float(g[0]), float(g[1]), float(g[2])),
                roll                   = float(g[3]),
                gripper_opening_length = float(g[4]),
                obj_height             = float(g[5]),
            )

            if env.last_grasp_metrics is not None:
                m = env.last_grasp_metrics
                contact_count     = int(m.get("left_contacts", 0)) + int(m.get("right_contacts", 0))
                bilateral_contact = bool(m.get("bilateral_contact", False))
                weld_triggered    = (bool(m["weld_triggered"]) if "weld_triggered" in m else None)
                table_contact     = (bool(m["table_contact"])  if "table_contact"  in m else None)
                final_z           = (float(m["final_z"])       if "final_z"        in m else None)
                lifted            = (bool(m["lifted"])         if "lifted"         in m else None)

            grasp_index = cand_idx_i
            if ok:
                success        = True
                used_rank      = rank_pos
                failure_reason = None
                break
            else:
                failure_reason = ("no_lift" if rank_pos == self.cfg.n_grasp_attempts - 1
                                  else "attempt_failed")

        extra["grasp_yaw"] = (round(chosen_yaw, 4) if chosen_yaw is not None else None)

        pos_after  = env.get_obj_pos(obj_id)
        dz         = float(pos_after[2]) - float(pos_before[2])
        slip       = float(np.linalg.norm(pos_after[:2] - pos_before[:2]))
        fell_off   = bool(float(pos_after[2]) < TABLE_TOP_Z - 0.10)
        scene_file = self._save_scene(scene, method.name, env, obj_id)

        return TrialResult(
            object=scene.obj_name, seed=scene.seed, method=method.name,
            execution_mode=self.cfg.execution_mode,
            stability_valid=True, skip_reason=None,
            n_candidates=n_cands,
            grasp_rank_order=tried_indices,
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
        ), extra

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save_scene(
        self,
        scene:  SceneConfig,
        method: str,
        env:    EnvironmentSoArm,
        obj_id: int,
    ) -> Optional[Path]:
        """Save pre-grasp scene state for later replay/video generation."""
        import json as _json
        try:
            scenes_dir = self.logger.run_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            fname = scenes_dir / f"{scene.obj_name}_{scene.difficulty}_seed{scene.seed:04d}_{method}.json"
            state = {
                "version":    2,
                "object":     scene.obj_name,
                "ycb_name":   scene.ycb_name,
                "difficulty": scene.difficulty,
                "seed":       scene.seed,
                "method":     method,
                "spawn_x":    scene.spawn_x,
                "spawn_y":    scene.spawn_y,
                "spawn_z":    scene.spawn_z,
                "spawn_yaw":  scene.spawn_yaw,
                "clutter_ycb_names": scene.clutter_ycb_names,
                "obj_pos":    env.get_obj_pos(obj_id).tolist(),
                "qpos":       env.data.qpos.tolist(),
                "qvel":       env.data.qvel.tolist(),
            }
            with open(fname, "w") as f:
                _json.dump(state, f)
            return fname
        except Exception:
            return None

    @staticmethod
    def _print_progress(
        r: TrialResult, extra: dict, done: int, total: int, elapsed: float
    ) -> None:
        pct   = done / total * 100
        eta_s = (elapsed / done) * (total - done) if done else 0
        sym   = "✓" if r.success else ("~" if not r.stability_valid else "✗")
        diff  = (extra.get("difficulty") or "?")[:1].upper()
        dz_s  = f"dz={r.dz:+.3f}  " if r.dz is not None else ""
        print(
            f"  [{sym}][{diff}] {r.object:10s} s={r.seed:3d}  {r.method:10s}  "
            f"yaw={extra.get('spawn_yaw', 0.0):+.2f}  "
            f"clutter={extra.get('clutter_count', 0)}  "
            f"{dz_s}"
            f"{done}/{total} ({pct:.0f}%)  ETA {eta_s/60:.1f}min"
            if r.stability_valid else
            f"  [{sym}][{diff}] {r.object:10s} s={r.seed:3d}  {r.method:10s}  "
            f"SKIPPED ({r.skip_reason})  {done}/{total} ({pct:.0f}%)"
        )
