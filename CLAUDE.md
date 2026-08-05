# TANGO — Claude Code project context

## Project overview

**TANGO** (Transport-Aligned Next-Grasp Optimizer): SO-ARM101 6-DoF robotic grasping with
OT-CFM grasp generation, VLM semantic grounding, and LGGSN pairwise reranking.
All evaluation runs in MuJoCo simulation (`tango` conda env).
The physical robot path goes through `robots/` (sim-to-real trajectory replay).

## Pipeline stages

| Stage | Description | Key flag |
|-------|-------------|----------|
| 1 | GR-ConvNet top-1 only (baseline) | `--stage 1` |
| 2 | 6-DoF grasp sampling, no ranking | `--stage 2` |
| 3 | VLM semantic grounding + sampling | `--stage 3` |
| 4 | Stage 3 + LGGSN pairwise reranker **(current best)** | `--stage 4` |

## Directory layout

```
tango_robot/        MuJoCo SO-ARM101 environment (env_soarm.py is the core)
tango/              VLM policy (policy.py, gpt_utils.py, visual_prompt.py)
robots/             Robot backend abstraction layer (sim-to-real)
  base.py           RobotBackend ABC + GraspResult dataclass
  mujoco_backend.py MujocoBackend wrapping EnvironmentSoArm
  soarm_real_backend.py SOARMRealBackend wrapping lerobot FeetechMotorsBus
  trajectory.py     TrajectoryPoint / Trajectory / Recorder / Replayer
benchmark/          BenchmarkRunner, TrialLogger, methods registry
  runner.py         OBJECT_REGISTRY, SpawnConfig, SamplingConfig
grasp_6dof/         Open3D 6-DoF grasp generation and validation
  grasp_generator_6dof.py   mesh → grasp candidate set (JSON)
  generate_grasps_open3d.py Open3D pipeline
  models/           LGGSN checkpoints (see Model Checkpoints below)
  dataset/          CSV and JSON grasp datasets
scripts/            One-off utilities, eval scripts, data collection
  record_trajectory.py   Sim grasp → Trajectory JSON
  replay_trajectory.py   Replay Trajectory JSON on sim or hardware
  quick_eval.sh          Fast/full eval loop (calls demo.py)
cameras/            CameraBase, RealSenseCamera stub, SimulatedCamera
datasets/           GraspAction / GraspEpisode data types (episode.py)
world_model/        MLP-based world model reranker
legacy/pybullet_panda/  Archived Panda+PyBullet baseline (do not modify)
```

## Conda environment

All commands: `conda run -n tango <cmd>`

Headless rendering: `MUJOCO_GL=egl` (set automatically in all scripts via
`os.environ.setdefault("MUJOCO_GL", "egl")`).

## Common commands

### Single demo (Stage 4)
```bash
conda run -n tango python demo.py \
  --stage 4 --prompt Banana --seed 1 --once --verbose 1 2>&1 | \
  grep -E 'LGGSN grasp scores|Final action|Done pick'
```

### Quick eval (35 trials)
```bash
bash scripts/quick_eval.sh           # fast: 7 objects × 5 seeds (corrected 2026-08-05; was documented as 4)
bash scripts/quick_eval.sh full      # paper: 7 objects × 25 seeds
bash scripts/quick_eval.sh fast 3    # stage 3 baseline
```
Objects iterated (`scripts/quick_eval.sh:9`): Banana, TomatoSoupCan, Pear, MustardBottle,
Scissors, CrackerBox, PowerDrill — all 7 of `OBJECT_REGISTRY`'s effective members, not a
4-object subset.

### Benchmark runner
```bash
conda run -n tango python scripts/run_benchmark.py \
  --config configs/benchmark/default.yaml
```

### Train LGGSN (pairwise BPR)
```bash
conda run -n tango python train_lggsn_pairwise.py
```

### 6-DoF grasp generation
```bash
conda run -n tango python grasp_6dof/grasp_generator_6dof.py \
  --obj <mesh.ply> --out grasp_6dof/dataset/<name>.json \
  --world-pos 0.38,0.0,0.027
```

### Trajectory recording (sim → JSON)
```bash
conda run -n tango python scripts/record_trajectory.py \
  --obj banana --seed 42 --n-tries 5 --out trajs/banana_42.json
# --save-all  : save even on failure (for debugging)
```

### Trajectory replay (sim verification or hardware)
```bash
# Sim verification
conda run -n tango python scripts/replay_trajectory.py \
  --traj trajs/banana_42.json --speed 0.5 --vis

# Physical SO-ARM101
conda run -n tango python scripts/replay_trajectory.py \
  --traj trajs/banana_42.json --backend real --port /dev/ttyUSB0 \
  --max-delta 30   # safety clamp: max 30° per joint per command
```

## Key constants (tango_robot/env_soarm.py)

```python
TABLE_TOP_Z         = 0.785          # metres
OBJECT_INIT_HEIGHT  = TABLE_TOP_Z + 0.10   # drop height for spawn
GRIPPER_MOVING_HEIGHT = TABLE_TOP_Z + 0.20 # clearance height during approach
GRIP_OPEN   = 1.0                    # internal angle (radians)
GRIP_CLOSED = 0.05
HOME_QPOS   = [0.0, -0.4, 0.8, -0.4, 0.0]   # arm home (radians)
ARM_JOINTS  = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
ROBOT_BASE_EULER = "0 0 -1.5707963267948966"   # -90° about Z, added 2026-07-10
TARGET_ZONE_POS  = [0.25, -0.30, TABLE_TOP_Z]  # tray position, changed 2026-07-10
```

**⚠️ `ROBOT_BASE_EULER`/`TARGET_ZONE_POS` are a breaking geometry change (2026-07-10, Phase 3 real-robot
work)**: the arm mount used to have no rotation (HOME reached toward world +X, parallel to `table_top`'s
edge rather than into it) and the tray sat at `(0.20, 0.25)`. Both were changed so HOME reaches into the
table (needed for real-hardware trajectory replay to stay within a physically safe rotation range — see
`paperA_data/README.md`'s "🔧 BREAKING CHANGE" entry). **Every success-rate number in `paper_final.tex` and
`paperA_data/formal_results/` was computed under the OLD geometry** and does not necessarily still hold —
only a quick n=9/strategy directional check has been re-run under the new geometry so far. Do not assume old
numbers transfer; re-verify before citing them in new work.

## Grasp modes

Always use `physics_weld_after_bilateral` for evaluation. The other modes are for
debugging only — `demo_attach` artificially inflates success rates.

```python
GRASP_MODE_PHYSICS_WELD = "physics_weld_after_bilateral"  # ← use this
GRASP_MODE_PHYSICS      = "physics"                       # legacy alias
GRASP_MODE_DEMO_ATTACH  = "demo_attach"                   # contaminates results
```

## Benchmark objects

```python
OBJECT_REGISTRY = {
    "banana":   "YcbBanana",
    "pear":     "YcbPear",
    "mustard":  "YcbMustardBottle",
    "cracker":  "YcbCrackerBox",
    "drill":    "YcbPowerDrill",
    "can":      "YcbTomatoSoupCan",
    "cylinder": "YcbMediumClamp",
}
```

## LGGSN model checkpoints (grasp_6dof/models/)

**⚠️ Corrected 2026-08-05 — this table was stale**: `demo.py`'s actual Stage-4 default
(`demo.py:64`) is `lggsn_pairwise_live_v5d.pt`, **not** `v2` as previously stated here.
`grasp_6dof/models/` also contains 10+ further versions (v3 through v11_ik) never listed
in this table — all `.pt` files are gitignored, so none of them are reproducible from a
clean checkout; treat any checkpoint not in the table below as unaudited.

| File | Status | Val pair_acc | Features |
|------|--------|-------------|---------|
| `lggsn_pairwise_live.pt` | v1, deprecated (inflated accuracy, noisy negatives) | 0.710 (independently re-verified 2026-08-01; provenance incomplete — see below) | 12-dim |
| `lggsn_pairwise_live_v2.pt` | independently verified, sha256-checked | **0.658** (`research_agent_pilots/lggsn_suite/`, commit `ceb2029`) | 14-dim |
| `lggsn_pairwise_live_v5d.pt` | **`demo.py`'s actual current Stage-4 default** — accuracy **unverified** | ~0.750 *(cited from training logs/commit messages only — this project's own standalone eval suite cannot score it: needs `local_point_density`/`normal_consistency`/`contact_width_ratio` per-candidate point-cloud features plus a query-embedding architecture the evaluator doesn't implement; see `research_agent_pilots/lggsn_suite/eval_outputs/matrix_summary.json`'s `ext_v5d` entry, `status: BLOCKED`)* | 17+dim, point-cloud + query-embedding |
| `lggsn_geom_only_live.pt` | legacy single-label | — | 12-dim |

v2 uses 14 features: base 12 + `dist_to_centroid` + `z_rel` (min-max height within episode).
BPR pairwise + margin_0.00 is the current best-performing training configuration among the
**independently verified** checkpoints. v5d is architecturally newer and cited as more
accurate, but that number has never been reproduced outside its own training run — if a
result depends on which checkpoint was active, check `LGGSN_CKPT`/`demo.py`'s actual default
explicitly rather than assuming v2, and flag v5d-based results as resting on an unverified
accuracy claim until someone extends the standalone evaluator to support point-cloud features.

## robots/ abstraction layer

`RobotBackend` (ABC in `robots/base.py`) is the boundary between the planner and hardware.
All public units: joint positions in **radians**, gripper in **metres** [0, 0.10].

```
MujocoBackend       → wraps EnvironmentSoArm; hooks via env._step_hook for recording
SOARMRealBackend    → wraps FeetechMotorsBus (lerobot); lazy import; degrees↔radians
TrajectoryRecorder  → begin() / snap(backend) / end() — called from _step_hook
TrajectoryReplayer  → wall-clock timing from .t offsets; speed multiplier
```

Do NOT import `robots/` from `tango_robot/` or `benchmark/` — the planner and benchmark
use `EnvironmentSoArm` directly and are intentionally unaffected by this layer.

## Success metric

`success rate = "Done pick" count / total attempts`

Parsed from demo.py stdout. The benchmark logger writes per-trial JSONL to `results/`.

## What NOT to do

- Do not modify `legacy/pybullet_panda/` — archived, not used.
- Do not use `GRASP_MODE_DEMO_ATTACH` in any evaluation — results are invalid.
- Do not `import robots` from inside `tango_robot/` or `tango/` — circular imports.
- Do not call `env._step_hook` directly; only `MujocoBackend.execute_grasp()` sets it.
- Do not edit `paper_final.tex` — **submitted to RA-L 2026-07-11** (Editor-in-Chief: Tamim Asfour). This is
  the actual manuscript on file with the journal; do not modify it further while under review. If reviewer
  revisions are requested later, treat that as a distinct, deliberate edit pass, not a routine change.
  - Compiled length at submission: **7 pages** (max allowed with overlength fee is 8; base free limit is 6),
    18 references, double-anonymous compliant (no author block, no institution names, no self-citing links —
    verified by grep before submission). Page count was reduced from an earlier 8-page draft by trimming
    prose (Related Work, Discussion, Scissors/Limitations/Future Work/Conclusion) and fixing a misplaced
    `\balance` command — no results, tables, or statistics were cut.
  - If ever asked to produce a revised/resubmitted version, recompile with
    `latexmk -pdf -interaction=nonstopmode -halt-on-error paper_final.tex && pdfinfo paper_final.pdf` and
    re-verify page count rather than trusting this note — it has gone stale before (multiple times), and note
    "revised version must retain the same keywords and in the exact same order as the previously rejected
    version" per RA-L's own submission form if this ever becomes a resubmission.
