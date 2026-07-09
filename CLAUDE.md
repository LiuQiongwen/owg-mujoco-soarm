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

### Quick eval (20 trials)
```bash
bash scripts/quick_eval.sh           # fast: 4 objects × 5 seeds
bash scripts/quick_eval.sh full      # paper: 4 objects × 25 seeds
bash scripts/quick_eval.sh fast 3    # stage 3 baseline
```

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
```

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

| File | Status | Val pair_acc | Features |
|------|--------|-------------|---------|
| `lggsn_pairwise_live.pt` | v1 (inflated accuracy, noisy negatives) | 0.766 | 12-dim |
| `lggsn_pairwise_live_v2.pt` | **active — use this** | 0.664 | 14-dim |
| `lggsn_geom_only_live.pt` | legacy single-label | — | 12-dim |

v2 uses 14 features: base 12 + `dist_to_centroid` + `z_rel` (min-max height within episode).
BPR pairwise + margin_0.00 is the current best-performing training configuration.

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
- Do not edit `paper_final.tex` without checking `paper_final.pdf` diff — it is the live submission draft.
  - Target venue: RA-L. Hard limit: **8 pages** (references excluded). Currently **5 pages** (verified by compiling commit 9cd0b21, 2026-07-09) — 3 pages of headroom, not "exactly 8" as this note previously said (stale since at least commit 00748d3/9cd0b21's table/ablation additions). Re-verify page count with `latexmk -pdf paper_final.tex && pdfinfo paper_final.pdf` after any substantial edit rather than trusting this note.
