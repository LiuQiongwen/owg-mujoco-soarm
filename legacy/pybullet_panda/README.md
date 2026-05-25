# Legacy: PyBullet + Panda Grasp Pipeline

This directory contains the original 6-DoF grasp validation code that used PyBullet simulation with a Franka Panda arm.

## Why archived

The project migrated to SO-ARM101 in MuJoCo as the primary robot pipeline.
PyBullet/Panda code is preserved here for reference and baseline comparisons.

## Contents

| Path | Description |
|------|-------------|
| `grasp_6dof/validate_grasps_panda.py` | Full Panda grasp validation with failure analysis system (9 categories), contact persistence tracking, frame capture, and `failure_summary.csv` output |
| `grasp_6dof/validate_grasps_panda.backup.py` | Earlier backup of validation script |
| `grasp_6dof/validate_grasps_panda_debug_broken.py` | Debug variant (broken/experimental) |
| `grasp_6dof/validate_grasps_panda.py.bak` | Bak file from patch session |
| `grasp_6dof/validate_grasps_pybullet.py` | Simpler PyBullet-only validator (no Panda arm) |
| `grasp_6dof/grasp_validator_panda.py` | Modular Panda validator class |
| `grasp_6dof/patch_validate.py` | Patch script for validation fixes |
| `grasp_6dof/bench_random_grasps.py` | Benchmarking random grasps in PyBullet |
| `owg_robot/env_panda.py` | PyBullet Panda environment wrapper |
| `owg_robot/robot_panda.py` | Panda robot controller |
| `scripts/gen_6dof_dataset.py` | Dataset generation using Panda validation |
| `scripts/run_grid.sh` | Grid sweep over grasp parameters (Panda) |
| `scripts/run_grid_objects.sh` | Multi-object grid sweep (Panda) |
| `scripts/sweep_panda_params.sh` | Parameter sweep for Panda tuning |

## Results achieved

- Banana: 75% success (6/8 grasps), PCA lateral prior confirmed (elongation=10.1)
- Cylinder: 25% success (2/8)
- Key finding: IK orientation fix + joint limits + `world_yaw` field resolved gripper rotation issues

## Primary pipeline (current)

See `benchmark/replay_soarm_grasp.py` and `owg_robot/` for SO-ARM101 MuJoCo pipeline.
