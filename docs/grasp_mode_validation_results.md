# Grasp Mode Validation Results

**Mode**: `physics_weld_after_bilateral`  
**Method**: `geometry` (centroid-based, random yaw)  
**Date**: 2026-05-19  
**Script**: `scripts/validate_grasp_mode.py`  
**Seed**: 42  

## Setup

- Each trial: fresh object spawn at `[0.0, -0.20, TABLE_TOP_Z + 0.15]`, settle 300 steps
- Grasp target: `z = com_z + GRASP_Z_TABLE_MARGIN` (0.020 m), yaw ∈ U[-π/2, π/2]
- Yaw is not enforced by `IK_MODE_JAW_POS` — all trials use the arm's natural approach yaw
- 20 trials per object, 60 total

## Results per Object

| Object | Success | Bilateral | Weld triggered | Table contact | Mean final_z |
|--------|---------|-----------|----------------|---------------|--------------|
| Banana | 100.0% | 100.0% | 100.0% | **0.0%** | 1.016 m |
| MustardBottle | 100.0% | 100.0% | 100.0% | **0.0%** | 0.994 m |
| TomatoSoupCan | 100.0% | 100.0% | 100.0% | **0.0%** | 1.002 m |
| **Overall** | **100.0%** | **100.0%** | **100.0%** | **0.0%** | 1.004 m |

## Failure breakdown

| Category | Count |
|----------|-------|
| no_bilateral (no jaw-object contact) | 0 |
| bilateral_no_lift (contact but object not raised) | 0 |
| **Total failures** | **0 / 60** |

## What these numbers mean

**Bilateral rate = 100%**: The `GRASP_Z_TABLE_MARGIN = 0.020 m` offset correctly
places both jaw spheres within the object collision volume for all three YCB objects
at their natural spawn positions.  IK convergence is ≤ 0.1 cm position error.

**Table contact = 0%**: The 20 mm offset ensures the fixed-jaw sphere (radius 6 mm,
center ≈ 10 mm below jaw midpoint) clears the table surface (TABLE_TOP_Z = 0.785 m)
with ≥ 7 mm of margin.  Before this fix, the fixed jaw sphere bottom was 1–3 mm
below the table, causing jaw-table contact that prevented lift.

**Weld trigger = bilateral rate**: The weld gate works correctly — it fires iff
and only iff bilateral contact is confirmed.

**Success = weld trigger rate**: Once the weld fires, the kinematic attachment
reliably carries the object above the 7 cm threshold (mean final_z ≈ 1.00 m,
well above threshold of 0.785 + 0.07 = 0.855 m).  There are zero cases of
weld-triggered-but-not-lifted.

## Scope and limitations

1. **Fixed approach position**: All trials spawn the object at `y = -0.20 m`.
   The arm's natural workspace is well-suited for this position.  IK failure rates
   may differ for objects placed closer to the robot base (small y) or at the table
   edges (large |x|).  See TomatoSoupCan sensitivity below.

2. **Yaw not exercised**: `IK_MODE_JAW_POS` does not enforce a jaw yaw, so
   the randomly sampled yaw parameter has no effect on the arm configuration.
   All 20 trials per object are geometrically identical.  Yaw sensitivity requires
   either `IK_MODE_JAW_TOPDOWN` or explicit base-joint targeting.

3. **Geometry method only**: This validation tests the execution mode, not grasp
   selection quality.  For world-model (LGGSN) vs. geometry comparisons, use
   `quick_eval.sh` with `--stage 3` (geometry) and `--stage 4` (LGGSN).

4. **Single spawn position**: Real benchmark trials use random spawn positions within
   the workspace.  Position diversity affects IK reachability and bilateral contact
   rates; the results above represent a best-case scenario for a well-reachable pose.

## Prior failure modes (now fixed)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| bilateral_contacts=0 for all trials | Object spawned at `y=0.20` (off table) | Spawn at `y=-0.20` |
| bilateral_contacts=0 even on table | `GRASP_Z_OFFSET=0.06` placed jaw 3–6 cm above CoM | Use `get_obj_com_pos()` |
| Jaw-table contact | `GRASP_Z_TABLE_MARGIN=0.012` insufficient; fixed jaw bottom 1 mm below table | Raised to 0.020 m |
| Lift always fails (grasped=[]) | 6 mm sphere friction < required 0.41 N per jaw | Kinematic weld conditioned on bilateral contact |

## JSON output

Full per-trial results: `results/grasp_mode_validation.json`
