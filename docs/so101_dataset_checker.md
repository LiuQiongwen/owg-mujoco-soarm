# SO-101 dataset checker

`scripts/check_so101_dataset.py` is a **read-only** structural validator for
SO-101 (SO-ARM101) demonstration episode datasets. It never writes to the
dataset it inspects; it only reads files and (optionally) writes a JSON
report to a path you specify outside the dataset.

It does **not** train models, run policies, control a robot, run simulation,
convert datasets, or repair datasets.

## Supported dataset format

The checker targets the "Format B / LeRobot-style" on-disk layout produced by
`datasets.writer.EpisodeWriter` and consumed by `datasets.reader.EpisodeReader`
(the repository's own lightweight episode-dataset interface — see
`datasets/episode.py` for the canonical `Episode` / `EpisodeStep` schema).
This is distinct from the third-party `lerobot` pip package's chunked
parquet/video dataset format (used by `paperA_data/scripts/lerobot_record.py`
and `scripts/record_sim_lerobot_episodes.py` for BC/ACT training data); that
format was not covered here because no such dataset exists on disk in this
repository and parsing it requires heavy optional dependencies
(`pyarrow`, `datasets`) outside the scope of a lightweight checker.

Expected directory layout:

```
<dataset_path>/
    meta.jsonl                 one JSON object per line: {"episode_id": <int>, ...}
    dataset_info.json          optional dataset-level metadata (informational only)
    episodes/
        ep_{episode_id:05d}/
            steps.npz            required arrays: obs (T, 10), action (T, 6), timestamp (T,)
            frames/               optional per-frame images
                step_{idx:04d}_rgb.npy
                step_{idx:04d}_depth.npy
```

### Observation fields (10-dim, per `datasets/episode.py::EpisodeStep.obs_vector()`)

`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll` (JointState, 5) +
`gripper_opening_m, gripper_is_closed` (GripperState, 2) +
`eef_x, eef_y, eef_z` (end-effector position, 3).

### Action fields (6-dim, per `datasets/episode.py::GraspAction.as_vector()`)

`eef_x, eef_y, eef_z, yaw, opening_m, obj_height`.

These field lists are hardcoded in the checker to mirror the dataclasses in
`datasets/episode.py` exactly, so they will drift if that schema changes.

## Assumptions

- `meta.jsonl` is the source of truth for which episodes exist; an episode is
  only checked if it has a `meta.jsonl` entry with an integer `episode_id`.
- Frame/row order in `obs` / `action` / `timestamp` within `steps.npz` is the
  frame-index order (guaranteed by construction in `EpisodeWriter`, which
  appends steps sequentially). There is no separate `frame_index` array to
  cross-check against.
- `frames/` (per-frame image files) are optional — `EpisodeWriter` only
  writes them when `record_frames=True`, and even then a given step's `rgb`
  may be `None`. Missing frame files are therefore reported as **warnings**,
  not errors.
- `dataset_info.json` is informational only; a mismatch against the actual
  episode count is a warning, and a missing file is a (non-fatal) warning.

## Command examples

```bash
conda run -n tango python scripts/check_so101_dataset.py \
  --dataset-path data/lerobot \
  --output /tmp/so101_dataset_report.json

conda run -n tango python scripts/check_so101_dataset.py \
  --dataset-path /path/to/some/format-b/dataset
```

`--output` is optional; the JSON report is always printed to stdout.

## JSON output fields

| Field | Meaning |
|---|---|
| `schema_version` | Report schema version string |
| `dataset_path` | Path passed on the CLI |
| `valid` | `true` iff `errors` is empty |
| `dataset_format` | Fixed identifier: `"so101_lerobot_style_v1"` |
| `number_of_episodes` | Count of entries in `meta.jsonl` |
| `number_of_frames` | Sum of `n_frames` across all episodes |
| `observation_fields` | Canonical 10-dim observation field names |
| `action_fields` | Canonical 6-dim action field names |
| `action_dimension` | Expected action dimensionality (6) |
| `errors` | Dataset- and episode-level errors (each episode error is prefixed `episode <id>: `) |
| `warnings` | Dataset- and episode-level warnings, same prefixing |
| `per_episode_summary` | List of `{episode_id, n_frames, has_steps_file, observation_dim, action_dim, errors, warnings}` |

## Exit-code behavior

| Exit code | Meaning |
|---|---|
| `0` | Dataset passed all checks (`valid: true`) |
| `1` | Dataset failed one or more checks (`valid: false`), path exists |
| `2` | `--dataset-path` does not exist (a message is also printed to stderr) |

## Checks performed

1. Dataset path exists and is a directory.
2. `meta.jsonl` exists, is parseable line-by-line JSON, and yields at least
   one episode with a valid integer `episode_id` (duplicates flagged).
3. Each episode's `steps.npz` exists and contains `obs`, `action`, `timestamp`.
4. Each episode has at least one frame (`T > 0`).
5. `obs` / `action` / `timestamp` row counts agree within an episode.
6. `observation_dim` / `action_dim` match the canonical schema (10 / 6), and
   `action_dim` is consistent across all episodes in the dataset.
7. All numeric values in `obs`, `action`, `timestamp` are finite (no NaN/Inf).
8. `timestamp` is non-decreasing within an episode.
9. When `frames/` is present, per-frame filename indices are ordered and
   every expected `step_{idx:04d}_{rgb,depth}.npy` file is checked for
   existence (missing ones reported as warnings).
10. `dataset_info.json`, if present, is cross-checked against the actual
    episode count (mismatch is a warning).

## Known limitations

- Only the repository's own Format B ("LeRobot-style") layout is supported;
  the third-party `lerobot` pip package's parquet/video dataset format is
  **not** parsed by this tool (see "Supported dataset format" above).
- Frame-index ordering is only independently verified when a `frames/`
  directory is present; otherwise it is trusted to match array row order.
- The checker reports structural/numeric validity only. It makes no claim
  about semantic correctness (e.g. whether grasps in the dataset actually
  succeeded, or whether action values are physically plausible) beyond
  finiteness and dimensionality.
- This tool has not been run against a real production dataset — no such
  dataset was present in this repository at the time of writing. All
  behavior above has only been verified against synthetic datasets built at
  runtime in `tests/test_so101_dataset_checker.py` (both valid and broken
  cases). These are generated on the fly rather than checked in, because the
  repository root `.gitignore` excludes `*.npz` and a committed dataset
  fixture would silently lose its `steps.npz` files on a fresh clone.
