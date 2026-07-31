# Task: SO-101 Dataset Checker

## Goal

Implement a lightweight, read-only checker for the SO-101 demonstration dataset format used by this repository.

The checker must inspect dataset structure and produce a machine-readable validation report without modifying the dataset.

## Scope

This task is only for dataset validation.

Do not implement:

- behavioral cloning;
- model training;
- policy evaluation;
- robot control;
- simulation experiments;
- dataset conversion;
- dataset repair.

## Repository inspection

Before implementing, inspect the repository for:

- existing SO-101 or LeRobot dataset loading code;
- dataset paths and configuration;
- episode representation;
- observation and action field names;
- image or video references;
- timestamps and frame indices.

Reuse existing interfaces where appropriate. Do not invent a conflicting dataset format.

If no usable SO-101 dataset is present locally, implement and test the checker using a minimal synthetic fixture.

## Allowed paths

You may add or modify only:

- scripts/check_so101_dataset.py
- tests/test_so101_dataset_checker.py
- tests/fixtures/so101_dataset/
- docs/so101_dataset_checker.md
- .agents/tasks/active/so101_dataset_checker.md

## Forbidden paths

Do not modify:

- data/
- datasets/
- results/
- checkpoints/
- paper/
- grasp_6dof/models/
- .agents/locks/
- existing experiment outputs;
- existing robot calibration files.

## Required checks

Where supported by the discovered dataset format, check:

1. Dataset path exists and is readable.
2. Dataset contains at least one episode.
3. Each episode contains at least one frame.
4. Required observation fields exist.
5. Required action fields exist.
6. Observation and action lengths are consistent.
7. Action dimensions are consistent across frames.
8. Numeric values are finite.
9. Frame indices are ordered.
10. Timestamps are ordered when available.
11. Referenced image or video files exist when applicable.
12. Empty, malformed, or incomplete episodes are reported.
13. Dataset files are not modified.

## CLI

Provide a command similar to:

python scripts/check_so101_dataset.py \
  --dataset-path <path> \
  --output /tmp/so101_dataset_report.json

The exact arguments may be adapted to the repository’s existing dataset interface.

## Output

Write a JSON report containing at least:

- schema_version;
- dataset_path;
- valid;
- dataset_format;
- number_of_episodes;
- number_of_frames;
- observation_fields;
- action_fields;
- action_dimension;
- errors;
- warnings;
- per_episode_summary.

The command must return:

- exit code 0 when the dataset passes;
- nonzero exit code when validation errors are found;
- a clear error when the dataset path is missing.

## Tests

Add lightweight tests covering:

- valid synthetic dataset;
- missing dataset path;
- empty dataset;
- inconsistent action dimensions;
- NaN or infinite numeric values;
- missing required fields.

Tests must use temporary directories or fixtures and must not require:

- GPU;
- network access;
- real robot;
- Docker;
- sudo.

## Documentation

Document:

- supported dataset format;
- assumptions;
- command examples;
- JSON output fields;
- exit-code behavior;
- known limitations.

## Acceptance criteria

- `python -m compileall` passes;
- targeted pytest tests pass;
- only allowed paths are changed;
- no raw dataset is modified;
- valid fixture returns exit code 0;
- invalid fixture returns nonzero;
- output is valid JSON;
- no experimental result is claimed.

## Prohibited actions

Do not:

- run full training;
- connect to the robot;
- modify raw data;
- download datasets;
- use sudo;
- use Docker;
- commit;
- push;
- fabricate validation results.
