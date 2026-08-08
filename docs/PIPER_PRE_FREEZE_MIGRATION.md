# Piper pre-freeze integration

This stage fixes contracts and provenance without treating current Piper
outcomes as formal labels. The active path is:

```text
OWG candidate -> Piper feasibility adapter -> execution backend
              -> phase logs -> provisional outcome store
```

The provisional store is intentionally disconnected from candidate-critic
training. `piper_integration.training.require_frozen_sample` is the mandatory
admission gate for every new formal critic loader. It rejects missing fields,
`pre-freeze`, provisional, training-ineligible, version-mismatched, and
`legacy_execution_confounded` records.

## Current contracts

- Candidates have stable IDs, target-instance IDs, explicit pose frame and
  convention, score, and local point-cloud indices.
- Feasibility fields use `null` for unavailable measurements. Callers must not
  replace missing values with zero.
- Execution configuration is fully serializable and deterministically hashed.
- Transit planning receives explicit constraints; the interface has no hidden
  `SAFE_TRANSIT_Z` default.
- Every outcome carries execution and embodiment hashes, source commit, model
  variant, seed, object/candidate IDs, schema version, and Phase 2Y status.

## Freeze boundary

Do not change `pre-freeze` records to `frozen`. After Phase 2Y, Gate 3, and
Hardware Gate 1 pass, define a new `piper-execution-v1` configuration and run a
fresh collection. Historical Pear/Mustard data should remain available with
`legacy_execution_confounded: true` for diagnostics only.

The legacy executor still contains its historical fixed transit waypoint. New
integration code must implement `TransitPlanner.plan_transit(candidate,
constraints)` and record those constraints before that executor can be adopted
under frozen semantics.
