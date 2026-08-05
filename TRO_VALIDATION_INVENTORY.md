# T-RO validation artifact inventory

Read-only inventory, no `paper_tro.tex` edits yet. Answers each row the user asked for.

## Headline finding

**Scenario A, with one structural complication.** All of Stage 1's fixtures, ground truth,
runner, and results exist, are internally consistent, and were independently re-verified in this
pass (re-ran `run_validation_suite.py` from a clean export, got the exact numbers claimed). Stage
2's external-pipeline work (GraspGen, Sim-Grasp) also exists as a real, honest, mostly-null result.
**None of it is on the branch this session has been writing `paper_tro.tex` from
(`feat/lggsn-statistical-analysis`)** — it's on local `main`, commit `1dba924` (2026-08-03) and
never merged into this feature branch, which itself forked from an earlier point on `main`
(`e508ff3`, before `1dba924` existed). This is a branch-divergence problem, not data loss — nothing
was deleted or reset.

**Second complication**: `main`'s `causal_validity_audit/auto_tagger.py` is strictly more capable
than the one on `feat/lggsn-statistical-analysis` — it has the subscript-assignment recognition
(the entire Category 12 mechanism, added specifically for a real Sim-Grasp pattern) and a
fail-closed cross-module helper check that this branch's version lacks outright. The validation
numbers below describe `main`'s tool. They do not automatically describe this branch's tool until
the two are reconciled.

## Inventory table

| 项目 | 结论 |
|---|---|
| 测试夹具 | **存在，人工标注**。`causal_validity_audit/test_fixtures/` (main tip), 12 categories, 14 fixture functions, 30 labeled fields, `ground_truth.json` with documented expected outcomes (including two *intentionally* expected failures, not surprises). |
| 工具输出 | **存在**。`run_validation_suite.py` prints per-field predicted vs. expected provenance, a full confusion matrix, and confidence is tracked in the tagger's `field_confidence` dict (used, not just computed). |
| 指标 | **实际计算过，且刚重新验证过**。Accuracy 0.933, Precision/Recall/F1 0.889 each (EXECUTION_DERIVED as positive class), FPR 0.048, n=30. Re-ran independently in this pass from a clean `git archive main` export — numbers reproduce exactly. |
| 外部流水线 | **真实运行过，诚实的部分-null结果**。GraspGen: code-level-verified clean (no execution boundary exists in the released repo at all — a pure scoring library, nothing to run the tagger against). Sim-Grasp: real repo, drove one genuine tool extension (subscript-assignment recognition), and produced a caught near-miss (a label-writing function almost misreported as a violation, correctly ruled out after call-graph tracing — labels are exempt from the criterion by the project's own corollary). No confirmed real-world violation found in either. Dex-Net not attempted. |
| 版本信息 | Commit `1dba924` on local `main` (not on `origin/main` either — `main` itself is 116 commits ahead of `origin/main`, unpushed). Depends on `causal_validity_audit/auto_tagger.py` + `provenance.py` at that same point in `main`'s history. |
| 可复现性 | **是**。Verified in this pass: exported `main` via `git archive` to a clean temp dir (no working-tree checkout, no risk to `feat/lggsn-statistical-analysis`'s state), ran `run_validation_suite.py`, got bit-identical numbers to `VALIDATION_RESULTS.md`'s claims. |
| 论文可用性 | Stage 1 = real quantitative evidence (fixture-based detection-rate benchmark), citable as such. Stage 2 = boundary/specificity checks + one real tool-development finding + one caught near-miss — explicitly NOT "external validation" in the sense of "found and confirmed a real bug in someone else's code." `AUDIT_TOOL_VALIDATION_PLAN.md` itself says this plainly: "if Stage 2 is resumed later... do not restart the search for a target from scratch." Both are usable, but must be described at the right strength — Stage 1 as detection-rate evidence, Stage 2 as boundary characterization, not as "the tool was validated against real-world violations." |

## What's actually different between `main` and this branch (`feat/lggsn-statistical-analysis`)

- `causal_validity_audit/auto_tagger.py`: `main` has the Category 12 subscript-assignment
  recognition and a cross-module fail-closed helper check; this branch does not.
- `causal_validity_audit/provenance.py`: this branch has today's session's own additions
  (`pc_stats_local` registration, the `ALL_FIELDS` collision guard fix) that `main` does not.
  These are unrelated to Stage 1/2 and don't affect the validation numbers either way.
- `retrospective_audit.py`, `commit_marker.py`: identical on both branches.
- Everything else Stage 1/2 needs (`test_fixtures/`, `run_validation_suite.py`,
  `VALIDATION_RESULTS.md`, `AUDIT_TOOL_VALIDATION_PLAN.md`) exists only on `main`.

## Recommended path (not yet executed — this is inventory only, per instruction)

Reconcile before writing anything into `paper_tro.tex`: merge/cherry-pick `main`'s
`causal_validity_audit/` state (specifically the `auto_tagger.py` improvements, `test_fixtures/`,
`run_validation_suite.py`, `VALIDATION_RESULTS.md`) into this branch, re-run the suite once more
post-merge to confirm nothing regressed, *then* write Stage 1's numbers and Stage 2's honestly-scoped
boundary-check summary into `paper_tro.tex`'s external-validation section. This is almost entirely a
writing task at that point, not new research — matches the "Scenario A: 几乎是纯写作收益" case.
