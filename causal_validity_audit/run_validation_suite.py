#!/usr/bin/env python3
"""
Stage 1 of AUDIT_TOOL_VALIDATION_PLAN.md: measure auto_tagger.py's per-field
classification accuracy, precision/recall/F1 for EXECUTION_DERIVED detection,
and false-positive rate, against test_fixtures/ground_truth.json's
human-assigned labels.

Usage:
  conda run -n tango python3 causal_validity_audit/run_validation_suite.py
  (no MuJoCo/conda dependency actually required -- this only imports ast-based
  auto_tagger.py -- but run inside the project's usual env for consistency)
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causal_validity_audit.auto_tagger import tag_file

FIXTURES_DIR = Path(__file__).resolve().parent / "test_fixtures"
GROUND_TRUTH_PATH = FIXTURES_DIR / "ground_truth.json"


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())["entries"]

    # Cache TagResult per (file, function) since multiple fields share one call.
    tag_cache = {}
    rows = []
    for entry in ground_truth:
        key = (entry["file"], entry["function"])
        if key not in tag_cache:
            path = str(FIXTURES_DIR / entry["file"])
            tag_cache[key] = tag_file(path, entry["function"])
        result = tag_cache[key]
        predicted = result.field_provenance.get(entry["field"], "MISSING")
        rows.append({
            **entry,
            "predicted": predicted,
            "marker_found": result.marker_found,
            "correct": predicted == entry["expected"],
        })

    n = len(rows)
    tp = sum(1 for r in rows if r["expected"] == "EXECUTION_DERIVED" and r["predicted"] == "EXECUTION_DERIVED")
    fn = sum(1 for r in rows if r["expected"] == "EXECUTION_DERIVED" and r["predicted"] != "EXECUTION_DERIVED")
    fp = sum(1 for r in rows if r["expected"] == "PRE_EXECUTION" and r["predicted"] == "EXECUTION_DERIVED")
    tn = sum(1 for r in rows if r["expected"] == "PRE_EXECUTION" and r["predicted"] == "PRE_EXECUTION")
    missing = sum(1 for r in rows if r["predicted"] == "MISSING")

    accuracy = sum(r["correct"] for r in rows) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    print("=" * 72)
    print("Stage 1 validation results")
    print("=" * 72)
    print(f"n = {n} labeled fields across {len(tag_cache)} fixture functions")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision (EXECUTION_DERIVED as positive class): {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"False positive rate: {false_positive_rate:.3f}")
    print()
    print("Confusion matrix:")
    print(f"                    predicted EXEC   predicted PRE   predicted MISSING")
    print(f"  true EXECUTION_DERIVED   {tp:>5}            {fn:>5}            "
          f"{sum(1 for r in rows if r['expected']=='EXECUTION_DERIVED' and r['predicted']=='MISSING'):>5}")
    print(f"  true PRE_EXECUTION      {fp:>5}            {tn:>5}            "
          f"{sum(1 for r in rows if r['expected']=='PRE_EXECUTION' and r['predicted']=='MISSING'):>5}")
    if missing:
        print(f"\nWARNING: {missing} field(s) had no prediction at all (marker not found, or field "
              f"name not present in the analyzed return dict) -- investigate before trusting the "
              f"metrics above, this usually means a fixture/ground-truth mismatch, not a tool result.")

    print()
    print("Per-category breakdown:")
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    for cat in sorted(by_category):
        cat_rows = by_category[cat]
        n_correct = sum(r["correct"] for r in cat_rows)
        print(f"  Category {cat:>2}: {n_correct}/{len(cat_rows)} correct")

    print()
    print("All disagreements (predicted != expected):")
    disagreements = [r for r in rows if not r["correct"]]
    if not disagreements:
        print("  (none)")
    for r in disagreements:
        expected_outcome = r.get("expected_tool_outcome", "pass")
        flag = "" if expected_outcome != "pass" else "  <-- UNEXPECTED, investigate"
        print(f"  [{r['file']}::{r['function']}::{r['field']}] "
              f"expected={r['expected']} got={r['predicted']} "
              f"(category {r['category']}, documented outcome: {expected_outcome}){flag}")

    print()
    unexpected = [r for r in disagreements if r.get("expected_tool_outcome", "pass") == "pass"]
    if unexpected:
        print(f"*** {len(unexpected)} UNEXPECTED disagreement(s) on cases predicted to pass. ***")
        print("*** These are real bugs or gaps in the tagger, not documented limitations. ***")
        sys.exit(1)
    else:
        print("No unexpected disagreements -- all misses match documented, expected limitations "
              "(category 8's unknown-entry-method miss, category 9's static-config false positive).")


if __name__ == "__main__":
    main()
