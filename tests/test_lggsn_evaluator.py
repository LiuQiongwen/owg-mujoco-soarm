# -*- coding: utf-8 -*-
"""Tests for research_agent_pilots/lggsn_suite/evaluator.py.

Split by dependency: evaluator.py imports torch LAZILY, only inside
evaluate_checkpoint()/main() -- module-level import (this file's import of
`evaluator` itself) needs no torch, so the output-path-policy and
blocked-record tests run under the research-agent venv. Tests that actually
call evaluate_checkpoint() (real torch forward pass) are gated with
pytest.importorskip("torch") and only run for real under the tango env.

Covers requirement list items:
  16. output confined to assigned artifact directory
  17. evaluator never imports train_lggsn_pairwise.py
  18. evaluator never mutates checkpoint or dataset files
  19. same dataset split used for all ablations
"""
import inspect
import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SUITE_DIR = os.path.join(_REPO_ROOT, "research_agent_pilots", "lggsn_suite")
sys.path.insert(0, _SUITE_DIR)

import evaluator  # noqa: E402
import checkpoint_registry as reg  # noqa: E402


# ── 16. output path policy (torch-free, always runs) ───────────────────────

@pytest.mark.parametrize("bad_dir", [
    "results/lggsn", "checkpoints/lggsn", "data/lggsn", "datasets/lggsn", "paperA_data/lggsn", "paper_final/lggsn",
])
def test_protected_output_dirs_are_rejected(bad_dir):
    with pytest.raises(evaluator.OutputPathPolicyError):
        evaluator._assert_output_dir_allowed(os.path.join(_REPO_ROOT, bad_dir), _REPO_ROOT)


def test_evaluator_own_output_dir_is_allowed():
    ok_dir = os.path.join(_REPO_ROOT, "research_agent_pilots", "lggsn_suite", "eval_outputs", "base")
    evaluator._assert_output_dir_allowed(ok_dir, _REPO_ROOT)  # must not raise


def test_outside_repo_output_dir_is_allowed_but_not_a_protected_prefix(tmp_path):
    # Paths outside the repo entirely (e.g. /tmp scratch during dev) aren't
    # matched against repo-relative prefixes at all -- must not raise.
    evaluator._assert_output_dir_allowed(str(tmp_path / "somewhere"), _REPO_ROOT)


# ── blocked records stay structurally visible ───────────────────────────────

def test_describe_blocked_returns_structured_record_with_reason():
    for name in ("ext_v5", "ext_v5d", "ext2_v6", "v10_baseline", "v11_ik_full"):
        record = evaluator.describe_blocked(name)
        assert record["checkpoint_name"] == name
        assert record["status"] == "BLOCKED"
        assert record["reason"]  # never an empty/bare reason


# ── 17. never imports train_lggsn_pairwise.py (static check, torch-free) ───
#
# Checked as actual `import`/`from ... import` statements (via AST), not a
# bare substring search -- both modules' own docstrings legitimately name
# these modules (to explain why they are deliberately NOT imported), which
# would false-positive a substring check.

def _imported_module_names(source: str) -> set:
    import ast

    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_evaluator_source_never_imports_train_lggsn_pairwise():
    imported = _imported_module_names(inspect.getsource(evaluator))
    assert not any(name == "train_lggsn_pairwise" for name in imported), imported


def test_eval_core_source_never_imports_train_lggsn_pairwise():
    sys.path.insert(0, _SUITE_DIR)
    import eval_core
    imported = _imported_module_names(inspect.getsource(eval_core))
    assert not any(name == "train_lggsn_pairwise" for name in imported), imported


def test_evaluator_source_never_imports_causal_validity_audit():
    imported = _imported_module_names(inspect.getsource(evaluator))
    assert not any(
        name == "causal_validity_audit" or name.startswith("causal_validity_audit.") for name in imported
    ), imported


def test_evaluator_module_level_source_has_no_top_level_torch_import():
    """evaluator.py's own module-level imports must stay torch-free (torch
    is imported lazily, inside evaluate_checkpoint()) -- this is what lets
    the output-path-policy and blocked-record tests above run under the
    research-agent venv at all. Checked structurally (parsed AST of
    top-level statements only), not by sys.modules introspection -- another
    test in the same pytest session may have already imported torch for
    unrelated reasons, which would make a sys.modules check meaningless."""
    import ast

    tree = ast.parse(inspect.getsource(evaluator))
    top_level_imports = []
    for node in tree.body:  # only module-level statements, not nested in functions
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.append(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in top_level_imports), top_level_imports


# ── 17b. runtime confirmation train_lggsn_pairwise.py is never imported ────

def test_running_evaluator_as_subprocess_never_loads_train_lggsn_pairwise():
    """Runs evaluator.py end-to-end (subprocess, real interpreter) against
    a checkpoint that requires torch, and confirms train_lggsn_pairwise is
    never present in the resulting process's loaded modules. Uses
    sys.modules introspection via a wrapper so this works even without
    torch available in THIS test process."""
    pytest.importorskip("torch")  # only meaningful (and only run for real) in the tango env
    code = (
        "import sys, os\n"
        f"sys.path.insert(0, {_SUITE_DIR!r})\n"
        "import evaluator\n"
        f"evaluator.evaluate_checkpoint('base', repo_root={_REPO_ROOT!r}, "
        "output_dir=os.environ['OUT_DIR'])\n"
        "assert 'train_lggsn_pairwise' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    import tempfile
    with tempfile.TemporaryDirectory() as out_dir:
        env = dict(os.environ)
        env["OUT_DIR"] = out_dir
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


# ── 18. never mutates checkpoint or dataset files ───────────────────────────

def test_evaluate_checkpoint_does_not_mutate_checkpoint_or_dataset(tmp_path):
    pytest.importorskip("torch")
    sys.path.insert(0, _SUITE_DIR)
    import eval_core as ec

    entry = reg.CHECKPOINTS_BY_NAME["base"]
    ckpt_path = os.path.join(_REPO_ROOT, entry.relative_path)
    dataset_path = os.path.join(_REPO_ROOT, entry.dataset_identifier)
    before_ckpt = ec.file_sha256(ckpt_path)
    before_dataset = ec.file_sha256(dataset_path)

    evaluator.evaluate_checkpoint("base", repo_root=_REPO_ROOT, output_dir=str(tmp_path / "out"))

    after_ckpt = ec.file_sha256(ckpt_path)
    after_dataset = ec.file_sha256(dataset_path)
    assert before_ckpt == after_ckpt
    assert before_dataset == after_dataset


# ── 19. same dataset split used for all ablations ───────────────────────────

def test_evaluate_checkpoint_repeat_run_produces_identical_digest(tmp_path):
    """End-to-end version of eval_core's unit-level digest-determinism test
    -- runs the real evaluator (real torch forward pass) twice and confirms
    byte-identical metrics (except wall-clock runtime_seconds)."""
    pytest.importorskip("torch")
    m1 = evaluator.evaluate_checkpoint("nozrel", repo_root=_REPO_ROOT, output_dir=str(tmp_path / "run1"))
    m2 = evaluator.evaluate_checkpoint("nozrel", repo_root=_REPO_ROOT, output_dir=str(tmp_path / "run2"))
    assert m1["deterministic_digest"] == m2["deterministic_digest"]
    m1 = dict(m1)
    m2 = dict(m2)
    m1.pop("runtime_seconds")
    m2.pop("runtime_seconds")
    assert m1 == m2


def test_all_four_matrix_checkpoints_use_the_same_dataset_and_split_size(tmp_path):
    pytest.importorskip("torch")
    results = {}
    for name in reg.MATRIX_NAMES:
        out_dir = str(tmp_path / name)
        metrics = evaluator.evaluate_checkpoint(name, repo_root=_REPO_ROOT, output_dir=out_dir)
        results[name] = metrics

    dataset_shas = {m["dataset_sha256"] for m in results.values()}
    group_counts = {m["eligible_group_count"] for m in results.values()}
    val_pair_counts = {m["eligible_pair_count"] for m in results.values()}
    assert len(dataset_shas) == 1, f"different datasets used: {dataset_shas}"
    assert len(group_counts) == 1, f"different episode splits: {group_counts}"
    assert len(val_pair_counts) == 1, f"different pair counts: {val_pair_counts}"
