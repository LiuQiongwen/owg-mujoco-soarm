from pathlib import Path

from causal_validity_audit.auto_tagger import tag_file


def _write_case(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "case.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_cross_function_env_state_read_is_execution_derived(tmp_path):
    path = _write_case(
        tmp_path,
        """
def post_execution_label(env):
    return env.success

def score(env, candidate):
    CAUSAL_VALIDITY_COMMIT_POINT()
    return {"score": post_execution_label(env)}
""",
    )
    result = tag_file(str(path), "score")
    assert result.field_provenance["score"] == "EXECUTION_DERIVED"


def test_training_label_remains_non_execution_feature(tmp_path):
    path = _write_case(
        tmp_path,
        """
def score(candidate, success_label):
    return {"candidate": candidate, "label": success_label}
""",
    )
    result = tag_file(str(path), "score")
    assert result.field_provenance == {
        "candidate": "PRE_EXECUTION",
        "label": "PRE_EXECUTION",
    }


def test_unresolved_cross_module_call_fails_closed(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text(
        "def post_execution_label(env):\n    return env.success\n",
        encoding="utf-8",
    )
    scorer = _write_case(
        tmp_path,
        """
from helper import post_execution_label

def score(env, candidate):
    CAUSAL_VALIDITY_COMMIT_POINT()
    return {"score": post_execution_label(env)}
""",
    )
    result = tag_file(str(scorer), "score")
    assert result.field_provenance["score"] == "EXECUTION_DERIVED"
