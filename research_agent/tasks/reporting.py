"""JSON report writers. Every agent output and the final report are
persisted verbatim as JSON files inside the run directory."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from research_agent.models import RunPaths


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_artifact(path: Path, model: BaseModel) -> None:
    """Write `path` via temporary-file-plus-atomic-replace so a reader never
    observes a partially written artifact (required for plan.json, which a
    concurrent `status` invocation may read while a run is in flight)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(model.model_dump_json(indent=2) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def write_final_report(run_paths: RunPaths, report: BaseModel) -> None:
    save_json_artifact(run_paths.report_path, report)
