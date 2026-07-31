"""JSON report writers. Every agent output and the final report are
persisted verbatim as JSON files inside the run directory."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from research_agent.models import RunPaths


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_artifact(path: Path, model: BaseModel) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n")


def write_final_report(run_paths: RunPaths, report: BaseModel) -> None:
    save_json_artifact(run_paths.report_path, report)
