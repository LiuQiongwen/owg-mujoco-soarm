#!/usr/bin/env python3
"""Parse Claude Code's outer JSON and extract the model's implementation JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outer")
    ap.add_argument("inner")
    args = ap.parse_args()
    outer = json.loads(Path(args.outer).read_text())
    if not isinstance(outer, dict) or not isinstance(outer.get("result"), str):
        raise SystemExit("invalid Claude outer JSON: missing string result")
    inner = json.loads(outer["result"])
    if not isinstance(inner, dict):
        raise SystemExit("implementation output is not a JSON object")
    Path(args.inner).write_text(json.dumps(inner, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
