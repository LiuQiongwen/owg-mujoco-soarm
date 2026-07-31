#!/usr/bin/env python3
"""Independent, side-effect-limited verifier for the local agent workflow."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def manifest(root: Path, protected: list[str]) -> dict:
    out = {}
    for rel in protected:
        p = root / rel
        if p.is_file():
            out[rel] = {"type":"file", "sha256":sha256(p), "size":p.stat().st_size}
        elif p.is_dir():
            files = {}
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                files[str(f.relative_to(root))] = {"sha256":sha256(f), "size":f.stat().st_size}
            out[rel] = {"type":"directory", "files":files}
        else:
            out[rel] = {"type":"missing"}
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--expected", choices=["PASS", "REVISE", "BLOCKED"], default="PASS")
    ap.add_argument("--command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    root, run = Path(args.root).resolve(), Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    protected = ["results", "data", "paperA_data", "trajs", "grasp_6dof/models", ".agents/locks"]
    before = json.loads((run / "protected_before.json").read_text()) if (run / "protected_before.json").exists() else manifest(root, protected)
    after = manifest(root, protected)
    (run / "protected_after.json").write_text(json.dumps(after, indent=2, sort_keys=True) + "\n")
    changed = before != after
    command_ok = True
    exit_code = 0
    if args.command:
        proc = subprocess.run(args.command, cwd=root, text=True, capture_output=True)
        (run / "verify.stdout").write_text(proc.stdout)
        (run / "verify.stderr").write_text(proc.stderr)
        (run / "verify.exit_code").write_text(str(proc.returncode) + "\n")
        exit_code = proc.returncode
        command_ok = exit_code == 0
    verdict = "PASS" if args.expected == "PASS" and command_ok and not changed else ("REVISE" if args.expected == "REVISE" else "BLOCKED")
    required_artifacts_present = all((run / p).exists() for p in ("verifier.stdout", "verifier.stderr", "verifier.exit_code"))
    if not required_artifacts_present:
        verdict = "BLOCKED"
    result = {"verdict": verdict, "protected_data_unchanged": not changed, "command_ok": command_ok, "required_artifacts_present": required_artifacts_present, "exit_code": exit_code}
    (run / "verifier.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if verdict == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
