#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TASK=${1:?usage: orchestrate.sh TASK.md [--dry-run|--safe-readonly]}
MODE=${2:-}
DRY=0; SAFE_READONLY=0
case "$MODE" in
  --dry-run) DRY=1 ;;
  --safe-readonly) SAFE_READONLY=1 ;;
  "") ;;
  *) echo "REFUSE: unsupported mode $MODE" >&2; exit 30 ;;
esac
TASK=$(cd "$ROOT" && realpath "$TASK")
[[ -f "$TASK" ]] || { echo "TASK_NOT_FOUND: $TASK" >&2; exit 2; }
TASK_ID=$(basename "$TASK" .md)
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)_${TASK_ID}_$RANDOM}
RUN="$ROOT/.agents/runs/$RUN_ID"
[[ -e "$RUN" ]] && { echo "REFUSE: run_id exists: $RUN_ID" >&2; exit 20; }
if [[ "${CONFIRMATORY:-0}" == "1" ]]; then
  LOCK="$ROOT/.agents/locks/${TASK_ID}.lock"
  ( set -o noclobber; : > "$LOCK" ) 2>/dev/null || { echo "REFUSE: confirmatory lock exists: $LOCK" >&2; exit 21; }
fi
mkdir -p "$RUN"
atomic_json(){ local file=$1; local tmp="${file}.tmp.$$"; cat > "$tmp"; python3 -m json.tool "$tmp" >/dev/null; mv -f "$tmp" "$file"; }
run_cmd(){
  local name=$1; local stdin_file=$2; shift 2
  set +e
  timeout --signal=TERM --kill-after=2s "${TASK_TIMEOUT_SECONDS:-30}s" "$@" <"$stdin_file" >"$RUN/$name.stdout" 2>"$RUN/$name.stderr"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$RUN/$name.exit_code"
  return 0
}
run_safe_readonly(){
  cp "$TASK" "$RUN/task.md"
  cat "$ROOT/.agents/prompts/project_context.md" > "$RUN/combined_planner_prompt.md"
  printf '\n===== ROLE_PROMPT =====\n' >> "$RUN/combined_planner_prompt.md"
  cat "$ROOT/.agents/prompts/planner.md" >> "$RUN/combined_planner_prompt.md"
  printf '\n===== TASK =====\n' >> "$RUN/combined_planner_prompt.md"
  cat "$TASK" >> "$RUN/combined_planner_prompt.md"
  cat "$ROOT/.agents/prompts/project_context.md" > "$RUN/combined_executor_prompt.md"
  printf '\n===== ROLE_PROMPT =====\n' >> "$RUN/combined_executor_prompt.md"
  cat "$ROOT/.agents/prompts/executor.md" >> "$RUN/combined_executor_prompt.md"
  printf '\n===== TASK =====\n' >> "$RUN/combined_executor_prompt.md"
  cat "$TASK" >> "$RUN/combined_executor_prompt.md"
  cat "$ROOT/.agents/prompts/project_context.md" > "$RUN/combined_reviewer_prompt.md"
  printf '\n===== ROLE_PROMPT =====\n' >> "$RUN/combined_reviewer_prompt.md"
  cat "$ROOT/.agents/prompts/reviewer.md" >> "$RUN/combined_reviewer_prompt.md"
  printf '\n===== TASK =====\n' >> "$RUN/combined_reviewer_prompt.md"
  cat "$TASK" >> "$RUN/combined_reviewer_prompt.md"
  git -C "$ROOT" status --porcelain=v1 > "$RUN/git_before.txt"
  git -C "$ROOT" diff --binary > "$RUN/user_changes_before.patch"
  git -C "$ROOT" diff --cached --binary > "$RUN/user_staged_before.patch"
  git -C "$ROOT" ls-files --others --exclude-standard > "$RUN/user_untracked_before.txt"
  printf '%s\n' \
    'codex exec --sandbox read-only --output-schema <run>/plan.schema.json --output-last-message <run>/plan.raw.json - < <run>/combined_planner_prompt.md' \
    'claude --print --system-prompt <project_context.md> --output-format json --permission-mode plan --tools Read,Grep,Glob < <run>/combined_executor_prompt.md' \
    'codex exec --sandbox read-only --output-schema <run>/review.schema.json --output-last-message <run>/review.raw.json - < <run>/combined_reviewer_prompt.md' \
    > "$RUN/commands.txt"
  atomic_json "$RUN/state.json" <<EOF
{"schema_version":"1.0","run_id":"$RUN_ID","task_id":"$TASK_ID","verdict":"","summary":"safe-readonly scaffold created; commands not executed","issues":[],"artifacts":["commands.txt","combined_planner_prompt.md","combined_executor_prompt.md","combined_reviewer_prompt.md"],"status":"SAFE_READONLY_MODE_SCAFFOLDED"}
EOF
}

write_stage_failure(){
  local stage=$1
  local summary=$2
  atomic_json "$RUN/state.json" <<EOF
{"schema_version":"1.0","run_id":"$RUN_ID","task_id":"$TASK_ID","verdict":"BLOCKED","summary":"$summary","issues":["$stage"],"artifacts":[],"status":"BLOCKED"}
EOF
  exit 40
}

if (( SAFE_READONLY )); then
  run_safe_readonly
  echo "SAFE_READONLY_SCAFFOLDED run_id=$RUN_ID run_dir=$RUN"
  exit 0
fi

cp "$TASK" "$RUN/task.md"
cat "$ROOT/.agents/prompts/project_context.md" > "$RUN/combined_planner_prompt.md"
printf '\n===== ROLE_PROMPT =====\n' >> "$RUN/combined_planner_prompt.md"
cat "$ROOT/.agents/prompts/planner.md" >> "$RUN/combined_planner_prompt.md"
printf '\n===== TASK =====\n' >> "$RUN/combined_planner_prompt.md"
cat "$TASK" >> "$RUN/combined_planner_prompt.md"
discover_cli(){
  local name=$1; shift
  local out="$RUN/${name}_help.stdout.log" err="$RUN/${name}_help.stderr.log" code="$RUN/${name}_help.exit_code"
  if ! command -v "$name" >/dev/null 2>&1; then
    : >"$out"; printf 'command unavailable\n' >"$err"; printf '127\n' >"$code"; return 0
  fi
  set +e
  timeout --signal=TERM --kill-after=2s "${CLI_DISCOVERY_TIMEOUT_SECONDS:-10}s" \
    env CI=1 NO_COLOR=1 "$@" </dev/null >"$out" 2>"$err"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$code"
  return 0
}
discover_cli codex codex exec --help
discover_cli claude claude --help
python3 - "$RUN" <<'PY'
import json, pathlib, re, sys
run = pathlib.Path(sys.argv[1])
def cap(name):
    code = int((run/f"{name}_help.exit_code").read_text().strip())
    text = (run/f"{name}_help.stdout.log").read_text(errors="replace")
    args = [a for a in ("--system-file", "--input", "--prompt", "--stdin") if re.search(r"(?<!\w)"+re.escape(a)+r"(?!\w)", text)]
    status = "UNAVAILABLE" if code == 127 else ("TIMEOUT" if code == 124 else ("OK" if code == 0 else "ERROR"))
    return {"available": code != 127, "discovery_status": status, "supports_stdin": "--stdin" in args, "supported_prompt_arguments": args, "help_exit_code": code}
(run/"cli_capabilities.json").write_text(json.dumps({"codex":cap("codex"),"claude":cap("claude")}, indent=2)+"\n")
PY
git -C "$ROOT" status --short > "$RUN/git_before.txt"
git -C "$ROOT" diff --binary > "$RUN/user_changes_before.patch"
python3 "$ROOT/.agents/scripts/verify.py" --root "$ROOT" --run-dir "$RUN" --expected PASS >/dev/null || true
cp "$RUN/protected_after.json" "$RUN/protected_before.json"
atomic_json "$RUN/state.json" <<EOF
{"schema_version":"1.0","run_id":"$RUN_ID","task_id":"$TASK_ID","verdict":"","summary":"initialized","issues":[],"artifacts":[],"round":0,"phase":"initialized","status":"RUNNING"}
EOF

if (( DRY )); then
  command -v codex >/dev/null 2>&1 && codex_ok=true || codex_ok=false
  command -v claude >/dev/null 2>&1 && claude_ok=true || claude_ok=false
  atomic_json "$RUN/cli_verdict.json" <<EOF
{"schema_version":"1.0","run_id":"$RUN_ID","task_id":"$TASK_ID","verdict":"PASS","summary":"CLI discovery dry-run completed","issues":[],"artifacts":["cli_capabilities.json"],"dry_run":true,"codex_cli_available":$codex_ok,"claude_cli_available":$claude_ok,"max_rounds":${MAX_ROUNDS:-2},"timeout_seconds":${TASK_TIMEOUT_SECONDS:-30}}
EOF
  atomic_json "$RUN/state.json" <<EOF
{"schema_version":"1.0","run_id":"$RUN_ID","task_id":"$TASK_ID","verdict":"PASS","summary":"Dry-run completed without agent invocation","issues":[],"artifacts":["cli_verdict.json","cli_capabilities.json"],"round":0,"phase":"dry_run","status":"PASS"}
EOF
  echo "DRY_RUN_PASS run_id=$RUN_ID run_dir=$RUN"; exit 0
fi

echo "Execution is intentionally disabled in the minimal scaffold. Use --dry-run first." >&2
exit 30
