#!/usr/bin/env bash
# Drive the whole training pipeline on a remote Colab runtime from your terminal.
#
#   read -rs OPENAI_API_KEY && export OPENAI_API_KEY   # prompt silently; keeps the
#                                                      # key out of ~/.zsh_history
#   ./scripts/colab_run.sh            # full pipeline, then pull the adapters down
#   ./scripts/colab_run.sh watch      # live tail; safe to Ctrl-C and re-run
#   ./scripts/colab_run.sh status     # one-shot snapshot
#   ./scripts/colab_run.sh fetch      # just download artefacts from a finished run
#   ./scripts/colab_run.sh stop       # tear the runtime down
#
# Requires the Google Colab CLI (released June 2026):
#   uv tool install google-colab-cli   # or: pip install google-colab-cli
#
# The CLI runs a keep-alive daemon, so the runtime is not reclaimed while idle --
# which is what previously destroyed a finished distillation run. Even so, every
# stage checkpoints to Drive, so a lost runtime never means paying for the same
# OpenAI calls twice.
set -euo pipefail

SESSION="${COLAB_SESSION:-skincare}"
GPU="${COLAB_GPU:-T4}"
POLL_SECONDS="${POLL_SECONDS:-90}"                 # T4 | L4 | G4 | A100 | H100
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need() { command -v "$1" >/dev/null || { echo "missing '$1' -- uv tool install google-colab-cli"; exit 1; }; }

cmd_status() {
  colab status -s "$SESSION" || true
  echo "--- last pipeline events ---"
  colab exec -s "$SESSION" <<'PY' || true
from pathlib import Path
p = Path('/content/drive/MyDrive/skincare_data/pipeline_status.jsonl')
print(''.join(p.read_text().splitlines(keepends=True)[-12:]) if p.exists() else 'no status yet')
PY
}

cmd_watch() {
  local out
  while true; do
    out="$(colab exec -s "$SESSION" <<'PY' 2>/dev/null
from pathlib import Path
log = Path('/content/pipeline.log')
if log.exists():
    print(''.join(log.read_text().splitlines(keepends=True)[-12:]), end='')
else:
    print('waiting for the pipeline to start...')
if Path('/content/PIPELINE_DONE').exists():
    print('###DONE###')
if Path('/content/PIPELINE_FAILED').exists():
    print('###FAILED### ' + Path('/content/PIPELINE_FAILED').read_text().strip())
PY
)"
    printf '\033[2J\033[H'          # clear, so the tail reads like a live view
    echo "=== $(date '+%H:%M:%S')  session '$SESSION' ==="
    echo "$out" | grep -v '^###'
    if grep -q '###DONE###' <<<"$out"; then
      echo; echo "pipeline finished. downloading artefacts..."; cmd_fetch; return 0
    fi
    if grep -q '###FAILED###' <<<"$out"; then
      echo; echo "pipeline FAILED:"; grep '###FAILED###' <<<"$out" | sed 's/###FAILED### //'
      echo "full log:  colab download -s $SESSION /content/pipeline.log ./pipeline.log"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
}

cmd_fetch() {
  mkdir -p models/llm reports
  for a in sft-lora grpo; do
    colab download -s "$SESSION" "/content/drive/MyDrive/skincare_models/$a" "models/llm/$a" \
      && echo "pulled $a" || echo "skip $a (not produced yet)"
  done
  colab download -s "$SESSION" /content/drive/MyDrive/skincare_models/manifest.json \
      models/llm/manifest.json || true
  for f in sft.jsonl rl.jsonl rl_test.jsonl distill_summary.json; do
    colab download -s "$SESSION" "/content/drive/MyDrive/skincare_data/$f" "data/processed/$f" || true
  done
  colab log -s "$SESSION" -o reports/colab_run.ipynb || true
  # The exported session log is the one artefact that could still capture a secret
  # (e.g. if someone pasted a key into a REPL by hand). Refuse to leave it lying
  # around unchecked -- reports/*.ipynb is gitignored, but check anyway.
  if [ -f reports/colab_run.ipynb ] && grep -qE 'sk-[A-Za-z0-9_-]{20,}' reports/colab_run.ipynb; then
    echo
    echo "!! An API key appears inside reports/colab_run.ipynb."
    echo "!! It is gitignored, but delete it and rotate the key to be safe:"
    echo "!!   rm reports/colab_run.ipynb"
    echo "!!   https://platform.openai.com/settings/organization/api-keys"
  else
    echo "session log clean (no key material found)"
  fi
  echo "artefacts are in models/llm/ and data/processed/; full log in reports/colab_run.ipynb"
}

case "${1:-run}" in
  status) need colab; cmd_status ;;
  watch)  need colab; cmd_watch ;;
  fetch)  need colab; cmd_fetch ;;
  stop)   need colab; colab stop -s "$SESSION"; echo "runtime stopped" ;;
  run)
    need colab
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY in your shell before running}"

    echo "==> creating runtime '$SESSION' on $GPU"
    colab sessions | grep -q "$SESSION" || colab new -s "$SESSION" --gpu "$GPU"
    colab status -s "$SESSION"

    echo "==> mounting Drive"
    colab drivemount -s "$SESSION" || true

    echo "==> uploading the real product catalogue (skipped if already on Drive)"
    colab exec -s "$SESSION" <<'PY'
from pathlib import Path
d = Path('/content/drive/MyDrive/skincare_data')
missing = [f for f in ('products.parquet', 'chunks.parquet') if not (d / f).exists()]
print('MISSING ' + ','.join(missing) if missing else 'catalogue already on Drive')
PY
    for f in products.parquet chunks.parquet; do
      if [ -f "data/processed/$f" ]; then
        colab upload -s "$SESSION" "data/processed/$f" \
          "/content/drive/MyDrive/skincare_data/$f" || true
      fi
    done

    echo "==> uploading pipeline script"
    colab upload -s "$SESSION" scripts/run_pipeline.py /content/run_pipeline.py

    # Transfer the key as an uploaded file, never as text inside `colab exec`.
    # Anything typed into an exec cell lands in the session history, and
    # `colab log` later exports that history to a notebook -- which is how a key
    # ends up committed to a public repo. Uploading keeps it out of the log.
    echo "==> transferring OPENAI_API_KEY as a file (kept out of the session log)"
    KEYFILE="$(mktemp)"; chmod 600 "$KEYFILE"
    printf '%s' "$OPENAI_API_KEY" > "$KEYFILE"
    trap 'rm -f "$KEYFILE"' EXIT INT TERM
    colab upload -s "$SESSION" "$KEYFILE" /content/.env_key
    rm -f "$KEYFILE"; trap - EXIT INT TERM
    colab exec -s "$SESSION" <<'PY'
key = open('/content/.env_key').read().strip()
print('key received on runtime, length', len(key))
PY

    # `colab exec` blocks and times out; the pipeline runs for hours. So launch it
    # detached on the runtime and poll instead. The kernel outlives the exec call,
    # so a start_new_session child keeps running after this returns.
    echo "==> launching the pipeline in the background on the runtime"
    colab exec -s "$SESSION" <<'PY'
import os, subprocess
for marker in ('/content/PIPELINE_DONE', '/content/PIPELINE_FAILED'):
    if os.path.exists(marker):
        os.remove(marker)
subprocess.Popen(
    'nohup python -u /content/run_pipeline.py > /content/pipeline.log 2>&1 &',
    shell=True, start_new_session=True)
print('pipeline launched (detached)')
PY

    echo "==> polling every ${POLL_SECONDS}s. Ctrl-C here does NOT stop the remote run."
    echo "    Resume watching later with:  ./scripts/colab_run.sh watch"
    cmd_watch
    cmd_fetch
    echo
    echo "Done. Stop the runtime when you no longer need it:  ./scripts/colab_run.sh stop"
    ;;
  *) echo "usage: $0 [run|watch|status|fetch|stop]"; exit 1 ;;
esac
