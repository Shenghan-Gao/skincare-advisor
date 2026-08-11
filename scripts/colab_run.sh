#!/usr/bin/env bash
# Drive the whole training pipeline on a remote Colab runtime from your terminal.
#
#   read -rs OPENAI_API_KEY && export OPENAI_API_KEY   # prompt silently; keeps the
#                                                      # key out of ~/.zsh_history
#   ./scripts/colab_run.sh            # full pipeline, then pull the artefacts down
#   ./scripts/colab_run.sh retrain    # redo SFT + GRPO + eval with current settings
#   ./scripts/colab_run.sh watch      # live tail; safe to Ctrl-C and re-run
#   ./scripts/colab_run.sh status     # one-shot snapshot
#   ./scripts/colab_run.sh fetch      # just download artefacts from a finished run
#   ./scripts/colab_run.sh stop       # tear the runtime down
#
# Any stage list can also be given directly, with FORCE to redo finished work:
#   FORCE=sft,grpo,eval ./scripts/colab_run.sh run sft grpo manifest eval
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
GPU="${COLAB_GPU:-L4}"                             # T4 | L4 | A100 | H100
POLL_SECONDS="${POLL_SECONDS:-90}"
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
    print(''.join(log.read_text().splitlines(keepends=True)[-14:]), end='')
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
  mkdir -p models/llm reports data/processed
  for a in sft-lora grpo; do
    colab download -s "$SESSION" "/content/drive/MyDrive/skincare_models/$a" "models/llm/$a" \
      && echo "pulled $a" || echo "skip $a (not produced yet)"
  done
  colab download -s "$SESSION" /content/drive/MyDrive/skincare_models/manifest.json \
      models/llm/manifest.json || true
  for f in sft.jsonl rl.jsonl rl_test.jsonl distill_summary.json; do
    colab download -s "$SESSION" "/content/drive/MyDrive/skincare_data/$f" "data/processed/$f" || true
  done
  # The evaluation table is what the report quotes, so it comes down with the rest
  # rather than being left on Drive for someone to copy by hand.
  for f in llm_eval.json llm_eval.md; do
    colab download -s "$SESSION" "/content/drive/MyDrive/skincare_reports/$f" "reports/$f" \
      && echo "pulled $f" || true
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
  if [ -f reports/llm_eval.md ]; then
    echo; echo "--- held-out evaluation ---"; cat reports/llm_eval.md
  fi
  echo "artefacts are in models/llm/, data/processed/ and reports/"
}

cmd_run() {
  need colab
  local stages="$*"
  # Only the distillation stage spends money, and only it needs a key. Requiring one
  # for a pure retrain would block the common case for no reason.
  local needs_key=0
  if [ -z "$stages" ] || grep -qw distil <<<"$stages"; then needs_key=1; fi

  if [ "$needs_key" = 1 ]; then
    # The key is deliberately NOT stored inside this repository. It lives in the
    # macOS Keychain, or failing that in ~/.config/skincare-advisor/ -- both outside
    # the project directory, so anything with access to the repo folder (an agent
    # working through a file bridge, a sync client, a shared drive) cannot read it.
    if [ -z "${OPENAI_API_KEY:-}" ] && command -v security >/dev/null 2>&1; then
      OPENAI_API_KEY="$(security find-generic-password -a "$USER" -s skincare-openai -w 2>/dev/null || true)"
      [ -n "$OPENAI_API_KEY" ] && export OPENAI_API_KEY && echo "loaded key from macOS Keychain"
    fi
    if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$HOME/.config/skincare-advisor/openai_key" ]; then
      OPENAI_API_KEY="$(cat "$HOME/.config/skincare-advisor/openai_key")"
      export OPENAI_API_KEY; echo "loaded key from ~/.config/skincare-advisor/"
    fi
    if [ -z "${OPENAI_API_KEY:-}" ]; then
      echo "No OPENAI_API_KEY found. Store it once, outside this repo:  make set-key"
      exit 1
    fi
  else
    echo "==> stages [$stages] do not include distillation; no API key needed"
  fi

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

  if [ "$needs_key" = 1 ]; then
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
  fi

  echo "==> launching: stages [${stages:-all}]  FORCE [${FORCE:-none}]"
  # `colab exec` blocks and times out; the pipeline runs for hours. So launch it
  # detached on the runtime and poll instead. The kernel outlives the exec call,
  # so a start_new_session child keeps running after this returns.
  #
  # Note this heredoc is deliberately unquoted so FORCE/stages are interpolated by
  # the local shell. Keep the Python body free of other dollar signs.
  colab exec -s "$SESSION" <<PY
import os, subprocess
for marker in ('/content/PIPELINE_DONE', '/content/PIPELINE_FAILED'):
    if os.path.exists(marker):
        os.remove(marker)
env = "FORCE='${FORCE:-}' "
if os.path.exists('/content/.env_key'):
    env += "OPENAI_API_KEY=\$(cat /content/.env_key) "
cmd = env + "nohup python -u /content/run_pipeline.py ${stages} > /content/pipeline.log 2>&1 &"
print('launching:', cmd.replace('\$(cat /content/.env_key)', '<key>'))
subprocess.Popen(cmd, shell=True, start_new_session=True)
print('pipeline launched (detached)')
PY

  echo "==> polling every ${POLL_SECONDS}s. Ctrl-C here does NOT stop the remote run."
  echo "    Resume watching later with:  ./scripts/colab_run.sh watch"
  cmd_watch
  echo
  echo "Done. Stop the runtime when you no longer need it:  ./scripts/colab_run.sh stop"
}

case "${1:-run}" in
  status) need colab; cmd_status ;;
  watch)  need colab; cmd_watch ;;
  fetch)  need colab; cmd_fetch ;;
  stop)   need colab; colab stop -s "$SESSION"; echo "runtime stopped" ;;
  retrain)
    # The common case after a training-config fix: rebuild the adapters and the
    # held-out table, reusing the distilled data and the FAISS index as they are.
    export FORCE="${FORCE:-sft,grpo,eval}"
    cmd_run sft grpo manifest eval ;;
  run)    shift || true; cmd_run "$@" ;;
  *) echo "usage: $0 [run [stages...]|retrain|watch|status|fetch|stop]"; exit 1 ;;
esac
