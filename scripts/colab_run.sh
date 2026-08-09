#!/usr/bin/env bash
# Drive the whole training pipeline on a remote Colab runtime from your terminal.
#
#   ./scripts/colab_run.sh            # full pipeline, then pull the adapters down
#   ./scripts/colab_run.sh status     # what is the remote doing right now
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
GPU="${COLAB_GPU:-T4}"                 # T4 | L4 | G4 | A100 | H100
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
  echo "artefacts are in models/llm/ and data/processed/; full log in reports/colab_run.ipynb"
}

case "${1:-run}" in
  status) need colab; cmd_status ;;
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

    echo "==> exporting OPENAI_API_KEY onto the runtime (ephemeral VM, torn down after)"
    colab exec -s "$SESSION" <<PY
import os
os.environ['OPENAI_API_KEY'] = "$OPENAI_API_KEY"
open('/content/.env_key','w').write("$OPENAI_API_KEY")
print('key set, length', len(os.environ['OPENAI_API_KEY']))
PY

    echo "==> running the pipeline (index -> distil -> SFT -> GRPO). This takes hours."
    echo "    Progress prints here; every stage also checkpoints to Drive."
    colab exec -s "$SESSION" <<'PY'
import os, runpy
os.environ.setdefault('OPENAI_API_KEY', open('/content/.env_key').read().strip())
runpy.run_path('/content/run_pipeline.py', run_name='__main__')
PY

    cmd_fetch
    echo
    echo "Done. Stop the runtime when you no longer need it:  ./scripts/colab_run.sh stop"
    ;;
  *) echo "usage: $0 [run|status|fetch|stop]"; exit 1 ;;
esac
