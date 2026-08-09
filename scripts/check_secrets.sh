#!/usr/bin/env bash
# Scan for credentials before anything leaves this machine.
#
#   ./scripts/check_secrets.sh              # staged changes (what a commit would ship)
#   ./scripts/check_secrets.sh --worktree   # every tracked + untracked file
#   ./scripts/check_secrets.sh --history    # the whole git history
#   ./scripts/check_secrets.sh --file PATH  # one file, before uploading it anywhere
#
# Exit code 1 means something was found. Wired into `make check-secrets`, into
# `make test`, and into the pre-commit hook installed by `make hooks`.
#
# This exists because the repository is public and a leaked key is scraped within
# minutes. It is deliberately a script rather than a habit: the one path that
# nearly leaked a key here (`colab log` exporting a session transcript that
# contained an exec cell) was one nobody thought to check by hand.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# name|regex  -- keep patterns specific enough not to fire on prose
PATTERNS=(
  "OpenAI key|sk-[A-Za-z0-9_-]{20,}"
  "OpenAI project key|sk-proj-[A-Za-z0-9_-]{20,}"
  "Anthropic key|sk-ant-[A-Za-z0-9_-]{20,}"
  "GitHub PAT|gh[pousr]_[A-Za-z0-9]{30,}"
  "GitHub fine-grained PAT|github_pat_[A-Za-z0-9_]{50,}"
  "Hugging Face token|hf_[A-Za-z0-9]{30,}"
  "AWS access key|AKIA[0-9A-Z]{16}"
  "Google API key|AIza[0-9A-Za-z_-]{30,}"
  "Slack token|xox[baprs]-[A-Za-z0-9-]{10,}"
  "Private key block|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)

# Files that legitimately contain key-shaped example text.
EXCLUDE_RE='(^|/)(\.git/|\.venv/|node_modules/|__pycache__/|uv\.lock$|scripts/check_secrets\.sh$)'

MODE="${1:---staged}"
TARGET="${2:-}"
hits=0

report() {  # file, line, label, snippet
  printf '  %s:%s  [%s]  %s\n' "$1" "$2" "$3" "$4"
  hits=$((hits + 1))
}

scan_text() {  # label_prefix, text_on_stdin
  local src="$1" line label re
  while IFS= read -r entry; do
    label="${entry%%|*}"; re="${entry#*|}"
    while IFS=: read -r line content; do
      [ -z "${line:-}" ] && continue
      report "$src" "$line" "$label" "$(printf '%s' "$content" | cut -c1-70)"
    done < <(grep -nE "$re" /dev/stdin 2>/dev/null <<<"$BUF" || true)
  done < <(printf '%s\n' "${PATTERNS[@]}")
}

scan_file() {
  local f="$1"
  [[ "$f" =~ $EXCLUDE_RE ]] && return 0
  [ -f "$f" ] || return 0
  file "$f" 2>/dev/null | grep -q "text\|JSON\|empty" || return 0
  BUF="$(cat "$f" 2>/dev/null)" || return 0
  scan_text "$f"
}

case "$MODE" in
  --staged)
    echo "scanning staged changes..."
    # Refuse obviously-sensitive filenames outright, even if the contents look clean.
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      case "$f" in
        .env|*/.env|.env_key|*/.env_key|*.pem|*.key|*_rsa|*_ed25519)
          report "$f" 0 "sensitive filename" "staged for commit"; continue ;;
      esac
      scan_file "$f"
    done < <(git diff --cached --name-only --diff-filter=ACM 2>/dev/null)
    ;;
  --worktree)
    echo "scanning working tree..."
    while IFS= read -r f; do scan_file "$f"; done \
      < <(git ls-files --cached --others --exclude-standard 2>/dev/null)
    ;;
  --history)
    echo "scanning full git history..."
    BUF="$(git log --all -p 2>/dev/null)"
    scan_text "git-history"
    ;;
  --file)
    [ -n "$TARGET" ] || { echo "usage: $0 --file PATH"; exit 2; }
    echo "scanning $TARGET..."
    scan_file "$TARGET"
    ;;
  *) echo "usage: $0 [--staged|--worktree|--history|--file PATH]"; exit 2 ;;
esac

if [ "$hits" -gt 0 ]; then
  cat <<'MSG'

BLOCKED: credential-shaped strings found above.

  1. remove them from the files
  2. if any of it was ever real, rotate it now -- scrubbing is not enough:
       OpenAI  https://platform.openai.com/settings/organization/api-keys
       GitHub  https://github.com/settings/tokens
  3. if it already reached a commit, the history needs rewriting; ask before pushing
MSG
  exit 1
fi
echo "clean -- no credential material found"
