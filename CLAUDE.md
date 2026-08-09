# Working rules for this repository

## Secrets — check before anything leaves this machine

**This repository is public. A leaked key is scraped within minutes.**

Run the scan before every push, and before uploading any file anywhere —
GitHub, Colab, Drive, chat, a shared folder:

```bash
./scripts/check_secrets.sh --staged      # what a commit would ship
./scripts/check_secrets.sh --worktree    # everything on disk
./scripts/check_secrets.sh --file PATH   # one file, before uploading it
./scripts/check_secrets.sh --history     # whole git history
```

`make test` runs the worktree scan first, and `make hooks` installs a pre-commit
hook, so the check does not depend on anyone remembering it.

**Watch the indirect paths.** The near-miss here was not a key pasted into source:
it was `colab log` exporting a session transcript into `reports/*.ipynb`, where an
earlier `colab exec` cell had set the key. Anything that captures a transcript, a
log, a notebook output, or an environment dump can carry a secret. Treat every
generated artefact as suspect before it is committed or uploaded.

**If a real key is ever exposed, rotate it.** Deleting the file or rewriting
history is not sufficient — assume it was captured.

Never write a key into: source files, notebooks, `colab exec` cell text, commit
messages, or shell commands that land in `~/.zsh_history`.

**Keys are never stored inside this repository — not even in a gitignored file.**
Anything with read access to the project directory (an agent working through a
file bridge, a sync client, a shared drive) can read a `.env` sitting there.
`make set-key` puts the key in the macOS Keychain, falling back to
`~/.config/skincare-advisor/` — both outside the project tree. Colab Secrets is
the equivalent for notebooks.

## Before pushing

```bash
make test          # runs the secret scan, then the suite
```

`main` must always be runnable: Colab clones it, and the grader clones it.

## Language

Code comments, docstrings, CLI output, `README.md`, `docs/TECH_DESIGN.md` and
`docs/DEPLOY.md` are in **English** — the grader reads them.

Internal team documents stay in **Chinese**: `CONTRIBUTING.md`,
`docs/TEAM_TASKS.md`, `docs/HANDOFF.md`, and the `MEMBER_*` handoff notes.

## Frozen interfaces

Changing any of these breaks other people's work, sometimes silently. Raise it
with the team before touching them:

| What | Where |
|---|---|
| Data contract | `app/schemas.py` |
| Label space | `src/skincare/config.py` — `CONCERNS`, `SKIN_TYPES` |
| Prompt template | `src/skincare/llm/prompts.py` — training and serving must not drift |
| evidence_id format | `src/skincare/rag/ingest.py` — `{product_id}:{desc\|ing\|rev}:{n}` |

File ownership is in `docs/TEAM_TASKS.md`. Only edit what you own.

## Verify, don't assume

Every acceptance check is a command, not a judgement call:

```bash
make test                                  # suite + secret scan
make validate                              # Member A's data deliverables
python scripts/verify_handoff.py vision <ckpt> | rag
python -m skincare.eval.run_eval --self-test
make verify                                # clean install from pyproject, as the grader does
```

A test that has never failed has not been shown to work. When adding a check,
prove it fires on a bad input.
