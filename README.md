# AI Skincare Advisor

## Quickstart

Two commands, on a clean clone, with nothing else installed:

```bash
docker compose up --build
curl localhost:8000/health
```

Then open <http://localhost:8000/docs> for the interactive API, or
<http://localhost:8501> for the demo UI.

A fresh clone has no weights and no index, so the container ships a configuration that
still **answers differently for different input**: real query construction, real
retrieval over the synthetic catalogue in `fixtures/`, a templated writer, and the real
rule-based safety layer. Send two different profiles and you get two different sets of
products, with citations that resolve and the pregnancy and medical-boundary rules
firing. `/analyze-skin` is the one endpoint answering from a fixture — the classifier is
43 MB of weights a fresh clone does not have.

To serve the trained stack, put the adapter under `models/` and the FAISS index under
`data/processed/` (both gitignored, hundreds of megabytes) and set
`USE_MOCK_RETRIEVAL=0 USE_STUB_GENERATOR=0 USE_MOCK_VISION=0`.

Try an endpoint:

```bash
curl -s -X POST localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"profile": {"query": "I have oily skin and keep getting breakouts"}}'
```

Gen AI course group project — **Group 2**.
CNN skin analysis + retrieval-grounded, **post-trained** LLM recommendations.

Three things we actually train:

| Pillar | What | Course anchor |
|---|---|---|
| **1. Vision** | CNN skin classifier (scratch CNN + fine-tuned ResNet) | Module 3, Assignment 2 |
| **2. LLM post-training** | pretrained base → **LoRA SFT** → **RL (GRPO, verifiable rewards)** | Module 8, Modules 9–11, Assignment 5 |
| **3. Generative images** | diffusion img2img augmentation + ablation | Modules 4–7, Assignments 3–4 |

Glue (course tech, no extra training): RAG retrieval (M1–2), FastAPI + Docker (M1/A1), safety & ethics (L1), Streamlit UI.

## Quickstart (2 minutes, no GPU, no models)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ui]"
cp .env.example .env          # USE_MOCKS=1 by default

make api        # http://localhost:8000/docs
make ui         # http://localhost:8501
make test       # 31 tests: contract + rewards + eval harness + data build
```

`USE_MOCKS=1` makes the API answer from `fixtures/`. **This is what lets four
people work in parallel on day 1** — the UI, Docker, safety and evaluation work
all run before a single model exists.

## Build order

| Stage | Command | Needs GPU |
|---|---|---|
| 0. API skeleton | `make api` | no |
| 1. Vision | `python -m skincare.vision.train --kind transfer` | yes |
| 2. RAG | `python -m skincare.rag.ingest && python -m skincare.rag.index` | no |
| 3. SFT | `python -m skincare.llm.data_build --n 800 --mode sft` then `sft_lora` | yes |
| 4. RL | `python -m skincare.llm.grpo_train` | yes |
| 5. Eval | `python -m skincare.eval.llm_eval --split ...` | yes |

Flip `USE_MOCKS=0` once stages 1–4 produce checkpoints.

## Docs

**New teammates start here:** [`CONTRIBUTING.md`](CONTRIBUTING.md) — onboarding steps, file ownership, frozen contracts, handoff and secret-key discipline. *(Internal team document, written in Chinese.)*

**Reading order:**
1. `docs/TECH_DESIGN.md` — the working manual: architecture, file ownership, frozen contracts, per-module commands and acceptance checks
2. `docs/TEAM_TASKS.md` — per-person tasks, the seven-day cadence, report section ownership *(internal team document, written in Chinese)*
3. [`docs/DEPLOY.md`](docs/DEPLOY.md) — deployment, benchmarking and the L4 report tables
4. `docs/HANDOFF.md` — handoff protocol, checkpoint contract, and why editing code can break model loading *(internal team document, written in Chinese)*

## Layout

```
app/        FastAPI service + schemas.py  <- FROZEN CONTRACT
src/skincare/
  vision/   Pillar 1: CNN
  rag/      retrieval
  llm/      Pillar 2: prompts, rewards, sft_lora, grpo_train, dpo_train
  augment/  Pillar 3: diffusion img2img augmentation
  safety/   L1 guardrails
  eval/     harness (decoupled) + judge + run_eval
configs/    vision hyperparameters (the only knob teammates turn)
scripts/    validate_data.py (data handoff) / verify_handoff.py (model handoff)
ui/         Streamlit demo
fixtures/   mock responses (parallel work enabler)
```
