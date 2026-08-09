# Technical Design Document v4 · AI Skin Analysis + Post-Trained Skincare Advisor

> Group 2 · Gen AI Course Project
> **This is a working manual, not a proposal.** For every module it states: who owns it, which files it
> touches, which commands to run, what it delivers, and how it is accepted.

---

## 0. How to read this document

| You are | Must read | Optional |
|---|---|---|
| **Anna (main line)** | everything | — |
| **Member A (data + safety + report editing)** | §1–§3, §5, §6.4, §6.6, §8, §10 | §6.2 |
| **Member B (model training)** | §1–§3, §5, §6.1, §6.4, §8, §11 | §6.7 |
| **Member C (augmentation + evaluation)** | §1–§3, §5, §6.3, §6.7, §8, §9 | §6.2 |

Companion documents: `TEAM_TASKS.md` (per-person tasks and schedule) and `HANDOFF.md` (handoff protocol and checkpoint contract). Both are internal team documents written in Chinese.

---

## 1. The project in one sentence

The user uploads a selfie and describes what they want in natural language → **a CNN identifies the skin condition** → **RAG retrieves evidence from real products and reviews**
→ **an LLM post-trained with LoRA fine-tuning plus reinforcement learning** generates personalized, explainable recommendations with evidence citations
→ the safety module attaches disclaimers and filters contraindications → served by FastAPI with a Streamlit front end, delivered in Docker.

**There are three things we genuinely train ourselves** (this is what makes the project more than an "API wrapper"):

| | Pillar | Content | Course anchor | Owner |
|---|---|---|---|---|
| **1** | Vision | CNN skin classification (hand-built CNN + transfer-learning fine-tuning) | M3 / A2 | B |
| **2** | LLM post-training | pretrained base → **LoRA SFT** → **RL (GRPO)** | M8 / M9–11 / A5 | Anna |
| **3** | Generative images | diffusion img2img data augmentation + ablation study | M4–M7 / A3–A4 | C |

**The glue layer** (course technology, no extra training required): RAG retrieval (M1–M2), FastAPI + Docker (M1/A1),
safety and ethics (L1), the Streamlit front end, and the evaluation suite.

---

## 2. Terminology clarifications (use the right words in the report)

| Claim | Correct? | Correct phrasing |
|---|---|---|
| "we pre-trained with LoRA" | ❌ no such thing | **off-the-shelf pretrained base → LoRA supervised fine-tuning (SFT) → RL post-training**. We adapt; we do not pre-train |
| "we trained a GPT" | ❌ | We **fine-tuned** an open-source base model (Qwen2.5-1.5B) and used GPT for teacher distillation and as a judge |
| "RLHF" | ⚠️ imprecise | We use **GRPO + verifiable rewards**, which needs no human preference labelling — closer to what DeepSeek-R1 did |
| "the diffusion model generates product images" | ❌ | Diffusion is used for **training-data augmentation**; the goal is to raise classification F1, not to produce display images |

---

## 3. System architecture

```
        User (selfie + natural-language request)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
 ┌──────────────────────┐  ┌────────────────────────────┐
 │ Pillar 1: CNN        │  │ UserProfile                │
 │ skin analysis        │  │ budget / prefs / pregnancy │
 └───────────┬──────────┘  └─────────────┬──────────────┘
             │ SkinAnalysis              │
             └─────────────┬─────────────┘
                           ▼
        ┌────────────────────────────┐   ┌────────────────────────────┐
        │ RAG retrieval              │◄──│ product / ingredient /     │
        │ fused query construction   │   │ review FAISS vector store  │
        └──────────────────┬─────────┘   └────────────────────────────┘
                           │ Evidence[] + Product[]
                           ▼
        ┌────────────────────────────┐
        │ Pillar 2: post-trained LLM │
        │ base → LoRA SFT → GRPO     │
        └──────────────────┬─────────┘
                           │ AdvisorResponse
                           ▼
        ┌────────────────────────────┐
        │ Safety module (L1)         │  contraindication filter + disclaimer
        └──────────────────┬─────────┘
                           ▼
        FastAPI ──► Streamlit front end

  Offline: Pillar 3 diffusion augmentation ──► synthetic samples ──► retrain Pillar 1 (ablation study)
```

**Key design decision: the CNN is not inside the GRPO training loop.** The skin profiles in the RL training samples are synthetic
(see `llm/data_build.py::sample_profile`); the CNN only enters the chain **at serving time**.
That is exactly why B optimizing the CNN and Anna training GRPO can proceed fully in parallel.

---

## 4. Code structure and file ownership

**Editing someone else's file = a conflict. Check this table before you touch anything.**

```
app/                          [Anna] FastAPI service
  schemas.py                  ⚠️ frozen contract; changes need whole-group agreement
  main.py  deps.py  routers/{health,skin,recommend}.py
src/skincare/
  config.py                   ⚠️ frozen: CONCERNS / SKIN_TYPES label space
  vision/                     [B trains / Anna maintains the code]
    model.py                  SimpleCNN + TransferNet (dual head)
    data.py  train.py  infer.py
  rag/                        [A builds the store / B tunes it]
    ingest.py                 A: raw CSV → products/chunks
    index.py  retrieve.py     B: vectorization and retrieval tuning
  llm/                        [Anna only]
    prompts.py                ⚠️ frozen: training and serving share one template
    rewards.py                verifiable rewards (the intellectual core of the project)
    data_build.py  sft_lora.py  grpo_train.py  dpo_train.py  generate.py
  augment/                    [C only]
    diffusion_aug.py          diffusion img2img augmentation
  safety/guard.py             [A] contraindication filter + disclaimer
  eval/                       [C only]
    harness.py                model-loading decoupling layer (reads the manifest)
    judge.py  run_eval.py  rag_eval.py  vision_eval.py
configs/vision_*.yaml         [B] the only hyperparameter knob
scripts/
  validate_data.py            acceptance check for A's deliverables
  verify_handoff.py           acceptance check for B's deliverables
fixtures/                     mock responses + 10 verified evaluation samples
tests/                        contract / rewards / evaluation self-check (31 of them)
data/{raw,processed,knowledge} data (not committed to git)
models/{vision,llm/manifest.json}
```

---

## 5. Frozen contracts (touch these four and everything downstream breaks)

> What is frozen is the **interface**, not the implementation. Architecture, backbone, hyperparameters, index contents and product count can all change freely.

| # | Frozen item | File | What happens if you change it |
|---|---|---|---|
| 1 | **Data contract** | `app/schemas.py` | rework across the whole project |
| 2 | **Label space** | `config.py::CONCERNS`/`SKIN_TYPES` | checkpoints, reward functions and already-generated training data all break together |
| 3 | **Prompt template** | `llm/prompts.py` | training/serving inputs diverge → **silent degradation** (no error, just worse results) |
| 4 | **evidence_id format** | naming convention in `rag/ingest.py` | the grounding reward becomes distorted |

### 5.1 Core data structures (`app/schemas.py`)

```python
SkinAnalysis:   skin_type(oily|dry|combination|normal), skin_type_confidence,
                concerns[6 x ConcernScore], model_version
Evidence:       evidence_id, product_id, source(description|review|ingredient), text, score
Product:        product_id, name, brand, category, price_usd, rating, ingredients[]
UserProfile:    query, budget_usd, preferences[], avoid_ingredients[], pregnant
Recommendation: product_id, name, brand, price_usd, reason,
                key_ingredients[], cited_evidence[], matched_concerns[]
AdvisorResponse:analysis, recommendations[], routine_note, disclaimer,
                safety_flags[], generator
```

### 5.2 Label space (6 concerns — **do not add a 7th**)

```python
SKIN_TYPES = ["oily", "dry", "combination", "normal"]
CONCERNS   = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]
```

### 5.3 evidence_id format

```
{product_id}:{desc|rev|ing}:{index}      e.g. P1001:rev:3  /  P1001:desc:0
```

### 5.4 Mock mode (the infrastructure that makes parallel development possible)

Three layers of mock infrastructure let four people start work simultaneously on day one:

| File | What it stands in for | Who benefits |
|---|---|---|
| `fixtures/mock_skin_analysis.json`, `mock_advisor_response.json` | the entire API response (`USE_MOCKS=1`) | front end / deployment / safety |
| `fixtures/mock_catalog.json` (12 products / 48 evidence items) | the real FAISS index | **Anna's SFT/GRPO pipeline** |
| `fixtures/eval_samples.jsonl` (10 known answers) | the model being evaluated | **Member C's evaluator** |

**With no models and no real data whatsoever, all four workstreams can still move forward.**

---

## 6. Detailed module design

Every module is specified the same way: **owner / files / commands / contract / acceptance**.

### 6.1 Pillar 1: CNN skin analysis — M3 / A2

**Owner** B (training and tuning) · **Files** `src/skincare/vision/`, `configs/vision_*.yaml`

Having two models is deliberate — the report needs the comparison:

- `SimpleCNN` — four hand-written convolutional blocks (Conv→BN→ReLU→MaxPool) + global pooling + dual head.
  This is the evidence of "understanding from first principles" that the rubric rewards, so **do not delete it**.
- `TransferNet` — a pretrained ResNet/EfficientNet backbone + the same dual head. **Fine-tuning a pretrained model is exactly L2.**

**Dual-head design**: skin type as a single-label task (CrossEntropy) plus the 6 concerns as a multi-label task (BCEWithLogits),
both produced by one forward pass (`model.py::multitask_loss`).

```bash
python -m skincare.vision.train --config configs/vision_transfer.yaml
python -m skincare.vision.train --config configs/vision_simple.yaml   # control baseline, do not skip it
```

**Hyperparameters change only through yaml, never by editing .py.** Suggested sweep: backbone ∈ {resnet18, resnet50} × lr ∈ {1e-4, 3e-4, 1e-3}.

**Checkpoint contract** (relied on by `infer.py`):
```python
{"state_dict": ..., "kind": "transfer"|"simple", "metrics": {...}, "config": {...}}
```
The checkpoint is **self-describing**: the architecture is read from `config.backbone` rather than guessed from the current code —
so if you swap the backbone and retrain, Anna can load it without changing any code (already verified with resnet18/resnet50/SimpleCNN).
A mismatched label space raises `CheckpointMismatch` and states the reason explicitly.

**Acceptance** `python scripts/verify_handoff.py vision models/vision/<run_name>.pt`
**Metrics** accuracy, macro-F1, per-class confusion matrix; plus the hand-built CNN vs transfer-learning comparison table.

---

### 6.2 Pillar 2: the LLM post-training stack — M8 / M9–11 / A5 [project core]

**Owner** Anna · **Files** `src/skincare/llm/`

```
Qwen2.5-1.5B-Instruct (off-the-shelf base; we do not pre-train it)
   │
   ├─ Stage 1  LoRA supervised fine-tuning (SFT)  ← M8: learn the domain's structured explanation style
   │
   └─ Stage 2  RL post-training with GRPO         ← M9–11 / A5: optimize quality with verifiable rewards
```
1.5B was chosen so that **a free Colab T4 can run the whole thing**; with a better GPU you can swap in a 7–8B model.

#### (a) Data construction `data_build.py` [implemented]

Synthesize user profiles → retrieve real evidence → teacher distillation → **filter by reward** → split the dataset.

```bash
# Day one: run the whole pipeline at zero cost (synthetic catalog + offline fake teacher; no data from A, no API key)
python -m skincare.llm.data_build --n 60 --mock-retrieval --dry-teacher

# Build the real SFT data (requires OPENAI_API_KEY)
python -m skincare.llm.data_build --n 800 --mode sft

# Build RL data only (no target answers needed, costs nothing)
python -m skincare.llm.data_build --n 600 --mode rl
```

**It produces three files**
| File | Purpose | Who uses it |
|---|---|---|
| `sft.jsonl` | carries teacher target answers, for LoRA SFT | Anna |
| `rl.jsonl` | prompt + reward context only, for GRPO | Anna |
| `rl_test.jsonl` | held-out set (15% by default) | **Member C's evaluation** |

**Three key design decisions**
1. **Teacher filtering is what decides whether SFT works at all**: any teacher output scoring below `--threshold` (0.8 by default) is discarded outright.
   Not filtering means teaching the model noise. In practice this throws away answers where the ingredients do not match the concerns.
2. **Resumable runs**: every teacher result is written to `sft.cache.jsonl` immediately, so a crash and restart does not burn money twice.
3. **`--mock-retrieval` lets Anna work in parallel too**: `fixtures/mock_catalog.json`
   (12 products / 48 evidence items, covering all 6 concerns) stands in for the real index,
   and the flag is simply dropped once A delivers. **Apply the parallelism principle to yourself as well.**

The RL data only needs the prompt plus the reward context (`concerns` / `evidence_ids` / `product_ids` /
`pregnant` / `avoid`) — no target answers. That is precisely where GRPO is less work than SFT.

#### (b) Stage 1: LoRA SFT `sft_lora.py`

PEFT LoRA (r=16, α=32, dropout=0.05) applied to all attention and MLP projection layers, via TRL's `SFTTrainer`.
If you run out of VRAM, switch to QLoRA (4-bit).

```bash
python -m skincare.llm.sft_lora --epochs 2
```

#### (c) Stage 2: GRPO `grpo_train.py`

**Why GRPO**: (1) it is **the algorithm used by DeepSeek-R1, which M11 studies closely**;
(2) it **requires no trained reward model** (our reward rules are verifiable); (3) it is far lighter than PPO — there is no value network.

Mechanism: for each prompt, sample a **group** of candidates → score them with the rewards → push probability mass toward the answers that score above the group average.
A KL penalty of `beta=0.04` prevents reward hacking; the RL learning rate must be two orders of magnitude smaller than the SFT one (1e-6).

```bash
python -m skincare.llm.grpo_train --steps 300 --group-size 8
```

#### (d) Verifiable rewards `rewards.py` — **the intellectual core of the project**

**No GPU required; this can be written and unit-tested on day one.**

| Reward term | Weight | Programmatic check |
|---|---|---|
| `format` | 0.15 | does it parse as the specified JSON schema |
| `ingredient_match` | 0.30 | do the recommended ingredients address the detected concerns (looked up in `ingredient_rules.json`) |
| `grounding` | 0.25 | do the ids in `cited_evidence` actually exist in the context — **penalizes hallucinated citations** |
| `product_validity` | 0.15 | does the recommended product come from retrieval rather than from the model's memory |
| `safety` | 0.15 | is a disclaimer present; pregnancy/allergy contraindicated ingredients are penalized heavily |

The rule table is maintained by **A** — **every rule A adds makes the RL training signal sharper**, which is a natural parallel interface.

#### (e) Fallback ladder

```
GRPO does not converge → DPO (dpo_train.py, offline preference pairs, results in half a day)
No time for DPO either → SFT-LoRA only (still fully covers A5's "fine-tune an LLM" requirement)
```
Sampled answers are scored with **the same `rewards.py`**, and the highest/lowest are taken as chosen/rejected —
so switching to DPO **wastes none of the work already done**.

#### (f) The interface handed to C

Once training finishes, just fill the paths into `models/llm/manifest.json`; C runs the evaluation:
```json
{"base": "Qwen/Qwen2.5-1.5B-Instruct", "sft": "models/llm/sft-lora", "grpo": "models/llm/grpo"}
```

---

### 6.3 Pillar 3: diffusion image augmentation — M4–M7 / A3–A4

**Owner** C · **File** `src/skincare/augment/diffusion_aug.py`

**The goal is not "generate good-looking skin images" but to prove via an ablation study that generative augmentation improves classification performance.**
The strongest sentence in the report: *"after adding N diffusion-synthesized samples, minority-class macro-F1 rose from 0.61 to 0.68"*

```bash
python -m skincare.augment.diffusion_aug --concern acne --n 200
```

**Method**: `StableDiffusionImg2ImgPipeline` for img2img repainting — starting from real minority-class images,
which preserves the semantics of skin texture. **Far more stable than training a GAN from scratch, and it produces results within a day.**
Use clinical, descriptive wording in the prompt, and try `strength` between 0.5 and 0.7.

**Two red lines**
1. **Augment train only; never touch val/test** — otherwise the metrics become meaningless and the defense will call it out.
2. Synthetic faces are **for training only** and must never be passed off as real users in the demo; the report must state the proportion of synthetic data and the share removed by manual screening.

**Fallback**: if img2img works poorly → a "classical augmentation vs diffusion augmentation" comparison → at minimum, deliver a class-imbalance analysis plus a classical-augmentation ablation.
**Every rung of that ladder still covers A3/A4.**

**Deliverable** `data/processed/vision_train_aug.csv` → B retrains with **identical hyperparameters**, and the difference between the two runs is the ablation conclusion.

---

### 6.4 RAG retrieval — M1–M2

**Owner** A (builds the store) + B (tunes it) · **Files** `src/skincare/rag/`

```bash
python -m skincare.rag.ingest    # A: raw CSV → products.parquet / chunks.parquet
python -m skincare.rag.index     # B: vectorize → FAISS
```

**Query construction is where the added value is realized** (`retrieve.py::build_query`): it fuses
the user's request, the CNN skin labels, and the expanded ingredient rules into a single query vector.
A general-purpose chatbot cannot see the photo and cannot access the product table — this is the concrete evidence of "going beyond an out-of-the-box LLM".

Hard filters (budget and so on) are applied in the retrieval layer, not left to the LLM to reason about.
**B may swap the embedding model and tune chunking and top-k, but must not change the `evidence_id` format.**

**Acceptance** `python scripts/verify_handoff.py rag`

---

### 6.5 Safety and ethics — L1

**Owner** A · **File** `src/skincare/safety/guard.py`

Filtering of pregnancy/allergy/comedogenic ingredients, a mandatory medical disclaimer, and refusal of out-of-scope medical questions.
It reads the very same `ingredient_rules.json` that A maintains, so it goes most smoothly if one person owns both.

The ethics chapter of the report must cover: facial images are **discarded after use and never persisted**, the boundaries of the medical disclaimer,
**recommendation bias analysis** (brand/price/skin-tone distribution), honest disclosure of synthetic data, and dataset licensing and provenance.

---

### 6.6 API, front end and deployment — A1 / L4

**Owner** Anna · **Files** `app/`, `ui/`, `docker/`

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | — | status + whether mock mode is on |
| `POST /analyze-skin` | image | `SkinAnalysis` |
| `POST /recommend` | `RecommendRequest` | `AdvisorResponse` |

```bash
make api          # :8000/docs
make ui           # :8501
make docker-up    # full stack
```
**L4 is a self-contained scoring item**: deploy to AWS/GCP and report local vs cloud **latency p50/p95, throughput req/s, and per-request cost**.
Mock mode is enough to get the deployment pipeline and the load-test script working first.

⚠️ Grading happens by running the repo on the professor's machine — **pin the dependencies and verify `make docker-build` in a clean environment**.

---

### 6.7 Evaluation suite

**Owner** C · **Files** `src/skincare/eval/`

**Core design: the evaluation tooling is decoupled from the model being evaluated**, so C never has to wait for Anna.

```bash
python -m skincare.eval.run_eval --self-test        # day one: self-check with no models
python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl \
       --variants base sft grpo                     # at the end: the three-way comparison
```

`harness.py` reads `models/llm/manifest.json` and **automatically skips any tier that is missing**;
`fixtures/eval_samples.jsonl` holds **10 samples with known answers** (perfect / fabricated citation / mismatched ingredients /
pregnancy-unsafe / malformed output / partial citations, and so on), each annotated with its expected score range.
**Passing the self-check means the evaluator itself is correct** — already verified at 10/10.

`judge.py` covers the subjective dimensions the rule-based rewards cannot reach (helpfulness / clarity / specificity / faithfulness):
temperature=0, randomized scoring order to avoid position bias, and a parse failure returns None rather than a score of 0.

---

## 7. Course coverage matrix

### 7.1 Modules 1–12

| Module | Topic | Where it lives in the project | Who |
|---|---|---|---|
| M1–M2 | GenAI foundations, embeddings, FastAPI/Docker | RAG sentence embeddings + the API foundation | A/B/Anna |
| M3 | FCNN / CNN | **Pillar 1** skin analysis | B |
| M4–M5 | VAE / GAN | **Pillar 3** control condition and discussion of generative augmentation | C |
| M6 | Normalizing Flows | ⚠️ not covered by this project (just explain the trade-off in the report) | — |
| M7 | Energy-based and diffusion models | **Pillar 3** diffusion img2img augmentation + ablation | C |
| M8 | Transformers / fine-tuning GPT | **Pillar 2** LoRA SFT | Anna |
| M9–M10 | RL foundations and policy optimization | **Pillar 2** GRPO | Anna |
| M11 | LLMs and reasoning (DeepSeek-R1) | **verifiable reward design** | Anna |
| M12 | Multimodality / deploying SD | image-plus-text multimodal input + SD for Pillar 3 | C/Anna |

### 7.2 Learning objectives L1–L5

| | Objective | Where it lives |
|---|---|---|
| L1 | Limitations and ethics | §6.5 safety module + the report's ethics chapter + synthetic-data disclosure |
| L2 | Domain fine-tuning of a pretrained model | CNN transfer-learning fine-tuning + LLM LoRA SFT |
| L3 | Implementing generative models in PyTorch | hand-built SimpleCNN + diffusion augmentation + Transformer post-training |
| L4 | Cloud deployment and benchmarking | FastAPI/Docker → AWS/GCP + benchmark tables |
| L5 | Team collaboration and delivery | four-person division of labor + report + demo + GitHub |

### 7.3 Assignments 1–5

| Assignment | Reused / extended as |
|---|---|
| A1 FastAPI/Docker/uv | the project's API and deployment foundation |
| A2 CNN | Pillar 1 (upgraded to multi-label + a transfer-learning comparison) |
| A3 VAE/GAN | Pillar 3's control experiment |
| A4 Diffusion | Pillar 3's main line |
| A5 Transformer + RL | Pillar 2 SFT + GRPO |

---

## 8. Data flow and handoff points

```
Member A ──vision_*.csv──────────────► Member B (trains the CNN)
         ──vision_*.csv──────────────► Member C (picks minority classes to augment)
         ──products/chunks.parquet───► Member B (builds the index) ──► Anna (RAG wired up)
         ──ingredient_rules.json─────► Anna (reward functions) + A's own safety module

Member C ──vision_train_aug.csv──────► Member B (retrain → ablation conclusion)
Anna     ──models/llm/manifest.json──► Member C (three-way evaluation)
Member B ──models/vision/*.pt────────► Anna (final integration / demo)
```

| # | Handoff | Deliverable | When | Workaround if it stalls |
|---|---|---|---|---|
| 1 | A → B, C | `vision_*.csv` + `class_distribution.md` | Day 2–3 | B/C get the pipeline working on a small sample subset first |
| 2 | A → B, Anna | `products/chunks.parquet` | before the MVP | Anna keeps working on the SFT format using `fixtures/` |
| 3 | C → B | `vision_train_aug.csv` | before the ablation | B delivers the baseline result first and adds the augmented run later |
| 4 | Anna → C | fill in `manifest.json` | Day 5 | **C works throughout with fixtures + the base tier and is never blocked** |
| 5 | B → Anna | `*.pt` + `_report.json` | before the demo | Anna gets by with `USE_MOCKS=1` |

**Every handoff has an acceptance command; if it does not pass, it does not count as delivered:**
```bash
python scripts/validate_data.py {vision|products|chunks|rules|all}   # A's deliverables
python scripts/verify_handoff.py vision models/vision/<run>.pt       # B's CNN
python scripts/verify_handoff.py rag                                 # B's index
python -m skincare.eval.run_eval --self-test                         # C's evaluator
make test                                                            # everyone, mandatory before merging
```

---

## 9. Evaluation plan and the report's headline results

| Module | Metrics | Comparison | Who |
|---|---|---|---|
| CNN | accuracy, macro-F1, confusion matrix | **hand-built CNN vs transfer learning** | B/C |
| Diffusion augmentation | minority-class macro-F1 | **before vs after augmentation (ablation)** | C |
| RAG | Precision@3, hallucination rate | embedding model A vs B | C |
| LLM post-training | the five reward components + the four LLM-judge dimensions | **base vs SFT vs GRPO** | C |
| Deployment | latency p50/p95, throughput, cost | local vs cloud | Anna |

**The report's three headline figures** (none of them optional):
1. **The three-stage improvement curve** of each reward component across base → SFT → GRPO (proves post-training works)
2. **The ablation bar chart** of minority-class F1 with and without diffusion-synthesized data (proves generative modelling has value)
3. **The model comparison table** of hand-built CNN vs transfer learning (proves both low-level understanding and the benefit of fine-tuning)

Because the rewards are rule-based, figure 1 is **fully reproducible and requires no human labelling** — which makes it the most convincing part.

---

## 10. Risks and fallbacks

| Risk | Trigger signal | Fallback |
|---|---|---|
| GRPO does not converge | reward curve oscillates or collapses | → DPO → SFT-LoRA only (still covers A5) |
| Out of VRAM | OOM | QLoRA 4-bit / switch to a smaller base / reduce group-size |
| Diffusion augmentation does not help | F1 does not improve, or drops | → classical-augmentation comparison → deliver only the class-imbalance analysis |
| Data is late | A does not deliver on time | everyone continues with `USE_MOCKS=1`; Anna advances the SFT format using fixtures |
| Docker will not run on the professor's machine | the build fails in a clean environment | **verify in a clean environment a day in advance**, with dependencies pinned |
| The label space gets changed | `CheckpointMismatch` | unify the label space immediately and retrain; do not work around it with `strict=False` |

**Top priority**: the **MVP (CNN + RAG + API wired together)** must exist within the first 3 days — that is the floor on our grade.
SFT/GRPO/diffusion are additive bonuses stacked on top, not the make-or-break part.

---

## 11. Environment and command cheat sheet

```bash
# One-time
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ui]"          # basics (enough for mock mode)
uv pip install -e ".[vision,rag,llm]"  # install these only when you need to train
cp .env.example .env                   # fill in OPENAI_API_KEY / HF_TOKEN

# Day to day
make api / make ui / make test / make docker-up

# Training (use a Colab / Kaggle GPU)
python -m skincare.vision.train --config configs/vision_transfer.yaml
python -m skincare.rag.ingest && python -m skincare.rag.index
python -m skincare.llm.data_build --n 800 --with-targets
python -m skincare.llm.sft_lora --epochs 2
python -m skincare.llm.grpo_train --steps 300
python -m skincare.augment.diffusion_aug --concern acne --n 200
python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl
```

**Environment notes**: the CNN, diffusion, and SFT/GRPO all need a GPU, so run them on Colab/Kaggle;
your own machine only runs the API and the front end. `USE_MOCKS=1` is the default — set it to 0 once real models exist.

---

## 12. Deliverables

- **A runnable system** (Docker, evaluable locally or in the cloud): photo + text → skin analysis → explainable recommendations
- **The final report**: including §9's three headline figures, methodology, ethics and limitations
- **A live demo**
- **A public GitHub repository**: complete code + reproduction instructions + data provenance statement

Report section ownership is listed in `TEAM_TASKS.md` §6 (a hard deliverable, not a "let's all write it together" arrangement).
