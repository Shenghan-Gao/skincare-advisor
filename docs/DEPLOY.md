# Deployment and Benchmarking (L4)

> Learning objective L4: **deploy the trained model on AWS/Azure/GCP and benchmark it against a local run.**
> This is a self-contained scoring item, and it **depends on nobody** — the whole flow already runs end to end in mock mode.
>
> Owner: Anna

---

## 0. Get it running locally first (10 minutes)

```bash
make docker-build
make docker-up
```

Open http://localhost:8000/docs and http://localhost:8501.

**This step is a hard grading requirement** — the professor will clone the repo and run it on his own machine.
So always verify in a **clean environment** (ideally a second machine, or start over after `docker system prune -af`).

> Known pitfalls (already fixed, recorded for reference):
> - `pip install -e .` must run only after `src/` exists, otherwise it fails with
>   `Getting requirements to build editable did not run successfully`
> - `docker-compose.yml` must not hard-depend on `.env` (it is gitignored, so a fresh clone has none);
>   every variable now carries a default value

---

## 1. Local benchmark

```bash
make api
make bench
```

Or pass the parameters by hand:

```bash
python scripts/benchmark.py --url http://localhost:8000 --label local \
       --n 50 --concurrency 4
```

This produces `reports/bench_local.json`, containing: cold start, latency p50/p95/p99, throughput, failure count.

---

## 2. Cloud deployment

Any of the three clouds works; the example below uses **AWS EC2** (the GCP Compute Engine steps are equivalent).

### 2.1 Launch an instance

- **API/retrieval only** (lightweight): a CPU instance such as `t3.medium` is enough
- **Running local LLM inference**: you need a GPU instance (e.g. `g4dn.xlarge`, T4 card)
- Open ports **8000** (API) and **8501** (front end) in the security group
- Choose Ubuntu 22.04+ as the OS

### 2.2 Install Docker and bring the services up

```bash
ssh -i <your-key>.pem ubuntu@<public-ip>

sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/Shenghan-Gao/skincare-advisor.git
cd skincare-advisor

docker compose --project-directory . -f docker/docker-compose.yml up -d --build
curl localhost:8000/health
```

### 2.3 Ship the trained models up

Model weights are not in git (too large); download them from Google Drive and drop them into `models/`:

```
models/
  llm/sft-lora/      # produced by Anna's training
  llm/grpo/
  vision/best.pt     # produced by teammate B's training
```

Then set `USE_MOCKS` to `0` in `.env` and restart the containers.

---

## 3. Cloud benchmark + comparison

Run the benchmark **from your own machine** against the cloud address (this measures true end-to-end latency from a user's point of view):

```bash
python scripts/benchmark.py --url http://<public-ip>:8000 --label cloud \
       --n 50 --concurrency 4 --hourly-cost 0.526
```

Set `--hourly-cost` to the instance's hourly price; the script converts it into a **cost per thousand requests** —
that is the number the report wants, not the raw wall-clock duration.

Generate the comparison table:

```bash
python scripts/benchmark.py --compare reports/bench_local.json reports/bench_cloud.json
```

This produces `reports/bench_comparison.md`, which can be pasted straight into the report.

---

## 4. The two tables the report needs

### 4.1 Inference service benchmark (generated automatically by the script)

| Metric | local | cloud |
|---|---|---|
| Cold start (ms) | | |
| Latency p50 / p95 / p99 (ms) | | |
| Throughput (req/s) | | |
| Cost per 1k requests (USD) | | |

### 4.2 Training time comparison (recorded by hand)

The L4 wording mentions "benchmarking against local training", so besides inference we also have to record training:

| Stage | Local (Mac CPU/MPS) | Colab T4 | Speedup |
|---|---|---|---|
| LoRA SFT (2 epochs) | | | |
| GRPO (300 steps) | | | |

**Where the numbers come from**: the training scripts print `train_runtime` — copy it directly.
The local column does not have to be run to completion — run 10 steps, measure the per-step time and extrapolate linearly,
then just note in the report that the value is extrapolated (spending several hours to measure it for real is not worth it).

---

## 5. Discussion points (extra credit in the report)

Pasting numbers is not enough; the rubric looks for **trade-off analysis**. Suggested discussion:

- **Cold start vs steady-state latency**: the first request has to load the retrieval index/model, so it is noticeably slower.
  What do you do about this in production (warm-up requests, a long-lived process)?
- **Cost-effectiveness of CPU vs GPU**: retrieval and the API are already fast enough on CPU; only LLM inference needs a GPU.
  Is it worth paying for a GPU instance? Could this be split into two services?
- **Why local training is not viable**: VRAM, time, and precision support (Mac's MPS does not support bf16).
- **Cost estimation**: from the cost per thousand requests, extrapolate the monthly cost of serving 1000 users.
