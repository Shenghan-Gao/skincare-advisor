"""End-to-end training pipeline, executed on a remote Colab runtime.

Run it from your laptop with scripts/colab_run.sh, which uploads this file and
calls `colab exec -f`. Nothing here needs a browser tab.

Every stage is resumable: results are mirrored to Google Drive as soon as they
exist, and a re-run restores them and skips the finished work. A recycled runtime
therefore costs time, never repeated OpenAI spend -- the distillation cache is
restored before any API call is made.

Stages
    1 restore   pull previous artefacts back from Drive
    2 index     build the FAISS index over the real 2,282-product catalogue
    3 distil    teacher distillation -> sft.jsonl / rl.jsonl / rl_test.jsonl
    4 sft       LoRA supervised fine-tuning
    5 grpo      RL post-training with verifiable rewards
    6 manifest  publish adapter paths for the evaluation owner
    7 eval      base / sft / grpo on the held-out split -> reports/llm_eval.md

Re-running a stage whose artefact already exists
------------------------------------------------
Every stage short-circuits when its output is present, which is what makes a
dropped runtime cheap. To deliberately redo one -- a changed hyper-parameter, a
fixed bug -- name it in FORCE:

    FORCE=sft,grpo,eval python scripts/run_pipeline.py

A forced stage deletes its artefact on both the runtime and Drive *before*
running, and is excluded from the restore that precedes it. Without that purge
the trainers' own --resume would silently continue from a checkpoint trained
under the old configuration, and the run would look successful while mixing two
different settings.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/Shenghan-Gao/skincare-advisor.git"
WORK = Path("/content/skincare")
DRIVE = Path("/content/drive")
MYDRIVE = DRIVE / "MyDrive"
DATA_DST = MYDRIVE / "skincare_data"
MODEL_DST = MYDRIVE / "skincare_models"
REPORT_DST = MYDRIVE / "skincare_reports"
SRC = WORK / "data" / "processed"
MODEL_SRC = WORK / "models" / "llm"
REPORT_SRC = WORK / "reports"

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "800"))

# SFT. max-len 3200 covers the longest distilled sample (3080 tokens). At 2048 the
# median sample (1973) sat right at the limit and TRL truncated from the right,
# which removed the trailing "disclaimer" field from 275 of 576 targets -- the
# model then never learned to emit it and held-out safety_reward stalled at 0.487.
SFT_EPOCHS = float(os.environ.get("SFT_EPOCHS", "2"))
SFT_MAX_LEN = int(os.environ.get("SFT_MAX_LEN", "3200"))

# GRPO. Measured on an L4: ~55 s per optimizer step at these settings, so 60 steps
# is about an hour. lr 2e-5 rather than the 1e-6 the script defaults to -- at 1e-6
# the KL stayed at 6e-4 over 20 steps, i.e. the policy never left the reference and
# the reward curve was flat for reasons unrelated to the reward design.
GRPO_STEPS = int(os.environ.get("GRPO_STEPS", "60"))
GRPO_GROUP = int(os.environ.get("GRPO_GROUP", "6"))
GRPO_ACCUM = int(os.environ.get("GRPO_ACCUM", "2"))
GRPO_LR = os.environ.get("GRPO_LR", "2e-5")
GRPO_MAXCOMP = int(os.environ.get("GRPO_MAXCOMP", "384"))

EVAL_VARIANTS = os.environ.get("EVAL_VARIANTS", "base sft grpo")
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "119"))

FORCE = {s.strip() for s in os.environ.get("FORCE", "").split(",") if s.strip()}

_t0 = time.time()


def log(msg: str):
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


def sh(cmd: str, check: bool = True) -> int:
    log(f"$ {cmd}")
    rc = subprocess.run(cmd, shell=True, cwd=str(WORK) if WORK.exists() else None).returncode
    if check and rc != 0:
        raise SystemExit(f"command failed ({rc}): {cmd}")
    return rc


def status(stage: str, state: str, **extra):
    """Machine-readable progress, mirrored to Drive so the laptop can poll it."""
    line = {"stage": stage, "state": state, "elapsed_s": round(time.time() - _t0, 1), **extra}
    print("STATUS " + json.dumps(line), flush=True)
    try:
        DATA_DST.mkdir(parents=True, exist_ok=True)
        with open(DATA_DST / "pipeline_status.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:
        pass


def purge(*pairs):
    """Delete an artefact from the runtime and from Drive, before a forced re-run."""
    for local, remote in pairs:
        for p in (local, remote):
            if p.exists():
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
                log(f"purged {p}")


def backup():
    n = 0
    for pat in ("*.jsonl", "*.json"):
        for f in SRC.glob(pat):
            shutil.copy(f, DATA_DST / f.name); n += 1
    if (SRC / "index").exists():
        shutil.copytree(SRC / "index", DATA_DST / "index", dirs_exist_ok=True); n += 1
    for name in ("sft-lora", "grpo"):
        if (MODEL_SRC / name).exists():
            shutil.copytree(MODEL_SRC / name, MODEL_DST / name, dirs_exist_ok=True); n += 1
    if REPORT_SRC.exists():
        REPORT_DST.mkdir(parents=True, exist_ok=True)
        for f in REPORT_SRC.glob("llm_eval.*"):
            shutil.copy(f, REPORT_DST / f.name); n += 1
        # The evaluator caches one scored row at a time; carrying it across runs is
        # what makes a dropped runtime cost minutes rather than the whole variant.
        if (REPORT_SRC / ".eval_cache").exists():
            shutil.copytree(REPORT_SRC / ".eval_cache", REPORT_DST / "eval_cache",
                            dirs_exist_ok=True); n += 1
    log(f"backed up {n} artefact(s) to Drive")


def restore():
    SRC.mkdir(parents=True, exist_ok=True)
    n = 0
    for pat in ("*.jsonl", "*.json", "*.parquet"):
        for f in DATA_DST.glob(pat):
            shutil.copy(f, SRC / f.name); n += 1
    if (DATA_DST / "index").exists():
        shutil.copytree(DATA_DST / "index", SRC / "index", dirs_exist_ok=True); n += 1
    for name in ("sft-lora", "grpo"):
        # A forced stage must start from nothing, or the trainer's own --resume
        # picks up a checkpoint built under the previous configuration.
        if name.replace("sft-lora", "sft") in FORCE:
            log(f"skipping restore of {name}: stage is forced")
            continue
        if (MODEL_DST / name).exists():
            MODEL_SRC.mkdir(parents=True, exist_ok=True)
            shutil.copytree(MODEL_DST / name, MODEL_SRC / name, dirs_exist_ok=True); n += 1
    if REPORT_DST.exists() and "eval" not in FORCE:
        REPORT_SRC.mkdir(parents=True, exist_ok=True)
        for f in REPORT_DST.glob("llm_eval.*"):
            shutil.copy(f, REPORT_SRC / f.name); n += 1
        if (REPORT_DST / "eval_cache").exists():
            shutil.copytree(REPORT_DST / "eval_cache", REPORT_SRC / ".eval_cache",
                            dirs_exist_ok=True); n += 1
    log(f"restored {n} artefact(s) from Drive")


# ------------------------------------------------------------------ stages ---
def stage_setup():
    status("setup", "start", force=sorted(FORCE) or None)
    # Drive is mounted at session level by `colab drivemount` before this script is
    # launched. Calling drive.mount() again from a detached background process hangs,
    # because it expects an interactive kernel to complete the auth handshake.
    if not (DRIVE / "MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")
    for d in (DATA_DST, MODEL_DST, REPORT_DST):
        d.mkdir(parents=True, exist_ok=True)
    log("Drive ready")

    # A previous partial run may have left /content/skincare half-populated, and
    # `git clone` into a non-empty directory fails with exit 128. Handle all three
    # states explicitly rather than assuming a clean VM.
    if (WORK / ".git").exists():
        log("repo already cloned; syncing to origin/main")
        sh("git fetch -q --all", check=False)
        sh("git reset -q --hard origin/main", check=False)
    else:
        if WORK.exists():
            log(f"removing stale {WORK} before cloning")
            shutil.rmtree(WORK, ignore_errors=True)
        rc = subprocess.run(f"git clone -q {REPO} {WORK}", shell=True).returncode
        if rc != 0:
            raise SystemExit(
                f"git clone failed (exit {rc}). Check the runtime has network access "
                f"and that {REPO} is reachable anonymously.")
    if not (WORK / "pyproject.toml").exists():
        raise SystemExit(f"clone produced no pyproject.toml at {WORK}")
    os.chdir(WORK)
    sys.path[:0] = [str(WORK), str(WORK / "src")]

    sh('pip install -q -e ".[dev,rag,llm]"', check=True)
    # Colab ships torchao 0.10; modern PEFT raises on anything below 0.16 during
    # its LoRA dispatch probe. We do no quantisation, so removing it is cleanest.
    sh("pip uninstall -y -q torchao", check=False)

    restore()

    import torch
    gpu = torch.cuda.is_available()
    status("setup", "done", gpu=gpu,
           device=torch.cuda.get_device_name(0) if gpu else "cpu")
    if not gpu:
        raise SystemExit("no GPU on this runtime -- recreate it with --gpu l4")
    for f in ("products.parquet", "chunks.parquet"):
        if not (SRC / f).exists():
            raise SystemExit(f"missing {f}; upload it to Drive/skincare_data first")


def stage_index():
    if (SRC / "index" / "chunks.faiss").exists():
        status("index", "skipped", reason="already present")
        return
    status("index", "start")
    sh("python -m skincare.rag.index")
    sh("python scripts/verify_handoff.py rag", check=False)
    backup()
    status("index", "done")


def stage_distil():
    sft = SRC / "sft.jsonl"
    if sft.exists() and sum(1 for _ in open(sft)) > N_SAMPLES * 0.5:
        status("distil", "skipped", rows=sum(1 for _ in open(sft)))
        return
    status("distil", "start", n=N_SAMPLES)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set on the runtime")
    # No --mock-retrieval: this uses the real FAISS index built above.
    sh(f"python -m skincare.llm.data_build --n {N_SAMPLES} --mode both --inspect 2")
    backup()
    summary = SRC / "distill_summary.json"
    extra = json.loads(summary.read_text()) if summary.exists() else {}
    status("distil", "done", pass_rate=extra.get("pass_rate"), usage=extra.get("usage"))


def stage_sft():
    out = MODEL_SRC / "sft-lora"
    if "sft" in FORCE:
        purge((out, MODEL_DST / "sft-lora"))
    elif list(out.glob("adapter*")):
        status("sft", "skipped", reason="adapter already present")
        return
    status("sft", "start", epochs=SFT_EPOCHS, max_len=SFT_MAX_LEN)
    sh(f"python -m skincare.llm.sft_lora --base {BASE} --epochs {SFT_EPOCHS} "
       f"--bs 1 --accum 8 --max-len {SFT_MAX_LEN} --save-steps 20 --resume --out {out}")
    if not list(out.glob("adapter*")):
        raise SystemExit("SFT produced no adapter")
    backup()
    status("sft", "done")


def stage_grpo():
    out, sft = MODEL_SRC / "grpo", MODEL_SRC / "sft-lora"
    if "grpo" in FORCE:
        purge((out, MODEL_DST / "grpo"))
    elif list(out.glob("adapter*")):
        status("grpo", "skipped", reason="adapter already present")
        return
    if not list(sft.glob("adapter*")):
        raise SystemExit("GRPO needs the SFT adapter; run the sft stage first")
    status("grpo", "start", steps=GRPO_STEPS, group=GRPO_GROUP, lr=GRPO_LR)
    sh(f"python -m skincare.llm.grpo_train --base {BASE} --adapter {sft} "
       f"--steps {GRPO_STEPS} --group-size {GRPO_GROUP} --accum {GRPO_ACCUM} "
       f"--lr {GRPO_LR} --max-completion-length {GRPO_MAXCOMP} "
       f"--save-steps 15 --resume --out {out}")
    if not list(out.glob("adapter*")):
        raise SystemExit("GRPO produced no adapter")
    backup()
    status("grpo", "done")


def stage_manifest():
    mf = MODEL_SRC / "manifest.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    m = json.loads(mf.read_text()) if mf.exists() else {}
    m.update({"base": BASE, "sft": str(MODEL_DST / "sft-lora"), "grpo": str(MODEL_DST / "grpo")})
    mf.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    shutil.copy(mf, MODEL_DST / "manifest.json")
    status("manifest", "done", manifest=m)


def stage_eval():
    out = REPORT_SRC / "llm_eval.json"
    wanted = EVAL_VARIANTS.split()
    if "eval" in FORCE:
        purge((out, REPORT_DST / "llm_eval.json"),
              (out.with_suffix(".md"), REPORT_DST / "llm_eval.md"),
              (REPORT_SRC / ".eval_cache", REPORT_DST / "eval_cache"))
    elif out.exists() and all(v in json.loads(out.read_text()) for v in wanted):
        status("eval", "skipped", reason="all variants already scored")
        return
    status("eval", "start", variants=wanted, limit=EVAL_LIMIT)
    sh(f"python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl "
       f"--variants {EVAL_VARIANTS} --limit {EVAL_LIMIT} --out reports/llm_eval.json")
    backup()
    if not out.exists():
        raise SystemExit("evaluation produced no results file")
    results = json.loads(out.read_text())
    missing = [v for v in wanted if v not in results]
    if missing:
        raise SystemExit(f"evaluation is missing {missing}; the cache keeps the rest")
    log("\n" + (out.with_suffix(".md").read_text() if out.with_suffix(".md").exists() else ""))
    status("eval", "done", totals={v: round(results[v]["total"], 4) for v in wanted})


def main():
    stages = [("setup", stage_setup), ("index", stage_index), ("distil", stage_distil),
              ("sft", stage_sft), ("grpo", stage_grpo), ("manifest", stage_manifest),
              ("eval", stage_eval)]
    only = set(sys.argv[1:])
    unknown = only - {n for n, _ in stages}
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}")
    for name, fn in stages:
        if only and name not in only and name != "setup":
            continue
        try:
            fn()
        except SystemExit as e:
            status(name, "failed", error=str(e))
            Path("/content/PIPELINE_FAILED").write_text(f"{name}: {e}")
            raise
        except Exception as e:
            status(name, "failed", error=f"{type(e).__name__}: {e}")
            Path("/content/PIPELINE_FAILED").write_text(f"{name}: {e}")
            raise
    Path("/content/PIPELINE_DONE").write_text("ok")
    log("PIPELINE COMPLETE")
    status("pipeline", "complete")


if __name__ == "__main__":
    main()
