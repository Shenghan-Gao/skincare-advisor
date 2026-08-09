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
DATA_DST, MODEL_DST = MYDRIVE / "skincare_data", MYDRIVE / "skincare_models"
SRC, MODEL_SRC = WORK / "data" / "processed", WORK / "models" / "llm"

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "800"))
SFT_EPOCHS = float(os.environ.get("SFT_EPOCHS", "2"))
GRPO_STEPS = int(os.environ.get("GRPO_STEPS", "300"))

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
        if (MODEL_DST / name).exists():
            MODEL_SRC.mkdir(parents=True, exist_ok=True)
            shutil.copytree(MODEL_DST / name, MODEL_SRC / name, dirs_exist_ok=True); n += 1
    log(f"restored {n} artefact(s) from Drive")


# ------------------------------------------------------------------ stages ---
def stage_setup():
    status("setup", "start")
    # Drive is mounted at session level by `colab drivemount` before this script is
    # launched. Calling drive.mount() again from a detached background process hangs,
    # because it expects an interactive kernel to complete the auth handshake.
    if not (DRIVE / "MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")
    for d in (DATA_DST, MODEL_DST):
        d.mkdir(parents=True, exist_ok=True)
    log("Drive ready")

    if not (WORK / "pyproject.toml").exists():
        sh(f"git clone -q {REPO} {WORK}", check=True)
    else:
        sh("git pull -q", check=False)
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
        raise SystemExit("no GPU on this runtime -- recreate it with --gpu T4")
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
    if list(out.glob("adapter*")):
        status("sft", "skipped", reason="adapter already present")
        return
    status("sft", "start", epochs=SFT_EPOCHS)
    sh(f"python -m skincare.llm.sft_lora --base {BASE} --epochs {SFT_EPOCHS} "
       f"--bs 1 --accum 8 --max-len 2048 --out {out}")
    if not list(out.glob("adapter*")):
        raise SystemExit("SFT produced no adapter")
    backup()
    status("sft", "done")


def stage_grpo():
    out, sft = MODEL_SRC / "grpo", MODEL_SRC / "sft-lora"
    if list(out.glob("adapter*")):
        status("grpo", "skipped", reason="adapter already present")
        return
    status("grpo", "start", steps=GRPO_STEPS)
    sh(f"python -m skincare.llm.grpo_train --base {BASE} --adapter {sft} "
       f"--steps {GRPO_STEPS} --group-size 8 --accum 4 "
       f"--max-completion-length 512 --out {out}")
    if not list(out.glob("adapter*")):
        raise SystemExit("GRPO produced no adapter")
    backup()
    status("grpo", "done")


def stage_manifest():
    mf = MODEL_SRC / "manifest.json"
    m = json.loads(mf.read_text()) if mf.exists() else {}
    m.update({"base": BASE, "sft": str(MODEL_DST / "sft-lora"), "grpo": str(MODEL_DST / "grpo")})
    mf.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    shutil.copy(mf, MODEL_DST / "manifest.json")
    status("manifest", "done", manifest=m)


def main():
    stages = [("setup", stage_setup), ("index", stage_index), ("distil", stage_distil),
              ("sft", stage_sft), ("grpo", stage_grpo), ("manifest", stage_manifest)]
    only = set(sys.argv[1:])
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
