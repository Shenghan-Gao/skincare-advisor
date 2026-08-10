"""STAGE 1 of post-training: LoRA supervised fine-tuning (Module 8).

    python -m skincare.llm.sft_lora --epochs 2

Pipeline reminder for the report:
    off-the-shelf pretrained base  ->  LoRA SFT (here)  ->  RL post-training (GRPO)
We never pre-train from scratch; we adapt. LoRA = parameter-efficient fine-tuning.
"""
import argparse
from pathlib import Path

from skincare.config import MODELS, PROCESSED


def _precision():
    """Pick the precision the GPU can handle -- free Colab's T4 is Turing and has no bf16.
    Returns (bf16, fp16):
      A100/L4/H100 -> bf16     T4/P100 -> fp16     CPU/MPS -> both off (fp32)
    Without this check: enabling bf16 on a T4 errors out, while leaving both off falls back
    to fp32 and blows up GPU memory.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, False
        return (True, False) if torch.cuda.is_bf16_supported() else (False, True)
    except Exception:
        return False, False


BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # small enough to train on a free Colab T4
_BF16, _FP16 = _precision()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--data", default=str(PROCESSED / "sft.jsonl"))
    ap.add_argument("--out", default=str(MODELS / "llm" / "sft-lora"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    # Colab recycles runtimes mid-run. Saving only at epoch boundaries means a
    # disconnect 80% through epoch 1 leaves nothing on disk at all.
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in --out")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                    help="disable gradient checkpointing: ~40%% faster, more VRAM. "
                         "Turn it back on if the run OOMs.")
    ap.set_defaults(grad_ckpt=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda r: {"text": r["prompt"] + "\n" + r["completion"]})

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        max_length=args.max_len,          # renamed in TRL>=0.20 (used to be max_seq_length)
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=_BF16,
        fp16=_FP16,                       # fp16 on a T4; without it we fall back to fp32 and OOM
        gradient_checkpointing=args.grad_ckpt,
        report_to="none",
    )
    trainer = SFTTrainer(model=args.base, train_dataset=ds,
                         peft_config=peft_config, args=cfg)
    resume = bool(args.resume and list(Path(args.out).glob("checkpoint-*")))
    if resume:
        print(f"resuming from the newest checkpoint under {args.out}")
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(args.out)
    print("saved LoRA adapter ->", args.out)


if __name__ == "__main__":
    main()
