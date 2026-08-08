"""STAGE 1 of post-training: LoRA supervised fine-tuning (Module 8).

    python -m skincare.llm.sft_lora --epochs 2

Pipeline reminder for the report:
    off-the-shelf pretrained base  ->  LoRA SFT (here)  ->  RL post-training (GRPO)
We never pre-train from scratch; we adapt. LoRA = parameter-efficient fine-tuning.
"""
import argparse

from skincare.config import MODELS, PROCESSED


def _precision():
    """按显卡能力选精度 —— 免费 Colab 的 T4 是 Turing 架构,不支持 bf16。
    返回 (bf16, fp16):
      A100/L4/H100 -> bf16     T4/P100 -> fp16     CPU/MPS -> 都关(fp32)
    不做这个判断的话:T4 上开 bf16 直接报错,全关又会退回 fp32 把显存撑爆。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, False
        return (True, False) if torch.cuda.is_bf16_supported() else (False, True)
    except Exception:
        return False, False


BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # 小到免费 Colab T4 也能跑
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
        max_length=args.max_len,          # TRL>=0.20 改名(旧版叫 max_seq_length)
        logging_steps=10,
        save_strategy="epoch",
        bf16=_BF16,
        fp16=_FP16,                       # T4 用 fp16;不设的话会退回 fp32 撑爆显存
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = SFTTrainer(model=args.base, train_dataset=ds,
                         peft_config=peft_config, args=cfg)
    trainer.train()
    trainer.save_model(args.out)
    print("saved LoRA adapter ->", args.out)


if __name__ == "__main__":
    main()
