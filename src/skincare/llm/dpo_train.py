"""FALLBACK for stage 2: DPO (simpler, no sampling loop, no reward model).

Use this if GRPO will not converge in the time you have. Build preference pairs
offline by scoring N sampled answers with rewards.total_reward and keeping the
best as `chosen` and the worst as `rejected` -- so the SAME reward code is reused
and the report can still claim RL-style preference post-training.

    python -m skincare.llm.build_pairs   # TODO(Anna)
    python -m skincare.llm.dpo_train --epochs 1
"""
import argparse

from skincare.config import MODELS, PROCESSED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=str(PROCESSED / "pairs.jsonl"))
    ap.add_argument("--out", default=str(MODELS / "llm" / "dpo"))
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    ds = load_dataset("json", data_files=args.data, split="train")  # prompt/chosen/rejected
    cfg = DPOConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    learning_rate=5e-6, beta=0.1, bf16=True,
                    logging_steps=10, report_to="none")
    trainer = DPOTrainer(model=args.base, args=cfg, train_dataset=ds,
                         peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM"))
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
