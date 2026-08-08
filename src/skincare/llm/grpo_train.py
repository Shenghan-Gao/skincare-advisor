"""STAGE 2 of post-training: GRPO with verifiable rewards (Modules 9-11, A5).

    python -m skincare.llm.grpo_train --steps 300

Why GRPO: it is the algorithm from the DeepSeek-R1 paper we read in Module 11,
it needs no learned reward model (our rewards are rule-checkable), and it is far
lighter than PPO -- no value network to train.

The trainer samples a GROUP of completions per prompt, scores them with the
reward functions, and pushes probability mass toward the above-average ones.
"""
import argparse

from skincare.config import MODELS, PROCESSED
from skincare.llm import rewards as R


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


def _make_reward_fn(fn):
    """Adapt our single-completion scorers to TRL's batched signature."""
    def wrapped(completions, **kw):
        out = []
        for i, c in enumerate(completions):
            text = c[0]["content"] if isinstance(c, list) else c
            ctx = {k: (v[i] if isinstance(v, list) else v) for k, v in kw.items()}
            out.append(fn(text, **ctx))
        return out
    wrapped.__name__ = fn.__name__
    return wrapped


_BF16, _FP16 = _precision()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=str(MODELS / "llm" / "sft-lora"),
                    help="start from the SFT adapter -- RL on a raw base is much harder")
    ap.add_argument("--data", default=str(PROCESSED / "rl.jsonl"))
    ap.add_argument("--out", default=str(MODELS / "llm" / "grpo"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group-size", type=int, default=8, help="每个 prompt 采样几个候选")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=512)
    args = ap.parse_args()

    from pathlib import Path as _P

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset("json", data_files=args.data, split="train")

    # ---- 从 SFT adapter 继续训,而不是从裸基座开始 ----
    # 在裸基座上直接跑 RL 很难收敛:模型还不会输出规定的 JSON 结构,
    # 格式奖励长时间为 0,组内没有差异 -> 没有梯度信号。
    if args.adapter and _P(args.adapter).exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        dtype = torch.bfloat16 if _BF16 else (torch.float16 if _FP16 else torch.float32)
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
        peft_config = None                      # 已经有 adapter 了,不能再叠一层
        print(f"从 SFT adapter 继续: {args.adapter}")
    else:
        model = args.base
        peft_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")
        print(f"未找到 adapter({args.adapter}),从裸基座开始 —— RL 会更难收敛")

    cfg = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        num_generations=args.group_size,     # the "group" in GRPO
        per_device_train_batch_size=args.group_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=1e-6,                  # RL wants a much smaller LR than SFT
        max_completion_length=args.max_completion_length,
        beta=0.04,                           # KL penalty -> stops reward hacking
        temperature=0.9,
        logging_steps=5,
        save_steps=50,
        bf16=_BF16,
        fp16=_FP16,                       # T4 用 fp16;不设的话会退回 fp32 撑爆显存
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
        reward_funcs=[
            _make_reward_fn(R.format_reward),
            _make_reward_fn(R.ingredient_match_reward),
            _make_reward_fn(R.grounding_reward),
            _make_reward_fn(R.product_validity_reward),
            _make_reward_fn(R.safety_reward),
        ],
    )
    trainer.train()
    trainer.save_model(args.out)
    print("saved GRPO adapter ->", args.out)


if __name__ == "__main__":
    main()
