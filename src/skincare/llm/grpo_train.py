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
    ap.add_argument("--group-size", type=int, default=8,
                    help="how many candidate completions to sample per prompt")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=512)
    args = ap.parse_args()

    from pathlib import Path as _P

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset("json", data_files=args.data, split="train")

    # ---- Resume from the SFT adapter rather than starting from the raw base model ----
    # Running RL directly on the raw base converges poorly: the model cannot yet emit the
    # required JSON structure, so the format reward stays at 0 for a long time and every
    # completion in the group scores the same -> no gradient signal.
    if args.adapter and _P(args.adapter).exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        dtype = torch.bfloat16 if _BF16 else (torch.float16 if _FP16 else torch.float32)
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
        peft_config = None                      # adapter already loaded; do not stack another
        print(f"resuming from SFT adapter: {args.adapter}")
    else:
        model = args.base
        peft_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")
        print(f"no adapter found at {args.adapter}; starting from the raw base model "
              f"-- RL will be much harder to converge")

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
        fp16=_FP16,                       # fp16 on a T4; without it we fall back to fp32 and OOM
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
