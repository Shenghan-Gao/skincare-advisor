"""评估骨架 —— 组员 C 独占。

设计目标:**把"评估工具"和"被评估的模型"彻底解耦**,
这样 C 第一天就能把评估体系建完并验证,不用等 Anna 训完任何东西。

交接接口只有一个文件:models/llm/manifest.json
Anna 训完一档就往里填一个路径;C 的脚本读它,缺的 key 自动跳过。
"""
import json
import os
from pathlib import Path

from skincare.config import MODELS

MANIFEST = MODELS / "llm" / "manifest.json"


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {"base": "Qwen/Qwen2.5-1.5B-Instruct"}
    return {k: v for k, v in json.load(open(MANIFEST)).items()
            if not k.startswith("_") and v}


def available_variants() -> list[str]:
    """当前能评的档位。Anna 没训完时只有 base/gpt,C 照样能跑。"""
    return list(load_manifest().keys())


def get_generator(variant: str):
    """返回 f(prompt:str) -> completion:str。"""
    m = load_manifest()
    if variant not in m:
        raise KeyError(f"manifest 里没有 '{variant}';当前可用:{list(m)}")
    target = m[variant]

    if variant == "gpt" or str(target).startswith("gpt-"):
        from openai import OpenAI
        client = OpenAI()

        def gen(prompt: str) -> str:
            r = client.chat.completions.create(
                model=target, temperature=0.3,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
        return gen

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = m.get("base", "Qwen/Qwen2.5-1.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")
    if variant != "base":                      # sft / grpo 是 LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, target)
    model.eval()

    def gen(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt",
                                      add_generation_prompt=True).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=512, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return gen


def fixture_generator(path: str = "fixtures/eval_samples.jsonl"):
    """假的"模型" —— 直接吐出固定样本。
    让 C 在没有任何模型、没有 GPU、没有 API key 的情况下把评估链路跑通。"""
    rows = [json.loads(l) for l in open(path)]
    it = iter(rows)

    def gen(prompt: str) -> str:
        return next(it)["completion"]
    return gen
