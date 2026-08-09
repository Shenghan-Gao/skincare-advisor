"""Evaluation harness -- owned exclusively by member C.

Design goal: **fully decouple the evaluation tooling from the model being evaluated**,
so C can build and validate the whole evaluation setup on day one, without waiting for
Anna to finish training anything.

The handoff interface is a single file: models/llm/manifest.json
Anna adds one path per checkpoint she finishes; C's scripts read it and silently skip
any key that is not there yet.
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
    """Variants that can be evaluated right now. Before Anna finishes training there is
    only base/gpt, and C can still run everything."""
    return list(load_manifest().keys())


def get_generator(variant: str):
    """Return f(prompt: str) -> completion: str."""
    m = load_manifest()
    if variant not in m:
        raise KeyError(f"'{variant}' is not in the manifest; currently available: {list(m)}")
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
    if variant != "base":                      # sft / grpo are LoRA adapters
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
    """A fake "model" -- it simply replays fixed samples.
    Lets C exercise the whole evaluation pipeline with no model, no GPU and no API key."""
    rows = [json.loads(l) for l in open(path)]
    it = iter(rows)

    def gen(prompt: str) -> str:
        return next(it)["completion"]
    return gen
