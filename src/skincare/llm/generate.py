"""Serving-time advisor. Three backends behind one interface so the demo can
switch generators live -- this is what produces the base/SFT/RL comparison."""
import json
import os

from app.schemas import AdvisorResponse, Recommendation, RetrievalResult, SkinAnalysis, UserProfile
from skincare.llm.prompts import build_messages
from skincare.llm.rewards import _parse


class Advisor:
    def __init__(self, backend: str | None = None, adapter: str | None = None):
        self.backend = backend or os.getenv("LLM_BACKEND", "openai")  # openai|local
        self.adapter = adapter or os.getenv("LLM_ADAPTER")
        self._pipe = None

    def _generate(self, messages) -> str:
        if self.backend == "openai":
            from openai import OpenAI
            client = OpenAI()
            r = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages, temperature=0.3)
            return r.choices[0].message.content
        if self._pipe is None:
            self._pipe = self._load_local()
        out = self._pipe(messages, max_new_tokens=512, do_sample=False)
        return out[0]["generated_text"][-1]["content"]

    def _load_local(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        base = os.getenv("LLM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        tok = AutoTokenizer.from_pretrained(base)
        # Explicit device and dtype. bfloat16 + device_map="auto" is right on the
        # training GPU and fails on Apple Silicon: MPS has no bf16, and "auto" needs
        # accelerate, which lives in the llm extra that cannot install on macOS.
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        print(f"[advisor] loading {base}"
              f"{' + ' + self.adapter if self.adapter else ''} on {device} ({dtype})",
              flush=True)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
        model = model.to(device).eval()
        return pipeline("text-generation", model=model, tokenizer=tok, device=device)

    def recommend(self, profile: UserProfile, analysis: SkinAnalysis | None,
                  retrieval: RetrievalResult) -> AdvisorResponse:
        messages = build_messages(
            profile.model_dump(),
            analysis.model_dump(mode="json") if analysis else None,
            [e.model_dump() for e in retrieval.evidence][:12],
        )
        raw = self._generate(messages)
        obj = _parse(raw) or {}
        recs = [Recommendation(**{**r, "brand": r.get("brand", ""),
                                  "price_usd": r.get("price_usd")})
                for r in obj.get("recommendations", []) if r.get("product_id")]
        return AdvisorResponse(
            analysis=analysis, recommendations=recs,
            routine_note=obj.get("routine_note", ""),
            generator=f"{self.backend}:{self.adapter or 'base'}",
        )
