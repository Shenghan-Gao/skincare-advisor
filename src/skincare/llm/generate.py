"""Serving-time advisor. Three backends behind one interface so the demo can
switch generators live -- this is what produces the base/SFT/RL comparison."""
import json
import os

from pydantic import ValidationError

from app.schemas import AdvisorResponse, Recommendation, RetrievalResult, SkinAnalysis, UserProfile
from skincare.llm.prompts import build_messages
from skincare.llm.rewards import _parse

# What an empty answer says when the generator, not the safety guard, emptied it.
# The guard always leaves a flag behind when it removes something; a generation
# failure left nothing at all, so the interface fell back to blaming the budget or
# the avoid-list -- a cause it cannot see and, here, guessed wrong.
GENERATION_FAILURE_NOTE = (
    "No recommendation could be read from the advisor model's reply, so this answer is "
    "empty for a generation reason -- not because your budget or your avoid-list ruled "
    "every product out."
)


# Set LLM_SERVE_AS_TRAINED=0 to send a post-trained adapter the system turn anyway,
# which is what the published evaluation's "format B" did.
_SERVE_AS_TRAINED = os.getenv("LLM_SERVE_AS_TRAINED", "1") == "1"


class Advisor:
    def __init__(self, backend: str | None = None, adapter: str | None = None):
        self.backend = backend or os.getenv("LLM_BACKEND", "openai")  # openai|local
        self.adapter = adapter or os.getenv("LLM_ADAPTER")
        self._pipe = None

    # 512 was chosen to match the evaluation harness. It is too small for the served
    # request, which carries the skin analysis and the preferences the harness never
    # sends: the object gets cut mid-way, and a truncated outer object leaves the first
    # *complete* thing in the text as one recommendation, so the reply parses and
    # arrives with no recommendations at all. Serving correctness wins over matching a
    # table that is already labelled as produced under the old configuration.
    MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))

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
        out = self._pipe(messages, max_new_tokens=self.MAX_NEW_TOKENS, do_sample=False)
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
        # Serve the adapter the way it was trained. sft_lora.py trains on
        # build_user_prompt(...) + "\n" + completion: one user string, no system turn
        # and no chat template. build_messages adds a system turn, which is right for
        # an API model and out of distribution for ours -- the held-out run measured it
        # at 36% and 28% parseable against 87% under the training format. Serving it the
        # wrong shape is the same drift Section 7 is about, arriving from the other side.
        # The base model keeps the system turn, since the schema is all it has to go on.
        if self.backend == "local" and self.adapter and _SERVE_AS_TRAINED:
            messages = [m for m in messages if m.get("role") != "system"]
        raw = self._generate(messages)
        obj = _parse(raw)
        parsed = obj is not None
        if not parsed:
            print(f"[advisor] the reply was not valid JSON, so no recommendation could be "
                  f"read from it ({len(raw or '')} characters of output)", flush=True)
        obj = obj or {}
        # Model output is untrusted input, and the old construction treated it as a
        # keyword-argument source: a recommendation missing `reason`, or carrying a
        # key that is not a field, raised ValidationError and turned one bad item into
        # a 500 for the whole request. That is the wrong failure for a demo -- two good
        # recommendations should still reach the user. Build each field explicitly,
        # coerce the types the schema expects, and drop what will not validate.
        # The prompt shows the model an evidence_id, a product_id and a chunk of text.
        # It never shows a product name, a brand or a price, so the model cannot know
        # them and writes something plausible instead -- three different serums all
        # called "Hydrating Serum", or "Product Name Unknown". Those are facts the
        # catalogue owns, so take them from retrieval and keep the model's prose.
        # prompts.py is a frozen interface and adding names to it would put training
        # and serving out of step again, which is the whole lesson of Section 7.
        catalogue = {p.product_id: p for p in retrieval.products}

        recs = []
        for r in obj.get("recommendations") or []:
            if not isinstance(r, dict) or not r.get("product_id"):
                continue
            pid = str(r["product_id"])
            known = catalogue.get(pid)
            if known is None:
                # Outside the top-k summary. The id may still be valid (evidence spans
                # more products than products[] lists), so keep the row and say so.
                print(f"[advisor] {pid} is not in the retrieved product list; "
                      f"falling back to the model's own name for it", flush=True)
            try:
                recs.append(Recommendation(
                    product_id=pid,
                    name=str(known.name if known else (r.get("name") or "")),
                    brand=str(known.brand if known else (r.get("brand") or "")),
                    price_usd=known.price_usd if known else r.get("price_usd"),
                    reason=str(r.get("reason") or ""),
                    key_ingredients=[str(i) for i in (r.get("key_ingredients") or [])],
                    cited_evidence=[str(c) for c in (r.get("cited_evidence") or [])],
                    matched_concerns=[str(c) for c in (r.get("matched_concerns") or [])],
                ))
            except ValidationError as exc:
                print(f"[advisor] dropped a malformed recommendation: {exc}", flush=True)

        # An empty list must never leave the generator looking like a considered
        # answer. Everything else that empties it -- the pregnancy block, the user's
        # avoid-list, the medical-boundary refusal -- writes a flag or a note saying
        # so, and the UI reads that to explain itself. Without one line here, a reply
        # that did not parse is indistinguishable from "the catalogue had nothing for
        # you", and the demo tells the user their budget is at fault.
        note = str(obj.get("routine_note") or "")
        if not recs:
            # "parsed, but the object has no recommendations key at all" is the
            # signature of a cut-off reply: the outer object never closed, so the first
            # complete object in the text was one recommendation, and that is what came
            # back. Name it, because the repair is a bigger token budget and nothing else.
            truncated = parsed and "recommendations" not in obj
            print(f"[advisor] generation produced no usable recommendation "
                  f"(json parsed={parsed}, items={len(obj.get('recommendations') or [])}"
                  f"{', reply looks truncated -- raise LLM_MAX_NEW_TOKENS' if truncated else ''})",
                  flush=True)
            note = note or GENERATION_FAILURE_NOTE
        return AdvisorResponse(
            analysis=analysis, recommendations=recs,
            routine_note=note,
            generator=f"{self.backend}:{self.adapter or 'base'}",
        )
