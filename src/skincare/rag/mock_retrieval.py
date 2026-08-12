"""A stand-in for Retriever -- a synthetic catalog in place of the real FAISS index.

Why it exists: the SFT/GRPO pipeline would otherwise have to wait for the data workstream to
deliver products/chunks.parquet. With this stand-in, we can run and test the whole
"build data -> distil -> filter -> train" pipeline on day one, and swap in the real data
later by changing a single line. **This applies the parallel-work principle to the post-training
workstream.**

The interface matches rag.retrieve.Retriever exactly (duck typing), so the two are
interchangeable.
"""
import json
import re
import random

from app.schemas import Evidence, Product, RetrievalResult, SkinAnalysis, UserProfile
from skincare.config import FIXTURES, SEED


class MockRetriever:
    def __init__(self, path=None, seed: int = SEED):
        data = json.loads((path or (FIXTURES / "mock_catalog.json")).read_text()
                          if hasattr(path or FIXTURES, "read_text")
                          else open(path or FIXTURES / "mock_catalog.json").read())
        self._products = data["products"]
        self._chunks = data["chunks"]
        self.rng = random.Random(seed)

    def product_table(self, product_ids) -> dict[str, Product]:
        """Mirrors Retriever.product_table so the two stay interchangeable."""
        by_id = {p["product_id"]: p for p in self._products}
        return {pid: Product(**{k: v for k, v in by_id[pid].items()
                                if not k.startswith("_")})
                for pid in dict.fromkeys(product_ids) if pid in by_id}

    def search(self, profile: UserProfile, analysis: SkinAnalysis | None,
               top_k: int = 3, n_chunks: int = 12) -> RetrievalResult:
        # Concern vocabulary for the shipped catalogue. The real Retriever gets this from the
        # embedding model; the mock needs its own so /recommend still responds to the user's
        # words when no analysis is attached, which the API allows.
        TERMS = {
            "acne": ("acne", "pimple", "breakout", "blemish", "zit", "blackhead"),
            "dryness": ("dry", "flaky", "flake", "tight", "dehydrated", "rough"),
            "redness": ("red", "irritat", "sensitive", "rosacea", "inflam"),
            "wrinkles": ("wrinkle", "fine line", "aging", "ageing", "firm", "sag"),
            "dark_spots": ("dark spot", "pigment", "uneven", "dull", "brighten", "melasma"),
            "large_pores": ("pore", "texture", "oily", "shine", "sebum"),
        }

        wanted = set(analysis.top_concerns() if analysis else [])
        q = (profile.query or "").lower()
        for concern, terms in TERMS.items():
            if any(t in q for t in terms):
                wanted.add(concern)
        words = set(re.findall(r"[a-z]{4,}", q))

        scored = []
        for p in self._products:
            if profile.budget_usd and p["price_usd"] > profile.budget_usd:
                continue
            overlap = len(wanted & set(p["_concerns"]))
            text = " ".join([str(p.get("name", "")), str(p.get("brand", "")),
                             " ".join(str(x) for x in (p.get("ingredients") or []))]).lower()
            hits = sum(1 for w in words if w in text)
            scored.append((overlap, hits, p["product_id"], p))

        # A total order with no randomness. The same request must return the same products,
        # and a different request must be able to return different ones. The previous version
        # added jitter from an RNG that advanced on every call, so identical requests diverged.
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

        chosen = ([p for ov, _, _, p in scored[:top_k] if ov > 0]
                  or [p for _, _, _, p in scored[:top_k]])
        ids = {p["product_id"] for p in chosen}

        evidence = [Evidence(evidence_id=c["evidence_id"], product_id=c["product_id"],
                             source=c["source"], text=c["text"], score=1.0)
                    for c in self._chunks if c["product_id"] in ids][:n_chunks]
        products = [Product(**{k: v for k, v in p.items() if not k.startswith("_")})
                    for p in chosen]
        return RetrievalResult(products=products, evidence=evidence)
