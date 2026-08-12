"""A stand-in for Retriever -- a synthetic catalog in place of the real FAISS index.

Why it exists: Anna's SFT/GRPO pipeline would otherwise have to wait for teammate A to
deliver products/chunks.parquet. With this stand-in, Anna can run and test the whole
"build data -> distil -> filter -> train" pipeline on day one, and swap in the real data
later by changing a single line. **This applies the parallel-work principle to Anna's own
workstream.**

The interface matches rag.retrieve.Retriever exactly (duck typing), so the two are
interchangeable.
"""
import json
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
        wanted = set(analysis.top_concerns() if analysis else [])

        # Score by overlap with the user's concerns so that retrieval stays relevant to the
        # profile -- if the results were random, the rewards would carry no signal.
        scored = []
        for p in self._products:
            overlap = len(wanted & set(p["_concerns"]))
            if profile.budget_usd and p["price_usd"] > profile.budget_usd:
                continue
            scored.append((overlap, overlap + self.rng.random() * 0.1, p))
        scored.sort(key=lambda x: -x[1])
        # Keep only products that actually overlap a concern; fall back to plain top_k only
        # when nothing is relevant (otherwise the reward signal gets diluted).
        chosen = [p for ov, _, p in scored[:top_k] if ov > 0] or [p for _, _, p in scored[:top_k]]
        ids = {p["product_id"] for p in chosen}

        evidence = [Evidence(evidence_id=c["evidence_id"], product_id=c["product_id"],
                             source=c["source"], text=c["text"], score=1.0)
                    for c in self._chunks if c["product_id"] in ids][:n_chunks]
        products = [Product(**{k: v for k, v in p.items() if not k.startswith("_")})
                    for p in chosen]
        return RetrievalResult(products=products, evidence=evidence)
