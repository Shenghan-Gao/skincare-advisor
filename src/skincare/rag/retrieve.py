"""Query-time retrieval: fuse the user's words with the CNN's skin labels.

That fusion is the concrete 'value beyond an off-the-shelf LLM' claim in the
proposal -- a plain chatbot cannot see the photo or the product table.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.schemas import Evidence, Product, RetrievalResult, SkinAnalysis, UserProfile
from skincare.config import KNOWLEDGE, PROCESSED
from skincare.rag.index import EMBED_MODEL

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_rules() -> dict:
    with open(KNOWLEDGE / "ingredient_rules.json") as f:
        return json.load(f)


def build_query(profile: UserProfile, analysis: SkinAnalysis | None) -> str:
    parts = [profile.query]
    rules = load_rules()
    if analysis:
        parts.append(f"skin type {analysis.skin_type.value}")
        for concern in analysis.top_concerns():
            parts.append(concern.replace("_", " "))
            parts += rules["concern_to_ingredients"].get(concern, [])[:3]
    if profile.preferences:
        parts += profile.preferences
    return " ".join(p for p in parts if p)


def index_embedding_model(index_dir: str | Path) -> str:
    """Return the encoder recorded with an index, with legacy MiniLM fallback."""
    config_path = Path(index_dir) / "index_config.json"
    if not config_path.exists():
        return EMBED_MODEL
    return json.loads(config_path.read_text()).get("embedding_model", EMBED_MODEL)


@lru_cache(maxsize=2)
def get_reranker(model_name: str = RERANK_MODEL, device: str | None = None):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device=device)


class Retriever:
    def __init__(self, index_dir: str = str(PROCESSED / "index"),
                 device: str | None = None,
                 rerank_model: str | None = None):
        import faiss
        d = Path(index_dir)
        self.index = faiss.read_index(str(d / "chunks.faiss"))
        self.meta = pd.read_parquet(d / "chunks_meta.parquet")
        self.products = pd.read_parquet(PROCESSED / "products.parquet").set_index("product_id")
        self.embedding_model = index_embedding_model(d)
        self.device = device
        self.rerank_model = rerank_model

    def search(self, profile: UserProfile, analysis: SkinAnalysis | None,
               top_k: int = 3, n_chunks: int = 30) -> RetrievalResult:
        from skincare.rag.index import embed_texts
        query = build_query(profile, analysis)
        qv = embed_texts([query], model_name=self.embedding_model,
                         device=self.device)
        scores, idx = self.index.search(qv, n_chunks)

        hits = self.meta.iloc[idx[0]].copy()
        hits["score"] = scores[0]
        # hard filters the LLM must not have to reason about
        if profile.budget_usd:
            ok = self.products[self.products["price_usd"] <= profile.budget_usd].index
            hits = hits[hits["product_id"].isin(ok)]

        if self.rerank_model and not hits.empty:
            reranker = get_reranker(self.rerank_model, self.device)
            hits["score"] = reranker.predict(
                [[query, text] for text in hits["text"].tolist()],
                show_progress_bar=False,
            )
            hits = hits.sort_values("score", ascending=False)

        evidence = [Evidence(evidence_id=r.evidence_id, product_id=r.product_id,
                             source=r.source, text=r.text, score=float(r.score))
                    for r in hits.itertuples()]
        top_ids = list(dict.fromkeys(e.product_id for e in evidence))[:top_k]
        products = [Product(product_id=pid, **self.products.loc[pid].to_dict())
                    for pid in top_ids if pid in self.products.index]
        return RetrievalResult(products=products, evidence=evidence)
