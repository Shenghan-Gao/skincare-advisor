import json

import numpy as np
import pandas as pd

from app.schemas import UserProfile
from skincare.rag.index import EMBED_MODEL, text_for_embedding
from skincare.rag.retrieve import index_embedding_model


def test_legacy_index_defaults_to_minilm(tmp_path):
    assert index_embedding_model(tmp_path) == EMBED_MODEL


def test_index_uses_recorded_embedding_model(tmp_path):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    (tmp_path / "index_config.json").write_text(
        json.dumps({"embedding_model": model_name, "dimension": 768})
    )
    assert index_embedding_model(tmp_path) == model_name


def test_embedding_text_can_be_shortened_without_changing_source():
    source = "one two three four five"
    assert text_for_embedding(source, 3) == "one two three"
    assert text_for_embedding(source) == source


def test_optional_reranker_changes_product_order(monkeypatch):
    from skincare.rag import index as index_module
    from skincare.rag import retrieve as retrieve_module

    class FakeIndex:
        def search(self, query, n_chunks):
            return np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]])

    class FakeReranker:
        def predict(self, pairs, show_progress_bar=False):
            return np.array([0.1, 0.9, 0.2])

    retriever = retrieve_module.Retriever.__new__(retrieve_module.Retriever)
    retriever.index = FakeIndex()
    retriever.meta = pd.DataFrame([
        {"evidence_id": "P1:rev:0", "product_id": "P1", "source": "review", "text": "one"},
        {"evidence_id": "P2:rev:0", "product_id": "P2", "source": "review", "text": "two"},
        {"evidence_id": "P3:rev:0", "product_id": "P3", "source": "review", "text": "three"},
    ])
    retriever.products = pd.DataFrame([
        {"product_id": pid, "name": pid, "brand": "brand", "category": "skin",
         "price_usd": 10.0, "rating": 4.0, "ingredients": ["water"]}
        for pid in ("P1", "P2", "P3")
    ]).set_index("product_id")
    retriever.embedding_model = EMBED_MODEL
    retriever.device = "cpu"
    retriever.rerank_model = "fake-reranker"

    monkeypatch.setattr(index_module, "embed_texts", lambda *args, **kwargs: np.zeros((1, 3)))
    monkeypatch.setattr(retrieve_module, "get_reranker", lambda *args, **kwargs: FakeReranker())
    result = retriever.search(UserProfile(query="test"), None, top_k=3, n_chunks=3)
    assert [product.product_id for product in result.products] == ["P2", "P3", "P1"]
