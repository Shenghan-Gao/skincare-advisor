"""Embed chunks (Module 1-2: word/sentence embeddings) and build a FAISS index.

    python -m skincare.rag.index
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from skincare.config import PROCESSED

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def get_encoder(model_name: str = EMBED_MODEL):
    """One encoder per process. Without the cache every retrieval call reloaded the
    weights from disk -- roughly a second each, paid once per user request at serve
    time and once per sample when building the SFT set."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = EMBED_MODEL,
                show_progress: bool | None = None) -> np.ndarray:
    """show_progress defaults to on only for bulk work; a single query stays quiet."""
    if show_progress is None:
        show_progress = len(texts) > 256
    model = get_encoder(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=show_progress,
                        normalize_embeddings=True).astype("float32")


def build_index(out_dir=PROCESSED / "index"):
    import faiss
    chunks = pd.read_parquet(PROCESSED / "chunks.parquet")
    vecs = embed_texts(chunks["text"].tolist())
    index = faiss.IndexFlatIP(vecs.shape[1])   # normalised -> inner product == cosine
    index.add(vecs)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "chunks.faiss"))
    chunks.to_parquet(out_dir / "chunks_meta.parquet")
    print(f"indexed {len(chunks)} chunks, dim={vecs.shape[1]}")


if __name__ == "__main__":
    build_index()
