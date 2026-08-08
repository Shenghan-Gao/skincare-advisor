"""Embed chunks (Module 1-2: word/sentence embeddings) and build a FAISS index.

    python -m skincare.rag.index
"""
import numpy as np
import pandas as pd
from skincare.config import PROCESSED

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=True,
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
