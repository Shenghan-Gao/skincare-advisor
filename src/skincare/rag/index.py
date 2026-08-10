"""Embed chunks (Module 1-2: word/sentence embeddings) and build a FAISS index.

    python -m skincare.rag.index
    python -m skincare.rag.index --model sentence-transformers/all-mpnet-base-v2 \
        --out-dir data/processed/index_mpnet --device cpu --batch-size 8 \
        --encode-chunk-size 128
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from skincare.config import PROCESSED

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def text_for_embedding(text: str, max_words: int | None = None) -> str:
    """Shorten embedding input without changing stored evidence or its ID."""
    return " ".join(text.split()[:max_words]) if max_words else text


@lru_cache(maxsize=4)
def get_encoder(model_name: str = EMBED_MODEL, device: str | None = None):
    """One encoder per process. Without the cache every retrieval call reloaded the
    weights from disk -- roughly a second each, paid once per user request at serve
    time and once per sample when building the SFT set."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: list[str], model_name: str = EMBED_MODEL,
                show_progress: bool | None = None,
                device: str | None = None,
                batch_size: int = 64) -> np.ndarray:
    """show_progress defaults to on only for bulk work; a single query stays quiet."""
    if show_progress is None:
        show_progress = len(texts) > 256
    model = get_encoder(model_name, device=device)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress,
                        normalize_embeddings=True).astype("float32")


def build_index(out_dir=PROCESSED / "index", model_name: str = EMBED_MODEL,
                device: str | None = None, batch_size: int = 64,
                encode_chunk_size: int | None = None,
                max_words: int | None = None):
    import faiss
    out_dir = Path(out_dir)
    chunks = pd.read_parquet(PROCESSED / "chunks.parquet")
    texts = chunks["text"].tolist()
    if max_words:
        # Keep one vector and one metadata row per evidence_id. Only the text used
        # for embedding is shortened; returned evidence retains the original text.
        texts = [text_for_embedding(text, max_words) for text in texts]
    if not texts:
        raise ValueError("cannot build an index from an empty chunks table")

    block_size = encode_chunk_size or len(texts)
    index = None
    dimension = None
    for start in range(0, len(texts), block_size):
        block = texts[start:start + block_size]
        vecs = embed_texts(block, model_name=model_name, device=device,
                           batch_size=batch_size, show_progress=False)
        if index is None:
            dimension = int(vecs.shape[1])
            index = faiss.IndexFlatIP(dimension)  # normalised -> IP == cosine
        index.add(vecs)
        if encode_chunk_size:
            print(f"encoded {min(start + block_size, len(texts))}/{len(texts)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "chunks.faiss"))
    chunks.to_parquet(out_dir / "chunks_meta.parquet")
    (out_dir / "index_config.json").write_text(
        json.dumps({
            "embedding_model": model_name,
            "dimension": dimension,
            "max_words": max_words,
        }, indent=2)
    )
    print(f"indexed {len(chunks)} chunks, dim={dimension}, model={model_name}, "
          f"max_words={max_words}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index for RAG chunks")
    parser.add_argument("--model", default=EMBED_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--out-dir", default=str(PROCESSED / "index"), help="Index output directory")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or mps")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--encode-chunk-size", type=int, default=None,
                        help="Encode this many texts at a time before adding them to FAISS")
    parser.add_argument("--max-words", type=int, default=None,
                        help="Embed at most this many words per evidence row")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args.out_dir, args.model, args.device, args.batch_size,
                args.encode_chunk_size, args.max_words)
