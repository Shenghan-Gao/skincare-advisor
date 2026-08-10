"""Compare RAG indexes with a reproducible concern/ingredient proxy metric.

This is an engineering screen, not the final human-labelled Precision@3 owned by
the evaluation workstream. It helps choose which index and top-k deserve that
more expensive evaluation.

    python -m skincare.rag.evaluate \
        --index minilm=data/processed/index \
        --index mpnet=data/processed/index_mpnet \
        --device mps
"""
import argparse
import json
import time
from pathlib import Path

from app.schemas import SkinAnalysis, UserProfile
from skincare.config import MODELS
from skincare.rag.retrieve import Retriever, load_rules

CASES = [
    ("acne", "oily", "gentle treatment for acne and breakouts"),
    ("acne", "oily", "oil-free moisturizer for acne-prone skin"),
    ("dark_spots", "combination", "serum to fade dark spots"),
    ("dark_spots", "combination", "brightening skincare for post-acne marks"),
    ("redness", "dry", "soothing moisturizer for facial redness"),
    ("redness", "dry", "calming skincare for irritated skin"),
    ("large_pores", "oily", "product to reduce the look of large pores"),
    ("large_pores", "oily", "pore-clearing skincare for oily skin"),
    ("wrinkles", "dry", "anti-aging serum for wrinkles"),
    ("wrinkles", "dry", "treatment for fine lines and firming"),
    ("dryness", "dry", "rich moisturizer for very dry skin"),
    ("dryness", "dry", "hydrating serum for dehydrated skin"),
]


def ingredient_match(ingredients: list[str], targets: list[str]) -> bool:
    values = [str(value).lower() for value in ingredients]
    return any(target in value for target in targets for value in values)


def evaluate_index(label: str, index_dir: str, top_ks: list[int],
                   n_chunks: int, device: str | None,
                   rerank_model: str | None = None) -> dict:
    retriever = Retriever(index_dir, device=device, rerank_model=rerank_model)
    targets = load_rules()["concern_to_ingredients"]

    # Exclude one-time model loading from query latency.
    concern, skin_type, query = CASES[0]
    retriever.search(
        UserProfile(query=query, budget_usd=100),
        SkinAnalysis(skin_type=skin_type, skin_type_confidence=0.9,
                     concerns=[{"concern": concern, "score": 0.9}]),
        top_k=max(top_ks), n_chunks=n_chunks,
    )

    by_k = {}
    for top_k in top_ks:
        matches = 0
        returned = 0
        elapsed = 0.0
        unique_ids = set()
        details = []
        for concern, skin_type, query in CASES:
            profile = UserProfile(query=query, budget_usd=100)
            analysis = SkinAnalysis(
                skin_type=skin_type,
                skin_type_confidence=0.9,
                concerns=[{"concern": concern, "score": 0.9}],
            )
            started = time.perf_counter()
            result = retriever.search(profile, analysis, top_k=top_k,
                                      n_chunks=n_chunks)
            elapsed += time.perf_counter() - started

            rows = []
            for product in result.products:
                hit = ingredient_match(product.ingredients, targets[concern])
                matches += int(hit)
                returned += 1
                unique_ids.add(product.product_id)
                rows.append({"product_id": product.product_id, "name": product.name,
                             "ingredient_match": hit})
            details.append({"concern": concern, "query": query, "products": rows})

        denominator = len(CASES) * top_k
        by_k[str(top_k)] = {
            "ingredient_proxy_precision": matches / returned if returned else 0.0,
            "result_fill_rate": returned / denominator,
            "mean_warm_query_seconds": elapsed / len(CASES),
            "unique_products": len(unique_ids),
            "details": details,
        }

    return {
        "label": label,
        "index_dir": index_dir,
        "embedding_model": retriever.embedding_model,
        "dimension": retriever.index.d,
        "indexed_chunks": retriever.index.ntotal,
        "n_chunks": n_chunks,
        "rerank_model": rerank_model,
        "metrics_by_top_k": by_k,
    }


def parse_index(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("index must use LABEL=PATH")
    return tuple(value.split("=", 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="append", type=parse_index, required=True)
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--n-chunks", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--output", default=str(MODELS / "rag" / "retrieval_proxy_comparison.json"))
    args = parser.parse_args()

    top_ks = args.top_k or [3, 5]
    report = {
        "metric_note": (
            "Ingredient proxy only; do not report as human-labelled Precision@k. "
            "Final Precision@3 requires C's manually labelled query-product set."
        ),
        "case_count": len(CASES),
        "indexes": [
            evaluate_index(label, path, top_ks, args.n_chunks, args.device,
                           args.rerank_model)
            for label, path in args.index
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    for index in report["indexes"]:
        for top_k, metrics in index["metrics_by_top_k"].items():
            print({
                "index": index["label"],
                "top_k": int(top_k),
                "ingredient_proxy_precision": round(metrics["ingredient_proxy_precision"], 4),
                "result_fill_rate": round(metrics["result_fill_rate"], 4),
                "mean_warm_query_seconds": round(metrics["mean_warm_query_seconds"], 4),
                "unique_products": metrics["unique_products"],
            })
    print(f"saved {output}")


if __name__ == "__main__":
    main()
