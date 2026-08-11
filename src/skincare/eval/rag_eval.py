"""Retrieval evaluation utilities owned by member C.

The reported RAG comparison used blinded human relevance labels on 42 unique
query-product pairs across 8 queries, then computed macro-averaged Precision@3 for
MiniLM and MPNet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def groundedness_rate(
    cited_evidence_ids: Iterable[str], valid_evidence_ids: Iterable[str]
) -> float:
    """Return the share of citations that resolve to real evidence IDs.

    An empty citation list returns 0.0 because no generated claim is grounded by a
    verifiable citation.
    """
    cited = [str(x) for x in cited_evidence_ids]
    if not cited:
        return 0.0
    valid = {str(x) for x in valid_evidence_ids}
    return sum(item in valid for item in cited) / len(cited)


def precision_at_k(
    relevance_by_product: dict[str, int], ranked_product_ids: list[str], k: int = 3
) -> float:
    """Compute Precision@k for one query from binary human relevance labels."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = [str(product_id) for product_id in ranked_product_ids[:k]]
    if len(top) < k:
        raise ValueError(f"Ranking contains only {len(top)} products, fewer than k={k}")
    return sum(int(relevance_by_product.get(product_id, 0)) for product_id in top) / k


def load_human_labels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load blind human labels as ``query_id -> product_id -> {0,1}``."""
    df = pd.read_csv(path)
    required = {"query_id", "product_id", "relevant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label CSV is missing columns: {sorted(missing)}")

    labels: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        if pd.isna(row["relevant"]):
            raise ValueError(
                "Missing relevance label for "
                f"query={row['query_id']} product={row['product_id']}"
            )
        relevant = int(row["relevant"])
        if relevant not in {0, 1}:
            raise ValueError("relevant must contain only 0/1 labels")
        qid = str(row["query_id"])
        product_id = str(row["product_id"])
        if product_id in labels.setdefault(qid, {}):
            raise ValueError(f"Duplicate query-product pair: {qid} / {product_id}")
        labels[qid][product_id] = relevant
    return labels


def _extract_rankings(payload: dict) -> dict[str, dict[str, list[str]]]:
    """Normalize supported ranking JSON shapes to system -> query -> product IDs.

    Supported shapes:
      {"systems": {"minilm": {"per_query": [{"query_id": ..., "ranked_product_ids": [...]}]}}}
    or
      {"minilm": {"Q001": ["P1", "P2", "P3"], ...}, "mpnet": {...}}
    """
    if "systems" in payload:
        systems = payload["systems"]
        normalized: dict[str, dict[str, list[str]]] = {}
        for system_name, system_payload in systems.items():
            per_query = system_payload.get("per_query", [])
            normalized[system_name] = {
                str(item["query_id"]): [str(p) for p in item["ranked_product_ids"]]
                for item in per_query
            }
        return normalized

    normalized = {}
    for system_name, query_map in payload.items():
        if not isinstance(query_map, dict):
            continue
        normalized[system_name] = {
            str(query_id): [str(p) for p in product_ids]
            for query_id, product_ids in query_map.items()
        }
    return normalized


def evaluate_rankings(
    labeled_csv: str | Path,
    rankings_json: str | Path,
    *,
    k: int = 3,
) -> dict:
    """Evaluate multiple retrieval systems against blinded human relevance labels."""
    labels = load_human_labels(labeled_csv)
    payload = json.loads(Path(rankings_json).read_text(encoding="utf-8"))
    systems = _extract_rankings(payload)
    if not systems:
        raise ValueError("No retrieval systems found in rankings JSON")

    result = {"query_count": len(labels), "top_k": k, "systems": {}}
    for system_name, rankings in systems.items():
        per_query = []
        for query_id in sorted(labels):
            if query_id not in rankings:
                raise ValueError(f"System {system_name!r} is missing ranking for {query_id}")
            ranked = rankings[query_id]
            score = precision_at_k(labels[query_id], ranked, k=k)
            relevant_at_k = int(round(score * k))
            per_query.append(
                {
                    "query_id": query_id,
                    "relevant_at_k": relevant_at_k,
                    f"precision_at_{k}": score,
                    "ranked_product_ids": ranked[:k],
                }
            )

        macro = sum(item[f"precision_at_{k}"] for item in per_query) / len(per_query)
        result["systems"][system_name] = {
            f"precision_at_{k}": macro,
            "per_query": per_query,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute blinded human-labelled RAG Precision@k")
    ap.add_argument("--labels", required=True, help="CSV containing query_id, product_id, relevant")
    ap.add_argument(
        "--rankings",
        required=True,
        help="JSON containing ranked product IDs per query/system",
    )
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="reports/rag_precision_at_3.json")
    args = ap.parse_args()

    result = evaluate_rankings(args.labels, args.rankings, k=args.k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
