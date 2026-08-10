"""Create a blinded query-product relevance sheet for Member C.

Seven queries x top-3 from two systems produces at most 42 query-product pairs,
matching the team's target of roughly 40 manual relevance judgements.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from app.schemas import SkinAnalysis, UserProfile
from skincare.config import MODELS, SEED
from skincare.rag.retrieve import Retriever

CASES = [
    {
        "query_id": "Q001", "query": "gentle treatment and moisturizer for acne-prone skin",
        "skin_type": "oily", "concerns": ["acne"], "budget_usd": 50,
        "preferences": ["fragrance-free", "non-comedogenic"],
    },
    {
        "query_id": "Q002", "query": "brightening serum to fade dark spots and post-acne marks",
        "skin_type": "combination", "concerns": ["dark_spots"], "budget_usd": 80,
        "preferences": [],
    },
    {
        "query_id": "Q003", "query": "calming moisturizer for redness and easily irritated skin",
        "skin_type": "dry", "concerns": ["redness"], "budget_usd": 60,
        "preferences": ["fragrance-free"],
    },
    {
        "query_id": "Q004", "query": "pore-clearing skincare for oily skin with large pores",
        "skin_type": "oily", "concerns": ["large_pores"], "budget_usd": 40,
        "preferences": [],
    },
    {
        "query_id": "Q005", "query": "anti-aging treatment for wrinkles and fine lines",
        "skin_type": "dry", "concerns": ["wrinkles"], "budget_usd": 100,
        "preferences": [],
    },
    {
        "query_id": "Q006", "query": "rich hydrating moisturizer for very dry flaky skin",
        "skin_type": "dry", "concerns": ["dryness"], "budget_usd": 50,
        "preferences": ["fragrance-free"],
    },
    {
        "query_id": "Q007", "query": "routine for acne and dark spots on oily skin",
        "skin_type": "oily", "concerns": ["acne", "dark_spots"], "budget_usd": 70,
        "preferences": ["non-comedogenic"],
    },
    {
        "query_id": "Q008", "query": "affordable soothing cream for red and very dry skin",
        "skin_type": "dry", "concerns": ["redness", "dryness"], "budget_usd": 35,
        "preferences": ["fragrance-free"],
    },
]


def parse_index(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("index must use LABEL=PATH")
    return tuple(value.split("=", 1))


def schemas(case):
    profile = UserProfile(
        query=case["query"], budget_usd=case["budget_usd"],
        preferences=case["preferences"],
    )
    analysis = SkinAnalysis(
        skin_type=case["skin_type"], skin_type_confidence=0.9,
        concerns=[{"concern": concern, "score": 0.9} for concern in case["concerns"]],
    )
    return profile, analysis


def generate(indexes, output: Path, device: str | None, top_k: int, n_chunks: int):
    retrievers = {
        label: Retriever(path, device=device) for label, path in indexes
    }
    output.mkdir(parents=True, exist_ok=True)
    private_mapping = {"top_k": top_k, "n_chunks": n_chunks, "systems": {}}
    blinded_rows = []

    for case in CASES:
        profile, analysis = schemas(case)
        product_pool = {}
        private_mapping["systems"][case["query_id"]] = {}
        for label, retriever in retrievers.items():
            result = retriever.search(profile, analysis, top_k=top_k, n_chunks=n_chunks)
            ranked_ids = [product.product_id for product in result.products]
            private_mapping["systems"][case["query_id"]][label] = ranked_ids
            for product in result.products:
                evidence = [
                    item for item in result.evidence if item.product_id == product.product_id
                ][:2]
                product_pool.setdefault(product.product_id, {
                    "product_id": product.product_id,
                    "name": product.name,
                    "brand": product.brand,
                    "category": product.category,
                    "price_usd": product.price_usd,
                    "ingredients": "; ".join(product.ingredients[:20]),
                    "evidence_excerpt": " | ".join(
                        f"{item.source}: {item.text[:300]}" for item in evidence
                    ),
                })

        products = list(product_pool.values())
        random.Random(SEED + int(case["query_id"][1:])).shuffle(products)
        for number, product in enumerate(products, start=1):
            blinded_rows.append({
                "candidate_id": f"{case['query_id']}-C{number:02d}",
                "query_id": case["query_id"],
                "query": case["query"],
                "skin_type": case["skin_type"],
                "concerns": ";".join(case["concerns"]),
                "budget_usd": case["budget_usd"],
                "preferences": ";".join(case["preferences"]),
                **product,
                "relevant": "",
                "notes": "",
            })

    query_fields = [
        "query_id", "query", "skin_type", "concerns", "budget_usd", "preferences"
    ]
    with (output / "queries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fields)
        writer.writeheader()
        for case in CASES:
            writer.writerow({
                **case,
                "concerns": ";".join(case["concerns"]),
                "preferences": ";".join(case["preferences"]),
            })

    with (output / "candidates_for_c.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blinded_rows[0]))
        writer.writeheader()
        writer.writerows(blinded_rows)

    (output / "candidate_sources_private.json").write_text(
        json.dumps(private_mapping, indent=2)
    )
    (output / "README.md").write_text(
        "# RAG relevance labelling\n\n"
        "Open `candidates_for_c.csv` and inspect each query-product pair. "
        "Set `relevant` to `1` when the product is relevant to the query/profile, "
        "otherwise set it to `0`. Add an optional short reason in `notes`. "
        "Do not open `candidate_sources_private.json` before labelling because it "
        "reveals which retrieval system returned each product.\n"
    )
    print(f"saved {len(CASES)} queries and {len(blinded_rows)} blinded pairs to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="append", type=parse_index, required=True)
    parser.add_argument("--output", default=str(MODELS / "rag" / "human_eval"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-chunks", type=int, default=30)
    args = parser.parse_args()
    generate(args.index, Path(args.output), args.device, args.top_k, args.n_chunks)


if __name__ == "__main__":
    main()
