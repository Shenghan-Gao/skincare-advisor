"""Score a completed blinded relevance sheet against the private system mapping."""
import argparse
import csv
import json
from pathlib import Path

from skincare.config import MODELS


def read_labels(path: Path) -> dict[tuple[str, str], int]:
    labels = {}
    missing = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row["relevant"].strip()
            key = (row["query_id"], row["product_id"])
            if value not in {"0", "1"}:
                missing.append(row["candidate_id"])
                continue
            labels[key] = int(value)
    if missing:
        raise ValueError(
            f"{len(missing)} candidates are not labelled 0/1; first missing: {missing[:5]}"
        )
    return labels


def score_mapping(mapping: dict, labels: dict[tuple[str, str], int]) -> dict:
    top_k = int(mapping["top_k"])
    per_system = {}
    query_ids = sorted(mapping["systems"])
    system_names = sorted({
        name for query in mapping["systems"].values() for name in query
    })
    for system_name in system_names:
        rows = []
        for query_id in query_ids:
            product_ids = mapping["systems"][query_id][system_name]
            relevant = 0
            for product_id in product_ids:
                key = (query_id, product_id)
                if key not in labels:
                    raise ValueError(f"missing label for {query_id}, {product_id}")
                relevant += labels[key]
            rows.append({
                "query_id": query_id,
                "relevant_at_3": relevant,
                "precision_at_3": relevant / top_k,
                "ranked_product_ids": product_ids,
            })
        per_system[system_name] = {
            "precision_at_3": sum(row["precision_at_3"] for row in rows) / len(rows),
            "per_query": rows,
        }
    return {
        "query_count": len(query_ids),
        "top_k": top_k,
        "systems": per_system,
    }


def main():
    default_dir = MODELS / "rag" / "human_eval"
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(default_dir / "candidates_for_c.csv"))
    parser.add_argument("--mapping", default=str(default_dir / "candidate_sources_private.json"))
    parser.add_argument("--output", default=str(default_dir / "precision_at_3.json"))
    args = parser.parse_args()

    labels = read_labels(Path(args.labels))
    mapping = json.loads(Path(args.mapping).read_text())
    result = score_mapping(mapping, labels)
    Path(args.output).write_text(json.dumps(result, indent=2))
    for system_name, metrics in result["systems"].items():
        print({"system": system_name, "precision_at_3": metrics["precision_at_3"]})
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
