"""Convert Member A's cleaned CSV handoff into the parquet files the contract requires.

Why this exists: the delivered package contains ``products_clean.csv`` and
``chunks_clean.csv``, but ``skincare.rag.retrieve`` and ``scripts/validate_data.py``
read ``data/processed/products.parquet`` and ``chunks.parquet``. Regenerating them with
``python -m skincare.rag.ingest`` needs the raw Sephora exports, which are not part of
the handoff, so the cleaned CSVs are converted directly instead.

The only non-trivial step is ``ingredients``: CSV cannot hold a list, so it is stored as
a JSON array string and must be parsed back into ``list[str]`` for the contract to hold.

    python scripts/csv_to_parquet.py
    python scripts/csv_to_parquet.py --src <dir> --out data/processed
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

DEFAULT_SRC = ROOT / "data" / "processed" / "member_a_processed_handoff" / "sephora"
DEFAULT_OUT = ROOT / "data" / "processed"


def parse_ingredients(value) -> list[str]:
    """Turn the stored ingredient string back into a list.

    Accepts a JSON array (what the handoff uses), a Python-literal list, or a plain
    comma-separated string, so the converter survives a change of export format.
    """
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [str(v).strip().lower() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            continue
    return [p.strip().lower() for p in text.split(",") if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(src / "products_clean.csv")
    products["ingredients"] = products["ingredients"].apply(parse_ingredients)
    products["price_usd"] = pd.to_numeric(products["price_usd"], errors="coerce")
    products["rating"] = pd.to_numeric(products["rating"], errors="coerce")
    products.to_parquet(out / "products.parquet", index=False)
    empty = int((products["ingredients"].apply(len) == 0).sum())
    print(f"products.parquet  {len(products):6d} rows   empty ingredient lists: {empty}")

    chunks = pd.read_csv(src / "chunks_clean.csv")
    chunks.to_parquet(out / "chunks.parquet", index=False)
    print(f"chunks.parquet    {len(chunks):6d} rows   sources: "
          f"{chunks['source'].value_counts().to_dict()}")

    orphan = int((~chunks["product_id"].isin(set(products["product_id"]))).sum())
    print(f"\nreferential integrity: {orphan} chunk(s) point at a missing product")
    print("next: python scripts/validate_data.py products && "
          "python scripts/validate_data.py chunks")


if __name__ == "__main__":
    main()
