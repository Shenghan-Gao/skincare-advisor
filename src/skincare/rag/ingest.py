"""Build the knowledge base from raw Kaggle CSVs.

TEAMMATE B owns this file end to end.
Input : data/raw/  (Sephora products + reviews, Amazon reviews)
Output: data/processed/products.parquet, data/processed/chunks.parquet
        chunks columns -> evidence_id, product_id, source, text
"""
import pandas as pd
from skincare.config import PROCESSED, RAW


def load_products(path=RAW / "product_info.csv") -> pd.DataFrame:
    """TODO(B): normalise columns to:
    product_id, name, brand, category, price_usd, rating, ingredients(list[str])
    - parse the ingredient string into a lowercase list
    - drop rows with no ingredients
    """
    raise NotImplementedError


def load_reviews(path=RAW, max_per_product: int = 20) -> pd.DataFrame:
    """TODO(B): concat review shards -> product_id, review_text, rating, skin_type.
    Cap per product so popular items don't dominate retrieval."""
    raise NotImplementedError


def build_chunks(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """TODO(B): one row per retrievable snippet.
    evidence_id must be stable and unique -- e.g. f"{product_id}:rev:{i}".
    The reward function checks generated text against these ids."""
    raise NotImplementedError


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    products = load_products()
    reviews = load_reviews()
    chunks = build_chunks(products, reviews)
    products.to_parquet(PROCESSED / "products.parquet")
    chunks.to_parquet(PROCESSED / "chunks.parquet")
    print(f"products={len(products)} chunks={len(chunks)}")


if __name__ == "__main__":
    main()
