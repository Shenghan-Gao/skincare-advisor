"""Member A: build the skincare knowledge base from raw Sephora CSVs.

Raw source used for this project:
  Kaggle dataset: "Sephora Products and Skincare Reviews"
  Owner: Nady Inky
  Slug: nadyinky/sephora-products-and-skincare-reviews
  Data card: collected via Python scraper in March 2023.

Expected raw files under ``data/raw/sephora``:
  - product_info.csv
  - reviews_*.csv

Outputs under ``data/processed``:
  - products.parquet
  - chunks.parquet
  - cleaning_audit.json

The downstream contract is frozen:
  products: product_id, name, brand, category, price_usd, rating, ingredients
  chunks:   evidence_id, product_id, source, text

Important: ``evidence_id`` must stay ``{product_id}:{desc|ing|rev}:{index}``.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from skincare.config import PROCESSED, RAW

SPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,30}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")


def _clean_scalar(value) -> str:
    if pd.isna(value):
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def _literal_list(value) -> list[str]:
    """Parse Sephora cells that are often stored as a stringified Python list."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple)):
            return [_clean_scalar(x) for x in obj if _clean_scalar(x)]
    except (ValueError, SyntaxError):
        pass
    return [text]


def parse_ingredients(value) -> list[str]:
    """Normalize ingredient text to a deduplicated lowercase list.

    Project cleaning rule: split ingredient blocks by comma, strip whitespace,
    lowercase, remove fragments longer than 60 characters, and deduplicate.
    """
    out: list[str] = []
    seen: set[str] = set()
    for block in _literal_list(value):
        block = block.replace("\n", ",").replace("\r", ",").replace(";", ",")
        for part in block.split(","):
            token = SPACE_RE.sub(" ", part).strip(" .:-\t'\"[]").lower()
            if not token or len(token) > 60:
                continue
            if token in {"ingredients", "ingredient list", "key ingredients"}:
                continue
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


def _parse_highlights(value) -> str:
    return "; ".join(_literal_list(value))


def load_products(path: Path = RAW / "sephora" / "product_info.csv") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean Sephora product_info.csv.

    Returns ``(products, metadata_for_chunks, audit)``.  ``metadata_for_chunks``
    is internal only and is not part of the downstream product contract.
    """
    raw = pd.read_csv(path, low_memory=False)
    audit = Counter(raw_products=len(raw))

    mask = raw["primary_category"].astype("string").fillna("").str.contains(
        "skincare", case=False, na=False
    )
    df = raw.loc[mask].copy()
    audit["skincare_products"] = len(df)

    df["_ingredients"] = df["ingredients"].map(parse_ingredients)
    df["_price"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df["_rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["_review_count"] = pd.to_numeric(df.get("reviews"), errors="coerce").fillna(-1)

    audit["dropped_empty_ingredients"] = int((df["_ingredients"].map(len) == 0).sum())
    audit["dropped_invalid_price"] = int((~(df["_price"] > 0)).sum())
    df = df[(df["_ingredients"].map(len) > 0) & (df["_price"] > 0)].copy()

    # Keep the duplicate product row with the largest review count.
    audit["duplicate_rows_before_product_id_dedup"] = int(df["product_id"].duplicated(keep=False).sum())
    df = (
        df.sort_values(["product_id", "_review_count"], ascending=[True, False])
        .drop_duplicates("product_id", keep="first")
    )

    products = pd.DataFrame(
        {
            "product_id": df["product_id"].astype(str).str.strip(),
            "name": df["product_name"].map(_clean_scalar),
            "brand": df["brand_name"].map(_clean_scalar),
            "category": df["secondary_category"].map(_clean_scalar),
            "price_usd": df["_price"].astype(float),
            # Missing rating stays missing; it is never imputed as 0.
            "rating": df["_rating"].astype("Float64"),
            "ingredients": df["_ingredients"],
        }
    )
    products = products[products["product_id"].ne("")].reset_index(drop=True)
    audit["final_products"] = len(products)
    audit["missing_ratings"] = int(products["rating"].isna().sum())

    metadata = pd.DataFrame(
        {
            "product_id": df["product_id"].astype(str).str.strip(),
            "name": df["product_name"].map(_clean_scalar),
            "brand": df["brand_name"].map(_clean_scalar),
            "category": df["secondary_category"].map(_clean_scalar),
            "highlights": df["highlights"].map(_parse_highlights),
            "ingredients_text": df["_ingredients"].map(lambda xs: ", ".join(xs)),
        }
    ).drop_duplicates("product_id")

    return products, metadata, dict(audit)


def _remove_pii_vectorized(text: pd.Series) -> pd.Series:
    text = text.str.replace(EMAIL_RE, "[EMAIL]", regex=True)
    text = text.str.replace(HANDLE_RE, "[HANDLE]", regex=True)
    text = text.str.replace(PHONE_RE, "[PHONE]", regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def load_reviews(
    path: Path = RAW / "sephora",
    valid_product_ids: set[str] | None = None,
    max_per_product: int = 20,
    candidate_pool: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Clean the five Sephora review shards.

    For speed, each shard first keeps a generous helpfulness-based candidate
    pool per product.  PII/language normalization is then applied to that pool,
    exact normalized review text is deduplicated globally, and the final top
    ``max_per_product`` reviews are retained per product.

    The assignment explicitly permits an ASCII-ratio threshold for the English
    filter.  Here, text with more than 15% non-ASCII characters is excluded.
    """
    review_files = sorted(path.glob("reviews_*.csv"))
    if not review_files:
        raise FileNotFoundError(f"No reviews_*.csv found under {path}")

    valid_product_ids = valid_product_ids or set()
    audit = Counter()
    pools: list[pd.DataFrame] = []
    usecols = ["product_id", "review_text", "helpfulness", "total_feedback_count"]

    for file in review_files:
        raw = pd.read_csv(file, usecols=usecols, low_memory=False)
        audit["review_rows_raw"] += len(raw)
        raw["product_id"] = raw["product_id"].astype(str).str.strip()
        if valid_product_ids:
            raw = raw[raw["product_id"].isin(valid_product_ids)]
        audit["review_rows_for_retained_products"] += len(raw)

        raw["review_text"] = (
            raw["review_text"].astype("string").fillna("")
            .str.replace(r"\s+", " ", regex=True).str.strip()
        )
        raw = raw[raw["review_text"].str.len().ge(20)]
        audit["review_rows_length_ok"] += len(raw)
        raw["helpfulness"] = pd.to_numeric(raw["helpfulness"], errors="coerce").fillna(0.0)
        raw["feedback"] = pd.to_numeric(raw["total_feedback_count"], errors="coerce").fillna(0.0)

        # A 3x cushion over the final top-20 makes the expensive text-cleaning
        # stage tractable while leaving ample room for language/duplicate drops.
        shard_pool = (
            raw.sort_values(["product_id", "helpfulness", "feedback"], ascending=[True, False, False])
            .groupby("product_id", group_keys=False)
            .head(max(candidate_pool, max_per_product))
        )
        pools.append(shard_pool[["product_id", "review_text", "helpfulness", "feedback"]])

    pool = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    audit["review_candidate_pool"] = len(pool)
    if pool.empty:
        return pd.DataFrame(columns=["product_id", "text", "helpfulness", "feedback"]), dict(audit)

    text = pool["review_text"].str.slice(0, 1500)
    pool["text"] = _remove_pii_vectorized(text)

    non_ascii_ratio = pool["text"].str.count(r"[^\x00-\x7F]") / pool["text"].str.len().clip(lower=1)
    pool = pool[non_ascii_ratio <= 0.15].copy()
    audit["review_rows_after_english_heuristic"] = len(pool)

    before = len(pool)
    pool = pool.drop_duplicates("text", keep="first")
    audit["duplicate_review_text_removed"] = before - len(pool)

    reviews = (
        pool.sort_values(["product_id", "helpfulness", "feedback"], ascending=[True, False, False])
        .groupby("product_id", group_keys=False)
        .head(max_per_product)
        .reset_index(drop=True)
    )
    audit["final_reviews"] = len(reviews)
    audit["products_with_reviews"] = int(reviews["product_id"].nunique())
    return reviews[["product_id", "text", "helpfulness", "feedback"]], dict(audit)


def build_chunks(products: pd.DataFrame, metadata: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Create description, ingredient, and review evidence chunks."""
    rows: list[dict[str, str]] = []
    valid = set(products["product_id"].astype(str))

    for r in metadata.itertuples(index=False):
        pid = str(r.product_id)
        if pid not in valid:
            continue
        # product_info.csv has no dedicated free-text description field.  A
        # grounded metadata description is built from brand/name/category/highlights.
        description = "; ".join(
            x
            for x in [
                f"{r.brand} {r.name}".strip(),
                f"Category: {r.category}" if r.category else "",
                f"Highlights: {r.highlights}" if r.highlights else "",
            ]
            if x
        )
        if len(description) >= 20:
            rows.append(
                {
                    "evidence_id": f"{pid}:desc:0",
                    "product_id": pid,
                    "source": "description",
                    "text": description[:1500],
                }
            )

        ingredient_text = f"Ingredients: {r.ingredients_text}"
        if len(ingredient_text) >= 20:
            rows.append(
                {
                    "evidence_id": f"{pid}:ing:0",
                    "product_id": pid,
                    "source": "ingredient",
                    "text": ingredient_text[:1500],
                }
            )

    for pid, group in reviews.groupby("product_id", sort=True):
        group = group.sort_values(["helpfulness", "feedback"], ascending=[False, False]).head(20)
        for i, r in enumerate(group.itertuples(index=False)):
            rows.append(
                {
                    "evidence_id": f"{pid}:rev:{i}",
                    "product_id": str(pid),
                    "source": "review",
                    "text": str(r.text)[:1500],
                }
            )

    return pd.DataFrame(rows, columns=["evidence_id", "product_id", "source", "text"]).drop_duplicates(
        "evidence_id"
    ).reset_index(drop=True)


def main() -> None:
    raw_dir = RAW / "sephora"
    PROCESSED.mkdir(parents=True, exist_ok=True)

    products, metadata, product_audit = load_products(raw_dir / "product_info.csv")
    reviews, review_audit = load_reviews(raw_dir, set(products["product_id"]))
    chunks = build_chunks(products, metadata, reviews)

    # pyarrow is included in the rag optional dependencies in pyproject.toml.
    products.to_parquet(PROCESSED / "products.parquet", index=False)
    chunks.to_parquet(PROCESSED / "chunks.parquet", index=False)

    audit = {
        **product_audit,
        **review_audit,
        "chunk_rows": len(chunks),
        "chunk_products": int(chunks["product_id"].nunique()),
        "chunk_source_counts": {k: int(v) for k, v in chunks["source"].value_counts().to_dict().items()},
        "price_min": float(products["price_usd"].min()),
        "price_max": float(products["price_usd"].max()),
    }
    (PROCESSED / "cleaning_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"products={len(products):,}")
    print(f"reviews={len(reviews):,}")
    print(f"chunks={len(chunks):,}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
