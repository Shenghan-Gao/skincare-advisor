# Sephora Data Reproducibility — Member A

This file is the short reproducibility note for the RAG data handoff.

## Raw dataset

- **Dataset:** Sephora Products and Skincare Reviews
- **Owner:** Nady Inky
- **Kaggle slug:** `nadyinky/sephora-products-and-skincare-reviews`
- **Collection date stated on Kaggle data card:** March 2023
- **Raw archive SHA-256 used by Member A:** `da149ba6114abbe9c197fe7e0b072bd0e234f64fe956a423c5a2b12cd6a829e6`

The archive used in this project contains:

```text
product_info.csv
reviews_0-250.csv
reviews_250-500.csv
reviews_500-750.csv
reviews_750-1250.csv
reviews_1250-end.csv
```

Put them under:

```text
data/raw/sephora/
```

## Reproduce the formal outputs

```bash
python -m skincare.rag.ingest
python scripts/validate_data.py products
python scripts/validate_data.py chunks
```

Expected validated counts for the supplied raw archive:

- **2,282 products**
- **43,089 chunks**

## Why the handoff also contains CSV files

`products_clean.csv` and `chunks_clean.csv` are easy-to-inspect CSV exports of the same cleaned logical tables represented by the current `src/skincare/rag/ingest.py` pipeline. The formal project artifacts are `products.parquet` and `chunks.parquet`. Converting the clean tables from CSV to Parquet changes the storage format, not the logical records.

For the final project, use the validated Parquet files downstream and keep this note plus the raw-source information so another teammate can reproduce them from scratch.
