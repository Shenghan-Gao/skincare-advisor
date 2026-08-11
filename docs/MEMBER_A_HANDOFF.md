# Member A Data + Safety Handoff

This note records the **actual datasets inspected**, the cleaning decisions, the files handed to downstream teammates, and one training-code compatibility issue that must be resolved before CNN training.

## 1. What Member A owns

Member A owns the data-preparation layer (`src/skincare/rag/ingest.py`, `data/knowledge/*`, `data/processed/*`) and the deterministic safety layer (`src/skincare/safety/guard.py`). The frozen application schemas, label space, LLM prompt template, and training code are not modified here.

## 2. Sephora product/review data

### Raw source inspected

**Reproducible source:**

- Kaggle dataset: **Sephora Products and Skincare Reviews**
- Dataset owner: **Nady Inky**
- Kaggle dataset slug: `nadyinky/sephora-products-and-skincare-reviews`
- Kaggle data card states that the data were collected with a Python scraper in **March 2023**.
- The raw archive used by Member A contains exactly six files: `product_info.csv` plus five `reviews_*.csv` shards.
- Raw archive SHA-256 used for this handoff: `da149ba6114abbe9c197fe7e0b072bd0e234f64fe956a423c5a2b12cd6a829e6`.

To reproduce locally, download that Kaggle dataset and place the raw files here:

```text
data/raw/sephora/product_info.csv
data/raw/sephora/reviews_0-250.csv
data/raw/sephora/reviews_250-500.csv
data/raw/sephora/reviews_500-750.csv
data/raw/sephora/reviews_750-1250.csv
data/raw/sephora/reviews_1250-end.csv
```

Then run:

```bash
python -m skincare.rag.ingest
python scripts/validate_data.py products
python scripts/validate_data.py chunks
```

**CSV / Parquet provenance:** `products_clean.csv` and `chunks_clean.csv` in the handoff package are CSV exports from the same Member A cleaning pipeline represented by the current `src/skincare/rag/ingest.py`. They were provided as easy-to-inspect interchange copies because the handoff environment did not have a Parquet engine. The formal downstream artifacts are `products.parquet` and `chunks.parquet`; serializing the cleaned CSV tables to Parquet does not change the logical records. The validated handoff counts are **2,282 products** and **43,089 chunks**.

`Sephora Products and Skincare Reviews` contains:

- `product_info.csv`: **8,494** products total.
- Skincare rows before ingredient/price cleaning: **2,420**.
- Five `reviews_*.csv` shards: **1,094,411** review rows in total.

### Product cleaning

`src/skincare/rag/ingest.py`:

1. Keeps rows whose `primary_category` contains `Skincare`.
2. Parses Sephora's stringified ingredient lists.
3. Splits ingredient blocks on comma/newline/semicolon, trims whitespace, lowercases, removes empty or >60-character sentence-like fragments, and deduplicates while preserving order.
4. Drops rows with no usable ingredients.
5. Converts `price_usd` to numeric and drops missing/non-positive prices.
6. Converts `rating` to numeric but leaves missing ratings as null rather than 0.
7. Deduplicates `product_id`, retaining the row with the largest review count.

Observed clean product result from the supplied archive: **2,282 unique skincare products**, including **61 missing ratings**; price range **$3–$495**.

### Review cleaning

The supplied five review shards contain **1,094,411** rows. After limiting to retained product IDs, **1,071,484** rows remain; **1,070,053** meet the >=20-character rule. To keep preprocessing tractable, each shard first retains a generous helpfulness-ranked candidate pool; normalization is then applied globally.

Cleaning includes:

- <=15% non-ASCII character heuristic for English filtering.
- 20-character minimum and 1,500-character maximum.
- Exact normalized-text deduplication.
- Regex removal of e-mail addresses, @handles, and North-American-style phone numbers.
- Final helpfulness-ranked cap of **20 reviews per product**.

Observed clean review result: **38,525 reviews across 2,146 products**.

### RAG chunks

Three grounded source types are generated:

- `description`: product name + brand + category + Sephora highlights. (The supplied product file does not contain a dedicated long-form description column.)
- `ingredient`: cleaned ingredient list.
- `review`: retained review text.

Observed result: **43,089 chunks** = 38,525 review + 2,282 description + 2,282 ingredient chunks, covering all **2,282** retained products.

The frozen evidence ID pattern is preserved exactly:

`{product_id}:{desc|rev|ing}:{index}`

Examples: `P1001:desc:0`, `P1001:ing:0`, `P1001:rev:3`.

## 3. Vision data: what the downloads actually contain

### Dataset A: Facial Skin Condition Dataset

The supplied archive contains only **45 images (15 people x 3 views)** plus `id` and `gender`. It does **not** contain the required skin-type labels or the six frozen concern labels. It is therefore excluded from the main CNN dataset rather than being assigned invented labels.

### Dataset B: Facial Skin Analysis and Type Classification

This is the usable main source:

- **4,093 images** with skin-type folders: oily, dry, combination, normal.
- Two spreadsheets provide detailed 0–5 concern scores for only **200 annotation rows / 188 unique source-image groups**.
- The spreadsheets cover oily/dry/normal only; combination images do not have detailed concern annotations.

Cleaning in `scripts/build_vision_data.py`:

1. Reject unreadable, <100px, or non-RGB images.
2. Remove within-class pHash near-duplicates at Hamming distance <=5.
3. When an almost-identical image is assigned conflicting skin-type labels, remove both copies rather than choosing a label.
4. Normalize Roboflow augmentation filenames back to a common source-image group.
5. Clip documented 0–5 severity values into [0,5] (one supplied value was outside the stated scale).
6. Convert concern severity to binary with a documented threshold: score >=2 => positive; 0–1 => negative.
7. Propagate a spreadsheet annotation only to augmentation variants of that same source image.
8. Re-split 70/15/15 at the **source-image group** level, stratified by skin type and whether concern labels exist. This avoids augmented copies crossing splits.
9. Leave unannotated concern targets **blank/NaN**, never fake them as 0.

Observed audit:

- Source images: **4,093**
- Invalid/small/non-RGB: **0**
- Within-class pHash duplicates removed: **163**
- Cross-class conflicting near-identical images removed: **49** (26 detected pairs)
- Original vendor split had **786 source-image groups crossing train/valid/test**, so it was not reused.
- Final main dataset: **3,881 images**
- Train: **2,718**; validation: **584**; test: **579**
- Final skin-type counts: normal 1,312; dry 1,146; oily 1,094; combination 329
- Concern-labeled images after safe propagation: **495**

See `data/processed/class_distribution.md` for split-level counts.

### Dataset C: Skin Defects (Acne / Bags / Redness)

The archive contains **90 images / 30 people**, three views per person: 10 acne, 10 bags, 10 redness.

- `bags` is dropped because it is outside the frozen concern space.
- **60 acne/redness images** are retained in `vision_concern_aux.csv` as auxiliary positive labels.
- They are **not merged into the main CSVs** because this source has no skin-type target. B may use them later only if the training loader/loss explicitly supports a missing skin-type target.

## 4. IMPORTANT: partial-label compatibility issue for B

The main 3,881-image dataset has reliable skin-type labels for every row, but detailed concern labels exist for only 495 images. This is a property of the supplied source, not a cleaning failure.

Current `src/skincare/vision/data.py` converts missing concern cells to `float(NaN)`, and current `multitask_loss` in `src/skincare/vision/model.py` computes BCE over the whole tensor. That will make the concern loss become NaN.

**B needs to mask missing concern labels before training.** Member A did not edit B-owned training files. A minimal implementation is:

```python
# data.py: use -1 as the missing-label sentinel
vals = [float(row[c]) if pd.notna(row[c]) else -1.0 for c in CONCERNS]
y_concern = torch.tensor(vals, dtype=torch.float32)
```

```python
# model.py: BCE only where a concern label exists
mask = y_concern >= 0
safe_target = y_concern.clamp_min(0)
raw = nn.functional.binary_cross_entropy_with_logits(
    concern_logits, safe_target, reduction="none"
)
bce = (raw * mask).sum() / mask.sum().clamp_min(1)
```

This preserves all 3,881 images for skin-type training while learning concern heads only from actually annotated targets.

## 5. Ingredient rules and safety

`data/knowledge/ingredient_rules.json` now contains 8–12 retrieval/reward cues for every frozen concern plus pregnancy hard-block strings, pregnancy caution strings, irritant cautions, comedogenic heuristics, and skin-type caution mappings. `data/knowledge/sources.md` records the medical/dermatology sources and limitations.

`src/skincare/safety/guard.py`:

- hard-removes recommendations whose exposed key ingredients match pregnancy-unsafe rules when `pregnant=True`;
- hard-removes user-specified avoid ingredients;
- adds softer pregnancy, irritant, and possible pore-clogging cautions;
- adds a cosmetic-only disclaimer;
- refuses obvious diagnosis/prescription requests at the final deterministic layer.

Known limitation: `Recommendation` exposes `key_ingredients`, not necessarily the complete formulation. Safety matching is therefore a conservative final guard, not a guarantee of clinical suitability.

## 6. Files to hand off

Member A -> B/C:

- `vision_train.csv`
- `vision_val.csv`
- `vision_test.csv`
- `class_distribution.md`
- optional `vision_concern_aux.csv`

Member A -> B/Anna:

- `products.parquet`
- `chunks.parquet`
- `cleaning_audit.json`

Member A -> Anna:

- `data/knowledge/ingredient_rules.json`
- `data/knowledge/sources.md`
- `src/skincare/safety/guard.py`

## 7. Reproduction commands

After placing raw data under the paths documented by each script:

```bash
uv pip install -e ".[dev,vision,rag]"
python scripts/build_vision_data.py
python -m skincare.rag.ingest
python scripts/validate_data.py vision
python scripts/validate_data.py products
python scripts/validate_data.py chunks
python scripts/validate_data.py rules
pytest -q
```

`pyarrow` was added to the `rag` extra because pandas needs a Parquet engine, and `openpyxl` was added to the `vision` extra because the supplied concern labels are Excel files.
