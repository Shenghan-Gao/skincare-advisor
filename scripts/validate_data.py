"""Data handoff acceptance check -- member A runs this before delivering anything; if it
does not pass, the delivery does not count.

    python scripts/validate_data.py vision
    python scripts/validate_data.py products
    python scripts/validate_data.py chunks
    python scripts/validate_data.py rules
    python scripts/validate_data.py all

What it checks is the "contract" (column names, types, uniqueness, referential integrity),
not whether the data is any good.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pandas as pd  # noqa: E402
from skincare.config import CONCERNS, KNOWLEDGE, PROCESSED, SKIN_TYPES  # noqa: E402

OK, BAD, WARN = "  [ok]", "  [FAIL]", "  [warn]"
EVIDENCE_RE = re.compile(r"^[^:]+:(desc|rev|ing):\d+$")


def _fail(msg):
    print(BAD, msg)
    return False


def validate_vision() -> bool:
    print("== vision label tables ==")
    ok = True
    frames = {}
    for split in ["train", "val", "test"]:
        p = PROCESSED / f"vision_{split}.csv"
        if not p.exists():
            ok = _fail(f"missing file {p}"); continue
        df = pd.read_csv(p)
        frames[split] = df
        need = ["filepath", "skin_type"] + CONCERNS
        missing = [c for c in need if c not in df.columns]
        if missing:
            ok = _fail(f"{split}: missing columns {missing}"); continue

        bad_type = set(df["skin_type"].dropna().unique()) - set(SKIN_TYPES)
        if bad_type:
            ok = _fail(f"{split}: invalid skin_type {bad_type}; must be one of {SKIN_TYPES}")
        for c in CONCERNS:
            vals = set(pd.to_numeric(df[c], errors="coerce").dropna().unique())
            if not vals <= {0, 1}:
                ok = _fail(f"{split}: column {c} must be 0/1, found {sorted(vals)[:5]}")
        n_missing = sum(1 for f in df["filepath"] if not (ROOT / str(f)).exists())
        if n_missing:
            ok = _fail(f"{split}: {n_missing}/{len(df)} filepath entries point at no file")
        else:
            print(OK, f"{split}: {len(df)} rows, all columns present, every image path exists")

    if len(frames) == 3:
        seen, overlap = {}, 0
        for split, df in frames.items():
            for f in df["filepath"]:
                if f in seen and seen[f] != split:
                    overlap += 1
                seen[f] = split
        if overlap:
            ok = _fail(f"{overlap} files appear in more than one split -- data leakage, "
                       "validation metrics will be inflated")
        else:
            print(OK, "no file appears in more than one split")

        train = frames["train"]
        print(OK, f"train skin-type distribution: {train['skin_type'].value_counts().to_dict()}")
        for c in CONCERNS:
            pos = int(pd.to_numeric(train[c], errors="coerce").fillna(0).sum())
            if pos < 20:
                print(WARN, f"concern '{c}' has only {pos} positive samples in train -- "
                            "too few, consider collecting more data")
    if not (PROCESSED / "class_distribution.md").exists():
        print(WARN, "missing class_distribution.md (needed for the report)")
    return ok


def validate_products() -> bool:
    print("== product table ==")
    p = PROCESSED / "products.parquet"
    if not p.exists():
        return _fail(f"missing file {p}")
    df = pd.read_parquet(p)
    need = ["product_id", "name", "brand", "category", "price_usd", "rating", "ingredients"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return _fail(f"missing columns {missing}")
    ok = True
    if df["product_id"].duplicated().any():
        ok = _fail(f"product_id has {int(df['product_id'].duplicated().sum())} duplicate(s)")
    if df["ingredients"].isna().any() or (df["ingredients"].apply(len) == 0).any():
        ok = _fail("some products have no ingredients -- they should be dropped in cleaning")
    sample = df["ingredients"].iloc[0]
    if not isinstance(sample, (list, tuple)) and not hasattr(sample, "__len__"):
        ok = _fail(f"ingredients must be a list, got {type(sample).__name__}")
    else:
        joined = " ".join(map(str, sample))
        if joined != joined.lower():
            ok = _fail("ingredients must be entirely lowercase")
        if any(len(str(i)) > 60 for i in sample):
            print(WARN, "an ingredient is longer than 60 characters -- likely an uncut sentence")
    if (pd.to_numeric(df["price_usd"], errors="coerce") <= 0).any():
        ok = _fail("price_usd contains values <= 0")
    if ok:
        print(OK, f"{len(df)} products, ids unique, ingredients parsed")
        print(OK, f"price range ${df['price_usd'].min():.2f} – ${df['price_usd'].max():.2f}")
    return ok


def validate_chunks() -> bool:
    print("== retrieval chunk table ==")
    p = PROCESSED / "chunks.parquet"
    if not p.exists():
        return _fail(f"missing file {p}")
    df = pd.read_parquet(p)
    need = ["evidence_id", "product_id", "source", "text"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return _fail(f"missing columns {missing}")
    ok = True
    if df["evidence_id"].duplicated().any():
        ok = _fail(f"evidence_id has {int(df['evidence_id'].duplicated().sum())} duplicate(s) "
                   "-- the reward function would score citations wrongly, this must be fixed")
    bad_fmt = df[~df["evidence_id"].astype(str).str.match(EVIDENCE_RE)]
    if len(bad_fmt):
        ok = _fail(f"{len(bad_fmt)} evidence_id(s) are malformed (expected e.g. P1001:rev:3), "
                   f"first one: {bad_fmt['evidence_id'].iloc[0]}")
    bad_src = set(df["source"].unique()) - {"description", "review", "ingredient"}
    if bad_src:
        ok = _fail(f"invalid source {bad_src}")
    pp = PROCESSED / "products.parquet"
    if pp.exists():
        valid = set(pd.read_parquet(pp)["product_id"])
        orphan = int((~df["product_id"].isin(valid)).sum())
        if orphan:
            ok = _fail(f"{orphan} chunks have a product_id that is not in the product table "
                       "(referential integrity)")
        else:
            print(OK, "product_id references are complete")
    if (df["text"].astype(str).str.len() < 20).any():
        print(WARN, "some chunks are under 20 characters -- of essentially no retrieval value")
    per = df.groupby("product_id").size()
    if len(per) and per.max() > 25:
        print(WARN, f"one product has as many as {per.max()} chunks -- over the cap of 20, "
                    "it will crowd out the retrieval results")
    if ok:
        print(OK, f"{len(df)} chunks covering {df['product_id'].nunique()} products")
        print(OK, f"source distribution: {df['source'].value_counts().to_dict()}")
    return ok


def validate_rules() -> bool:
    print("== ingredient rule table ==")
    p = KNOWLEDGE / "ingredient_rules.json"
    if not p.exists():
        return _fail(f"missing file {p}")
    rules = json.load(open(p))
    ok = True
    for key in ["concern_to_ingredients", "pregnancy_unsafe", "common_irritants"]:
        if key not in rules:
            ok = _fail(f"missing field {key}")
    c2i = rules.get("concern_to_ingredients", {})
    for c in CONCERNS:
        n = len(c2i.get(c, []))
        if n == 0:
            ok = _fail(f"concern '{c}' has no ingredient mapping at all -- the reward "
                       "function will score it 0 forever")
        elif n < 8:
            print(WARN, f"'{c}' has only {n} ingredients, the target is 8–12")
        else:
            print(OK, f"'{c}': {n} ingredients")
    for k in ["pregnancy_unsafe", "common_irritants"]:
        if len(rules.get(k, [])) < 5:
            print(WARN, f"{k} has only {len(rules.get(k, []))} entries, consider adding more")
    if not (KNOWLEDGE / "sources.md").exists():
        print(WARN, "missing sources.md (the source of each rule, cited in the report)")
    return ok


MODES = {"vision": validate_vision, "products": validate_products,
         "chunks": validate_chunks, "rules": validate_rules}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "all":
        results = {k: fn() for k, fn in MODES.items()}
        print("\n== summary ==")
        for k, v in results.items():
            print(f"  {k:10s} {'PASS' if v else 'FAIL'}")
        sys.exit(0 if all(results.values()) else 1)
    if mode not in MODES:
        print(__doc__); sys.exit(1)
    passed = MODES[mode]()
    print(f"\n{'PASS -- ready to deliver' if passed else 'FAIL -- fix it before delivering'}\n")
    sys.exit(0 if passed else 1)
