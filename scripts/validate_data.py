"""数据交付验收 —— 组员 A 交东西之前先跑这个,跑不过不算交付。

    python scripts/validate_data.py vision
    python scripts/validate_data.py products
    python scripts/validate_data.py chunks
    python scripts/validate_data.py rules
    python scripts/validate_data.py all

它查的是"契约"(列名、类型、唯一性、引用完整性),不是"数据好不好"。
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
    print("== 视觉标签表 ==")
    ok = True
    frames = {}
    for split in ["train", "val", "test"]:
        p = PROCESSED / f"vision_{split}.csv"
        if not p.exists():
            ok = _fail(f"缺文件 {p}"); continue
        df = pd.read_csv(p)
        frames[split] = df
        need = ["filepath", "skin_type"] + CONCERNS
        missing = [c for c in need if c not in df.columns]
        if missing:
            ok = _fail(f"{split}: 缺列 {missing}"); continue

        bad_type = set(df["skin_type"].dropna().unique()) - set(SKIN_TYPES)
        if bad_type:
            ok = _fail(f"{split}: 非法 skin_type {bad_type};只能是 {SKIN_TYPES}")
        for c in CONCERNS:
            vals = set(pd.to_numeric(df[c], errors="coerce").dropna().unique())
            if not vals <= {0, 1}:
                ok = _fail(f"{split}: 列 {c} 必须是 0/1,发现 {sorted(vals)[:5]}")
        n_missing = sum(1 for f in df["filepath"] if not (ROOT / str(f)).exists())
        if n_missing:
            ok = _fail(f"{split}: {n_missing}/{len(df)} 个 filepath 找不到文件")
        else:
            print(OK, f"{split}: {len(df)} 行,列齐全,图片路径都存在")

    if len(frames) == 3:
        seen, overlap = {}, 0
        for split, df in frames.items():
            for f in df["filepath"]:
                if f in seen and seen[f] != split:
                    overlap += 1
                seen[f] = split
        if overlap:
            ok = _fail(f"{overlap} 个文件同时出现在多个 split —— 数据泄漏,验证集指标会虚高")
        else:
            print(OK, "split 之间无重复文件")

        train = frames["train"]
        print(OK, f"train 肤质分布: {train['skin_type'].value_counts().to_dict()}")
        for c in CONCERNS:
            pos = int(pd.to_numeric(train[c], errors="coerce").fillna(0).sum())
            if pos < 20:
                print(WARN, f"关注点 '{c}' 训练集只有 {pos} 个正样本 —— 太少,考虑补数据")
    if not (PROCESSED / "class_distribution.md").exists():
        print(WARN, "缺 class_distribution.md(报告要用)")
    return ok


def validate_products() -> bool:
    print("== 产品表 ==")
    p = PROCESSED / "products.parquet"
    if not p.exists():
        return _fail(f"缺文件 {p}")
    df = pd.read_parquet(p)
    need = ["product_id", "name", "brand", "category", "price_usd", "rating", "ingredients"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return _fail(f"缺列 {missing}")
    ok = True
    if df["product_id"].duplicated().any():
        ok = _fail(f"product_id 有 {int(df['product_id'].duplicated().sum())} 个重复")
    if df["ingredients"].isna().any() or (df["ingredients"].apply(len) == 0).any():
        ok = _fail("有产品成分为空 —— 应在清洗时丢弃")
    sample = df["ingredients"].iloc[0]
    if not isinstance(sample, (list, tuple)) and not hasattr(sample, "__len__"):
        ok = _fail(f"ingredients 必须是 list,现在是 {type(sample).__name__}")
    else:
        joined = " ".join(map(str, sample))
        if joined != joined.lower():
            ok = _fail("ingredients 必须全小写")
        if any(len(str(i)) > 60 for i in sample):
            print(WARN, "有成分长度 > 60 字符,可能是没切干净的句子")
    if (pd.to_numeric(df["price_usd"], errors="coerce") <= 0).any():
        ok = _fail("price_usd 有 <= 0 的值")
    if ok:
        print(OK, f"{len(df)} 个产品,id 唯一,成分已解析")
        print(OK, f"价格区间 ${df['price_usd'].min():.2f} – ${df['price_usd'].max():.2f}")
    return ok


def validate_chunks() -> bool:
    print("== 检索片段表 ==")
    p = PROCESSED / "chunks.parquet"
    if not p.exists():
        return _fail(f"缺文件 {p}")
    df = pd.read_parquet(p)
    need = ["evidence_id", "product_id", "source", "text"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return _fail(f"缺列 {missing}")
    ok = True
    if df["evidence_id"].duplicated().any():
        ok = _fail(f"evidence_id 有 {int(df['evidence_id'].duplicated().sum())} 个重复 "
                   "—— 奖励函数会误判,必须修")
    bad_fmt = df[~df["evidence_id"].astype(str).str.match(EVIDENCE_RE)]
    if len(bad_fmt):
        ok = _fail(f"{len(bad_fmt)} 个 evidence_id 格式不对(应为 P1001:rev:3 这种),"
                   f"例:{bad_fmt['evidence_id'].iloc[0]}")
    bad_src = set(df["source"].unique()) - {"description", "review", "ingredient"}
    if bad_src:
        ok = _fail(f"非法 source {bad_src}")
    pp = PROCESSED / "products.parquet"
    if pp.exists():
        valid = set(pd.read_parquet(pp)["product_id"])
        orphan = int((~df["product_id"].isin(valid)).sum())
        if orphan:
            ok = _fail(f"{orphan} 条片段的 product_id 不在产品表里(引用完整性)")
        else:
            print(OK, "product_id 引用完整")
    if (df["text"].astype(str).str.len() < 20).any():
        print(WARN, "有片段短于 20 字符,基本没检索价值")
    per = df.groupby("product_id").size()
    if len(per) and per.max() > 25:
        print(WARN, f"单个产品最多有 {per.max()} 条片段 —— 超过 20 的上限,会霸占检索结果")
    if ok:
        print(OK, f"{len(df)} 条片段,覆盖 {df['product_id'].nunique()} 个产品")
        print(OK, f"来源分布: {df['source'].value_counts().to_dict()}")
    return ok


def validate_rules() -> bool:
    print("== 成分规则表 ==")
    p = KNOWLEDGE / "ingredient_rules.json"
    if not p.exists():
        return _fail(f"缺文件 {p}")
    rules = json.load(open(p))
    ok = True
    for key in ["concern_to_ingredients", "pregnancy_unsafe", "common_irritants"]:
        if key not in rules:
            ok = _fail(f"缺字段 {key}")
    c2i = rules.get("concern_to_ingredients", {})
    for c in CONCERNS:
        n = len(c2i.get(c, []))
        if n == 0:
            ok = _fail(f"关注点 '{c}' 没有任何成分映射 —— 奖励函数对它永远给 0 分")
        elif n < 8:
            print(WARN, f"'{c}' 只有 {n} 个成分,目标 8–12 个")
        else:
            print(OK, f"'{c}': {n} 个成分")
    for k in ["pregnancy_unsafe", "common_irritants"]:
        if len(rules.get(k, [])) < 5:
            print(WARN, f"{k} 只有 {len(rules.get(k, []))} 条,建议补充")
    if not (KNOWLEDGE / "sources.md").exists():
        print(WARN, "缺 sources.md(每条规则的出处,报告要引用)")
    return ok


MODES = {"vision": validate_vision, "products": validate_products,
         "chunks": validate_chunks, "rules": validate_rules}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "all":
        results = {k: fn() for k, fn in MODES.items()}
        print("\n== 汇总 ==")
        for k, v in results.items():
            print(f"  {k:10s} {'PASS' if v else 'FAIL'}")
        sys.exit(0 if all(results.values()) else 1)
    if mode not in MODES:
        print(__doc__); sys.exit(1)
    passed = MODES[mode]()
    print(f"\n{'PASS —— 可以交付' if passed else 'FAIL —— 修好再交'}\n")
    sys.exit(0 if passed else 1)
