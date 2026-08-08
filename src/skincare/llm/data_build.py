"""构造 SFT / RL 数据集 —— 支柱二的入口(Anna 独占)。

    # 第一天就能跑:用合成目录,不需要 A 的数据、不需要 API key
    python -m skincare.llm.data_build --n 50 --mock-retrieval --dry-teacher

    # 真正造 SFT 数据(要 OPENAI_API_KEY)
    python -m skincare.llm.data_build --n 800 --mode sft

    # 造 RL 数据(不需要目标答案,也就不花钱)
    python -m skincare.llm.data_build --n 600 --mode rl

设计要点:
1. **训练与服务共用同一个 prompt 模板**(prompts.py)—— 漂移会导致静默退化。
2. **教师输出必须过滤**:低于阈值的直接丢。不过滤等于让模型学噪声,
   这是 SFT 有没有效果的分水岭。
3. **断点续跑**:每条结果即时写入 cache,中途挂了重跑不会重复烧钱。
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from skincare.config import PROCESSED, SEED
from skincare.llm.prompts import SYSTEM, build_user_prompt
from skincare.llm.rewards import reward_breakdown

random.seed(SEED)
SKIN_TYPES = ["oily", "dry", "combination", "normal"]
CONCERNS = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]
PREFS = ["fragrance-free", "vegan", "gentle", "lightweight", "non-comedogenic"]


# ----------------------------------------------------------------- 采样 ---
def sample_profile() -> tuple[dict, dict]:
    concerns = random.sample(CONCERNS, k=random.randint(1, 3))
    skin_type = random.choice(SKIN_TYPES)
    profile = {
        "query": f"I have {skin_type} skin and want help with "
                 + ", ".join(c.replace("_", " ") for c in concerns),
        "budget_usd": random.choice([None, 25, 40, 60, 100]),
        "preferences": random.sample(PREFS, k=random.randint(0, 2)),
        "avoid_ingredients": random.choice([[], [], ["fragrance"], ["essential oil"]]),
        "pregnant": random.random() < 0.15,
    }
    analysis = {
        "skin_type": skin_type,
        "skin_type_confidence": round(random.uniform(0.7, 0.95), 2),
        "concerns": [{"concern": c, "score": round(random.uniform(0.6, 0.95), 2)}
                     if c in concerns else {"concern": c, "score": round(random.uniform(0.0, 0.3), 2)}
                     for c in CONCERNS],
    }
    return profile, analysis


def get_retriever(mock: bool):
    if mock:
        from skincare.rag.mock_retrieval import MockRetriever
        return MockRetriever()
    from skincare.rag.retrieve import Retriever
    return Retriever()


def build_rows(n: int, mock: bool, top_k: int = 3) -> list[dict]:
    """一行 = 一个训练样本的 prompt + 奖励上下文(GRPO 只需要这些)。"""
    from app.schemas import SkinAnalysis, UserProfile

    retriever = get_retriever(mock)
    rows = []
    for _ in range(n):
        profile, analysis = sample_profile()
        res = retriever.search(UserProfile(**profile), SkinAnalysis(**analysis), top_k=top_k)
        ev = [e.model_dump() for e in res.evidence][:12]
        if not ev:
            continue
        active = [c["concern"] for c in analysis["concerns"] if c["score"] >= 0.5]
        rows.append({
            "prompt": build_user_prompt(profile, analysis, ev),
            # ---- 奖励上下文,rewards.py 在 GRPO 中读这些 ----
            "concerns": active,
            "evidence_ids": [e["evidence_id"] for e in ev],
            "product_ids": [p.product_id for p in res.products],
            "pregnant": profile["pregnant"],
            "avoid": profile["avoid_ingredients"],
        })
    return rows


# --------------------------------------------------------------- 教师蒸馏 ---
def _key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


def teacher_answer(row: dict, client, model: str, retries: int = 2,
                   stats: dict | None = None) -> str | None:
    """让强模型生成 SFT 目标答案。返回 None 表示这条应被丢弃。

    stats 用于累计 token 用量 —— 这样上量前你能按真实用量估成本,而不是靠猜。
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["prompt"]}]
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.2 + 0.2 * attempt,      # 重试时略微提高多样性
                response_format={"type": "json_object"})
            if stats is not None and getattr(r, "usage", None):
                stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + r.usage.prompt_tokens
                stats["completion_tokens"] = stats.get("completion_tokens", 0) + r.usage.completion_tokens
                stats["calls"] = stats.get("calls", 0) + 1
            return r.choices[0].message.content
        except Exception as e:
            if attempt == retries:
                print(f"    教师调用失败: {e}", file=sys.stderr)
                return None
    return None


def fake_teacher(row: dict) -> str:
    """离线假教师:直接照抄检索到的第一个产品,用于无 API key 时跑通链路。
    它的答案质量不高但格式正确,足以验证过滤与训练管线。"""
    pid = row["product_ids"][0] if row["product_ids"] else "P001"
    ev = [e for e in row["evidence_ids"] if e.startswith(pid)][:2]
    return json.dumps({
        "recommendations": [{
            "product_id": pid, "name": f"Product {pid}", "brand": "MockBrand",
            "reason": f"Matches your {', '.join(row['concerns']) or 'skin'} concerns [{ev[0] if ev else ''}].",
            "key_ingredients": ["niacinamide"],
            "cited_evidence": ev, "matched_concerns": row["concerns"],
        }],
        "routine_note": "Apply in the morning and finish with SPF.",
        "disclaimer": "Cosmetic suggestions only; not medical advice. See a dermatologist "
                      "for persistent or worsening skin problems.",
    }, ensure_ascii=False)


DIAGNOSIS = {
    "format": "教师没输出合法 JSON —— 检查 prompts.py 的 SYSTEM 是否够明确,"
              "以及是否用了 response_format={'type':'json_object'}",
    "ingredient_match": "推荐成分对不上关注点 —— 多半是 ingredient_rules.json 覆盖太少"
                        "(组员 A 的活),先把每个关注点补到 8-12 个成分再重跑",
    "grounding": "教师没引用给定的 evidence_id —— 在 prompt 里把'必须引用 evidence_id'"
                 "写得更硬,或在示例里给出引用格式",
    "product_validity": "教师在编产品 —— 强调只能从 evidence 里出现过的 product_id 中选",
    "safety": "缺免责声明或推荐了禁忌成分 —— 检查 SYSTEM 里的免责要求",
}


def distil(rows: list[dict], model: str, threshold: float, cache_path: Path,
           dry: bool, inspect: int = 0) -> list[dict]:
    """蒸馏 + 按奖励过滤 + 断点续跑 + 质量诊断。"""
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in open(cache_path):
            d = json.loads(line)
            cache[d["k"]] = d["completion"]
        print(f"  从 cache 恢复 {len(cache)} 条")

    client = None
    if not dry:
        from openai import OpenAI
        client = OpenAI()

    kept, dropped_rows, cached_hits = [], [], 0
    usage: dict = {}
    all_breakdowns = []
    with open(cache_path, "a") as cf:
        for i, row in enumerate(rows, 1):
            k = _key(row["prompt"])
            if k in cache:
                completion = cache[k]
                cached_hits += 1
            else:
                completion = (fake_teacher(row) if dry
                              else teacher_answer(row, client, model, stats=usage))
                if completion is None:
                    dropped_rows.append((row, None))
                    continue
                cf.write(json.dumps({"k": k, "completion": completion}, ensure_ascii=False) + "\n")
                cf.flush()

            ctx = dict(concerns=row["concerns"], evidence_ids=row["evidence_ids"],
                       product_ids=row["product_ids"], pregnant=row["pregnant"],
                       avoid=row["avoid"])
            b = reward_breakdown(completion, **ctx)
            all_breakdowns.append(b)
            if b["total"] < threshold:
                dropped_rows.append(({**row, "_completion": completion}, b))
                continue
            kept.append({**row, "completion": completion,
                         "teacher_score": round(b["total"], 3)})

            if i % 25 == 0:
                print(f"  {i}/{len(rows)}  保留 {len(kept)}  丢弃 {len(dropped_rows)}")

    total_n = len(kept) + len(dropped_rows)
    rate = len(kept) / total_n * 100 if total_n else 0
    print(f"\n{'='*56}")
    print(f"  通过率 {rate:.0f}%  (保留 {len(kept)} / 丢弃 {len(dropped_rows)}"
          f",阈值 {threshold},cache 命中 {cached_hits})")

    # ---- 各奖励分量分布:通过率低时告诉你是哪一项拖后腿 ----
    if all_breakdowns:
        print(f"\n  各分量均值(1.0 满分):")
        comps = ["format", "ingredient_match", "grounding", "product_validity", "safety"]
        means = {c: sum(b[c] for b in all_breakdowns) / len(all_breakdowns) for c in comps}
        for c in comps:
            bar = "█" * int(means[c] * 20)
            print(f"    {c:18s} {means[c]:.2f}  {bar}")
        worst = min(means, key=means.get)
        if means[worst] < 0.7:
            print(f"\n  ⚠️  最弱环节是 {worst}({means[worst]:.2f}):")
            for line in DIAGNOSIS[worst].split(" —— "):
                print(f"      {line}")

    # ---- token 用量与上量成本外推 ----
    if usage.get("calls"):
        pt, ct, n = usage["prompt_tokens"], usage["completion_tokens"], usage["calls"]
        print(f"\n  本次用量:{n} 次调用,输入 {pt:,} tok,输出 {ct:,} tok")
        print(f"  单条均值:输入 {pt//n:,} tok,输出 {ct//n:,} tok")
        print(f"  外推 800 条:输入约 {pt//n*800:,} tok,输出约 {ct//n*800:,} tok")
        print(f"  (按你所用模型的当前单价自行折算)")

    # ---- 人工抽检 ----
    if inspect and kept:
        print(f"\n{'='*56}\n  抽检 {min(inspect, len(kept))} 条保留样本(请肉眼判断质量)\n")
        for r in kept[:inspect]:
            print(f"  --- 教师分 {r['teacher_score']} ---")
            print(f"  {r['completion'][:600]}\n")
    if inspect and dropped_rows:
        shown = [(r, b) for r, b in dropped_rows if b][:2]
        if shown:
            print(f"{'='*56}\n  被丢弃的样本(看看为什么不合格)\n")
            for r, b in shown:
                low = [k for k in ["format","ingredient_match","grounding","product_validity","safety"]
                       if b[k] < 0.5]
                print(f"  --- 总分 {b['total']:.2f},拖后腿的分量:{low} ---")
                print(f"  {r.get('_completion','(教师调用失败,无输出)')[:400]}\n")
    # ---- 机器可读摘要:notebook 用它自动判断要不要继续烧钱 ----
    summary = {
        "kept": len(kept), "dropped": len(dropped_rows), "total": total_n,
        "pass_rate": round(rate / 100, 4), "threshold": threshold,
        "component_means": {c: round(sum(b[c] for b in all_breakdowns) / len(all_breakdowns), 4)
                            for c in ["format", "ingredient_match", "grounding",
                                      "product_validity", "safety"]} if all_breakdowns else {},
        "weakest": (min({c: sum(b[c] for b in all_breakdowns) / len(all_breakdowns)
                         for c in ["format", "ingredient_match", "grounding",
                                   "product_validity", "safety"]}.items(),
                        key=lambda kv: kv[1])[0] if all_breakdowns else None),
        "usage": usage, "dry_teacher": dry,
    }
    (cache_path.parent / "distill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"  摘要已写入 {cache_path.parent / 'distill_summary.json'}")
    print(f"{'='*56}")
    return kept


# -------------------------------------------------------------------- IO ---
def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入 {len(rows):5d} 条 -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--mode", choices=["sft", "rl", "both"], default="both")
    ap.add_argument("--mock-retrieval", action="store_true",
                    help="用合成目录,不需要组员 A 的真实数据")
    ap.add_argument("--dry-teacher", action="store_true",
                    help="用离线假教师,不调 API、不花钱(验证链路用)")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "gpt-4o-mini"))
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="教师答案低于此分直接丢弃 —— SFT 有效性的分水岭")
    ap.add_argument("--inspect", type=int, default=0,
                    help="打印 N 条教师答案供人工检查(小样本试跑时用 3)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--outdir", default=str(PROCESSED))
    args = ap.parse_args()

    out = Path(args.outdir)
    print(f"采样 {args.n} 条画像并检索证据"
          f"({'合成目录' if args.mock_retrieval else '真实索引'})…")
    rows = build_rows(args.n, args.mock_retrieval)
    print(f"  得到 {len(rows)} 条有效样本\n")

    random.shuffle(rows)
    n_test = max(1, int(len(rows) * args.test_frac))
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    if args.mode in ("rl", "both"):
        print("[RL 数据] 不需要目标答案")
        write_jsonl(train_rows, out / "rl.jsonl")
        write_jsonl(test_rows, out / "rl_test.jsonl")   # 组员 C 的评估留出集
        print()

    if args.mode in ("sft", "both"):
        print(f"[SFT 数据] 教师={'离线假教师' if args.dry_teacher else args.model}"
              f",过滤阈值={args.threshold}")
        kept = distil(train_rows, args.model, args.threshold,
                      out / "sft.cache.jsonl", args.dry_teacher, args.inspect)
        write_jsonl(kept, out / "sft.jsonl")

    print("\n下一步:")
    print("  python -m skincare.llm.sft_lora --epochs 2")
    print("  python -m skincare.llm.grpo_train --steps 300")


if __name__ == "__main__":
    main()
