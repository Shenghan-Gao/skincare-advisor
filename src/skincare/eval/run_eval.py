"""评估总入口 —— 组员 C 拥有,产出报告里的主结果表。

C 的第一天(无模型、无 GPU、无 API key):
    python -m skincare.eval.run_eval --self-test
有了留出集之后:
    python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl
    python -m skincare.eval.run_eval --split ... --variants base sft grpo
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from skincare.eval.harness import available_variants, fixture_generator, get_generator
from skincare.llm.rewards import reward_breakdown

COMPONENTS = ["format", "ingredient_match", "grounding", "product_validity", "safety", "total"]


def score_rows(gen_fn, rows: list[dict]) -> dict:
    agg = defaultdict(list)
    for row in rows:
        completion = gen_fn(row["prompt"])
        b = reward_breakdown(
            completion,
            concerns=row.get("concerns"),
            evidence_ids=row.get("evidence_ids"),
            product_ids=row.get("product_ids"),
            pregnant=row.get("pregnant", False),
            avoid=row.get("avoid", []),
        )
        for k, v in b.items():
            agg[k].append(v)
    return {k: sum(v) / len(v) for k, v in agg.items()}


def markdown_table(results: dict) -> str:
    heads = ["variant"] + COMPONENTS
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "---|" * len(heads)]
    for name, r in results.items():
        lines.append("| " + " | ".join([name] + [f"{r.get(c, 0):.3f}" for c in COMPONENTS]) + " |")
    return "\n".join(lines)


def self_test(path="fixtures/eval_samples.jsonl") -> bool:
    """不需要任何模型:用已知答案的样本验证评估链路正确。
    这是 C 第一天就能跑的命令。"""
    rows = [json.loads(l) for l in open(path)]
    bad = 0
    print(f"自检 {len(rows)} 个样本\n")
    for c in rows:
        got = reward_breakdown(c["completion"], **c["ctx"])
        errs = [f"{k}={got[k]:.2f} 期望[{lo},{hi}]"
                for k, (lo, hi) in c["expect"].items() if not lo <= got[k] <= hi]
        bad += bool(errs)
        print(f"  {'OK ' if not errs else 'BAD'} {c['case']:24s} {c['why']}")
        for e in errs:
            print(f"       {e}")
    print(f"\n{'评估链路正确' if not bad else f'{bad} 个样本不符,评估器有问题'}")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="无模型自检(第一天用这个)")
    ap.add_argument("--split", help="留出集 jsonl")
    ap.add_argument("--variants", nargs="+", help="默认评 manifest 里所有可用的")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="reports/llm_eval.json")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.split:
        raise SystemExit("需要 --split 或 --self-test")

    rows = [json.loads(l) for l in open(args.split)][: args.limit]
    variants = args.variants or available_variants()
    print(f"评估 {len(rows)} 条,档位:{variants}\n")

    results = {}
    for v in variants:
        try:
            gen = fixture_generator() if v == "fixture" else get_generator(v)
        except Exception as e:
            print(f"  跳过 {v}:{e}")
            continue
        results[v] = score_rows(gen, rows)
        print(f"  {v:8s} total={results[v]['total']:.3f}")

    table = markdown_table(results)
    print("\n" + table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    Path(args.out).with_suffix(".md").write_text(table + "\n")
    print(f"\n已保存 {args.out} 与 .md")


if __name__ == "__main__":
    main()
