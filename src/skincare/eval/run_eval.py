"""Main evaluation entry point -- owned by member C; produces the headline results table
for the report.

C's first day (no model, no GPU, no API key):
    python -m skincare.eval.run_eval --self-test
Once a held-out split exists:
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
    """Needs no model at all: checks the evaluation pipeline against samples whose scores
    are already known. This is the command C can run on day one."""
    rows = [json.loads(l) for l in open(path)]
    bad = 0
    print(f"self-testing {len(rows)} samples\n")
    for c in rows:
        got = reward_breakdown(c["completion"], **c["ctx"])
        errs = [f"{k}={got[k]:.2f} expected [{lo},{hi}]"
                for k, (lo, hi) in c["expect"].items() if not lo <= got[k] <= hi]
        bad += bool(errs)
        print(f"  {'OK ' if not errs else 'BAD'} {c['case']:24s} {c['why']}")
        for e in errs:
            print(f"       {e}")
    verdict = ("evaluation pipeline is correct" if not bad
               else f"{bad} sample(s) out of range -- the evaluator itself is broken")
    print(f"\n{verdict}")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="self-test with no model (use this on day one)")
    ap.add_argument("--split", help="held-out split, jsonl")
    ap.add_argument("--variants", nargs="+",
                    help="defaults to every variant available in the manifest")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="reports/llm_eval.json")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any variant already scored in --out and redo everything")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.split:
        raise SystemExit("either --split or --self-test is required")

    rows = [json.loads(l) for l in open(args.split)][: args.limit]
    variants = args.variants or available_variants()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Each variant costs a full model load plus one generation per row, so a run that
    # only writes at the end loses every completed variant when the session drops.
    # Persist after each one and skip what is already scored.
    results = {}
    if out.exists() and not args.fresh:
        results = json.load(open(out))
        done = [v for v in variants if v in results]
        if done:
            print(f"resuming: {', '.join(done)} already scored in {out}")

    def flush():
        json.dump(results, open(out, "w"), indent=2)
        out.with_suffix(".md").write_text(markdown_table(results) + "\n")

    todo = [v for v in variants if v not in results]
    print(f"evaluating {len(rows)} rows; variants to run: {todo or '(none -- all cached)'}\n")

    for v in todo:
        try:
            gen = fixture_generator() if v == "fixture" else get_generator(v)
        except Exception as e:
            print(f"  skipping {v}: {e}")
            continue
        results[v] = score_rows(gen, rows)
        flush()
        print(f"  {v:8s} total={results[v]['total']:.3f}   (saved)")

    flush()
    print("\n" + markdown_table({v: results[v] for v in variants if v in results}))
    print(f"\nsaved {out} and the matching .md")


if __name__ == "__main__":
    main()
