"""Build the SFT / RL datasets -- the entry point for pillar two (owned by Anna).

    # Runnable on day one: synthetic catalog, needs neither A's data nor an API key
    python -m skincare.llm.data_build --n 50 --mock-retrieval --dry-teacher

    # Build real SFT data (requires OPENAI_API_KEY)
    python -m skincare.llm.data_build --n 800 --mode sft

    # Build RL data (no target answers needed, so it costs nothing)
    python -m skincare.llm.data_build --n 600 --mode rl

Design notes:
1. **Training and serving share one prompt template** (prompts.py) -- drift between the two
   causes silent degradation.
2. **Teacher outputs must be filtered**: anything below the threshold is discarded. Skipping
   the filter means training the model on noise, and this is what decides whether SFT works
   at all.
3. **Resumable**: every result is written to the cache immediately, so a crash mid-run can be
   restarted without paying for the same calls twice.
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


# -------------------------------------------------------------- sampling ---
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
    """One row = one training sample's prompt plus reward context (all GRPO needs)."""
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
            # ---- reward context; rewards.py reads these during GRPO ----
            "concerns": active,
            "evidence_ids": [e["evidence_id"] for e in ev],
            # Every product the model can actually see, not just the top-k summary
            # rows. SYSTEM says "Recommend products ONLY from the evidence", and the
            # evidence block shown to the model spans more products than res.products
            # does -- scoring against the narrower list punished the model for obeying
            # the instruction it was given.
            "product_ids": list(dict.fromkeys(e["product_id"] for e in ev)),
            "pregnant": profile["pregnant"],
            "avoid": profile["avoid_ingredients"],
        })
    return rows


# ---------------------------------------------------- teacher distillation ---
def _key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


def cache_path_for(outdir: Path, model: str, dry: bool) -> Path:
    """One cache file per (SYSTEM prompt, teacher model).

    The per-row key is a hash of the *user* prompt only, so a cache shared across
    SYSTEM revisions would happily serve completions produced by the old SYSTEM --
    you would tune the prompt to lift the pass rate, rerun, and see the identical
    number, with nothing in the output saying why. Putting the fingerprint in the
    filename makes a SYSTEM or model change start a fresh cache instead.

    A legacy sft.cache.jsonl is adopted under the current fingerprint on first use,
    so the calls already paid for are not paid for twice. That adoption happens only
    while no fingerprinted cache exists yet -- otherwise the very first SYSTEM edit
    would reseed the new cache from the old completions, which is the bug this
    function exists to prevent.
    """
    fp = hashlib.sha1(f"{SYSTEM}\x00{'dry' if dry else model}".encode()).hexdigest()[:8]
    path = outdir / f"sft.cache.{fp}.jsonl"
    legacy = outdir / "sft.cache.jsonl"
    if not path.exists() and legacy.exists() and not any(outdir.glob("sft.cache.*.jsonl")):
        path.write_text(legacy.read_text())
        print(f"  adopted legacy cache ({sum(1 for _ in open(legacy))} rows) -> {path.name}")
    return path


def teacher_answer(row: dict, client, model: str, retries: int = 2,
                   stats: dict | None = None) -> str | None:
    """Have the strong model generate an SFT target answer. None means: drop this row.

    stats accumulates token usage, so cost for a full-scale run can be estimated from real
    measurements before scaling up instead of guessed.
    """
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["prompt"]}]
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.2 + 0.2 * attempt,      # nudge diversity up on each retry
                response_format={"type": "json_object"})
            if stats is not None and getattr(r, "usage", None):
                stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + r.usage.prompt_tokens
                stats["completion_tokens"] = stats.get("completion_tokens", 0) + r.usage.completion_tokens
                stats["calls"] = stats.get("calls", 0) + 1
            return r.choices[0].message.content
        except Exception as e:
            if attempt == retries:
                print(f"    teacher call failed: {e}", file=sys.stderr)
                return None
    return None


def fake_teacher(row: dict) -> str:
    """Offline fake teacher: simply copies the first retrieved product, so the pipeline can be
    exercised without an API key. Its answers are not high quality but they are correctly
    formatted, which is enough to validate the filtering and training pipeline."""
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
    "format": "The teacher is not emitting valid JSON —— check that SYSTEM in prompts.py is "
              "explicit enough, and that response_format={'type':'json_object'} is being used",
    "ingredient_match": "Recommended ingredients do not line up with the concerns —— usually "
                        "ingredient_rules.json has too little coverage (teammate A's task); "
                        "bring each concern up to 8-12 ingredients and rerun",
    "grounding": "The teacher is not citing the evidence_id values it was given —— state the "
                 "'you must cite evidence_id' rule more forcefully in the prompt, or show the "
                 "citation format in an example",
    "product_validity": "Recommended product_ids are outside the allowed set —— before "
                        "blaming the teacher, check that row['product_ids'] covers every "
                        "product shown in the evidence block; a narrower list penalises the "
                        "model for obeying the prompt. Only if grounding is also low is the "
                        "teacher genuinely inventing products",
    "safety": "Missing disclaimer, or a contraindicated ingredient was recommended —— check "
              "the disclaimer requirements in SYSTEM",
}


def distil(rows: list[dict], model: str, threshold: float, cache_path: Path,
           dry: bool, inspect: int = 0) -> list[dict]:
    """Distil, filter by reward, resume from cache, and diagnose quality."""
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in open(cache_path):
            d = json.loads(line)
            cache[d["k"]] = d["completion"]
        print(f"  restored {len(cache)} rows from cache")

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
                print(f"  {i}/{len(rows)}  kept {len(kept)}  dropped {len(dropped_rows)}")

    total_n = len(kept) + len(dropped_rows)
    rate = len(kept) / total_n * 100 if total_n else 0
    print(f"\n{'='*56}")
    print(f"  pass rate {rate:.0f}%  (kept {len(kept)} / dropped {len(dropped_rows)}"
          f", threshold {threshold}, cache hits {cached_hits})")

    # ---- Per-component reward distribution: when the pass rate is low, this shows which
    # ---- component is dragging it down.
    if all_breakdowns:
        print(f"\n  component means (1.0 = perfect):")
        comps = ["format", "ingredient_match", "grounding", "product_validity", "safety"]
        means = {c: sum(b[c] for b in all_breakdowns) / len(all_breakdowns) for c in comps}
        for c in comps:
            bar = "█" * int(means[c] * 20)
            print(f"    {c:18s} {means[c]:.2f}  {bar}")
        worst = min(means, key=means.get)
        if means[worst] < 0.7:
            print(f"\n  ⚠️  weakest component is {worst} ({means[worst]:.2f}):")
            for line in DIAGNOSIS[worst].split(" —— "):
                print(f"      {line}")

    # ---- Token usage and extrapolated cost for a full-scale run ----
    if usage.get("calls"):
        pt, ct, n = usage["prompt_tokens"], usage["completion_tokens"], usage["calls"]
        print(f"\n  this run: {n} calls, {pt:,} input tok, {ct:,} output tok")
        print(f"  per sample: {pt//n:,} input tok, {ct//n:,} output tok")
        print(f"  extrapolated to 800: ~{pt//n*800:,} input tok, ~{ct//n*800:,} output tok")
        print(f"  (convert to dollars using the current price of the model you chose)")

    # ---- Manual spot check ----
    if inspect and kept:
        print(f"\n{'='*56}\n  spot check: {min(inspect, len(kept))} kept samples "
              f"(judge the quality by eye)\n")
        for r in kept[:inspect]:
            print(f"  --- teacher score {r['teacher_score']} ---")
            print(f"  {r['completion'][:600]}\n")
    if inspect and dropped_rows:
        shown = [(r, b) for r, b in dropped_rows if b][:2]
        if shown:
            print(f"{'='*56}\n  dropped samples (see why they failed)\n")
            for r, b in shown:
                low = [k for k in ["format","ingredient_match","grounding","product_validity","safety"]
                       if b[k] < 0.5]
                print(f"  --- total {b['total']:.2f}, weak components: {low} ---")
                print(f"  {r.get('_completion','(teacher call failed, no output)')[:400]}\n")
    # ---- Machine-readable summary: the notebook uses it to decide automatically whether
    # ---- it is worth spending more on API calls.
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
    print(f"  summary written to {cache_path.parent / 'distill_summary.json'}")
    print(f"{'='*56}")
    return kept


# -------------------------------------------------------------------- IO ---
def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):5d} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--mode", choices=["sft", "rl", "both"], default="both")
    ap.add_argument("--mock-retrieval", action="store_true",
                    help="use the synthetic catalog; teammate A's real data is not required")
    ap.add_argument("--dry-teacher", action="store_true",
                    help="use the offline fake teacher: no API calls, no cost "
                         "(for validating the pipeline)")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "gpt-4o-mini"))
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="drop teacher answers scoring below this -- what decides whether "
                         "SFT works at all")
    ap.add_argument("--inspect", type=int, default=0,
                    help="print N teacher answers for manual review (use 3 on a small pilot run)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--outdir", default=str(PROCESSED))
    args = ap.parse_args()

    out = Path(args.outdir)
    print(f"Sampling {args.n} profiles and retrieving evidence "
          f"({'synthetic catalog' if args.mock_retrieval else 'real index'})...")
    rows = build_rows(args.n, args.mock_retrieval)
    print(f"  got {len(rows)} usable samples\n")

    random.shuffle(rows)
    n_test = max(1, int(len(rows) * args.test_frac))
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    if args.mode in ("rl", "both"):
        print("[RL data] no target answers needed")
        write_jsonl(train_rows, out / "rl.jsonl")
        write_jsonl(test_rows, out / "rl_test.jsonl")   # held-out set for teammate C's eval
        print()

    if args.mode in ("sft", "both"):
        print(f"[SFT data] teacher={'offline fake teacher' if args.dry_teacher else args.model}"
              f", filter threshold={args.threshold}")
        kept = distil(train_rows, args.model, args.threshold,
                      cache_path_for(out, args.model, args.dry_teacher),
                      args.dry_teacher, args.inspect)
        write_jsonl(kept, out / "sft.jsonl")

    print("\nNext steps:")
    print("  python -m skincare.llm.sft_lora --epochs 2")
    print("  python -m skincare.llm.grpo_train --steps 300")


if __name__ == "__main__":
    main()
