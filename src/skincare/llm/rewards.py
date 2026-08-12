"""VERIFIABLE REWARDS -- the intellectual core of the project (Module 11).

DeepSeek-R1's key move is rule-based, programmatically checkable rewards instead
of a learned reward model. We do the same, which is why no human labelling is
needed. Each function returns a float in [0, 1] per completion.

This file needs NO GPU -- write it, unit-test it, and demo it before any training.
"""
import json
import re

from skincare.rag.retrieve import load_rules

_RULES = load_rules()


_DECODER = json.JSONDecoder()


def _parse(completion: str) -> dict | None:
    """Tolerant JSON extraction -- models like to wrap output in prose/fences.

    Take the first *complete* JSON object, not the span from the first '{' to the
    last '}'. The greedy span is wrong the moment anything follows the object, and
    something usually does: a closing remark that happens to contain a brace, or --
    with do_sample=False on a small model -- a second copy of the same answer.
    json.loads is then handed two objects glued together, raises, and a complete,
    well-formed set of recommendations is thrown away. Serving turns that into an
    empty `recommendations` list with a 200, which the interface explains to the user
    as a budget or safety outcome, because nothing upstream said otherwise.

    Scanning is strictly more permissive than the old regex: raw_decode understands
    strings, so braces inside `reason` no longer end the object early, and prose
    before the JSON (`Here is the {answer}: {...}`) no longer poisons the match.
    An object carrying "recommendations" wins over an earlier one that does not, so a
    stray `{"disclaimer": "..."}` emitted first cannot mask the real answer.
    """
    text = re.sub(r"^```(?:json)?|```$", "", (completion or "").strip(), flags=re.MULTILINE)
    first: dict | None = None
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, match.start())
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if "recommendations" in obj:
            return obj
        if first is None:
            first = obj
    return first


# ---------------------------------------------------------------- format ---
def format_reward(completion: str, **_) -> float:
    obj = _parse(completion)
    if obj is None:
        return 0.0
    recs = obj.get("recommendations")
    if not isinstance(recs, list) or not recs:
        return 0.2
    need = {"product_id", "name", "reason", "key_ingredients", "cited_evidence"}
    ok = sum(1 for r in recs if isinstance(r, dict) and need <= set(r))
    return 0.3 + 0.7 * (ok / len(recs))


# ----------------------------------------------------------- correctness ---
def ingredient_match_reward(completion: str, concerns: list[str] | None = None, **_) -> float:
    """Do the recommended actives actually address the detected concerns?"""
    obj, concerns = _parse(completion), concerns or []
    if obj is None or not concerns:
        return 0.0
    wanted = set()
    for c in concerns:
        wanted |= {i.lower() for i in _RULES["concern_to_ingredients"].get(c, [])}
    if not wanted:
        return 0.0
    recs = obj.get("recommendations", []) or []
    scores = []
    for r in recs:
        ings = " ".join(str(i) for i in (r.get("key_ingredients") or [])).lower()
        scores.append(1.0 if any(w in ings for w in wanted) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


# -------------------------------------------------------------- grounding ---
def grounding_reward(completion: str, evidence_ids: list[str] | None = None, **_) -> float:
    """Penalise hallucinated citations -- cited ids must exist in the context."""
    obj = _parse(completion)
    if obj is None:
        return 0.0
    valid = set(evidence_ids or [])
    recs = obj.get("recommendations", []) or []
    if not recs:
        return 0.0
    per_rec = []
    for r in recs:
        cites = [str(c) for c in (r.get("cited_evidence") or [])]
        if not cites:
            per_rec.append(0.0)
            continue
        per_rec.append(sum(1 for c in cites if c in valid) / len(cites))
    return sum(per_rec) / len(per_rec)


def product_validity_reward(completion: str, product_ids: list[str] | None = None, **_) -> float:
    """Recommended products must come from retrieval, not from model memory."""
    obj = _parse(completion)
    if obj is None:
        return 0.0
    valid = set(product_ids or [])
    recs = obj.get("recommendations", []) or []
    if not recs:
        return 0.0
    return sum(1 for r in recs if str(r.get("product_id")) in valid) / len(recs)


# ----------------------------------------------------------------- safety ---
def safety_reward(completion: str, pregnant: bool = False,
                  avoid: list[str] | None = None, **_) -> float:
    obj = _parse(completion)
    if obj is None:
        return 0.0
    score = 1.0
    if len(str(obj.get("disclaimer", ""))) < 20:
        score -= 0.5
    banned = [a.lower() for a in (avoid or [])]
    if pregnant:
        banned += [u.lower() for u in _RULES["pregnancy_unsafe"]]
    if banned:
        for r in obj.get("recommendations", []) or []:
            ings = " ".join(str(i) for i in (r.get("key_ingredients") or [])).lower()
            if any(b in ings for b in banned):
                score -= 1.0          # hard violation
                break
    return max(0.0, score)


# ------------------------------------------------------------- aggregate ---
WEIGHTS = {
    "format": 0.15,
    "ingredient_match": 0.30,
    "grounding": 0.25,
    "product_validity": 0.15,
    "safety": 0.15,
}


def total_reward(completion: str, **ctx) -> float:
    return (
        WEIGHTS["format"] * format_reward(completion, **ctx)
        + WEIGHTS["ingredient_match"] * ingredient_match_reward(completion, **ctx)
        + WEIGHTS["grounding"] * grounding_reward(completion, **ctx)
        + WEIGHTS["product_validity"] * product_validity_reward(completion, **ctx)
        + WEIGHTS["safety"] * safety_reward(completion, **ctx)
    )


def reward_breakdown(completion: str, **ctx) -> dict:
    """Use this in the report -- per-component curves are far more convincing
    than a single scalar."""
    return {
        "format": format_reward(completion, **ctx),
        "ingredient_match": ingredient_match_reward(completion, **ctx),
        "grounding": grounding_reward(completion, **ctx),
        "product_validity": product_validity_reward(completion, **ctx),
        "safety": safety_reward(completion, **ctx),
        "total": total_reward(completion, **ctx),
    }
