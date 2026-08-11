"""LLM-as-judge for subjective recommendation quality.

Owned by member C.

The rule-based reward answers questions such as whether citations exist, whether the
recommended ingredients match the detected concerns, and whether the response is safe.
This module complements those deterministic checks with four subjective dimensions:
helpfulness, clarity, specificity, and faithfulness.
"""
from __future__ import annotations

import json
import random
import re
from typing import Iterable

RUBRIC = """You are a strict reviewer of skincare recommendation quality.
Score the recommendation on each criterion from 1 to 5.

{criteria}

Return JSON only with exactly these keys:
{{"helpfulness": n, "clarity": n, "specificity": n, "faithfulness": n,
  "comment": "one concise sentence"}}
"""

DIMS = ["helpfulness", "clarity", "specificity", "faithfulness"]
_DESCRIPTIONS = {
    "helpfulness": "does it genuinely solve the problem the user described",
    "clarity": "is the explanation clear enough for an ordinary consumer to follow",
    "specificity": "is it concrete about ingredients and usage rather than generic filler",
    "faithfulness": "is it based only on the evidence given, with nothing invented",
}


def _rubric() -> str:
    """Return the rubric with criterion order shuffled to reduce rubric-position bias."""
    order = DIMS.copy()
    random.shuffle(order)
    criteria = "\n".join(f"{name}: {_DESCRIPTIONS[name]}" for name in order)
    return RUBRIC.format(criteria=criteria)


def _parse_score(text: str | None) -> dict | None:
    """Parse and validate one judge response; malformed outputs are excluded, not zeroed."""
    if not text:
        return None

    raw = text.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Defensive fallback for providers/models that wrap JSON in prose or code fences.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None

    parsed: dict[str, int | str] = {}
    for dim in DIMS:
        value = obj.get(dim)
        # bool is a subclass of int, so reject it explicitly.
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not numeric.is_integer() or not 1 <= numeric <= 5:
            return None
        parsed[dim] = int(numeric)

    parsed["comment"] = str(obj.get("comment", "")).strip()
    return parsed


def judge_one(prompt: str, completion: str, model: str = "gpt-4o-mini") -> dict | None:
    """Judge one recommendation with deterministic decoding.

    The batch helper :func:`judge_many` randomizes item order to reduce position bias.
    This function also shuffles criterion order in the rubric. A malformed judge response
    returns ``None`` so failed parses do not depress aggregate scores as artificial zeros.
    API/authentication errors are intentionally allowed to surface to the caller.
    """
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _rubric()},
            {
                "role": "user",
                "content": (
                    "USER REQUEST / CONTEXT:\n"
                    f"{prompt}\n\n"
                    "RECOMMENDATION TO SCORE:\n"
                    f"{completion}"
                ),
            },
        ],
    )
    return _parse_score(response.choices[0].message.content)


def judge_many(
    items: Iterable[tuple[str, str]],
    model: str = "gpt-4o-mini",
    seed: int | None = None,
) -> list[dict | None]:
    """Judge many ``(prompt, completion)`` pairs in randomized order.

    Returned scores are restored to the caller's original item order, which keeps later
    aggregation and joins deterministic while avoiding a fixed scoring sequence.
    """
    pairs = list(items)
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)

    scores: list[dict | None] = [None] * len(pairs)
    for idx in order:
        prompt, completion = pairs[idx]
        scores[idx] = judge_one(prompt, completion, model=model)
    return scores


def aggregate(scores: list[dict | None]) -> dict:
    """Mean subjective score by dimension, excluding failed/invalid judge calls."""
    valid = [s for s in scores if s]
    if not valid:
        return {d: None for d in DIMS}
    return {d: sum(s[d] for s in valid) / len(valid) for d in DIMS}

