"""LLM-as-judge -- the subjective quality dimensions the rule-based reward cannot cover.
Owned by member C.

The rule-based reward (rewards.py) asks "is anything cited, are the ingredients right, is
it safe". The judge asks "is this actually useful to read, is the explanation clear, does
it answer the question that was asked". The two are complementary and must be presented
separately in the report.
"""
import json
import re

RUBRIC = """You are a strict reviewer of skincare recommendation quality. Score the
recommendation below on each criterion from 1 to 5.

helpfulness: does it genuinely solve the problem the user described
clarity:     is the explanation clear enough for an ordinary consumer to follow
specificity: is it concrete about ingredients and usage rather than generic filler
faithfulness:is it based only on the evidence given, with nothing invented

Output JSON only:
{"helpfulness":n,"clarity":n,"specificity":n,"faithfulness":n,"comment":"one sentence"}
"""

DIMS = ["helpfulness", "clarity", "specificity", "faithfulness"]


def judge_one(prompt: str, completion: str, model: str = "gpt-4o-mini") -> dict:
    """TODO(C): call OpenAI with RUBRIC + prompt + completion concatenated, then parse the
    JSON that comes back.

    Key points:
      1. temperature=0, otherwise the score for the same answer drifts between runs
      2. **randomise the order in which items are scored** to avoid position bias
      3. on a parse failure return None rather than a score of 0, so the mean is not polluted
    """
    raise NotImplementedError


def aggregate(scores: list[dict]) -> dict:
    valid = [s for s in scores if s]
    if not valid:
        return {d: None for d in DIMS}
    return {d: sum(s[d] for s in valid) / len(valid) for d in DIMS}
