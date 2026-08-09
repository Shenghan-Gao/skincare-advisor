"""Member C's day-one validation: no model, no GPU, no API key required.

    pytest tests/test_eval_harness.py -v
"""
import json
from pathlib import Path

from skincare.llm.rewards import reward_breakdown

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "eval_samples.jsonl"


def load_cases():
    return [json.loads(l) for l in open(FIXTURES)]


def test_fixture_file_exists():
    assert FIXTURES.exists(), "fixtures/eval_samples.jsonl is missing"
    assert len(load_cases()) >= 8


def test_every_case_matches_expected_range():
    """Every known-answer sample must land inside its expected range — this proves the evaluator itself is correct."""
    failures = []
    for c in load_cases():
        got = reward_breakdown(c["completion"], **c["ctx"])
        for k, (lo, hi) in c["expect"].items():
            if not lo <= got[k] <= hi:
                failures.append(f"{c['case']}.{k}={got[k]:.2f} not in [{lo},{hi}]")
    assert not failures, "\n".join(failures)


def test_hallucination_is_detected():
    cases = {c["case"]: c for c in load_cases()}
    good = reward_breakdown(cases["perfect"]["completion"], **cases["perfect"]["ctx"])
    bad = reward_breakdown(cases["hallucinated_citation"]["completion"],
                           **cases["hallucinated_citation"]["ctx"])
    assert bad["grounding"] < good["grounding"], "the evaluator failed to detect a fabricated citation"


def test_markdown_table_renders():
    from skincare.eval.run_eval import markdown_table
    t = markdown_table({"base": {"total": 0.5}, "grpo": {"total": 0.8}})
    assert "base" in t and "grpo" in t and "|" in t
