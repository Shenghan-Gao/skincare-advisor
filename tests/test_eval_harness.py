"""组员 C 第一天的验证:不需要模型/GPU/API key。

    pytest tests/test_eval_harness.py -v
"""
import json
from pathlib import Path

from skincare.llm.rewards import reward_breakdown

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "eval_samples.jsonl"


def load_cases():
    return [json.loads(l) for l in open(FIXTURES)]


def test_fixture_file_exists():
    assert FIXTURES.exists(), "缺 fixtures/eval_samples.jsonl"
    assert len(load_cases()) >= 8


def test_every_case_matches_expected_range():
    """每个已知答案的样本都必须落在期望区间 —— 这验证评估器本身是对的。"""
    failures = []
    for c in load_cases():
        got = reward_breakdown(c["completion"], **c["ctx"])
        for k, (lo, hi) in c["expect"].items():
            if not lo <= got[k] <= hi:
                failures.append(f"{c['case']}.{k}={got[k]:.2f} 不在 [{lo},{hi}]")
    assert not failures, "\n".join(failures)


def test_hallucination_is_detected():
    cases = {c["case"]: c for c in load_cases()}
    good = reward_breakdown(cases["perfect"]["completion"], **cases["perfect"]["ctx"])
    bad = reward_breakdown(cases["hallucinated_citation"]["completion"],
                           **cases["hallucinated_citation"]["ctx"])
    assert bad["grounding"] < good["grounding"], "评估器没能识别伪造引用"


def test_markdown_table_renders():
    from skincare.eval.run_eval import markdown_table
    t = markdown_table({"base": {"total": 0.5}, "grpo": {"total": 0.8}})
    assert "base" in t and "grpo" in t and "|" in t
