import sys
import types

import skincare.eval.judge as judge


class _FakeResponse:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def _install_fake_openai(monkeypatch, content, captured):
    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse(content)

    class _OpenAI:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))


def test_judge_one_uses_deterministic_json_call(monkeypatch):
    captured = {}
    _install_fake_openai(
        monkeypatch,
        '{"helpfulness":5,"clarity":4,"specificity":3,"faithfulness":5,"comment":"good"}',
        captured,
    )

    got = judge.judge_one("dry skin", "Use a moisturizer.")
    assert got == {
        "helpfulness": 5,
        "clarity": 4,
        "specificity": 3,
        "faithfulness": 5,
        "comment": "good",
    }
    assert captured["temperature"] == 0
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "gpt-4o-mini"


def test_parse_failure_returns_none(monkeypatch):
    captured = {}
    _install_fake_openai(monkeypatch, "not json", captured)
    assert judge.judge_one("p", "c") is None


def test_out_of_range_score_returns_none(monkeypatch):
    captured = {}
    _install_fake_openai(
        monkeypatch,
        '{"helpfulness":0,"clarity":4,"specificity":3,"faithfulness":5,"comment":"bad"}',
        captured,
    )
    assert judge.judge_one("p", "c") is None


def test_judge_many_randomizes_calls_but_restores_output_order(monkeypatch):
    called = []

    def fake_judge_one(prompt, completion, model="gpt-4o-mini"):
        called.append(prompt)
        return {
            "helpfulness": int(prompt),
            "clarity": 1,
            "specificity": 1,
            "faithfulness": 1,
            "comment": completion,
        }

    monkeypatch.setattr(judge, "judge_one", fake_judge_one)
    items = [(str(i), f"c{i}") for i in range(1, 6)]
    got = judge.judge_many(items, seed=7)

    assert called != ["1", "2", "3", "4", "5"]
    assert [row["helpfulness"] for row in got] == [1, 2, 3, 4, 5]


def test_aggregate_ignores_none():
    scores = [
        {"helpfulness": 5, "clarity": 4, "specificity": 3, "faithfulness": 2},
        None,
        {"helpfulness": 3, "clarity": 2, "specificity": 1, "faithfulness": 4},
    ]
    assert judge.aggregate(scores) == {
        "helpfulness": 4.0,
        "clarity": 3.0,
        "specificity": 2.0,
        "faithfulness": 3.0,
    }
