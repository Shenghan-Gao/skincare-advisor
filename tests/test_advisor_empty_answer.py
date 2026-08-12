"""Why a live demo answered "0 products for your skin" with nothing to explain it.

Two distinct faults, both of which end as `recommendations: []` behind a 200:

1. `_parse` matched the span from the first '{' to the last '}'. Anything after the
   object -- a closing remark containing a brace, or a second copy of the answer,
   which greedy decoding on a 1.5B model produces readily -- made json.loads see two
   objects glued together and throw a *complete, valid* answer away.
2. Whatever the reason, the empty list came back indistinguishable from a considered
   one. The safety guard always leaves a flag when it removes something, so the UI
   reads "no flags + empty" as "your budget or your filters left nothing" and tells
   the user exactly that. A generation failure has to say so itself.

Against HEAD~ (the greedy regex, `routine_note=obj.get(...)`) every test here fails.
"""
import pytest

from app.schemas import Evidence, Product, RetrievalResult, UserProfile
from skincare.llm.generate import Advisor
from skincare.llm.rewards import _parse

GOOD = ('{"recommendations": [{"product_id": "P1001", "name": "x", "brand": "y", '
        '"reason": "cited [P1001:rev:3]", "key_ingredients": ["niacinamide"], '
        '"cited_evidence": ["P1001:rev:3"], "matched_concerns": ["acne"]}], '
        '"routine_note": "Use in the morning.", "disclaimer": "Cosmetic advice only."}')


@pytest.fixture
def retrieval():
    return RetrievalResult(
        products=[Product(product_id="P1001", name="Niacinamide 10%", brand="The Ordinary",
                          category="Treatments", price_usd=6.0,
                          ingredients=["niacinamide", "zinc pca"])],
        evidence=[Evidence(evidence_id="P1001:rev:3", product_id="P1001", source="review",
                           text="cleared my chin in a month", score=0.8)],
    )


def advise(monkeypatch, completion, retrieval):
    advisor = Advisor(backend="local")
    monkeypatch.setattr(advisor, "_generate", lambda messages: completion)
    return advisor.recommend(UserProfile(query="oily skin with acne"), None, retrieval)


# ----------------------------------------------------------------- _parse ---
def test_parse_keeps_the_answer_when_a_closing_remark_follows_it():
    """A brace in the trailing prose must not cost the whole answer."""
    obj = _parse(GOOD + "\n\nRemember to patch test {inner forearm} for 24 hours first.")
    assert obj is not None, "a complete object followed by prose was discarded"
    assert len(obj["recommendations"]) == 1


def test_parse_takes_the_first_object_when_the_model_repeats_itself():
    """Greedy decoding loops: the same object again, usually cut off by the token cap."""
    obj = _parse(GOOD + "\n" + GOOD)
    assert obj is not None, "two complete objects were read as one broken one"
    assert obj["routine_note"] == "Use in the morning."


def test_parse_survives_prose_before_the_object():
    obj = _parse("Here is the {result} you asked for:\n" + GOOD)
    assert obj is not None and len(obj["recommendations"]) == 1


def test_parse_prefers_the_object_that_carries_the_recommendations():
    obj = _parse('{"disclaimer": "Cosmetic advice only."}\n' + GOOD)
    assert obj is not None and obj["recommendations"][0]["product_id"] == "P1001"


def test_parse_still_returns_none_for_a_truncated_object():
    """The honest failure stays a failure -- the token cap is a separate problem."""
    assert _parse('{"recommendations": [{"product_id": "P1001", "reason": "because') is None


# -------------------------------------------------------------- recommend ---
def test_a_repeated_reply_still_reaches_the_user(monkeypatch, retrieval):
    resp = advise(monkeypatch, GOOD + "\n" + GOOD, retrieval)
    assert [r.product_id for r in resp.recommendations] == ["P1001"]
    assert resp.routine_note == "Use in the morning."


def test_an_unreadable_reply_says_so_instead_of_looking_considered(monkeypatch, retrieval):
    resp = advise(monkeypatch, "I would suggest a gentle cleanser and a moisturiser.", retrieval)
    assert resp.recommendations == []
    assert resp.routine_note.strip(), (
        "an empty answer with an empty note is read downstream as a budget/safety outcome"
    )
    assert "generation" in resp.routine_note.lower(), resp.routine_note


def test_a_reply_with_no_usable_item_says_so_too(monkeypatch, retrieval):
    """Parsed, but every recommendation lacks the one field the router keys on."""
    resp = advise(monkeypatch, '{"recommendations": [{"name": "Hydrating Serum"}]}', retrieval)
    assert resp.recommendations == []
    assert "generation" in resp.routine_note.lower(), resp.routine_note


def test_the_model_keeps_its_own_note_when_it_recommends_nothing(monkeypatch, retrieval):
    resp = advise(monkeypatch, '{"recommendations": [], "routine_note": "Nothing suits you."}',
                  retrieval)
    assert resp.routine_note == "Nothing suits you."
