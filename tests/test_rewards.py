"""Unit-test the reward functions. NO GPU NEEDED -- do this on day 1."""
import json

from skincare.llm import rewards as R

GOOD = json.dumps({
    "recommendations": [{
        "product_id": "P1", "name": "N", "brand": "B",
        "reason": "helps acne [E1]", "key_ingredients": ["niacinamide"],
        "cited_evidence": ["E1"], "matched_concerns": ["acne"],
    }],
    "routine_note": "am/pm",
    "disclaimer": "Cosmetic suggestions only, not medical advice; see a dermatologist.",
})
CTX = dict(concerns=["acne"], evidence_ids=["E1", "E2"], product_ids=["P1"])


def test_format_and_grounding():
    assert R.format_reward(GOOD) > 0.9
    assert R.grounding_reward(GOOD, **CTX) == 1.0
    assert R.product_validity_reward(GOOD, **CTX) == 1.0


def test_hallucinated_citation_is_punished():
    bad = GOOD.replace('"E1"', '"E999"')
    assert R.grounding_reward(bad, **CTX) < 0.5


def test_garbage_scores_zero():
    assert R.format_reward("I recommend some serums!") == 0.0


def test_pregnancy_violation():
    unsafe = GOOD.replace('"niacinamide"', '"retinol"')
    assert R.safety_reward(unsafe, pregnant=True) < 0.5
