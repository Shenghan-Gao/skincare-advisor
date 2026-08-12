"""The guard must judge the formula, not the model's summary of it.

`Recommendation.key_ingredients` is what the *model* chose to name. A user who
avoids fragrance is not protected by that: the model almost never lists fragrance,
so the avoid-list silently passed. `apply_safety` now takes an optional catalogue
of real ingredient lists. These tests pin both halves -- that the old two-argument
call still behaves exactly as it did, and that the new argument actually fires.
"""
import pytest

from app.schemas import AdvisorResponse, Recommendation, UserProfile
from skincare.safety.guard import apply_safety


def _response(key_ingredients):
    return AdvisorResponse(recommendations=[Recommendation(
        product_id="P1", name="Rose Mist", brand="TestBrand",
        reason="cited [P1:rev:0]", key_ingredients=key_ingredients,
        cited_evidence=["P1:rev:0"])])


FORMULA = ["water", "glycerin", "fragrance", "citronellol", "retinol"]


def test_avoid_list_misses_when_the_model_did_not_name_the_ingredient():
    """The behaviour before the fix, kept as a test so the gap stays visible."""
    resp = apply_safety(_response(["aloe", "glycerin"]),
                        UserProfile(query="hydrating", avoid_ingredients=["fragrance"]))
    assert resp.safety_flags == []
    assert len(resp.recommendations) == 1, "nothing to match against, so nothing removed"


def test_avoid_list_fires_once_the_guard_can_see_the_formula():
    resp = apply_safety(_response(["aloe", "glycerin"]),
                        UserProfile(query="hydrating", avoid_ingredients=["fragrance"]),
                        {"P1": FORMULA})
    assert resp.recommendations == [], "an avoided ingredient must remove the product"
    assert any("user-avoided" in f and "fragrance" in f for f in resp.safety_flags), \
        resp.safety_flags


def test_pregnancy_rule_reads_the_formula_too():
    resp = apply_safety(_response(["niacinamide"]),
                        UserProfile(query="fine lines", pregnant=True),
                        {"P1": FORMULA})
    assert resp.recommendations == []
    assert any("pregnancy-unsafe" in f and "retinol" in f for f in resp.safety_flags), \
        resp.safety_flags


def test_bracketed_inci_still_matches_through_the_catalogue():
    """_normalise has to survive the new path, not just the old one."""
    resp = apply_safety(_response(["shea butter"]),
                        UserProfile(query="dry", avoid_ingredients=["coconut oil"]),
                        {"P1": ["cocos nucifera (coconut) oil", "water"]})
    assert resp.recommendations == [], "bracketed INCI must still be caught"


@pytest.mark.parametrize("catalogue", [None, {}, {"P2": FORMULA}, {"P1": []}, {"P1": None}])
def test_absent_or_irrelevant_catalogue_never_raises(catalogue):
    """Missing, empty, null and wrong-product entries must all degrade quietly."""
    resp = apply_safety(_response(["aloe"]),
                        UserProfile(query="hydrating", avoid_ingredients=["fragrance"]),
                        catalogue)
    assert len(resp.recommendations) == 1
    assert resp.disclaimer, "the disclaimer is written regardless"
