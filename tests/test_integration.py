"""End-to-end integration test of the real pipeline (USE_MOCKS=0).

This exercises the genuine code path inside the recommend router: retrieval -> generation -> safety filtering.
Only the two "heavy dependencies" are swapped for stand-ins with the same interface:
  * USE_MOCK_RETRIEVAL=1 -> MockRetriever (fixtures/mock_catalog.json), so FAISS is not needed
  * USE_STUB_GENERATOR=1 -> StubAdvisor, so neither an OpenAI key nor a GPU is needed

Note: the environment variables are set with monkeypatch inside a single test and restored automatically,
otherwise they would contaminate test_contract.py (which sets USE_MOCKS to 1 at import time).
"""
import pytest
from fastapi.testclient import TestClient

from app.deps import get_generator, get_retriever
from app.main import app
from app.schemas import AdvisorResponse, SkinAnalysis, UserProfile
from skincare.llm.stub_advisor import StubAdvisor
from skincare.rag.mock_retrieval import MockRetriever
from skincare.safety.guard import apply_safety

client = TestClient(app)

# oily skin + acne + pores: lines up with P001/P002/P012 in mock_catalog
ACNE_ANALYSIS = {
    "skin_type": "oily", "skin_type_confidence": 0.91,
    "concerns": [{"concern": "acne", "score": 0.88}, {"concern": "large_pores", "score": 0.71},
                 {"concern": "dark_spots", "score": 0.22}, {"concern": "redness", "score": 0.15},
                 {"concern": "wrinkles", "score": 0.05}, {"concern": "dryness", "score": 0.10}],
}
# anti-wrinkle profile: retrieval is guaranteed to hit P009 (retinol), used to prove the pregnancy guardrail really blocks it
WRINKLE_ANALYSIS = {
    "skin_type": "combination", "skin_type_confidence": 0.80,
    "concerns": [{"concern": "wrinkles", "score": 0.90}, {"concern": "acne", "score": 0.05},
                 {"concern": "large_pores", "score": 0.10}, {"concern": "dark_spots", "score": 0.12},
                 {"concern": "redness", "score": 0.08}, {"concern": "dryness", "score": 0.20}],
}


@pytest.fixture(autouse=True)
def real_path_env(monkeypatch):
    """Take the real path and clear the lru_cache, so each case gets a fresh (identically seeded) retriever."""
    monkeypatch.setenv("USE_MOCKS", "0")
    monkeypatch.setenv("USE_MOCK_RETRIEVAL", "1")
    monkeypatch.setenv("USE_STUB_GENERATOR", "1")
    get_retriever.cache_clear(); get_generator.cache_clear()
    yield
    get_retriever.cache_clear(); get_generator.cache_clear()


def post(profile: dict, analysis: dict | None = ACNE_ANALYSIS, top_k: int = 3):
    r = client.post("/recommend", json={"profile": profile, "analysis": analysis, "top_k": top_k})
    assert r.status_code == 200, r.text
    return r.json()


def test_real_path_returns_valid_advisor_response():
    resp = AdvisorResponse(**post({"query": "oily skin with acne", "budget_usd": 40}))
    assert len(resp.recommendations) >= 1
    assert resp.disclaimer.strip(), "disclaimer must not be empty (the safety guardrail has to write it)"
    assert resp.generator == "stub" and resp.routine_note.strip()
    for rec in resp.recommendations:
        assert rec.product_id and rec.name and rec.reason
        assert rec.key_ingredients, "the guardrail uses key_ingredients to detect banned ingredients, so it must not be empty"


def test_budget_hard_filter_respected():
    resp = AdvisorResponse(**post({"query": "acne", "budget_usd": 20}))
    assert resp.recommendations
    for rec in resp.recommendations:
        assert rec.price_usd is not None and rec.price_usd <= 20


def test_cited_evidence_ids_come_from_the_actual_retrieval():
    """Every cited_evidence must be an id actually returned by this retrieval (zero hallucinated citations)."""
    profile = {"query": "oily skin with acne", "budget_usd": 40}
    resp = AdvisorResponse(**post(profile))
    retrieval = MockRetriever().search(UserProfile(**profile), SkinAnalysis(**ACNE_ANALYSIS), top_k=3)
    valid_ev = {e.evidence_id for e in retrieval.evidence}
    valid_pr = {p.product_id for p in retrieval.products}
    assert valid_ev and valid_pr
    for rec in resp.recommendations:
        assert rec.product_id in valid_pr, f"{rec.product_id} is not in the retrieval results"
        assert rec.cited_evidence, "a recommendation must provide citations"
        for eid in rec.cited_evidence:
            assert eid in valid_ev, f"hallucinated citation: {eid}"
            assert eid.startswith(rec.product_id), "a citation must belong to the product it recommends"


def test_pregnant_profile_gets_no_retinol_via_api():
    resp = AdvisorResponse(**post(
        {"query": "fine lines and wrinkles", "budget_usd": 40, "pregnant": True},
        analysis=WRINKLE_ANALYSIS))
    for rec in resp.recommendations:
        assert "retinol" not in " ".join(rec.key_ingredients).lower(), f"pregnancy-banned ingredient slipped through: {rec.name}"
        assert rec.product_id != "P009"
    assert resp.safety_flags, "a blocked recommendation must leave a safety_flag behind"
    assert any("pregnancy" in f for f in resp.safety_flags)


def test_safety_guard_actually_removes_a_retinol_recommendation():
    """Prove in two steps that the guardrail is not decorative: before it the stub really does recommend retinol, after it the item is gone."""
    profile = UserProfile(query="fine lines and wrinkles", budget_usd=40, pregnant=True)
    analysis = SkinAnalysis(**WRINKLE_ANALYSIS)
    retrieval = MockRetriever().search(profile, analysis, top_k=3)
    assert "P009" in {p.product_id for p in retrieval.products}, "retrieval was supposed to hit the retinol product"

    raw = StubAdvisor().recommend(profile, analysis, retrieval)
    before = {r.product_id for r in raw.recommendations}
    assert "P009" in before, "before the guardrail the stub should have recommended P009 (retinol)"

    guarded = apply_safety(raw, profile)
    after = {r.product_id for r in guarded.recommendations}
    assert "P009" not in after and before - after == {"P009"}
    assert any("pregnancy-unsafe" in f for f in guarded.safety_flags)


def test_avoid_ingredients_filter():
    resp = AdvisorResponse(**post(
        {"query": "acne", "budget_usd": 40, "avoid_ingredients": ["niacinamide"]}))
    for rec in resp.recommendations:
        assert "niacinamide" not in " ".join(rec.key_ingredients).lower()
    assert any("user-avoided" in f for f in resp.safety_flags)
