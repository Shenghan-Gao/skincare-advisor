import pandas as pd

from app.schemas import (
    AdvisorResponse,
    ConcernScore,
    Recommendation,
    SkinAnalysis,
    UserProfile,
)
from skincare.rag.ingest import build_chunks, parse_ingredients
from skincare.safety.guard import apply_safety, is_out_of_scope_medical_query


def test_parse_ingredients_normalizes_and_deduplicates():
    raw = "['Water, Niacinamide, Glycerin', 'NIACINAMIDE, A very long marketing sentence " \
          "that is deliberately longer than sixty characters so it should be removed']"
    out = parse_ingredients(raw)
    assert out[:3] == ["water", "niacinamide", "glycerin"]
    assert out.count("niacinamide") == 1
    assert all(len(x) <= 60 and x == x.lower() for x in out)


def test_build_chunks_uses_frozen_evidence_id_contract():
    products = pd.DataFrame([{
        "product_id": "P1", "name": "Serum", "brand": "Brand", "category": "Serum",
        "price_usd": 20.0, "rating": 4.5, "ingredients": ["niacinamide", "glycerin"],
    }])
    metadata = pd.DataFrame([{
        "product_id": "P1", "name": "Serum", "brand": "Brand", "category": "Serum",
        "highlights": "Hydrating; fragrance free", "ingredients_text": "niacinamide, glycerin",
    }])
    reviews = pd.DataFrame([{
        "product_id": "P1", "text": "This is a sufficiently long grounded customer review.",
        "helpfulness": 1.0, "feedback": 2.0,
    }])
    chunks = build_chunks(products, metadata, reviews)
    assert set(chunks["evidence_id"]) == {"P1:desc:0", "P1:ing:0", "P1:rev:0"}
    assert set(chunks["source"]) == {"description", "ingredient", "review"}


def _response_with_retinol() -> AdvisorResponse:
    analysis = SkinAnalysis(
        skin_type="combination",
        skin_type_confidence=0.9,
        concerns=[ConcernScore(concern="wrinkles", score=0.9)],
    )
    return AdvisorResponse(
        analysis=analysis,
        recommendations=[Recommendation(
            product_id="P9", name="Retinol Serum", brand="Demo", price_usd=30,
            reason="Grounded demo", key_ingredients=["retinol", "glycerin"],
            cited_evidence=["P9:ing:0"], matched_concerns=["wrinkles"],
        )],
    )


def test_pregnancy_guard_hard_removes_retinol():
    guarded = apply_safety(_response_with_retinol(), UserProfile(query="wrinkles", pregnant=True))
    assert guarded.recommendations == []
    assert any("pregnancy-unsafe" in flag for flag in guarded.safety_flags)
    assert guarded.disclaimer


def test_medical_boundary_refuses_diagnosis_request():
    assert is_out_of_scope_medical_query("Can you diagnose whether this is melanoma?")
    resp = _response_with_retinol()
    guarded = apply_safety(resp, UserProfile(query="Can you diagnose whether this is melanoma?"))
    assert guarded.recommendations == []
    assert any("medical-boundary" in flag for flag in guarded.safety_flags)
