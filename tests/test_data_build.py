"""Validation of the SFT/RL data construction — no API key, no GPU, no real data required.

    pytest tests/test_data_build.py -v
"""
import json

import pytest
from skincare.llm.data_build import build_rows, fake_teacher, sample_profile
from skincare.llm.rewards import total_reward
from skincare.rag.mock_retrieval import MockRetriever


def test_mock_retriever_returns_relevant_products():
    from app.schemas import SkinAnalysis, UserProfile
    r = MockRetriever()
    res = r.search(
        UserProfile(query="acne please"),
        SkinAnalysis(skin_type="oily", skin_type_confidence=0.9,
                     concerns=[{"concern": "acne", "score": 0.9}]),
        top_k=3)
    assert res.products and res.evidence
    ids = [e.evidence_id for e in res.evidence]
    assert len(ids) == len(set(ids)), "evidence_id must be unique"
    assert all(":" in i for i in ids), "evidence_id must follow the P001:rev:0 format"


def test_budget_filter_is_respected():
    from app.schemas import SkinAnalysis, UserProfile
    r = MockRetriever()
    res = r.search(UserProfile(query="wrinkles", budget_usd=15),
                   SkinAnalysis(skin_type="dry", skin_type_confidence=0.9,
                                concerns=[{"concern": "wrinkles", "score": 0.9}]), top_k=3)
    assert all(p.price_usd <= 15 for p in res.products)


def test_rows_carry_full_reward_context():
    rows = build_rows(8, mock=True)
    assert rows
    for r in rows:
        for k in ["prompt", "concerns", "evidence_ids", "product_ids", "pregnant", "avoid"]:
            assert k in r, f"missing reward-context field {k}"
        assert r["evidence_ids"], "a sample with no evidence should never be produced"
        # these evidence_ids must actually appear in the prompt, or the model cannot cite them
        for eid in r["evidence_ids"][:3]:
            assert eid in r["prompt"]


def test_fake_teacher_output_is_valid_and_scores_well():
    rows = build_rows(6, mock=True)
    for r in rows:
        c = fake_teacher(r)
        obj = json.loads(c)
        assert obj["recommendations"], "the fake teacher must produce recommendations"
        s = total_reward(c, concerns=r["concerns"], evidence_ids=r["evidence_ids"],
                         product_ids=r["product_ids"], pregnant=r["pregnant"],
                         avoid=r["avoid"])
        assert s > 0.5, f"fake-teacher answer scores too low ({s:.2f}), which means the pipeline is broken"


def test_filtering_actually_drops_bad_answers():
    """Filtering is what decides whether SFT works — verify it really does drop bad answers."""
    rows = build_rows(3, mock=True)
    r = rows[0]
    bad = json.dumps({"recommendations": [{
        "product_id": "P999", "name": "x", "brand": "y", "reason": "z",
        "key_ingredients": ["coconut oil"], "cited_evidence": ["FAKE:rev:9"],
        "matched_concerns": []}], "routine_note": "", "disclaimer": ""})
    score = total_reward(bad, concerns=r["concerns"], evidence_ids=r["evidence_ids"],
                         product_ids=r["product_ids"], pregnant=False, avoid=[])
    assert score < 0.8, "an answer with an invented product plus a fabricated citation must be blocked by the 0.8 threshold"


def test_sample_profile_shape():
    p, a = sample_profile()
    assert len(a["concerns"]) == 6, "all 6 concerns must be scored"
    assert a["skin_type"] in ["oily", "dry", "combination", "normal"]
