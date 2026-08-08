"""SFT/RL 数据构造的验证 —— 不需要 API key、不需要 GPU、不需要真实数据。

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
    assert len(ids) == len(set(ids)), "evidence_id 必须唯一"
    assert all(":" in i for i in ids), "evidence_id 必须符合 P001:rev:0 格式"


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
            assert k in r, f"缺奖励上下文字段 {k}"
        assert r["evidence_ids"], "没有证据的样本不应产生"
        # prompt 里必须真的出现这些 evidence_id,否则模型没法引用
        for eid in r["evidence_ids"][:3]:
            assert eid in r["prompt"]


def test_fake_teacher_output_is_valid_and_scores_well():
    rows = build_rows(6, mock=True)
    for r in rows:
        c = fake_teacher(r)
        obj = json.loads(c)
        assert obj["recommendations"], "假教师必须给出推荐"
        s = total_reward(c, concerns=r["concerns"], evidence_ids=r["evidence_ids"],
                         product_ids=r["product_ids"], pregnant=r["pregnant"],
                         avoid=r["avoid"])
        assert s > 0.5, f"假教师答案得分过低 {s:.2f},说明链路有问题"


def test_filtering_actually_drops_bad_answers():
    """过滤是 SFT 有没有效果的分水岭 —— 验证它真的会丢掉坏答案。"""
    rows = build_rows(3, mock=True)
    r = rows[0]
    bad = json.dumps({"recommendations": [{
        "product_id": "P999", "name": "x", "brand": "y", "reason": "z",
        "key_ingredients": ["coconut oil"], "cited_evidence": ["FAKE:rev:9"],
        "matched_concerns": []}], "routine_note": "", "disclaimer": ""})
    score = total_reward(bad, concerns=r["concerns"], evidence_ids=r["evidence_ids"],
                         product_ids=r["product_ids"], pregnant=False, avoid=[])
    assert score < 0.8, "编造产品+伪造引用的答案必须被 0.8 阈值挡下"


def test_sample_profile_shape():
    p, a = sample_profile()
    assert len(a["concerns"]) == 6, "必须给全部 6 个关注点打分"
    assert a["skin_type"] in ["oily", "dry", "combination", "normal"]
