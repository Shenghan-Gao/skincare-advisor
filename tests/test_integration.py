"""真实链路(USE_MOCKS=0)端到端集成测试。

跑的是 recommend router 里那条真代码路径:检索 → 生成 → 安全过滤。
只把两个"重依赖"换成同接口的替身:
  * USE_MOCK_RETRIEVAL=1 -> MockRetriever(fixtures/mock_catalog.json),不需要 FAISS
  * USE_STUB_GENERATOR=1 -> StubAdvisor,不需要 OpenAI key,也不需要 GPU

注意:环境变量只在单个测试内用 monkeypatch 设置并自动还原,
否则会污染 test_contract.py(它在 import 期就把 USE_MOCKS 设成 1)。
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

# 油皮 + 痘痘 + 毛孔:与 mock_catalog 里 P001/P002/P012 对得上
ACNE_ANALYSIS = {
    "skin_type": "oily", "skin_type_confidence": 0.91,
    "concerns": [{"concern": "acne", "score": 0.88}, {"concern": "large_pores", "score": 0.71},
                 {"concern": "dark_spots", "score": 0.22}, {"concern": "redness", "score": 0.15},
                 {"concern": "wrinkles", "score": 0.05}, {"concern": "dryness", "score": 0.10}],
}
# 抗皱画像:检索必然命中 P009(retinol),用来验证孕期护栏真的在拦人
WRINKLE_ANALYSIS = {
    "skin_type": "combination", "skin_type_confidence": 0.80,
    "concerns": [{"concern": "wrinkles", "score": 0.90}, {"concern": "acne", "score": 0.05},
                 {"concern": "large_pores", "score": 0.10}, {"concern": "dark_spots", "score": 0.12},
                 {"concern": "redness", "score": 0.08}, {"concern": "dryness", "score": 0.20}],
}


@pytest.fixture(autouse=True)
def real_path_env(monkeypatch):
    """走真实路径 + 清 lru_cache,保证每个用例拿到全新(同种子)的检索器。"""
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
    assert resp.disclaimer.strip(), "disclaimer 不能为空(安全护栏必须写入)"
    assert resp.generator == "stub" and resp.routine_note.strip()
    for rec in resp.recommendations:
        assert rec.product_id and rec.name and rec.reason
        assert rec.key_ingredients, "护栏靠 key_ingredients 判断禁用成分,不能为空"


def test_budget_hard_filter_respected():
    resp = AdvisorResponse(**post({"query": "acne", "budget_usd": 20}))
    assert resp.recommendations
    for rec in resp.recommendations:
        assert rec.price_usd is not None and rec.price_usd <= 20


def test_cited_evidence_ids_come_from_the_actual_retrieval():
    """每个 cited_evidence 都必须是本次检索真实返回的 id(零幻觉引用)。"""
    profile = {"query": "oily skin with acne", "budget_usd": 40}
    resp = AdvisorResponse(**post(profile))
    retrieval = MockRetriever().search(UserProfile(**profile), SkinAnalysis(**ACNE_ANALYSIS), top_k=3)
    valid_ev = {e.evidence_id for e in retrieval.evidence}
    valid_pr = {p.product_id for p in retrieval.products}
    assert valid_ev and valid_pr
    for rec in resp.recommendations:
        assert rec.product_id in valid_pr, f"{rec.product_id} 不在检索结果里"
        assert rec.cited_evidence, "推荐必须给出引用"
        for eid in rec.cited_evidence:
            assert eid in valid_ev, f"幻觉引用: {eid}"
            assert eid.startswith(rec.product_id), "引用必须属于它所推荐的产品"


def test_pregnant_profile_gets_no_retinol_via_api():
    resp = AdvisorResponse(**post(
        {"query": "fine lines and wrinkles", "budget_usd": 40, "pregnant": True},
        analysis=WRINKLE_ANALYSIS))
    for rec in resp.recommendations:
        assert "retinol" not in " ".join(rec.key_ingredients).lower(), f"孕期禁用成分漏网: {rec.name}"
        assert rec.product_id != "P009"
    assert resp.safety_flags, "被拦下的推荐必须留下 safety_flag"
    assert any("pregnancy" in f for f in resp.safety_flags)


def test_safety_guard_actually_removes_a_retinol_recommendation():
    """分两步证明护栏不是摆设:护栏前 stub 确实推了 retinol,护栏后它消失。"""
    profile = UserProfile(query="fine lines and wrinkles", budget_usd=40, pregnant=True)
    analysis = SkinAnalysis(**WRINKLE_ANALYSIS)
    retrieval = MockRetriever().search(profile, analysis, top_k=3)
    assert "P009" in {p.product_id for p in retrieval.products}, "检索本该命中视黄醇产品"

    raw = StubAdvisor().recommend(profile, analysis, retrieval)
    before = {r.product_id for r in raw.recommendations}
    assert "P009" in before, "护栏前 stub 应当推荐了 P009(retinol)"

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
