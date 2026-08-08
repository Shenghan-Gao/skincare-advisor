"""Lazy singletons. Heavy imports live INSIDE the functions on purpose so the
API still boots (in mock mode) on a machine with no torch installed."""
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def use_mocks() -> bool:
    return os.getenv("USE_MOCKS", "1") == "1"


def use_mock_retrieval() -> bool:
    """真实链路(USE_MOCKS=0)下,是否用 MockRetriever 顶替 FAISS 检索。

    组员 A 的 chunks.faiss / chunks_meta.parquet 还没交付,真实 Retriever 一构造就
    ImportError/FileNotFound。打开这个开关后走的仍是「检索→生成→安全过滤」真实代码路径,
    只是把索引换成 fixtures 里的合成目录。索引到位后置 0 即可。
    """
    return os.getenv("USE_MOCK_RETRIEVAL", "0") == "1"


def use_stub_generator() -> bool:
    """真实链路下,是否用 StubAdvisor 顶替 Advisor(OpenAI / 本地 LLM)。

    没有 API key、没有 GPU 时用它:输出仍是合法 AdvisorResponse,且引用的是本次检索
    真实返回的 evidence_id / product_id,因此奖励函数与安全护栏都能被真实检验。
    """
    return os.getenv("USE_STUB_GENERATOR", "0") == "1"


def load_fixture(name: str):
    with open(FIXTURES / name) as f:
        return json.load(f)


@lru_cache
def get_vision_model():
    from skincare.vision.infer import SkinClassifier
    return SkinClassifier(os.getenv("VISION_CKPT", "models/vision/best.pt"))


@lru_cache
def get_retriever():
    # 鸭子类型:MockRetriever.search 与 Retriever.search 签名/返回值完全一致,直接互换
    if use_mock_retrieval():
        from skincare.rag.mock_retrieval import MockRetriever
        return MockRetriever(os.getenv("MOCK_CATALOG") or None)
    from skincare.rag.retrieve import Retriever
    return Retriever(os.getenv("VECTOR_INDEX", "data/processed/index"))


@lru_cache
def get_generator():
    # 同上:StubAdvisor.recommend 与 Advisor.recommend 同签名,无需改 router
    if use_stub_generator():
        from skincare.llm.stub_advisor import StubAdvisor
        return StubAdvisor()
    from skincare.llm.generate import Advisor
    return Advisor()
