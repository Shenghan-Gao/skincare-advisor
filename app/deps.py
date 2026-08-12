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


def use_mock_vision() -> bool:
    """Whether /analyze-skin answers from a fixture instead of loading the CNN.

    Separate from USE_MOCKS so the shipped image can serve the real recommend path --
    real query construction, real retrieval, real safety filtering -- while still
    answering /analyze-skin from a fixture. The container has no torch: the classifier
    is 43 MB of weights that a fresh clone does not have, and installing torch to load
    weights that are not there buys nothing. Defaults to whatever USE_MOCKS says.
    """
    return os.getenv("USE_MOCK_VISION", os.getenv("USE_MOCKS", "1")) == "1"


def use_mock_retrieval() -> bool:
    """On the real pipeline (USE_MOCKS=0), whether MockRetriever stands in for FAISS search.

    Teammate A's chunks.faiss / chunks_meta.parquet have not been delivered yet, so the real
    Retriever raises ImportError/FileNotFound as soon as it is constructed. With this switch on
    we still execute the real "retrieve -> generate -> safety filter" code path; only the index
    is swapped for the synthetic catalog in fixtures. Set it back to 0 once the index lands.
    """
    return os.getenv("USE_MOCK_RETRIEVAL", "0") == "1"


def use_stub_generator() -> bool:
    """On the real pipeline, whether StubAdvisor stands in for Advisor (OpenAI / local LLM).

    Use it when there is no API key and no GPU: the output is still a valid AdvisorResponse,
    and it cites the evidence_id / product_id actually returned by this retrieval call, so the
    reward functions and the safety guardrails are genuinely exercised.
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
    # Duck typing: MockRetriever.search has exactly the same signature and return type as
    # Retriever.search, so the two are drop-in interchangeable.
    if use_mock_retrieval():
        from skincare.rag.mock_retrieval import MockRetriever
        return MockRetriever(os.getenv("MOCK_CATALOG") or None)
    from skincare.rag.retrieve import Retriever
    return Retriever(os.getenv("VECTOR_INDEX", "data/processed/index"))


@lru_cache
def get_generator():
    # Same idea: StubAdvisor.recommend shares Advisor.recommend's signature, so the router
    # needs no changes.
    if use_stub_generator():
        from skincare.llm.stub_advisor import StubAdvisor
        return StubAdvisor()
    from skincare.llm.generate import Advisor
    return Advisor()
