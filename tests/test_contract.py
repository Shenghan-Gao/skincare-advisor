"""Contract tests -- run these before every merge. They pass with USE_MOCKS=1,
so teammates can verify their work without any model files."""
import os

os.environ["USE_MOCKS"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_skin_shape():
    r = client.post("/analyze-skin", files={"image": ("a.jpg", b"fake-bytes", "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["skin_type"] in {"oily", "dry", "combination", "normal"}
    assert len(body["concerns"]) == 6


def test_recommend_shape():
    r = client.post("/recommend", json={"profile": {"query": "acne and dark spots"}, "top_k": 3})
    assert r.status_code == 200
    assert r.json()["recommendations"]
