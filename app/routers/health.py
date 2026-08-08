from fastapi import APIRouter
from app.deps import use_mocks

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "mock_mode": use_mocks()}
