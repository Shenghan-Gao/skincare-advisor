"""Pillar 1 endpoint -- CNN skin analysis."""
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas import SkinAnalysis
from app.deps import use_mocks, load_fixture

router = APIRouter(tags=["vision"])


@router.post("/analyze-skin", response_model=SkinAnalysis)
async def analyze_skin(image: UploadFile = File(...)):
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image")

    if use_mocks():
        return SkinAnalysis(**load_fixture("mock_skin_analysis.json"))

    from app.deps import get_vision_model
    return get_vision_model().predict_bytes(raw)
