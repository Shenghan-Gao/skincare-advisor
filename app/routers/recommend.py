"""Pillar 2 endpoint -- retrieval-grounded, post-trained advisor."""
from fastapi import APIRouter
from app.schemas import AdvisorResponse, RecommendRequest
from app.deps import use_mocks, load_fixture

router = APIRouter(tags=["advisor"])


@router.post("/recommend", response_model=AdvisorResponse)
def recommend(req: RecommendRequest):
    if use_mocks():
        return AdvisorResponse(**load_fixture("mock_advisor_response.json"))

    from app.deps import get_retriever, get_generator
    from skincare.safety.guard import apply_safety

    retrieval = get_retriever().search(req.profile, req.analysis, top_k=req.top_k)
    response = get_generator().recommend(req.profile, req.analysis, retrieval)
    return apply_safety(response, req.profile)
