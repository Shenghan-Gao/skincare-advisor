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

    retriever = get_retriever()
    retrieval = retriever.search(req.profile, req.analysis, top_k=req.top_k)
    response = get_generator().recommend(req.profile, req.analysis, retrieval)

    # The router is the only place that holds both the model's answer and the whole
    # catalogue, so it is where the two are reconciled. The prompt shows the model a
    # product_id and a chunk of text and never a name, a brand or a price, so those
    # come back invented -- three products called "Hydrating Serum". They are facts
    # the catalogue owns. The model keeps the prose and the citations.
    #
    # The lookup spans every product the evidence mentions, not the top-k summary in
    # retrieval.products: the evidence block is wider than that list, so the model can
    # recommend a product the summary never named, and a narrower lookup silently
    # leaves those rows with an invented name.
    ids = [e.product_id for e in retrieval.evidence]
    catalogue = getattr(retriever, "product_table", lambda _: {})(ids)
    for rec in response.recommendations:
        known = catalogue.get(rec.product_id)
        if known is None:
            continue
        rec.name, rec.brand, rec.price_usd = known.name, known.brand, known.price_usd

    # Hand the guard the real formulas too. Without them it can only inspect the
    # ingredients the model chose to name, and an avoid-list is only as good as the
    # list it is checked against.
    formulas = {pid: list(p.ingredients or []) for pid, p in catalogue.items()}
    return apply_safety(response, req.profile, formulas)
