"""A deterministic stand-in for Advisor -- runs the real pipeline with no API key and no GPU.

Why it exists: generate.Advisor either calls OpenAI (needs a key) or loads a local Qwen
(needs a GPU). When neither is available, the full "retrieve -> generate -> safety filter"
path cannot be verified end to end. This file assembles a **valid** AdvisorResponse in pure
Python under exactly the same key constraints as the real model:

  1. product_id may only come from the products returned by this retrieval call
     (no inventing products from memory);
  2. cited_evidence may only come from the evidence_id values returned by this retrieval
     call (no hallucinated citations);
  3. key_ingredients come from the catalog's real ingredient lists -- the safety guardrail
     relies on them to detect ingredients that are unsafe during pregnancy.

As a result, rewards.py awards full marks on grounding / product_validity / format, which
makes this an **upper-bound (oracle) baseline** for the training and evaluation pipeline.
"""
from app.schemas import (
    AdvisorResponse, Recommendation, RetrievalResult, SkinAnalysis, UserProfile,
)
from skincare.rag.retrieve import load_rules

DISCLAIMER = (
    "This tool offers cosmetic product suggestions only. It is not medical advice "
    "and cannot diagnose any skin condition. For persistent, painful, or worsening "
    "skin problems, please consult a licensed dermatologist."
)


class StubAdvisor:
    """Deterministic generator: restates the retrieval results as structured recommendations."""

    def __init__(self, name: str = "stub"):
        self.backend = name
        self.adapter = None

    @staticmethod
    def _matched_concerns(ingredients: list[str], wanted: list[str]) -> list[str]:
        """Map ingredients back to the concerns they address via ingredient_rules,
        keeping only the concerns the user actually cares about."""
        rules = load_rules()["concern_to_ingredients"]
        ings = " ".join(ingredients).lower()
        hit = [c for c, needles in rules.items() if any(n.lower() in ings for n in needles)]
        matched = [c for c in hit if c in wanted]
        if wanted:
            # Never claim a concern the user did not report. The previous fallback returned
            # hit[:1] whenever nothing intersected, so a self-tanner whose formula happens to
            # contain ascorbic acid was presented as "matches dark_spots" -- a concern scored
            # 0.2 and never asked about. An empty list is the honest answer there.
            return matched
        return hit[:1]

    def recommend(self, profile: UserProfile, analysis: SkinAnalysis | None,
                  retrieval: RetrievalResult) -> AdvisorResponse:
        wanted = analysis.top_concerns() if analysis else []

        by_product: dict[str, list] = {}
        for ev in retrieval.evidence:
            by_product.setdefault(ev.product_id, []).append(ev)

        recs: list[Recommendation] = []
        for product in retrieval.products:
            evs = by_product.get(product.product_id, [])
            cited = [e.evidence_id for e in evs][:3]
            snippet = evs[0].text if evs else ""
            concerns = self._matched_concerns(product.ingredients, wanted)
            recs.append(Recommendation(
                product_id=product.product_id, name=product.name, brand=product.brand,
                price_usd=product.price_usd,
                reason=(f"{product.name} ({product.brand}) matches "
                        f"{', '.join(concerns) if concerns else 'your profile'}. "
                        f"Evidence {', '.join(cited) if cited else '(none)'}: {snippet}").strip(),
                key_ingredients=list(product.ingredients),
                cited_evidence=cited, matched_concerns=concerns,
            ))

        return AdvisorResponse(
            analysis=analysis, recommendations=recs,
            routine_note=("Introduce one new active at a time, use it every other evening "
                          "for the first two weeks, and always apply sunscreen the next morning."),
            disclaimer=DISCLAIMER, generator=self.backend,
        )
