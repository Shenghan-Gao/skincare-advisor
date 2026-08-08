"""FROZEN INTERFACE CONTRACT.

Everyone codes against these types. Do not change a field without telling the
team -- this file is what lets 4 people work in parallel without blocking.

Owner: Anna (main line). Changes require a heads-up in the group chat.
"""
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------- vision ---
class SkinType(str, Enum):
    oily = "oily"
    dry = "dry"
    combination = "combination"
    normal = "normal"


CONCERNS = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]


class ConcernScore(BaseModel):
    concern: str = Field(..., description="one of CONCERNS")
    score: float = Field(..., ge=0.0, le=1.0)


class SkinAnalysis(BaseModel):
    """Output of Pillar 1 (CNN). Consumed by RAG query building + LLM prompt."""
    skin_type: SkinType
    skin_type_confidence: float = Field(..., ge=0.0, le=1.0)
    concerns: list[ConcernScore]
    model_version: str = "stub-0"

    def top_concerns(self, threshold: float = 0.5) -> list[str]:
        return [c.concern for c in self.concerns if c.score >= threshold]


# ------------------------------------------------------------------- rag ---
class Product(BaseModel):
    product_id: str
    name: str
    brand: str
    category: str
    price_usd: float | None = None
    rating: float | None = None
    ingredients: list[str] = []


class Evidence(BaseModel):
    """A retrieved snippet the LLM must ground its explanation in."""
    evidence_id: str
    product_id: str
    source: Literal["description", "review", "ingredient"]
    text: str
    score: float = 0.0


class RetrievalResult(BaseModel):
    products: list[Product]
    evidence: list[Evidence]


# --------------------------------------------------------------- advisor ---
class UserProfile(BaseModel):
    query: str = ""
    budget_usd: float | None = None
    preferences: list[str] = []          # e.g. ["fragrance-free", "vegan"]
    avoid_ingredients: list[str] = []
    pregnant: bool = False


class Recommendation(BaseModel):
    product_id: str
    name: str
    brand: str
    price_usd: float | None = None
    reason: str                          # must cite evidence_ids
    key_ingredients: list[str] = []
    cited_evidence: list[str] = []       # evidence_id list -> checked by reward fn
    matched_concerns: list[str] = []


class AdvisorResponse(BaseModel):
    analysis: SkinAnalysis | None = None
    recommendations: list[Recommendation]
    routine_note: str = ""
    disclaimer: str = ""
    safety_flags: list[str] = []
    generator: str = "stub"              # base | sft | grpo | dpo | gpt-api


# ------------------------------------------------------------ api models ---
class RecommendRequest(BaseModel):
    profile: UserProfile
    analysis: SkinAnalysis | None = None   # pass CNN output, or omit for text-only
    top_k: int = 3
