"""Member A: safety and ethics guardrails for the skincare advisor.

This module is deliberately rule-based and transparent.  It does not diagnose
skin disease.  It enforces the cosmetic-use boundary, hard-blocks a small set
of pregnancy-unsafe ingredient strings, respects user-specified avoid lists,
and emits caution flags for potential irritants/comedogenic ingredients.
"""
from __future__ import annotations

import re

from app.schemas import AdvisorResponse, UserProfile
from skincare.rag.retrieve import load_rules

DISCLAIMER = (
    "This tool provides cosmetic skincare product suggestions only. It is not medical advice "
    "and cannot diagnose or treat a skin disease. Ingredient suitability depends on formulation, "
    "concentration, allergies, and individual response. If symptoms are persistent, painful, rapidly "
    "worsening, infected, bleeding, or otherwise concerning, please seek care from a licensed clinician."
)

# Narrow, high-signal phrases that indicate a request for diagnosis/prescription
# rather than cosmetic product recommendation.  This is intentionally not a
# giant symptom list: the guard should avoid false refusals for ordinary skincare.
MEDICAL_BOUNDARY_RE = re.compile(
    r"\b("
    r"diagnos(?:e|is)|is this (?:cancer|melanoma|eczema|psoriasis|rosacea|infection)|"
    r"prescription|prescribe|antibiotic|steroid cream|isotretinoin|accutane|"
    r"skin cancer|melanoma|biopsy|infected|pus|bleeding|open wound|severe pain"
    r")\b",
    re.I,
)


def is_out_of_scope_medical_query(query: str) -> bool:
    return bool(MEDICAL_BOUNDARY_RE.search(query or ""))


def _dedupe_flags(flags: list[str]) -> list[str]:
    return list(dict.fromkeys(flag for flag in flags if flag))


def _contains_any(ingredient_text: str, needles: list[str]) -> list[str]:
    text = ingredient_text.lower()
    return [needle for needle in needles if needle and needle.lower() in text]


def apply_safety(resp: AdvisorResponse, profile: UserProfile) -> AdvisorResponse:
    rules = load_rules()
    flags: list[str] = list(resp.safety_flags)

    # Out-of-scope medical requests are refused at the final deterministic layer.
    if is_out_of_scope_medical_query(profile.query):
        resp.recommendations = []
        resp.routine_note = (
            "I can help compare cosmetic skincare products, but I cannot diagnose a skin condition "
            "or recommend prescription treatment from this app."
        )
        flags.append("medical-boundary refusal: query requires clinical assessment")
        resp.safety_flags = _dedupe_flags(flags)
        resp.disclaimer = DISCLAIMER
        return resp

    hard_pregnancy = [str(x).lower() for x in rules.get("pregnancy_unsafe", [])]
    pregnancy_caution = [str(x).lower() for x in rules.get("pregnancy_caution", [])]
    irritants = [str(x).lower() for x in rules.get("common_irritants", [])]
    comedogenic = [str(x).lower() for x in rules.get("comedogenic", [])]
    avoid = [str(x).strip().lower() for x in profile.avoid_ingredients if str(x).strip()]

    analysis_concerns = set()
    if resp.analysis is not None:
        analysis_concerns = set(resp.analysis.top_concerns())
    oily_or_acne = (resp.analysis is not None and resp.analysis.skin_type.value == "oily") or (
        "acne" in analysis_concerns
    )

    kept = []
    for rec in resp.recommendations:
        # The current response contract exposes key ingredients rather than the
        # full product formula.  This is a known limitation documented in report.
        ingredient_text = " | ".join(str(x).lower() for x in rec.key_ingredients)

        if profile.pregnant:
            unsafe_hits = _contains_any(ingredient_text, hard_pregnancy)
            if unsafe_hits:
                flags.append(
                    f"removed {rec.name}: pregnancy-unsafe ingredient ({', '.join(unsafe_hits[:3])})"
                )
                continue
            caution_hits = _contains_any(ingredient_text, pregnancy_caution)
            if caution_hits:
                flags.append(
                    f"pregnancy caution for {rec.name}: {', '.join(caution_hits[:3])}; discuss use with a clinician"
                )

        avoid_hits = _contains_any(ingredient_text, avoid)
        if avoid_hits:
            flags.append(f"removed {rec.name}: user-avoided ingredient ({', '.join(avoid_hits[:3])})")
            continue

        irritant_hits = _contains_any(ingredient_text, irritants)
        if irritant_hits and ("redness" in analysis_concerns or "dryness" in analysis_concerns):
            flags.append(
                f"potential irritation caution for {rec.name}: {', '.join(irritant_hits[:3])}"
            )

        comedogenic_hits = _contains_any(ingredient_text, comedogenic)
        if comedogenic_hits and oily_or_acne:
            flags.append(
                f"possible pore-clogging caution for {rec.name}: {', '.join(comedogenic_hits[:3])}"
            )

        kept.append(rec)

    resp.recommendations = kept
    resp.safety_flags = _dedupe_flags(flags)
    resp.disclaimer = DISCLAIMER
    return resp
