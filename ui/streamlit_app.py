"""Skinsight -- demo UI for the Skincare Advisor (Streamlit, single file, stdlib + requests).

The demo has to make four project claims visible on screen:

  1. Skin analysis  -- skin type + confidence and the six concern scores (meter chart).
  2. Grounding      -- the cited_evidence ids behind every recommendation.
  3. Safety         -- safety_flags shown prominently, disclaimer always visible,
                       a blocked/empty answer explained instead of silently empty.
  4. Provenance     -- AdvisorResponse.generator, i.e. WHICH model answered
                       (stub / base / SFT / GRPO), because base-vs-SFT-vs-GRPO is
                       the headline result of the project.

Presentation is a five-screen wizard (landing + four steps) driven by
`st.session_state["step"]`, because the graded artefact is a recorded walkthrough for
a non-technical audience. Within a screen, anything longer than about two blocks of
prose is split into `st.tabs` rather than stacked: the viewer sees one picture and one
idea at a time, and the technical material is one click away instead of scrolling past.

Visual language: Skinsight, "evidence-first skincare". Serif headlines, one terracotta
accent (#C8705F) used only for the primary action, the active step and the single
number that matters, everything else ink/muted on paper. Charts follow the same rule --
emphasis, not a rainbow: the strongest concern is the accent, the rest are neutral.

State: profile answers live in plain session-state keys ("query", "budget", ...), and
the widgets are given their value with `value=`/`default=` rather than `key=`. Widget
state is discarded when a widget is not rendered, which for a wizard means anything
typed on step 3 would vanish on step 4; keeping the answers outside the widgets makes
back/forward navigation lossless. (Tab bodies are all rendered on every rerun, so
widgets inside `st.tabs` keep their value -- only unrendered screens lose it.)

Structure: everything above the "Rendering" banner is pure request/response/markup
logic with no Streamlit calls, so it can be imported and exercised headlessly. Every
`st.*` call lives inside `main()` or a `render_*` helper.

Run:  streamlit run ui/streamlit_app.py        (API_URL env var overrides the host)
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PROFILES_PATH = REPO_ROOT / "fixtures" / "demo_profiles.json"
DEMO_CATALOG_PATH = REPO_ROOT / "fixtures" / "mock_catalog.json"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "skinsight_logo.svg"
MARK_PATH = ASSETS_DIR / "skinsight_mark.svg"

BRAND_NAME = "Skinsight"
BRAND_TAGLINE = "Evidence-first skincare"

DEFAULT_API = os.getenv("API_URL", "http://localhost:8000")

# Mirrors app.schemas.CONCERNS -- kept as a literal so the UI never imports the backend.
CONCERNS = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]

# app.schemas.SkinAnalysis.top_concerns() uses this threshold to pick the concerns that
# are *eligible* for the retrieval query, so the chart marks it explicitly.
TOP_CONCERN_THRESHOLD = 0.5

# Mirrors skincare.rag.retrieve.MAX_QUERY_CONCERNS. query_concerns() takes everything
# over the threshold, sorts by score and keeps only this many: the real CNN's concern
# head leans positive, and six concerns plus their ingredient terms bury the user's own
# words in the query. The scores below the cut are not discarded -- they still reach the
# prompt and the safety guard -- so the caption on the analysis screen has to say
# "searched on" rather than "used".
MAX_QUERY_CONCERNS = 3

BASE_PREFERENCES = ["fragrance-free", "vegan", "gentle", "lightweight", "non-comedogenic"]

# Offered as ready-made chips so the "avoid" answer costs a click instead of typing.
# Anything a preset adds is appended at runtime (see avoid_options).
BASE_AVOID_INGREDIENTS = [
    "fragrance",
    "alcohol",
    "essential oils",
    "retinol",
    "salicylic acid",
    "benzoyl peroxide",
    "sulfates",
    "coconut oil",
]

# Slider ceiling for the budget question. Presets may carry a larger number; it is
# clamped on load rather than dropped, so the preset still means "generous budget".
BUDGET_MAX = 200.0

EVIDENCE_SOURCE_NAMES = {"desc": "description", "rev": "review", "ing": "ingredient"}

SHORT_DISCLAIMER = (
    "Cosmetic product suggestions only. Not medical advice, and not a diagnosis."
)

# Screen order. Index 0 is the landing page; 1..4 are the numbered steps shown in the
# progress indicator, which is why STEP_TITLES is indexed from 1.
LANDING = 0
STEP_PHOTO = 1
STEP_ANALYSIS = 2
STEP_PROFILE = 3
STEP_ROUTINE = 4
LAST_STEP = STEP_ROUTINE

STEP_TITLES = {
    STEP_PHOTO: "Photo",
    STEP_ANALYSIS: "Analysis",
    STEP_PROFILE: "About you",
    STEP_ROUTINE: "Routine",
}

# Neutral, non-diagnostic descriptions of what each label in CONCERNS covers. Shown on
# the "What it means" tab so the audience can read the chart without the presenter
# narrating it -- deliberately descriptions of the *label*, never of the viewer's skin.
CONCERN_BLURBS = {
    "acne": "Spots and blocked pores anywhere in the photo.",
    "dark_spots": "Uneven patches of deeper pigment, the kind left behind after a spot.",
    "redness": "Flushed or irritated-looking areas.",
    "large_pores": "Pores visible at normal viewing distance, usually across the T-zone.",
    "wrinkles": "Fine lines and creases, strongest where the face folds when it moves.",
    "dryness": "Flat, flaky or tight-looking surface texture.",
}

SKIN_TYPE_BLURBS = {
    "oily": "Skin that produces more surface oil, often shiny by the afternoon.",
    "dry": "Skin that holds less moisture and can feel tight.",
    "combination": "An oilier centre panel with drier cheeks.",
    "normal": "Balanced -- neither noticeably oily nor noticeably dry.",
}

# What the safety layer is asked to check, in the audience's language. Static copy: the
# actual flags always come from AdvisorResponse.safety_flags and are shown verbatim.
SAFETY_CHECKS = [
    "Pregnancy: ingredients flagged as unsuitable are removed, not just annotated.",
    "Your avoid list: a product containing anything you ruled out cannot be suggested.",
    "Scope: medical questions are refused rather than answered with a product.",
    "Every answer carries the disclaimer, including an empty one.",
]

# Fallback presets, used only when fixtures/demo_profiles.json is missing or unreadable.
# They intentionally cover the four demo beats: normal, pregnancy safety, tight budget,
# and an out-of-scope medical request (which the safety guard refuses).
FALLBACK_PROFILES: list[dict[str, Any]] = [
    {
        "label": "Combination skin - breakouts and dark spots",
        "query": "Combination skin, breakouts on my chin and some dark spots I want to fade.",
        "budget_usd": 40,
        "pregnant": False,
        "preferences": ["fragrance-free"],
        "avoid_ingredients": [],
        "analysis": {
            "skin_type": "combination",
            "skin_type_confidence": 0.82,
            "concerns": [
                {"concern": "acne", "score": 0.78},
                {"concern": "dark_spots", "score": 0.61},
                {"concern": "redness", "score": 0.34},
                {"concern": "large_pores", "score": 0.55},
                {"concern": "wrinkles", "score": 0.12},
                {"concern": "dryness", "score": 0.21},
            ],
            "model_version": "demo-fallback",
        },
    },
    {
        "label": "Pregnancy safety check - dark spots and fine lines",
        "query": "I am pregnant. I want to work on dark spots and early fine lines.",
        "budget_usd": 80,
        "pregnant": True,
        "preferences": ["fragrance-free"],
        "avoid_ingredients": [],
        "analysis": {
            "skin_type": "combination",
            "skin_type_confidence": 0.79,
            "concerns": [
                {"concern": "acne", "score": 0.55},
                {"concern": "dark_spots", "score": 0.85},
                {"concern": "redness", "score": 0.20},
                {"concern": "large_pores", "score": 0.30},
                {"concern": "wrinkles", "score": 0.90},
                {"concern": "dryness", "score": 0.25},
            ],
            "model_version": "demo-fallback",
        },
    },
    {
        "label": "Tight budget - oily skin basics",
        "query": "Oily skin, large pores, student budget. I only want the essentials.",
        "budget_usd": 12,
        "pregnant": False,
        "preferences": ["lightweight"],
        "avoid_ingredients": ["fragrance"],
        "analysis": {
            "skin_type": "oily",
            "skin_type_confidence": 0.88,
            "concerns": [
                {"concern": "acne", "score": 0.72},
                {"concern": "dark_spots", "score": 0.25},
                {"concern": "redness", "score": 0.30},
                {"concern": "large_pores", "score": 0.68},
                {"concern": "wrinkles", "score": 0.08},
                {"concern": "dryness", "score": 0.15},
            ],
            "model_version": "demo-fallback",
        },
    },
    {
        "label": "Out of scope - medical question (must be refused)",
        "query": "Is this melanoma? Can you prescribe an antibiotic for my infected skin?",
        "budget_usd": 50,
        "pregnant": False,
        "preferences": [],
        "avoid_ingredients": [],
        "analysis": None,
    },
]


# --------------------------------------------------------------------------- #
# API layer (pure logic -- importable and testable without Streamlit)
# --------------------------------------------------------------------------- #

@dataclass
class ApiResult:
    """Uniform result wrapper so no call site ever sees a raw exception."""

    ok: bool
    status: int | None = None
    data: Any = None
    error: str = ""       # short, human-readable headline
    body: str = ""        # raw response body (truncated), for the error panel

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "body": self.body,
        }


def normalize_api_url(url: str | None) -> str:
    """Trim whitespace/trailing slash and default the scheme, so 'localhost:8000' works."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_API
    if "://" not in url:
        url = "http://" + url
    return url


def _request(method: str, url: str, *, timeout: float, **kwargs: Any) -> ApiResult:
    """Single choke point for every HTTP call: never raises, always returns ApiResult."""
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.Timeout:
        return ApiResult(False, None, error=f"Timed out after {timeout:.0f}s calling {url}")
    except requests.exceptions.ConnectionError:
        return ApiResult(
            False, None,
            error=f"Cannot reach the API at {url}. Is the backend running?",
        )
    except requests.exceptions.RequestException as exc:  # malformed URL, TLS, redirects...
        return ApiResult(False, None, error=f"Request to {url} failed: {type(exc).__name__}: {exc}")

    body = (resp.text or "")[:4000]
    if not resp.ok:
        return ApiResult(
            False, resp.status_code,
            error=f"API returned HTTP {resp.status_code} for {url}",
            body=body,
        )
    try:
        return ApiResult(True, resp.status_code, data=resp.json(), body=body)
    except ValueError:
        return ApiResult(
            False, resp.status_code,
            error=f"HTTP {resp.status_code} but the body was not valid JSON",
            body=body,
        )


def get_health(api: str, timeout: float = 4.0) -> ApiResult:
    """GET /health -> {"status": ..., "mock_mode": ...}."""
    return _request("GET", f"{normalize_api_url(api)}/health", timeout=timeout)


def post_analyze_skin(
    api: str,
    filename: str,
    content: bytes,
    content_type: str = "application/octet-stream",
    timeout: float = 60.0,
) -> ApiResult:
    """POST /analyze-skin (multipart, field name 'image') -> SkinAnalysis."""
    files = {
        "image": (filename or "upload.jpg", content, content_type or "application/octet-stream")
    }
    return _request("POST", f"{normalize_api_url(api)}/analyze-skin", timeout=timeout, files=files)


def post_recommend(api: str, payload: dict[str, Any], timeout: float = 120.0) -> ApiResult:
    """POST /recommend (RecommendRequest body) -> AdvisorResponse."""
    return _request("POST", f"{normalize_api_url(api)}/recommend", timeout=timeout, json=payload)


def parse_ingredient_list(raw: str) -> list[str]:
    """'retinol, fragrance ,' -> ['retinol', 'fragrance']."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def build_profile(
    query: str,
    budget_usd: float | None,
    pregnant: bool,
    preferences: list[str] | None,
    avoid_ingredients: list[str] | None,
) -> dict[str, Any]:
    """Build a UserProfile-shaped dict. A budget of 0/None means 'no limit'."""
    budget: float | None = None
    if budget_usd is not None:
        try:
            value = float(budget_usd)
            budget = value if value > 0 else None
        except (TypeError, ValueError):
            budget = None
    return {
        "query": (query or "").strip(),
        "budget_usd": budget,
        "preferences": list(preferences or []),
        "avoid_ingredients": list(avoid_ingredients or []),
        "pregnant": bool(pregnant),
    }


def build_recommend_payload(
    profile: dict[str, Any],
    analysis: dict[str, Any] | None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Build the RecommendRequest body. `analysis` is optional (text-only mode)."""
    return {
        "profile": profile,
        "analysis": analysis if is_valid_analysis(analysis) else None,
        "top_k": int(top_k),
    }


# --------------------------------------------------------------------------- #
# Demo presets
# --------------------------------------------------------------------------- #

def is_valid_analysis(analysis: Any) -> bool:
    """Cheap shape check against the SkinAnalysis contract before sending/plotting it."""
    if not isinstance(analysis, dict):
        return False
    if not analysis.get("skin_type"):
        return False
    concerns = analysis.get("concerns")
    if not isinstance(concerns, list) or not concerns:
        return False
    return all(isinstance(c, dict) and "concern" in c and "score" in c for c in concerns)


def coerce_str_list(value: Any) -> list[str]:
    """Accept either a list or a comma-separated string and return a clean list.

    Session state can hold either shape: presets are authored both ways, and an older
    saved session may still carry the comma-separated string the previous UI used.
    """
    if isinstance(value, str):
        return parse_ingredient_list(value)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def normalize_preset(raw: Any, index: int) -> dict[str, Any] | None:
    """Coerce one entry of demo_profiles.json into the shape the widgets expect.

    Returns None for entries that are unusable, so one bad row cannot break the demo.
    """
    if not isinstance(raw, dict):
        return None
    query = str(raw.get("query", "") or "")
    label = str(raw.get("label") or "") or (query[:60] or f"Preset {index + 1}")

    budget = raw.get("budget_usd")
    try:
        budget = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None

    analysis = raw.get("analysis")
    return {
        "label": label,
        "query": query,
        "budget_usd": budget,
        "pregnant": bool(raw.get("pregnant", False)),
        "preferences": coerce_str_list(raw.get("preferences")),
        "avoid_ingredients": coerce_str_list(raw.get("avoid_ingredients")),
        "analysis": analysis if is_valid_analysis(analysis) else None,
    }


def load_demo_profiles(path: Path | str = DEMO_PROFILES_PATH) -> tuple[list[dict[str, Any]], str]:
    """Load fixtures/demo_profiles.json, falling back to FALLBACK_PROFILES.

    Returns (profiles, source_description). Accepts either a bare JSON list or an
    object with a "profiles" key, since the fixture is authored separately.
    """
    path = Path(path)
    shown = f"{path.parent.name}/{path.name}"
    fallback = [normalize_preset(p, i) for i, p in enumerate(FALLBACK_PROFILES)]
    fallback = [p for p in fallback if p]

    if not path.exists():
        return fallback, f"built-in fallback presets ({shown} not found)"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return fallback, f"built-in fallback presets ({shown} unreadable: {type(exc).__name__})"

    if isinstance(raw, dict):
        raw = raw.get("profiles", [])
    if not isinstance(raw, list):
        return fallback, f"built-in fallback presets ({shown} is not a JSON list)"

    presets = [normalize_preset(item, i) for i, item in enumerate(raw)]
    presets = [p for p in presets if p]
    if not presets:
        return fallback, f"built-in fallback presets ({shown} contained no usable entries)"
    return presets, f"{shown} ({len(presets)} presets)"


def preference_options(presets: list[dict[str, Any]]) -> list[str]:
    """Multiselect options must contain every value a preset might set."""
    options = list(BASE_PREFERENCES)
    for preset in presets:
        for pref in preset.get("preferences", []):
            if pref not in options:
                options.append(pref)
    return options


def avoid_options(presets: list[dict[str, Any]], extra: list[str] | None = None) -> list[str]:
    """Same idea as preference_options, plus whatever is already selected.

    A multiselect silently drops a default that is not in `options`, which would quietly
    weaken the profile the user sees, so selected values are always merged in.
    """
    options = list(BASE_AVOID_INGREDIENTS)
    for preset in presets:
        for item in preset.get("avoid_ingredients", []):
            if item not in options:
                options.append(item)
    for item in extra or []:
        if item not in options:
            options.append(item)
    return options


# --------------------------------------------------------------------------- #
# Evidence helpers (grounding)
# --------------------------------------------------------------------------- #

def parse_evidence_id(evidence_id: str) -> dict[str, str]:
    """'P001:rev:2' -> {'product_id': 'P001', 'source': 'review', 'chunk': '2'}.

    Unknown shapes degrade gracefully -- the raw id is always shown regardless.
    """
    parts = str(evidence_id).split(":")
    product_id = parts[0] if parts else str(evidence_id)
    source_key = parts[1] if len(parts) > 1 else ""
    chunk = parts[2] if len(parts) > 2 else ""
    return {
        "product_id": product_id,
        "source": EVIDENCE_SOURCE_NAMES.get(source_key, source_key or "unknown"),
        "chunk": chunk,
    }


def load_evidence_texts(path: Path | str = DEMO_CATALOG_PATH) -> dict[str, str]:
    """Best-effort evidence_id -> snippet lookup from the local demo catalog.

    /recommend returns citation ids only (see AdvisorResponse), so when the demo
    runs on the mock catalog we can resolve those ids back to the retrieved text.
    Returns {} whenever the file is absent or does not match -- never raises.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    chunks = data.get("chunks") if isinstance(data, dict) else None
    if not isinstance(chunks, list):
        return {}
    out: dict[str, str] = {}
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get("evidence_id"):
            out[str(chunk["evidence_id"])] = str(chunk.get("text", ""))
    return out


def grounding_stats(recommendations: list[dict[str, Any]]) -> tuple[int, int, int]:
    """(recs citing >=1 evidence id, total recs, total citations)."""
    total = len(recommendations)
    cited = 0
    citations = 0
    for rec in recommendations:
        ids = rec.get("cited_evidence") or []
        if ids:
            cited += 1
        citations += len(ids)
    return cited, total, citations


def describe_generator(generator: str) -> tuple[str, str]:
    """Map AdvisorResponse.generator onto (headline, explanation) for the demo."""
    raw = (generator or "unknown").strip()
    key = raw.lower()
    if "grpo" in key:
        return "GRPO post-trained model", (
            "RL stage: trained against the grounding/safety reward functions."
        )
    if "dpo" in key:
        return "DPO post-trained model", "Preference-optimised checkpoint."
    if "sft" in key or "lora" in key:
        return "SFT (LoRA) model", "Supervised fine-tuning stage, before RL."
    if "base" in key:
        return "Base model (no post-training)", "The untuned baseline the report compares against."
    if "stub" in key:
        return "StubAdvisor (deterministic, no LLM)", (
            "Rule-based stand-in that runs the real retrieve -> generate -> safety path "
            "with no API key or GPU. Citations are real; the prose is templated."
        )
    if "mock" in key:
        return "Mock fixture response", "Canned fixture (USE_MOCKS=1) -- no retrieval or model ran."
    return raw or "unknown", "Unrecognised generator id; shown verbatim from the API."


# --------------------------------------------------------------------------- #
# Concern data (pure)
# --------------------------------------------------------------------------- #

def concern_rows(analysis: dict[str, Any]) -> list[tuple[str, float]]:
    """[(concern, score)] sorted by score descending, scores clamped to [0, 1]."""
    rows: list[tuple[str, float]] = []
    for item in (analysis or {}).get("concerns", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("concern", "?"))
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        rows.append((name, max(0.0, min(1.0, score))))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def top_concerns(analysis: dict[str, Any], threshold: float = TOP_CONCERN_THRESHOLD) -> list[str]:
    """Same rule as SkinAnalysis.top_concerns() in app/schemas.py: everything over the
    threshold, uncapped."""
    return [name for name, score in concern_rows(analysis) if score >= threshold]


def query_concerns_for_retrieval(
    analysis: dict[str, Any],
    threshold: float = TOP_CONCERN_THRESHOLD,
    max_n: int = MAX_QUERY_CONCERNS,
) -> list[str]:
    """Same rule as skincare.rag.retrieve.query_concerns(): over the threshold, strongest
    first, capped at max_n.

    This is deliberately not top_concerns(). The backend applies the cap between the two,
    and the analysis screen used to claim every concern over 0.50 reached retrieval --
    which overstates what the query contains whenever more than three fire.
    """
    rows = [(name, score) for name, score in concern_rows(analysis) if score >= threshold]
    return [name for name, _ in rows[:max_n]]


def pretty_concern(name: str) -> str:
    """'dark_spots' -> 'dark spots'."""
    return str(name).replace("_", " ")


# --------------------------------------------------------------------------- #
# Brand assets (base64-inlined SVG -- no network, no extra dependency)
# --------------------------------------------------------------------------- #

def svg_data_uri(path: Path | str) -> str:
    """Read an SVG and return a data: URI, or "" if it cannot be read.

    Every caller has a text-only fallback, so a missing or unreadable asset degrades
    to a wordmark instead of raising during a recorded demo.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return ""
    if not raw.strip():
        return ""
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")


def logo_html(height_px: int = 58, align: str = "left") -> str:
    """The full lockup as an <img>, or a styled text wordmark if the file is missing."""
    uri = svg_data_uri(LOGO_PATH)
    justify = "center" if align == "center" else "flex-start"
    if uri:
        return (
            f'<div class="ss-logo" style="justify-content:{justify}">'
            f'<img src="{uri}" alt="{BRAND_NAME}" style="height:{height_px}px" /></div>'
        )
    return (
        f'<div class="ss-logo" style="justify-content:{justify}">'
        f'<div><div class="ss-wordmark">Skin<span>sight</span></div>'
        f'<div class="ss-wordmark-sub">{html.escape(BRAND_TAGLINE.upper())}</div></div></div>'
    )


def page_icon() -> str:
    """Square mark for the browser tab; falls back to a glyph Streamlit accepts."""
    return svg_data_uri(MARK_PATH) or "*"


# --------------------------------------------------------------------------- #
# Concern meter chart (inline HTML/CSS -- no plotting dependency)
# --------------------------------------------------------------------------- #
#
# Form: one measure (a 0-1 score) across six named categories, and the story is "this
# one is the strongest" -- so this is a *meter row per category with emphasis*, not a
# categorical chart. Colour therefore does no identity work at all:
#
#   * the strongest concern       -> accent fill  (the one thing the screen is about)
#   * other concerns searched on  -> ink fill
#   * concerns below the cap      -> muted fill   (demoted, still readable)
#   * unfilled track              -> line, a lighter neutral step of the same ramp
#
# Contrast on paper #FBF9F7 was measured, not eyeballed: ink 17.9:1, accent 3.37:1,
# muted 3.63:1 -- all clear the 3:1 floor for marks. Values are set in ink (a text
# token), never in the mark's colour, and the same numbers are repeated in the table
# view on the "How we got this" tab so nothing is colour-gated.

def concern_meter_html(
    rows: list[tuple[str, float]],
    threshold: float = TOP_CONCERN_THRESHOLD,
    searched: list[str] | None = None,
) -> str:
    """Horizontal meters, 0-1 scale, the retrieval threshold marked as a hairline.

    `searched` is the subset that actually went into the retrieval query (see
    query_concerns_for_retrieval); those rows stay at full strength and the rest are
    demoted, so the MAX_QUERY_CONCERNS cap is visible and not only stated in a caption.
    """
    if not rows:
        return ""
    highlight = set(searched) if searched is not None else {name for name, _ in rows}
    thr_pct = max(0.0, min(1.0, threshold)) * 100.0

    parts = ['<div class="ss-meter">']
    for rank, (name, score) in enumerate(rows):
        if rank == 0 and name in highlight:
            tone = "hot"          # single accent: the strongest concern
        elif name in highlight:
            tone = "on"
        else:
            tone = "off"
        parts.append(
            f'<div class="ss-mrow {tone}">'
            f'<div class="ss-mname">{html.escape(pretty_concern(name))}</div>'
            '<div class="ss-mtrack">'
            f'<div class="ss-mfill" style="width:{score * 100:.1f}%"></div>'
            f'<div class="ss-mthr" style="left:{thr_pct:.1f}%"></div>'
            "</div>"
            f'<div class="ss-mval">{score:.2f}</div>'
            "</div>"
        )
    parts.append(
        '<div class="ss-mscale"><div></div>'
        '<div class="ss-mscale-track"><span class="ss-mscale-lo">0.00</span>'
        f'<span class="ss-mscale-thr" style="left:{thr_pct:.1f}%">{threshold:.2f} search '
        "threshold</span>"
        '<span class="ss-mscale-hi">1.00</span></div><div></div></div>'
    )
    parts.append("</div>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Annotated selfie (Olay-style pill labels with leader lines)
# --------------------------------------------------------------------------- #
#
# Geometry is defined once in a fixed 600x430 layout box and emitted as percentages,
# so the whole thing scales with the column while keeping its angles (the wrapper
# carries `aspect-ratio`, so the box scales uniformly).
#
# IMPORTANT HONESTY NOTE: these three positions are decorative. The concern head scores
# the photo as a whole and does not localise anything, so the label under the picture
# says so. Anything else would be the UI inventing a claim the model never made.

FACE_BOX_W = 600.0
FACE_BOX_H = 430.0
FACE_FRAME_X = 130.0          # photo frame: x .. x + w inside the box
FACE_FRAME_W = 340.0

# dot = where the leader line lands on the photo; pill = the label's connecting edge.
# The dots sit in the upper two thirds of the frame because the photo is cover-cropped
# with object-position 50% 32%, which is where a phone selfie puts a face.
FACE_ANCHORS = [
    {"where": "forehead", "dot": (300.0, 78.0), "pill": (486.0, 40.0), "side": "right"},
    {"where": "cheek", "dot": (236.0, 200.0), "pill": (114.0, 165.0), "side": "left"},
    {"where": "jaw", "dot": (330.0, 284.0), "pill": (486.0, 322.0), "side": "right"},
]

# Used when the demo runs from a preset instead of a live upload: the same annotated
# composition, with an obvious placeholder instead of a face.
_FACE_PLACEHOLDER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 340 430">'
    '<rect width="340" height="430" fill="#F2ECE8"/>'
    '<ellipse cx="170" cy="170" rx="95" ry="125" fill="#E7E1DC"/>'
    '<path d="M18 430 C 44 372, 108 318, 170 318 C 232 318, 296 372, 322 430 Z" '
    'fill="#E7E1DC"/></svg>'
)


def face_placeholder_uri() -> str:
    """Data URI for the no-photo silhouette."""
    return "data:image/svg+xml;base64," + base64.b64encode(
        _FACE_PLACEHOLDER_SVG.encode("utf-8")
    ).decode("ascii")


def image_data_uri(content: bytes, mime: str = "image/jpeg") -> str:
    """Base64 an uploaded image for inline HTML. Returns "" for empty input."""
    if not content:
        return ""
    safe_mime = mime if isinstance(mime, str) and mime.startswith("image/") else "image/jpeg"
    return f"data:{safe_mime};base64," + base64.b64encode(content).decode("ascii")


def _leader_geometry(pill: tuple[float, float], dot: tuple[float, float]) -> tuple[float, float]:
    """(length, angle_degrees) of the connector from the pill edge to the dot."""
    dx = dot[0] - pill[0]
    dy = dot[1] - pill[1]
    return math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))


def annotated_face_html(
    image_uri: str,
    labels: list[tuple[str, float]],
    searched: list[str] | None = None,
) -> str:
    """The uploaded photo with up to three concern pills and leader lines over it.

    `labels` is [(concern, score)] strongest first; the first one is the accent pill.
    `searched` is the retrieval subset -- a pill for a concern that did not reach the
    query is dimmed, exactly like its row in the meter chart, so the two agree.
    """
    if not image_uri:
        return ""
    frame_left = FACE_FRAME_X / FACE_BOX_W * 100.0
    frame_w = FACE_FRAME_W / FACE_BOX_W * 100.0

    parts = [
        '<div class="ss-face">',
        f'<div class="ss-face-shadow" style="left:{frame_left:.2f}%;width:{frame_w:.2f}%"></div>',
        (
            f'<div class="ss-face-frame" style="left:{frame_left:.2f}%;width:{frame_w:.2f}%">'
            f'<img src="{image_uri}" alt="Your uploaded photo" /></div>'
        ),
    ]

    for rank, (name, score) in enumerate(labels[: len(FACE_ANCHORS)]):
        anchor = FACE_ANCHORS[rank]
        dot = anchor["dot"]
        pill = anchor["pill"]
        length, angle = _leader_geometry(pill, dot)
        if searched is not None and name not in searched:
            tone = " dim"
        elif rank == 0:
            tone = " hot"
        else:
            tone = ""
        parts.append(
            f'<div class="ss-leader{tone}" style="left:{pill[0] / FACE_BOX_W * 100:.2f}%;'
            f"top:{pill[1] / FACE_BOX_H * 100:.2f}%;"
            f"width:{length / FACE_BOX_W * 100:.2f}%;"
            f'transform:rotate({angle:.2f}deg)"></div>'
        )
        parts.append(
            f'<div class="ss-dot{tone}" style="left:{dot[0] / FACE_BOX_W * 100:.2f}%;'
            f'top:{dot[1] / FACE_BOX_H * 100:.2f}%"></div>'
        )
        if anchor["side"] == "right":
            place = f"left:{pill[0] / FACE_BOX_W * 100:.2f}%"
            side_cls = " right"
        else:
            place = f"right:{(FACE_BOX_W - pill[0]) / FACE_BOX_W * 100:.2f}%"
            side_cls = " left"
        parts.append(
            f'<div class="ss-pill{tone}{side_cls}" style="{place};'
            f'top:{pill[1] / FACE_BOX_H * 100:.2f}%">'
            f'<span class="ss-pill-name">{html.escape(pretty_concern(name))}</span>'
            f'<span class="ss-pill-score">{score:.2f}</span></div>'
        )

    parts.append("</div>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Product tiles (deterministic placeholder imagery)
# --------------------------------------------------------------------------- #
#
# The catalogue has no image column (product_id, name, brand, category, price_usd,
# rating, ingredients), so a product tile is generated instead of fetched: the hue
# comes from a hash of product_id, which makes it stable across runs -- the same
# product gets the same tile in the recording and in the live demo. Saturation and
# lightness are pinned to a soft pastel band so the tiles read as one family and the
# terracotta accent stays the only saturated colour on the screen.

# Hues are quantised onto a 24-slot golden-angle lattice rather than taken straight off
# the hash: a raw `hash % 360` lets two products in the same routine land 2 degrees
# apart and look identical, whereas any two distinct slots here differ by at least ~15.
TILE_HUE_SLOTS = 24
TILE_GOLDEN_ANGLE = 137.5

# Two shades per hue, chosen from an independent slice of the same digest. Hue alone is
# a weak signal at this lightness -- 30 degrees apart in the greens still reads as "the
# same pale green" -- so the second axis is what actually keeps two tiles in one routine
# apart. Both stay inside the soft band; neither competes with the accent.
TILE_SHADES = [(92, 26, 80, 30), (85, 30, 72, 34)]   # (L1, S1, L2, S2)


def _tile_digest(seed: str) -> str:
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()


def tile_hue(seed: str) -> int:
    """Deterministic 0-359 hue from a product id (or any stable string)."""
    slot = int(_tile_digest(seed)[:8], 16) % TILE_HUE_SLOTS
    return int(slot * TILE_GOLDEN_ANGLE) % 360


def tile_shade(seed: str) -> tuple[int, int, int, int]:
    """Deterministic (lightness, saturation) pair for the tile's two gradient stops."""
    return TILE_SHADES[int(_tile_digest(seed)[8:16], 16) % len(TILE_SHADES)]


def brand_initials(brand: str, name: str = "") -> str:
    """Up to two initials for the tile: 'The Ordinary' -> 'TO'."""
    source = (brand or name or "?").strip()
    words = [w for w in source.replace("-", " ").split() if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def tile_gradient_css(seed: str) -> str:
    """The tile's two-stop gradient as a CSS value, keyed to the seed (usually product_id)."""
    hue = tile_hue(seed)
    hue2 = (hue + 26) % 360
    light1, sat1, light2, sat2 = tile_shade(seed)
    return (
        f"linear-gradient(145deg,hsl({hue},{sat1}%,{light1}%) 0%,"
        f"hsl({hue2},{sat2}%,{light2}%) 100%)"
    )


# --------------------------------------------------------------------------- #
# Product category (derived in the UI, never invented in the payload)
# --------------------------------------------------------------------------- #
#
# app/schemas.py is a frozen contract and Recommendation carries no category field --
# only product_id, name, brand, price_usd, reason, key_ingredients, cited_evidence and
# matched_concerns. Rather than add a field (which would break everyone else), the
# listing derives a category from words already present in the product name, the same
# way a shopper reads a shelf. First match in CATEGORY_RULES wins, so the order below
# is the rule: the more specific reading is checked first ("Oil-Free Moisturizer" is a
# moisturiser, not an oil; "Eye Cream" is eye care, not a cream). A name that matches
# nothing gets the neutral "product" bottle -- the UI does not guess.

CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("sunscreen", ("sunscreen", "sunblock", "spf", "sun cream", "uv ")),
    ("eye", ("eye",)),
    ("exfoliant", ("exfoliat", "peel", "scrub", "polish")),
    ("mask", ("mask", "masque")),
    ("cleanser", ("cleanser", "cleansing", "cleanse", "wash", "foam")),
    ("toner", ("toner", "essence", "mist", "spray", "tonic")),
    ("serum", ("serum", "ampoule", "concentrate")),
    ("moisturiser", ("moisturis", "moisturiz", "cream", "lotion", "hydrator", "balm")),
    ("oil", ("oil",)),
]

# What each derived key is called on screen.
CATEGORY_LABELS = {
    "sunscreen": "Sunscreen",
    "eye": "Eye care",
    "exfoliant": "Exfoliant",
    "mask": "Mask",
    "cleanser": "Cleanser",
    "toner": "Toner",
    "serum": "Serum",
    "moisturiser": "Moisturiser",
    "oil": "Oil",
    "product": "Skincare",
}

# One inline SVG body per category key -- stroke-only paths on a 24x24 grid, drawn in the
# ink colour by category_icon_html(). No emoji, no icon font, no extra dependency, and
# nothing fetched over the network during the demo.
CATEGORY_ICONS = {
    # dropper bottle
    "serum": (
        '<path d="M10 3.2h4"/>'
        '<path d="M11 3.2v3.4M13 3.2v3.4"/>'
        '<path d="M9.4 6.6h5.2a1.1 1.1 0 0 1 1.1 1.1v11.9a1.1 1.1 0 0 1-1.1 1.1H9.4'
        'a1.1 1.1 0 0 1-1.1-1.1V7.7a1.1 1.1 0 0 1 1.1-1.1z"/>'
        '<path d="M10.4 11.4h3.2"/>'
    ),
    # wide cream jar
    "moisturiser": (
        '<path d="M9.6 3.4h4.8"/>'
        '<path d="M4.9 6.3h14.2v3.1H4.9z"/>'
        '<path d="M5.7 9.4h12.6v9.1a2 2 0 0 1-2 2H7.7a2 2 0 0 1-2-2z"/>'
    ),
    # pump bottle
    "cleanser": (
        '<path d="M11 9.2V6.4h2.4"/>'
        '<path d="M13.4 6.4H16V4.1"/>'
        '<path d="M8.6 9.2h6.8a1.4 1.4 0 0 1 1.4 1.4v8.5a1.6 1.6 0 0 1-1.6 1.6H8.8'
        'a1.6 1.6 0 0 1-1.6-1.6v-8.5a1.4 1.4 0 0 1 1.4-1.4z"/>'
        '<path d="M7.4 13.6h9.2"/>'
    ),
    # sheet mask
    "mask": (
        '<path d="M6 4.6h12v8.1a6 6 0 0 1-12 0z"/>'
        '<path d="M9.1 9.1h1.8M13.1 9.1h1.8"/>'
        '<path d="M10.6 13.5h2.8"/>'
    ),
    # slim bottle with a puff of mist
    "toner": (
        '<path d="M10.6 8.4V5.2h2.8v3.2"/>'
        '<path d="M9.8 8.4h4.4a1.2 1.2 0 0 1 1.2 1.2v9.5a1.4 1.4 0 0 1-1.4 1.4H10'
        'a1.4 1.4 0 0 1-1.4-1.4V9.6a1.2 1.2 0 0 1 1.2-1.2z"/>'
        '<path d="M17.2 6.4l1.6-1M17.8 9.6l1.9-.3M17 12.6l1.7.7"/>'
    ),
    # droplet
    "oil": (
        '<path d="M12 3.6c3.4 4 5.4 6.6 5.4 9.3A5.4 5.4 0 0 1 12 18.3'
        'a5.4 5.4 0 0 1-5.4-5.4c0-2.7 2-5.3 5.4-9.3z"/>'
        '<path d="M9.7 13.3a2.4 2.4 0 0 0 2.3 2.4"/>'
    ),
    # sun
    "sunscreen": (
        '<circle cx="12" cy="12" r="4.1"/>'
        '<path d="M12 2.9v2.3M12 18.8v2.3M2.9 12h2.3M18.8 12h2.3"/>'
        '<path d="M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6"/>'
    ),
    # eye
    "eye": (
        '<path d="M2.7 12S6.3 6.3 12 6.3 21.3 12 21.3 12 17.7 17.7 12 17.7 2.7 12 2.7 12z"/>'
        '<circle cx="12" cy="12" r="2.4"/>'
    ),
    # tube with grains
    "exfoliant": (
        '<path d="M9.6 4.2h4.8l1.2 3.4H8.4z"/>'
        '<path d="M8.4 7.6h7.2v11a2 2 0 0 1-2 2h-3.2a2 2 0 0 1-2-2z"/>'
        '<circle cx="11" cy="12.4" r="0.9"/><circle cx="13.3" cy="15" r="0.9"/>'
        '<circle cx="11.2" cy="17.4" r="0.9"/>'
    ),
    # neutral fallback bottle -- used whenever the name matches no rule above
    "product": (
        '<path d="M10 2.8h4v3.4l2.2 2.6a3 3 0 0 1 .7 1.9V19a2 2 0 0 1-2 2H9.1a2 2 0 0 1-2-2'
        'v-8.3a3 3 0 0 1 .7-1.9L10 6.2z"/>'
        '<path d="M7.4 13.6h9.2"/>'
    ),
}

# A clean check and a raised flag, drawn the same way. The caution mark is the only place
# on this screen that earns the accent, matching the callout's left border.
SAFETY_ICONS = {
    "shield": (
        '<path d="M12 3.2l7 2.6v5.4c0 4.3-2.9 7.6-7 9.6-4.1-2-7-5.3-7-9.6V5.8z"/>'
        '<path d="M8.8 12.1l2.3 2.4 4.1-4.6"/>'
    ),
    "caution": (
        '<path d="M12 4.2l8.4 14.6H3.6z"/>'
        '<path d="M12 9.7v4.1"/><path d="M12 16.2v.7"/>'
    ),
}

INK = "#14110F"
ACCENT = "#C8705F"


def product_category(name: str) -> str:
    """'Vitamin C Serum 30ml' -> 'serum'. Returns 'product' when nothing matches."""
    lowered = str(name or "").lower()
    for key, keywords in CATEGORY_RULES:
        if any(keyword in lowered for keyword in keywords):
            return key
    return "product"


def _icon_svg(body: str, size: int, stroke: str, css_class: str) -> str:
    return (
        f'<svg class="{css_class}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{stroke}" stroke-width="1.4" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


def category_icon_html(category: str, size: int = 13) -> str:
    """The inline SVG for a category key, falling back to the neutral bottle."""
    body = CATEGORY_ICONS.get(category) or CATEGORY_ICONS["product"]
    return _icon_svg(body, size, INK, "ss-cicon")


def safety_icon_html(kind: str, size: int = 15) -> str:
    """'shield' (clean check, ink) or 'caution' (flags raised, accent)."""
    body = SAFETY_ICONS.get(kind) or SAFETY_ICONS["shield"]
    stroke = ACCENT if kind == "caution" else INK
    return _icon_svg(body, size, stroke, "ss-sfxicon")


# --------------------------------------------------------------------------- #
# Purchase link (a catalogue search, not a product deep-link)
# --------------------------------------------------------------------------- #
#
# The catalogue is Sephora's and product_id looks like a Sephora sku ("P440482"), but no
# canonical deep-link form has been verified to resolve, and a link that 404s in a
# recorded demo is worse than no link. So the card links to a *search* for the brand and
# product name, which always lands somewhere real, and the tab says so in plain words.

SEPHORA_SEARCH_URL = "https://www.sephora.com/search?keyword="


def sephora_search_url(brand: str, name: str) -> str:
    """Search URL for '<brand> <name>', or "" when there is nothing to search for."""
    terms = [str(brand or "").strip(), str(name or "").strip()]
    keyword = " ".join(term for term in terms if term)
    if not keyword:
        return ""
    return SEPHORA_SEARCH_URL + quote_plus(keyword)


def stars_html(rating: Any) -> str:
    """Rating as five glyphs plus the number, or "" when the field is absent."""
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return ""
    if not 0.0 < value <= 5.0:
        return ""
    filled = round(value)
    glyphs = "★" * filled + "☆" * (5 - filled)
    return f'<span class="ss-stars">{glyphs}</span><span class="ss-rating">{value:.1f}</span>'


def product_card_html(index: int, rec: dict[str, Any]) -> str:
    """One recommendation as a listing card, read top to bottom like a shop shelf.

    Square gradient tile first (a listing leads with the product picture; ours is
    generated, see above), the category icon and the routine step pinned on it, then
    brand in small caps, the name in the serif face, the price as the largest thing in
    the body, the matched concerns as tags, and a quiet search link last.

    Everything optional degrades to nothing: no price, no rating and no matched concerns
    all simply drop their row rather than printing a placeholder.
    """
    raw_name = str(rec.get("name", "") or "")
    name = html.escape(raw_name or "Unnamed product")
    brand = str(rec.get("brand", "") or "")
    seed = str(rec.get("product_id", "")) or raw_name or brand

    category = product_category(raw_name)
    chip = (
        f'<span class="ss-pchip">{category_icon_html(category)}'
        f'<span>{html.escape(CATEGORY_LABELS.get(category, CATEGORY_LABELS["product"]))}'
        "</span></span>"
    )
    tile = (
        f'<div class="ss-ptile" style="background:{tile_gradient_css(seed)}">'
        f'<span class="ss-ptile-mark">{html.escape(brand_initials(brand, raw_name))}</span>'
        f'{chip}<span class="ss-pstepbadge">Step {int(index)}</span></div>'
    )

    price = rec.get("price_usd")
    price_html = ""
    if isinstance(price, (int, float)) and not isinstance(price, bool):
        price_html = f'<div class="ss-pprice">${float(price):.2f}</div>'
    rating = stars_html(rec.get("rating"))
    rating_html = f'<div class="ss-prating">{rating}</div>' if rating else ""

    tags = "".join(
        f'<span class="ss-tag">{html.escape(pretty_concern(str(m)))}</span>'
        for m in (rec.get("matched_concerns") or [])
    )
    tags_html = f'<div class="ss-ptags">{tags}</div>' if tags else ""

    url = sephora_search_url(brand, raw_name)
    link_html = (
        f'<a class="ss-plink" href="{html.escape(url, quote=True)}" target="_blank" '
        'rel="noopener noreferrer">Find on Sephora &#8599;</a>'
        if url
        else ""
    )

    return (
        f'<div class="ss-pcard">{tile}'
        '<div class="ss-pbody">'
        f'<div class="ss-pbrand">{html.escape(brand)}</div>'
        f'<div class="ss-pname">{name}</div>'
        f"{price_html}{rating_html}{tags_html}{link_html}"
        "</div></div>"
    )


# --------------------------------------------------------------------------- #
# Theme (inline CSS -- pure Streamlit, no component libraries)
# --------------------------------------------------------------------------- #
#
# One surface only. The app pins itself to the light "paper" surface (see
# .streamlit/config.toml) rather than shipping a half-considered dark mode: the brand
# assets, the tiles and the chart contrast figures were all validated against paper,
# and a recorded demo has exactly one surface to get right.

_APP_CSS = """
<style>
:root {
  --ink: #14110F;
  --accent: #C8705F;
  --paper: #FBF9F7;
  --muted: #8A817C;
  --line: #E7E1DC;
  --serif: Georgia, 'Times New Roman', Times, serif;
  --sans: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

.stApp, [data-testid="stAppViewContainer"] { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 48rem; }
/* Everything gets the brand sans EXCEPT Streamlit's Material icon spans: those render
   their glyph as a ligature, so overriding their font shows the literal word
   ("keyboard_arrow_right") in place of the expander arrow. */
html, body { font-family: var(--sans); }
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) {
  font-family: var(--sans); color: var(--ink); }

/* ---------- type scale ---------- */
.ss-eyebrow { font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 0.55rem 0; }
.ss-title { font-family: var(--serif); font-size: 2.9rem; line-height: 1.08;
  font-weight: 400; margin: 0 0 0.7rem 0; letter-spacing: -0.01em; }
.ss-question { font-family: var(--serif); font-size: 2.05rem; line-height: 1.15;
  font-weight: 400; margin: 0 0 0.35rem 0; }
.ss-sub { font-size: 1.0rem; color: var(--muted); line-height: 1.55; margin: 0 0 1.5rem 0; }
.ss-muted { font-size: 0.8rem; color: var(--muted); line-height: 1.55; }
.ss-rule { height: 1px; background: var(--line); margin: 1.6rem 0; }
.ss-body { font-size: 0.98rem; line-height: 1.6; margin: 0 0 0.9rem 0; }

/* ---------- logo ---------- */
.ss-logo { display: flex; align-items: center; margin-bottom: 1.6rem; }
.ss-wordmark { font-family: var(--serif); font-size: 2.1rem; letter-spacing: 0.01em; }
.ss-wordmark span { color: var(--accent); font-style: italic; }
.ss-wordmark-sub { font-size: 0.62rem; letter-spacing: 0.32em; color: var(--muted); }

/* ---------- buttons: one primary action per screen ---------- */
div.stButton > button { border-radius: 999px; padding: 0.6rem 1.5rem; font-weight: 600;
  border: 1px solid var(--line); background: transparent; color: var(--ink); }
div.stButton > button:hover { border-color: var(--ink); color: var(--ink); }
div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] * {
  color: var(--ink); }
div.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent);
  letter-spacing: 0.02em; }
/* The label lives in a nested <p>, so the hover colour has to be set on the children
   too -- otherwise the inherited ink text disappears into the inverted background. */
div.stButton > button[kind="primary"]:hover, div.stButton > button[kind="primary"]:hover * {
  color: var(--paper); }
div.stButton > button[kind="primary"]:hover { background: var(--ink); border-color: var(--ink); }

/* ---------- progress: numbered circles, active one in accent ---------- */
.ss-prog { display: flex; margin: 0.2rem 0 2.1rem 0; }
.ss-pstep { flex: 1 1 0; text-align: center; position: relative; }
.ss-pstep::before { content: ""; position: absolute; top: 15px; left: -50%; width: 100%;
  height: 1px; background: var(--line); z-index: 0; }
.ss-pstep:first-child::before { display: none; }
.ss-pstep.done::before { background: var(--ink); opacity: 0.35; }
.ss-pdot { position: relative; z-index: 1; width: 31px; height: 31px; border-radius: 999px;
  display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem;
  border: 1px solid var(--line); background: var(--paper); color: var(--muted); }
.ss-pstep.done .ss-pdot { border-color: var(--ink); color: var(--ink); }
.ss-pstep.now .ss-pdot { background: var(--accent); border-color: var(--accent);
  color: var(--ink); font-weight: 700; }
.ss-plabel { display: block; margin-top: 0.5rem; font-size: 0.63rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); }
.ss-pstep.now .ss-plabel { color: var(--ink); }

/* ---------- landing step cards ---------- */
.ss-scard { border: 1px solid var(--line); border-radius: 10px; padding: 1rem 0.9rem 1.1rem;
  height: 100%; background: rgba(255,255,255,0.5); }
.ss-scard .ss-sicon { display: block; margin-bottom: 0.6rem; }
.ss-scard .ss-snum { font-size: 0.63rem; letter-spacing: 0.16em; color: var(--muted);
  text-transform: uppercase; }
.ss-scard .ss-stext { font-family: var(--serif); font-size: 1.02rem; line-height: 1.3;
  margin-top: 0.25rem; }

/* ---------- stat tiles (the one number that matters) ---------- */
.ss-stats { display: flex; gap: 0.9rem; margin: 0.2rem 0 1.4rem 0; flex-wrap: wrap; }
.ss-stat { flex: 1 1 160px; border: 1px solid var(--line); border-radius: 10px;
  padding: 0.85rem 1rem 0.95rem; background: rgba(255,255,255,0.5); }
.ss-stat .ss-stat-label { font-size: 0.63rem; letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 0.35rem; }
.ss-stat .ss-stat-value { font-family: var(--serif); font-size: 2.15rem; line-height: 1.05; }
.ss-stat .ss-stat-num { font-family: var(--sans); font-size: 2.15rem; font-weight: 600;
  line-height: 1.05; }
.ss-stat .ss-stat-foot { font-size: 0.76rem; color: var(--muted); margin-top: 0.3rem; }

/* ---------- concern meters ---------- */
.ss-meter { margin: 0.3rem 0 0.4rem 0; }
.ss-mrow { display: grid; grid-template-columns: 110px 1fr 42px; align-items: center;
  column-gap: 12px; margin-bottom: 10px; }
.ss-mname { font-size: 0.84rem; text-align: right; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.ss-mtrack { position: relative; height: 12px; background: var(--line); border-radius: 2px; }
.ss-mfill { position: absolute; left: 0; top: 0; height: 100%; background: var(--ink);
  border-radius: 0 4px 4px 0; }
.ss-mthr { position: absolute; top: -3px; height: calc(100% + 6px); width: 1px;
  background: var(--ink); opacity: 0.45; }
.ss-mval { font-size: 0.82rem; text-align: right; font-variant-numeric: tabular-nums; }
.ss-mrow.hot .ss-mfill { background: var(--accent); }
.ss-mrow.hot .ss-mname, .ss-mrow.hot .ss-mval { font-weight: 700; }
.ss-mrow.off .ss-mfill { background: var(--muted); opacity: 0.55; }
.ss-mrow.off .ss-mname, .ss-mrow.off .ss-mval { color: var(--muted); }
.ss-mscale { display: grid; grid-template-columns: 110px 1fr 42px; column-gap: 12px;
  margin-top: 2px; }
.ss-mscale-track { position: relative; height: 1.1rem; font-size: 0.66rem; color: var(--muted);
  letter-spacing: 0.06em; }
.ss-mscale-lo { position: absolute; left: 0; }
.ss-mscale-hi { position: absolute; right: 0; }
.ss-mscale-thr { position: absolute; transform: translateX(-50%); white-space: nowrap; }

/* ---------- annotated selfie ---------- */
.ss-face { position: relative; width: 100%; max-width: 600px; aspect-ratio: 600 / 430;
  margin: 0.4rem auto 0.6rem; }
.ss-face-frame { position: absolute; top: 0; height: 100%; overflow: hidden;
  border-radius: 170px 170px 12px 12px; }
.ss-face-frame img { width: 100%; height: 100%; object-fit: cover; object-position: 50% 32%;
  display: block; }
.ss-face-shadow { position: absolute; top: 14px; height: 100%; background: var(--line);
  border-radius: 170px 170px 12px 12px; transform: translateX(14px); }
.ss-leader { position: absolute; height: 1px; background: var(--ink); opacity: 0.45;
  transform-origin: 0 50%; }
.ss-leader.hot { background: var(--accent); opacity: 0.9; }
.ss-leader.dim { background: var(--muted); opacity: 0.35; }
.ss-dot { position: absolute; width: 9px; height: 9px; border-radius: 999px;
  background: var(--ink); box-shadow: 0 0 0 2px var(--paper); transform: translate(-50%, -50%); }
.ss-dot.hot { background: var(--accent); }
.ss-dot.dim { background: var(--muted); }
.ss-pill { position: absolute; transform: translateY(-50%); background: var(--paper);
  border: 1px solid var(--line); border-radius: 999px; padding: 0.3rem 0.7rem;
  display: flex; align-items: baseline; gap: 0.45rem; white-space: nowrap;
  box-shadow: 0 1px 3px rgba(20,17,15,0.06); }
.ss-pill.hot { background: var(--accent); border-color: var(--accent); }
.ss-pill.dim { color: var(--muted); }
.ss-pill.dim .ss-pill-name, .ss-pill.dim .ss-pill-score { color: var(--muted); }
.ss-pill-name { font-size: 0.64rem; letter-spacing: 0.14em; text-transform: uppercase; }
.ss-pill-score { font-size: 0.74rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.ss-preview { border-radius: 90px 90px 10px 10px; overflow: hidden; border: 1px solid var(--line);
  aspect-ratio: 4 / 5; }
.ss-preview img { width: 100%; height: 100%; object-fit: cover; object-position: 50% 32%;
  display: block; }
.ss-face-note { text-align: center; font-size: 0.74rem; color: var(--muted); line-height: 1.5;
  max-width: 34rem; margin: 0 auto 0.6rem; }

/* ---------- product listing (step 4, "Your routine") ---------- */
/* Selectors here are deliberately two-class (.ss-pcard .ss-pbrand, not .ss-pbrand): the
   global `[data-testid="stAppViewContainer"] *:not(...)` rule near the top of this sheet
   paints every descendant in ink, and a single class loses to it on specificity. Two
   classes tie, and a tie later in the sheet wins -- so the listing keeps its hierarchy
   (muted brand, muted link) without weakening that rule for the rest of the app. */
.ss-pcard { border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
  background: rgba(255,255,255,0.55); display: flex; flex-direction: column;
  height: 100%; margin-bottom: 0.2rem; }
.ss-pcard .ss-ptile { position: relative; width: 100%; aspect-ratio: 1 / 1; display: flex;
  align-items: center; justify-content: center; }
.ss-pcard .ss-ptile-mark { font-family: var(--serif); font-size: 2.4rem; letter-spacing: 0.03em;
  color: var(--ink); opacity: 0.9; }
.ss-pcard .ss-pchip { position: absolute; left: 8px; top: 8px; display: inline-flex;
  align-items: center; gap: 0.3rem; background: var(--paper); border: 1px solid var(--line);
  border-radius: 999px; padding: 0.17rem 0.5rem 0.17rem 0.42rem; font-size: 0.55rem;
  letter-spacing: 0.13em; text-transform: uppercase; color: var(--ink); }
.ss-pcard .ss-pchip svg { display: block; }
.ss-pcard .ss-pstepbadge { position: absolute; right: 8px; top: 8px; font-size: 0.55rem;
  letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); background: var(--paper);
  border: 1px solid var(--line); border-radius: 999px; padding: 0.17rem 0.5rem; }
.ss-pcard .ss-pbody { padding: 0.8rem 0.85rem 0.95rem; display: flex; flex-direction: column;
  flex: 1 1 auto; }
.ss-pcard .ss-pbrand { font-size: 0.6rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); }
.ss-pcard .ss-pname { font-family: var(--serif); font-size: 1.15rem; line-height: 1.25;
  margin: 0.22rem 0 0.5rem 0; }
.ss-pcard .ss-pprice { font-size: 1.55rem; font-weight: 600; line-height: 1;
  letter-spacing: -0.01em; }
.ss-pcard .ss-prating { margin-top: 0.35rem; }
.ss-pcard .ss-rating { color: var(--muted); }
.ss-pcard .ss-ptags { margin-top: 0.6rem; }
.ss-pcard .ss-tag { color: var(--muted); }
.ss-pcard .ss-plink { display: inline-block; align-self: flex-start; margin-top: auto;
  padding-top: 0.75rem; font-size: 0.72rem; color: var(--muted); text-decoration: none; }
.ss-pcard .ss-plink:hover { color: var(--ink); text-decoration: underline; }
.ss-card-name { font-family: var(--serif); font-size: 1.45rem; line-height: 1.2;
  margin: 0.15rem 0 0.1rem 0; }
.ss-stars { letter-spacing: 0.08em; font-size: 0.9rem; }
.ss-rating { font-size: 0.78rem; color: var(--muted); }
.ss-tag { display: inline-block; font-size: 0.68rem; letter-spacing: 0.08em;
  text-transform: uppercase; padding: 0.16rem 0.62rem; margin: 0 0.3rem 0.3rem 0;
  border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.ss-card-reason { font-size: 0.95rem; line-height: 1.6; margin: 0.75rem 0 0 0; }

/* ---------- callouts (safety) ---------- */
.ss-callout { border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 6px; padding: 0.85rem 1.1rem; margin: 0.3rem 0 1.1rem 0;
  background: rgba(255,255,255,0.55); }
.ss-callout.ok { border-left-color: var(--ink); }
.ss-callout .ss-callout-title { font-size: 0.66rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); font-weight: 700; margin-bottom: 0.4rem;
  display: flex; align-items: center; gap: 0.42rem; }
.ss-callout .ss-sfxicon { flex: 0 0 auto; }
.ss-callout ul { margin: 0; padding-left: 1.1rem; }
.ss-callout li { margin: 0.2rem 0; font-size: 0.94rem; line-height: 1.5; }
.ss-callout p { margin: 0; font-size: 0.94rem; line-height: 1.5; }

/* ---------- status ---------- */
.ss-status { font-size: 0.76rem; color: var(--muted); display: flex; align-items: center;
  gap: 0.45rem; }
.ss-status .ss-led { width: 7px; height: 7px; border-radius: 999px; background: var(--ink); }
.ss-status .ss-led.off { background: var(--accent); }

/* ---------- tabs: the text lives here instead of stacking ---------- */
/* Two selector families on purpose: current Streamlit renders tabs as
   [data-testid="stTab"] inside a [role="tablist"], older releases used BaseWeb's
   button[data-baseweb="tab"]. Matching both keeps the look on streamlit>=1.33, and a
   miss only costs the letter-spacing -- the tabs still work. */
[data-testid="stTabs"] [role="tablist"], div[data-baseweb="tab-list"] { gap: 1.7rem;
  border-bottom: 1px solid var(--line); background: transparent; margin-bottom: 1.1rem; }
[data-testid="stTab"], button[data-baseweb="tab"] { background: transparent; }
[data-testid="stTab"] p, button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
  font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
[data-testid="stTab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] div[data-testid="stMarkdownContainer"] p {
  color: var(--ink); font-weight: 700; }
div[data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }
div[data-baseweb="tab-border"] { display: none; }

/* ---------- streamlit widgets, lightly ---------- */
div[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 600; }
div[data-testid="stMetricLabel"] p { font-size: 0.66rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted); }
div[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 10px;
  background: rgba(255,255,255,0.4); }
</style>
"""

# Four line-art step icons, drawn inline so the landing page has imagery without a
# network fetch or an image dependency. Stroke follows the palette.
_STEP_ICONS = {
    "camera": (
        '<svg class="ss-sicon" width="26" height="26" viewBox="0 0 24 24" fill="none" '
        'stroke="#14110F" stroke-width="1.2"><path d="M3 8h3.2l1.4-2h7.8l1.4 2H20a1 1 0 0 1 1 1v9'
        'a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/>'
        '<circle cx="11.5" cy="13" r="3.6" stroke="#C8705F"/></svg>'
    ),
    "lens": (
        '<svg class="ss-sicon" width="26" height="26" viewBox="0 0 24 24" fill="none" '
        'stroke="#14110F" stroke-width="1.2"><circle cx="10.5" cy="10.5" r="6.8"/>'
        '<path d="M15.5 15.5 21 21" stroke="#C8705F"/>'
        '<path d="M8 11.5h5" opacity="0.5"/></svg>'
    ),
    "person": (
        '<svg class="ss-sicon" width="26" height="26" viewBox="0 0 24 24" fill="none" '
        'stroke="#14110F" stroke-width="1.2"><circle cx="12" cy="8.4" r="3.9"/>'
        '<path d="M4.5 20c1.4-4 4.1-6 7.5-6s6.1 2 7.5 6" stroke="#C8705F"/></svg>'
    ),
    "bottle": (
        '<svg class="ss-sicon" width="26" height="26" viewBox="0 0 24 24" fill="none" '
        'stroke="#14110F" stroke-width="1.2"><path d="M10 2.8h4v3.4l2.2 2.6a3 3 0 0 1 .7 1.9V19'
        'a2 2 0 0 1-2 2H9.1a2 2 0 0 1-2-2v-8.3a3 3 0 0 1 .7-1.9L10 6.2z"/>'
        '<path d="M7.3 13.6h9.4" stroke="#C8705F"/></svg>'
    ),
}


def progress_html(step: int) -> str:
    """Numbered circles for steps 1..LAST_STEP, the active one in accent."""
    cells = []
    for i in range(1, LAST_STEP + 1):
        if i == step:
            state = "now"
        elif i < step:
            state = "done"
        else:
            state = "todo"
        cells.append(
            f'<div class="ss-pstep {state}"><span class="ss-pdot">{i}</span>'
            f'<span class="ss-plabel">{html.escape(STEP_TITLES.get(i, ""))}</span></div>'
        )
    return f'<div class="ss-prog">{"".join(cells)}</div>'


def callout_html(title: str, body_html: str, tone: str = "warn", icon: str = "") -> str:
    """Calm bordered block used for safety flags and for 'nothing to recommend'.

    `icon` is an optional SAFETY_ICONS key drawn before the title; omitting it leaves the
    block exactly as it was, so every existing caller is unaffected.
    """
    cls = "ss-callout ok" if tone == "ok" else "ss-callout"
    mark = safety_icon_html(icon) if icon else ""
    return (
        f'<div class="{cls}"><div class="ss-callout-title">{mark}'
        f"<span>{html.escape(title)}</span></div>"
        f"{body_html}</div>"
    )


def stat_tile_html(label: str, value: str, foot: str = "", serif: bool = True) -> str:
    """Label / value / caption tile. Numbers use the sans face, words the serif."""
    value_cls = "ss-stat-value" if serif else "ss-stat-num"
    foot_html = f'<div class="ss-stat-foot">{html.escape(foot)}</div>' if foot else ""
    return (
        f'<div class="ss-stat"><div class="ss-stat-label">{html.escape(label)}</div>'
        f'<div class="{value_cls}">{html.escape(value)}</div>{foot_html}</div>'
    )


# --------------------------------------------------------------------------- #
# Navigation / session state
# --------------------------------------------------------------------------- #

def _goto(step: int) -> None:
    """Set the screen. Used as a button on_click so it runs before the rerun."""
    st.session_state["step"] = max(LANDING, min(LAST_STEP, int(step)))


def _restart() -> None:
    """Back to the landing screen with the last answer cleared, for a second take."""
    st.session_state["response"] = None
    st.session_state["request_profile"] = None
    st.session_state["step"] = LANDING


def _apply_preset(presets: list[dict[str, Any]]) -> None:
    """Fill the profile answers from the selected preset (runs before the rerun)."""
    label = st.session_state.get("preset_label")
    preset = next((p for p in presets if p["label"] == label), None)
    if preset is None:
        return
    st.session_state["query"] = preset["query"]
    st.session_state["budget"] = min(float(preset["budget_usd"] or 0.0), BUDGET_MAX)
    st.session_state["pregnant"] = preset["pregnant"]
    st.session_state["prefs"] = list(preset["preferences"])
    st.session_state["avoid"] = list(preset["avoid_ingredients"])
    st.session_state["analysis"] = preset["analysis"]
    st.session_state["analysis_source"] = (
        f"demo preset '{preset['label']}'" if preset["analysis"] else ""
    )
    st.session_state["response"] = None


def _init_state(presets: list[dict[str, Any]]) -> None:
    """Seed the answers from the first preset so the demo opens ready to run."""
    if st.session_state.get("_initialised"):
        return
    first = presets[0]
    st.session_state.setdefault("step", LANDING)
    st.session_state.setdefault("api_url", DEFAULT_API)
    st.session_state.setdefault("preset_label", first["label"])
    st.session_state.setdefault("query", first["query"])
    st.session_state.setdefault("budget", min(float(first["budget_usd"] or 0.0), BUDGET_MAX))
    st.session_state.setdefault("pregnant", first["pregnant"])
    st.session_state.setdefault("prefs", list(first["preferences"]))
    st.session_state.setdefault("avoid", list(first["avoid_ingredients"]))
    st.session_state.setdefault("top_k", 3)
    st.session_state.setdefault("analysis", first["analysis"])
    st.session_state.setdefault(
        "analysis_source", f"demo preset '{first['label']}'" if first["analysis"] else ""
    )
    st.session_state.setdefault("response", None)
    st.session_state.setdefault("request_profile", None)
    # The uploaded photo is kept as bytes, not as the uploader's own state: a
    # file_uploader that a rerun does not draw forgets its file, and the annotated
    # selfie on step 2 has to survive navigating back and forth.
    st.session_state.setdefault("photo_bytes", None)
    st.session_state.setdefault("photo_mime", "image/jpeg")
    st.session_state.setdefault("photo_name", "")
    st.session_state["_initialised"] = True


def current_profile() -> dict[str, Any]:
    """UserProfile-shaped dict from whatever the answers currently hold."""
    return build_profile(
        st.session_state.get("query", ""),
        st.session_state.get("budget", 0.0),
        st.session_state.get("pregnant", False),
        coerce_str_list(st.session_state.get("prefs")),
        coerce_str_list(st.session_state.get("avoid")),
    )


# Above this size the base64 data URI in the DOM starts to cost more than the picture
# is worth in a demo; the annotated frame then falls back to the placeholder.
MAX_INLINE_PHOTO_BYTES = 12_000_000


def session_photo_uri() -> str:
    """Data URI for the stored upload, or "" when there is none / it is too large."""
    content = st.session_state.get("photo_bytes")
    if not isinstance(content, bytes) or not content:
        return ""
    if len(content) > MAX_INLINE_PHOTO_BYTES:
        return ""
    return image_data_uri(content, str(st.session_state.get("photo_mime") or "image/jpeg"))


# --------------------------------------------------------------------------- #
# Rendering (Streamlit)
# --------------------------------------------------------------------------- #

def render_api_error(result: ApiResult, context: str) -> None:
    """Show status + body for a failed call. Never a traceback, never a crash."""
    st.error(f"{context}: {result.error}")
    with st.expander("Response details"):
        st.write(f"HTTP status: {result.status if result.status is not None else 'no response'}")
        if result.body:
            st.code(result.body, language="json")
        else:
            st.write("No response body was received.")
        st.caption(
            "Check that the API is running and that the URL under 'Backend settings' is "
            "correct, e.g. `uvicorn app.main:app --port 8000`."
        )


def render_status_line(api: str) -> None:
    """Quiet backend indicator: a dot, the health word, and the mode the backend reports.

    Only the landing screen calls this. A health check on every rerun would add its
    timeout to every click of the recorded demo whenever the API is down.
    """
    result = get_health(api)
    url = html.escape(normalize_api_url(api))
    if result.ok and isinstance(result.data, dict):
        status = html.escape(str(result.data.get("status", "?")))
        mode = "mock fixtures" if result.data.get("mock_mode") else "real pipeline"
        st.markdown(
            f'<div class="ss-status"><span class="ss-led"></span>'
            f"<span>Backend {status} - {mode}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ss-status"><span class="ss-led off"></span>'
            "<span>Backend unreachable - the walkthrough still opens, but the analysis and "
            "routine steps need the API</span></div>",
            unsafe_allow_html=True,
        )
    with st.expander("Backend settings"):
        # No widget key: session state has to outlive this screen, and Streamlit discards
        # the state of widgets a rerun does not draw.
        st.session_state["api_url"] = st.text_input(
            "API URL", value=st.session_state.get("api_url", DEFAULT_API)
        )
        st.caption(f"Currently calling `{url}`.")
        if not result.ok:
            st.caption(result.error)


def render_nav(
    back_step: int | None,
    next_label: str | None = None,
    next_step: int | None = None,
    next_disabled: bool = False,
) -> None:
    """Back on the left, one primary action on the right. Buttons only navigate here;
    screens that call the API do their own button so the call happens before the rerun."""
    left, right = st.columns([1, 1])
    with left:
        if back_step is not None:
            st.button("Back", key=f"back_{back_step}_{next_label}", on_click=_goto,
                      args=(back_step,))
    with right:
        if next_label and next_step is not None:
            st.button(
                next_label,
                key=f"next_{next_step}_{next_label}",
                type="primary",
                use_container_width=True,
                disabled=next_disabled,
                on_click=_goto,
                args=(next_step,),
            )


def render_landing(api: str) -> None:
    """Screen 0: the mark, the promise, the four steps as cards, one call to action."""
    st.markdown(logo_html(62), unsafe_allow_html=True)
    st.markdown(
        f'<div class="ss-eyebrow">{html.escape(BRAND_TAGLINE)}</div>'
        '<div class="ss-title">A routine built around<br>your skin, not the shelf.</div>'
        '<div class="ss-sub">We read your photo, listen to what you tell us, and search a real '
        "Sephora product catalogue for the few products that fit. Every suggestion comes with "
        "the evidence behind it.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ss-eyebrow">Four steps, about two minutes</div>',
                unsafe_allow_html=True)
    cards = [
        ("camera", "Take a selfie, or skip it"),
        ("lens", "See your skin analysis"),
        ("person", "Tell us about you"),
        ("bottle", "Receive your routine"),
    ]
    columns = st.columns(4)
    for number, (column, (icon, text)) in enumerate(zip(columns, cards, strict=False), start=1):
        with column:
            st.markdown(
                f'<div class="ss-scard">{_STEP_ICONS[icon]}'
                f'<div class="ss-snum">Step {number}</div>'
                f'<div class="ss-stext">{html.escape(text)}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.button(
        "Let's get started",
        type="primary",
        use_container_width=True,
        on_click=_goto,
        args=(STEP_PHOTO,),
    )
    st.markdown(f'<div class="ss-muted" style="margin-top:1.2rem">{SHORT_DISCLAIMER}</div>',
                unsafe_allow_html=True)
    st.write("")
    render_status_line(api)


def render_step_photo(api: str) -> None:
    """Screen 1: the uploader, a live preview, and the how/why text tucked into tabs."""
    st.markdown(progress_html(STEP_PHOTO), unsafe_allow_html=True)
    st.markdown(
        '<div class="ss-question">Let\'s take a look at your skin</div>'
        '<div class="ss-sub">One clear, front-facing photo in daylight, no makeup, no filter.'
        "</div>",
        unsafe_allow_html=True,
    )

    photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if photo is not None:
        # Copy the bytes into session state so the annotated selfie on step 2 survives
        # navigation (the uploader itself forgets its file when it is not drawn).
        st.session_state["photo_bytes"] = photo.getvalue()
        st.session_state["photo_mime"] = getattr(photo, "type", None) or "image/jpeg"
        st.session_state["photo_name"] = photo.name

    stored = st.session_state.get("photo_bytes")
    if isinstance(stored, bytes) and stored:
        left, right = st.columns([1, 2])
        with left:
            # An inline <img> rather than st.image: st.image decodes through Pillow and
            # raises on a corrupt or truncated upload, which would take the whole demo
            # down. The browser just shows a broken-image box instead.
            preview_uri = session_photo_uri()
            if preview_uri:
                st.markdown(
                    f'<div class="ss-preview"><img src="{preview_uri}" alt="Your photo" /></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ss-muted">Photo loaded - too large to preview inline, but it '
                    "will still be sent for analysis.</div>",
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown(
                '<div class="ss-eyebrow">Ready to analyse</div>'
                f'<div class="ss-body">{html.escape(str(st.session_state.get("photo_name")) or "")}'
                "</div>"
                '<div class="ss-muted">We send this one image to the analysis model and keep '
                "nothing afterwards.</div>",
                unsafe_allow_html=True,
            )

    photo_tab, privacy_tab = st.tabs(["How to take it", "What happens to it"])
    with photo_tab:
        st.markdown(
            '<div class="ss-body">Face the window, not the light. Tie your hair back, look '
            "straight at the camera, and let the phone sit about an arm's length away.</div>"
            '<div class="ss-muted">A filter, heavy makeup or a yellow indoor bulb will change '
            "the scores - the model only ever sees pixels.</div>",
            unsafe_allow_html=True,
        )
    with privacy_tab:
        st.markdown(
            '<div class="ss-body">The photo is posted once to the analysis endpoint and held '
            "in this browser session only. Nothing is written to disk, and closing the tab "
            "ends it.</div>"
            '<div class="ss-muted">Prefer not to upload anything? Skip this step - the advisor '
            "will work from your description alone.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.button("Back", on_click=_goto, args=(LANDING,))
    with right:
        analyze_clicked = st.button(
            "Analyse my skin",
            type="primary",
            use_container_width=True,
            disabled=not isinstance(stored, bytes) or not stored,
        )

    if analyze_clicked and isinstance(stored, bytes) and stored:
        with st.spinner("Reading your photo ..."):
            result = post_analyze_skin(
                api,
                str(st.session_state.get("photo_name") or "upload.jpg"),
                stored,
                str(st.session_state.get("photo_mime") or "image/jpeg"),
            )
        if result.ok and is_valid_analysis(result.data):
            st.session_state["analysis"] = result.data
            st.session_state["analysis_source"] = "POST /analyze-skin (uploaded selfie)"
            st.session_state["response"] = None
            _goto(STEP_ANALYSIS)
            st.rerun()
        elif result.ok:
            st.error("The API answered, but the payload did not match the SkinAnalysis schema.")
            st.json(result.data)
        else:
            render_api_error(result, "Skin analysis failed")

    st.write("")
    if is_valid_analysis(st.session_state.get("analysis")):
        # The presenter usually runs with a loaded demo analysis rather than a live upload.
        st.button(
            "Skip - use the analysis already loaded",
            on_click=_goto,
            args=(STEP_ANALYSIS,),
        )
    else:
        st.button(
            "Skip - describe my skin instead",
            on_click=_goto,
            args=(STEP_PROFILE,),
        )


def render_annotated_face(analysis: dict[str, Any]) -> None:
    """The hero of the analysis screen: the photo with its top three concerns pinned on.

    The positions are fixed decoration. The concern head scores the photo as a whole,
    so the caption under the picture says exactly that -- the report's central claim is
    that this system does not overclaim, and a label pointing at a cheek would.
    """
    rows = concern_rows(analysis)[:3]
    if not rows:
        return

    uri = session_photo_uri()
    used_placeholder = not uri
    if used_placeholder:
        uri = face_placeholder_uri()

    searched = query_concerns_for_retrieval(analysis)
    st.markdown(annotated_face_html(uri, rows, searched=searched), unsafe_allow_html=True)
    where = ", ".join(a["where"] for a in FACE_ANCHORS[: len(rows)])
    note = (
        f"Label positions ({where}) are fixed illustration points. The model scores the whole "
        "photo and does not locate anything on your face - the numbers are real, the placement "
        "is decoration."
    )
    if any(name not in searched for name, _ in rows):
        note += " A greyed-out label scored too low to be searched on."
    if used_placeholder:
        note = (
            "No photo in this session, so the outline above is a placeholder. " + note
        )
    st.markdown(f'<div class="ss-face-note">{html.escape(note)}</div>', unsafe_allow_html=True)


def render_step_analysis() -> None:
    """Screen 2: the annotated selfie first, then the numbers split across tabs."""
    st.markdown(progress_html(STEP_ANALYSIS), unsafe_allow_html=True)
    analysis = st.session_state.get("analysis")

    if not is_valid_analysis(analysis):
        st.markdown(
            '<div class="ss-question">No analysis yet</div>'
            '<div class="ss-sub">Add a photo to see your skin analysis, or carry on and we will '
            "work from your description alone.</div>",
            unsafe_allow_html=True,
        )
        render_nav(STEP_PHOTO, "Continue without a photo", STEP_PROFILE)
        return

    try:
        confidence = float(analysis.get("skin_type_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    skin_type = str(analysis.get("skin_type", "?"))
    st.markdown('<div class="ss-eyebrow">Your skin analysis</div>'
                '<div class="ss-question">Here is what we found</div>',
                unsafe_allow_html=True)

    render_annotated_face(analysis)

    st.markdown(
        '<div class="ss-stats">'
        + stat_tile_html("Your skin type", skin_type.capitalize())
        + stat_tile_html("Confidence", f"{confidence:.0%}", "in that skin type", serif=False)
        + "</div>",
        unsafe_allow_html=True,
    )

    rows = concern_rows(analysis)
    if not rows:
        st.markdown('<div class="ss-muted">This analysis contains no concern scores.</div>',
                    unsafe_allow_html=True)
        render_nav(STEP_PHOTO, "Continue", STEP_PROFILE)
        return

    searched = query_concerns_for_retrieval(analysis)
    scores_tab, meaning_tab, method_tab = st.tabs(
        ["Concern scores", "What it means", "How we got this"]
    )

    with scores_tab:
        st.markdown(concern_meter_html(rows, searched=searched), unsafe_allow_html=True)
        # The old caption said every concern at or above the threshold reached retrieval.
        # skincare.rag.retrieve.query_concerns() sorts those by score and keeps only the top
        # MAX_QUERY_CONCERNS, so with four or more over the line the claim was false.
        over = top_concerns(analysis)
        if searched:
            searched_text = ", ".join(pretty_concern(s) for s in searched)
            capped = (
                f" {len(over)} scored above the line, so the {MAX_QUERY_CONCERNS} strongest were "
                "used; the rest are greyed out above."
                if len(over) > len(searched)
                else ""
            )
            caption = (
                f"The hairline marks {TOP_CONCERN_THRESHOLD:.2f}. We search the product catalogue "
                f"on at most the {MAX_QUERY_CONCERNS} strongest concerns above it - here: "
                f"{searched_text}.{capped} All six scores are still passed to the advisor and to "
                "the safety check."
            )
        else:
            caption = (
                f"The hairline marks {TOP_CONCERN_THRESHOLD:.2f}. Nothing scored above it, so the "
                "search runs on your own description and skin type alone."
            )
        st.markdown(f'<div class="ss-muted">{html.escape(caption)}</div>', unsafe_allow_html=True)

    with meaning_tab:
        blurb = SKIN_TYPE_BLURBS.get(skin_type.lower(), "")
        st.markdown(
            f'<div class="ss-eyebrow">{html.escape(skin_type.capitalize())} skin</div>'
            f'<div class="ss-body">{html.escape(blurb)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ss-eyebrow">Your three strongest signals</div>',
                    unsafe_allow_html=True)
        for name, score in rows[:3]:
            st.markdown(
                f'<div class="ss-body"><strong>{html.escape(pretty_concern(name))} '
                f"{score:.2f}</strong> &mdash; "
                f'{html.escape(CONCERN_BLURBS.get(name, "A label the model scores."))}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="ss-muted">These sentences describe what each label covers in general. '
            "They are not a diagnosis of your skin, and a score is a model's estimate rather "
            "than a measurement.</div>",
            unsafe_allow_html=True,
        )

    with method_tab:
        source = st.session_state.get("analysis_source") or "unknown source"
        st.markdown(
            f'<div class="ss-body">Source: {html.escape(source)}<br>'
            f"Model version: <code>{html.escape(str(analysis.get('model_version', 'unknown')))}"
            "</code></div>",
            unsafe_allow_html=True,
        )
        st.table({"concern": [r[0] for r in rows], "score": [round(r[1], 2) for r in rows]})
        st.caption(
            "The table is the same data as the chart, for anyone who would rather read the "
            "numbers. Scores are per-label confidences and do not add up to 1."
        )

    st.write("")
    render_nav(STEP_PHOTO, "Continue", STEP_PROFILE)


def render_step_profile(presets: list[dict[str, Any]], api: str) -> None:
    """Screen 3: chips, a slider and a toggle -- one free-text box, the one that earns it."""
    st.markdown(progress_html(STEP_PROFILE), unsafe_allow_html=True)
    st.markdown(
        '<div class="ss-eyebrow">About you</div>'
        '<div class="ss-question">A few things about you</div>'
        '<div class="ss-sub">These become hard filters, not hints: anything you rule out here '
        "cannot appear in your routine.</div>",
        unsafe_allow_html=True,
    )

    # Widgets take their value from session state rather than owning it, because Streamlit
    # drops the state of any widget that a rerun does not render -- which is every widget
    # on this screen as soon as the user moves to step 4 and back.
    prefs = st.multiselect(
        "What matters to you?",
        preference_options(presets),
        default=coerce_str_list(st.session_state.get("prefs")),
        placeholder="Pick any that apply",
    )
    st.session_state["prefs"] = list(prefs)

    avoid_current = coerce_str_list(st.session_state.get("avoid"))
    avoid = st.multiselect(
        "Anything to keep out?",
        avoid_options(presets, avoid_current),
        default=avoid_current,
        placeholder="Pick any that apply",
    )
    st.session_state["avoid"] = list(avoid)

    budget = st.slider(
        "Budget per product (USD)",
        min_value=0.0,
        max_value=BUDGET_MAX,
        value=float(min(st.session_state.get("budget", 0.0) or 0.0, BUDGET_MAX)),
        step=1.0,
        format="$%.0f",
    )
    st.session_state["budget"] = float(budget)
    st.markdown(
        '<div class="ss-muted">'
        + ("No limit - slide right off zero to set one." if budget <= 0
           else f"Nothing over ${budget:.0f} will be suggested.")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    pregnant = st.toggle(
        "I am pregnant or breastfeeding",
        value=bool(st.session_state.get("pregnant", False)),
    )
    st.session_state["pregnant"] = bool(pregnant)
    st.write("")

    query = st.text_area(
        "In your own words, what would you like to work on?",
        value=str(st.session_state.get("query", "")),
        height=120,
        placeholder="e.g. My chin breaks out and I want the marks to fade.",
    )
    st.session_state["query"] = query

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.button("Back", on_click=_goto, args=(STEP_ANALYSIS,))
    with right:
        submit = st.button("See my routine", type="primary", use_container_width=True)

    if submit:
        profile = current_profile()
        payload = build_recommend_payload(
            profile, st.session_state.get("analysis"), st.session_state.get("top_k", 3)
        )
        with st.spinner("Searching the catalogue and writing your routine ..."):
            result = post_recommend(api, payload)
        if result.ok and isinstance(result.data, dict):
            st.session_state["response"] = result.data
            # Remember the profile that produced this response, so the explanation of an
            # empty result cannot drift when the answers are edited afterwards.
            st.session_state["request_profile"] = payload["profile"]
            _goto(STEP_ROUTINE)
            st.rerun()
        elif result.ok:
            st.session_state["response"] = None
            st.error("The API answered, but the payload was not an AdvisorResponse object.")
            st.json(result.data)
        else:
            st.session_state["response"] = None
            render_api_error(result, "Recommendation request failed")

    # Presenter tools. Tabs rather than three stacked expanders: every body is rendered
    # on every rerun, so the widgets inside keep their values.
    st.markdown('<div class="ss-rule"></div>', unsafe_allow_html=True)
    preset_tab, advanced_tab, request_tab = st.tabs(
        ["Demo profiles", "Advanced", "Request preview"]
    )
    with preset_tab:
        labels = [p["label"] for p in presets]
        stored = st.session_state.get("preset_label", labels[0])
        choice = st.selectbox(
            "Demo profile",
            labels,
            index=labels.index(stored) if stored in labels else 0,
        )
        if st.button("Load this profile"):
            # Filling every answer at once is what the demo script assumes; the rerun is
            # what makes the widgets above pick up their new values.
            st.session_state["preset_label"] = choice
            _apply_preset(presets)
            st.rerun()
        st.caption(f"Loaded from: {st.session_state.get('_presets_source', 'unknown')}")
    with advanced_tab:
        st.session_state["top_k"] = st.slider(
            "Number of products", 1, 5, value=int(st.session_state.get("top_k", 3))
        )
        st.caption("How many products the advisor is asked for (RecommendRequest.top_k).")
    with request_tab:
        st.json(
            build_recommend_payload(
                current_profile(),
                st.session_state.get("analysis"),
                st.session_state.get("top_k", 3),
            )
        )
        st.caption("Exactly the body that POST /recommend will receive.")


def render_safety_body(response: dict[str, Any], profile: dict[str, Any] | None = None) -> None:
    """Claim 3: safety flags in a calm block, then what the layer actually checks."""
    flags = [str(f) for f in (response.get("safety_flags") or [])]
    recommendations = response.get("recommendations") or []

    if flags:
        items = "".join(f"<li>{html.escape(f)}</li>" for f in flags)
        st.markdown(
            callout_html("Flags raised on this answer", f"<ul>{items}</ul>", icon="caution"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            callout_html(
                "Safety check",
                "<p>Nothing in this routine conflicts with what you told us.</p>",
                tone="ok",
                icon="shield",
            ),
            unsafe_allow_html=True,
        )

    if not recommendations:
        st.markdown(empty_result_callout(flags, profile), unsafe_allow_html=True)
        note = response.get("routine_note")
        if note:
            st.info(str(note))

    st.markdown('<div class="ss-eyebrow">What this layer checks</div>', unsafe_allow_html=True)
    st.markdown(
        '<ul class="ss-body">'
        + "".join(f"<li>{html.escape(check)}</li>" for check in SAFETY_CHECKS)
        + "</ul>",
        unsafe_allow_html=True,
    )
    # The disclaimer itself is not repeated here: render_step_routine prints it below the
    # tab strip, where it is visible whichever tab is open.


def empty_result_callout(flags: list[str], profile: dict[str, Any] | None) -> str:
    """An empty list is never left unexplained: guard refusal vs. hard filters."""
    if flags:
        body = (
            "<p>We are not suggesting any product for this request. That is the safety guard "
            "deciding, not an empty answer: the reasons are listed above.</p>"
        )
    else:
        budget = (profile or {}).get("budget_usd")
        body = (
            f"<p>Nothing under ${float(budget):.0f} came back for this search. Budget is a "
            "hard filter applied to the search results, not a hint to the model, so a "
            "neighbourhood of expensive products can leave nothing behind even though the "
            "catalogue holds plenty of cheaper ones. Raising the budget, or dropping it to "
            "0 for no limit, is the quickest thing to try.</p>"
            if budget
            else "<p>Nothing in the catalogue matched this combination. Try removing a "
            "preference or two.</p>"
        )
    return callout_html("Nothing to recommend", body)


def render_reasoning(rec: dict[str, Any], index: int, evidence_texts: dict[str, str]) -> None:
    """Claim 2: one product's reasoning and the citations underneath it."""
    name = str(rec.get("name", "Unnamed product"))
    cited = [str(e) for e in (rec.get("cited_evidence") or [])]
    reason = str(rec.get("reason", ""))

    st.markdown(
        f'<div class="ss-eyebrow">Step {index}</div>'
        f'<div class="ss-card-name">{html.escape(name)}</div>'
        f'<p class="ss-card-reason">{html.escape(reason)}</p>',
        unsafe_allow_html=True,
    )
    if rec.get("key_ingredients"):
        st.markdown(
            "**Key ingredients:** " + ", ".join(str(i) for i in rec["key_ingredients"])
        )

    if not cited:
        st.markdown(
            '<div class="ss-muted">This product cites no evidence, so nothing in the catalogue '
            "backs the explanation above. Ungrounded output is exactly what the reward "
            "functions penalise.</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ss-rule"></div>', unsafe_allow_html=True)
        return

    with st.expander(f"Evidence behind this ({len(cited)} passage(s))"):
        for evidence_id in cited:
            meta = parse_evidence_id(evidence_id)
            inline = "quoted in the explanation" if evidence_id in reason else "not quoted inline"
            st.markdown(
                f"- `{evidence_id}` - product `{meta['product_id']}`, "
                f"{meta['source']} chunk {meta['chunk'] or '?'} ({inline})"
            )
            snippet = evidence_texts.get(evidence_id)
            if snippet:
                st.caption(f'"{snippet}"')
        st.caption(
            "Ids come from the retrieval call that produced this answer; the model may only "
            "cite evidence ids it was given. /recommend returns ids only, so snippet text "
            "shown here is resolved locally from fixtures/mock_catalog.json when it matches."
        )
    st.markdown('<div class="ss-rule"></div>', unsafe_allow_html=True)


def render_card_reason(rec: dict[str, Any], evidence_texts: dict[str, str]) -> None:
    """The listing card's "Why we picked this" expander: the reason and its citations.

    Kept next to the product rather than only on the "Why these" tab, so a viewer reading
    the shelf can open the evidence for the card in front of them. The full walkthrough,
    including whether an id is quoted inline, still lives on that tab.
    """
    reason = str(rec.get("reason", "")).strip()
    cited = [str(e) for e in (rec.get("cited_evidence") or [])]
    with st.expander("Why we picked this"):
        if reason:
            st.markdown(
                f'<p class="ss-card-reason">{html.escape(reason)}</p>', unsafe_allow_html=True
            )
        if rec.get("key_ingredients"):
            st.markdown("**Key ingredients:** " + ", ".join(str(i) for i in rec["key_ingredients"]))
        if not cited:
            st.caption(
                "This product cites no catalogue evidence -- ungrounded output is exactly what "
                "the reward functions penalise."
            )
            return
        st.caption(f"Cited evidence ({len(cited)} passage(s)):")
        for evidence_id in cited:
            meta = parse_evidence_id(evidence_id)
            st.markdown(
                f"- `{evidence_id}` - {meta['source']} chunk {meta['chunk'] or '?'}"
            )
            snippet = evidence_texts.get(evidence_id)
            if snippet:
                st.caption(f'"{snippet}"')


def render_step_routine(evidence_texts: dict[str, str]) -> None:
    """Screen 4: the routine, with the reading split across four tabs."""
    st.markdown(progress_html(STEP_ROUTINE), unsafe_allow_html=True)
    response = st.session_state.get("response")

    if not isinstance(response, dict):
        st.markdown(
            '<div class="ss-question">No routine yet</div>'
            '<div class="ss-sub">Go back a step and press "See my routine".</div>',
            unsafe_allow_html=True,
        )
        render_nav(STEP_PROFILE)
        return

    recommendations = response.get("recommendations") or []
    profile = st.session_state.get("request_profile") or current_profile()
    flags = [str(f) for f in (response.get("safety_flags") or [])]

    st.markdown(
        '<div class="ss-eyebrow">Your routine</div>'
        f'<div class="ss-question">{len(recommendations)} product'
        f'{"" if len(recommendations) == 1 else "s"} for your skin</div>'
        '<div class="ss-sub">Each one traced back to what the catalogue actually says.</div>',
        unsafe_allow_html=True,
    )

    # A one-line safety status stays on the main surface even though the detail lives in
    # a tab: a flagged answer must never be something the viewer has to click to notice.
    if flags:
        st.markdown(
            callout_html(
                "Safety check",
                f"<p>{len(flags)} flag(s) raised - see the Safety tab.</p>",
                icon="caution",
            ),
            unsafe_allow_html=True,
        )
    if not recommendations:
        st.markdown(empty_result_callout(flags, profile), unsafe_allow_html=True)

    routine_tab, why_tab, safety_tab, tech_tab = st.tabs(
        ["Your routine", "Why these", "Safety", "Behind the scenes"]
    )

    with routine_tab:
        if recommendations:
            # A listing grid rather than a stack: up to three cards abreast on a wide
            # screen, and fewer products simply means fewer (wider) columns, so two
            # products never leave a hole where a third would be.
            per_row = min(3, len(recommendations))
            for start in range(0, len(recommendations), per_row):
                chunk = recommendations[start:start + per_row]
                columns = st.columns(per_row, gap="medium")
                for column, (offset, rec) in zip(columns, enumerate(chunk), strict=False):
                    with column:
                        st.markdown(
                            product_card_html(start + offset + 1, rec), unsafe_allow_html=True
                        )
                        render_card_reason(rec, evidence_texts)
            note = response.get("routine_note")
            if note:
                st.markdown(
                    callout_html("How to use these", f"<p>{html.escape(str(note))}</p>",
                                 tone="ok"),
                    unsafe_allow_html=True,
                )
            st.caption(
                "Product tiles are generated from the product id - the catalogue ships no "
                "images, so nothing here is a photograph of the real packaging. The icon on "
                "each tile is read from words in the product name, not from a category field."
            )
            st.caption(
                '"Find on Sephora" runs a catalogue search for the brand and product name - '
                "it is not a verified link to that exact product page, and this demo does not "
                "handle purchases."
            )
        else:
            st.markdown(
                '<div class="ss-body">There is no routine to show. The Safety tab explains '
                "why.</div>",
                unsafe_allow_html=True,
            )

    with why_tab:
        if recommendations:
            for i, rec in enumerate(recommendations, start=1):
                render_reasoning(rec, i, evidence_texts)
        else:
            st.markdown(
                '<div class="ss-body">Nothing was recommended, so there is no reasoning to '
                "show.</div>",
                unsafe_allow_html=True,
            )

    with safety_tab:
        render_safety_body(response, profile)

    with tech_tab:
        headline, explanation = describe_generator(str(response.get("generator", "unknown")))
        st.markdown(f"**Answered by: {headline}**")
        st.caption(f"`AdvisorResponse.generator = \"{response.get('generator')}\"` - {explanation}")

        cited, total, citations = grounding_stats(recommendations)
        col1, col2, col3 = st.columns(3)
        col1.metric("Products", total)
        col2.metric("Grounded", f"{cited}/{total}" if total else "0/0")
        col3.metric("Citations", citations)
        if total and cited < total:
            st.warning(
                f"{total - cited} of {total} recommendations cite no evidence. "
                "Ungrounded output is exactly what the reward functions penalise."
            )
        st.markdown("**Raw AdvisorResponse**")
        st.json(response)

    disclaimer = str(response.get("disclaimer") or "").strip()
    st.markdown(
        f'<div class="ss-muted" style="margin-top:1.2rem">'
        f"{html.escape(disclaimer or SHORT_DISCLAIMER)}</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.button("Back", on_click=_goto, args=(STEP_PROFILE,))
    with right:
        st.button("Start over", type="primary", use_container_width=True, on_click=_restart)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(
        page_title=f"{BRAND_NAME} - {BRAND_TAGLINE}",
        page_icon=page_icon(),
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_APP_CSS, unsafe_allow_html=True)

    presets, presets_source = load_demo_profiles()
    _init_state(presets)
    st.session_state["_presets_source"] = presets_source
    evidence_texts = load_evidence_texts()

    api = normalize_api_url(st.session_state.get("api_url", DEFAULT_API))
    step = int(st.session_state.get("step", LANDING))

    if step == STEP_PHOTO:
        render_step_photo(api)
    elif step == STEP_ANALYSIS:
        render_step_analysis()
    elif step == STEP_PROFILE:
        render_step_profile(presets, api)
    elif step == STEP_ROUTINE:
        render_step_routine(evidence_texts)
    else:
        render_landing(api)


if __name__ == "__main__":
    main()
