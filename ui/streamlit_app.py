"""Demo UI for the Skincare Advisor (Streamlit, single file, stdlib + requests only).

The demo has to make four project claims visible on screen:

  1. Skin analysis  -- skin type + confidence and the six concern scores (bar chart).
  2. Grounding      -- the cited_evidence ids behind every recommendation, expandable.
  3. Safety         -- safety_flags shown prominently, disclaimer always visible,
                       a blocked/empty answer explained instead of silently empty.
  4. Provenance     -- AdvisorResponse.generator, i.e. WHICH model answered
                       (stub / base / SFT / GRPO), because base-vs-SFT-vs-GRPO is
                       the headline result of the project.

Structure: everything above `main()` is pure request/response logic with no
Streamlit calls, so it can be imported and exercised headlessly against a running
API.  Every `st.*` call lives inside `main()` or a `render_*` helper.

Run:  streamlit run ui/streamlit_app.py        (API_URL env var overrides the host)
"""
from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PROFILES_PATH = REPO_ROOT / "fixtures" / "demo_profiles.json"
DEMO_CATALOG_PATH = REPO_ROOT / "fixtures" / "mock_catalog.json"

DEFAULT_API = os.getenv("API_URL", "http://localhost:8000")

# Mirrors app.schemas.CONCERNS -- kept as a literal so the UI never imports the backend.
CONCERNS = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]

# app.schemas.SkinAnalysis.top_concerns() uses this threshold to decide which
# concerns are handed to retrieval, so the chart marks it explicitly.
TOP_CONCERN_THRESHOLD = 0.5

BASE_PREFERENCES = ["fragrance-free", "vegan", "gentle", "lightweight", "non-comedogenic"]

EVIDENCE_SOURCE_NAMES = {"desc": "description", "rev": "review", "ing": "ingredient"}

SHORT_DISCLAIMER = (
    "Cosmetic product suggestions only. Not medical advice, and not a diagnosis."
)

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

    def _str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return parse_ingredient_list(value)
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    analysis = raw.get("analysis")
    return {
        "label": label,
        "query": query,
        "budget_usd": budget,
        "pregnant": bool(raw.get("pregnant", False)),
        "preferences": _str_list(raw.get("preferences")),
        "avoid_ingredients": _str_list(raw.get("avoid_ingredients")),
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
# Concern chart (horizontal bars, inline HTML -- no extra dependency)
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


# One measure, six categories -> horizontal bars, single hue, sorted, no gridlines.
# Blue #2a78d6 / #3987e5 both clear 3:1 on the light and the dark app surface, so the
# chart stays legible even if the Streamlit theme and the OS theme disagree.
# Track and text use currentColor/neutral alpha so they follow whatever theme is active.
_CHART_CSS = """
<style>
.sc-chart { --sc-bar: #2a78d6; margin: 0.2rem 0 0.1rem 0; }
@media (prefers-color-scheme: dark) { .sc-chart { --sc-bar: #3987e5; } }
.sc-chart .sc-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.sc-chart .sc-name { flex: 0 0 104px; font-size: 0.82rem; opacity: 0.85; text-align: right;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sc-chart .sc-track { flex: 1 1 auto; position: relative; height: 14px;
  background: rgba(128,128,128,0.18); border-radius: 0 4px 4px 0; }
.sc-chart .sc-bar { position: absolute; left: 0; top: 0; height: 100%;
  background: var(--sc-bar); border-radius: 0 4px 4px 0; }
.sc-chart .sc-thr { position: absolute; top: -2px; height: calc(100% + 4px);
  width: 1px; background: rgba(128,128,128,0.75); }
.sc-chart .sc-val { flex: 0 0 44px; font-size: 0.8rem; opacity: 0.75; text-align: right;
  font-variant-numeric: tabular-nums; }
</style>
"""


def concern_chart_html(
    rows: list[tuple[str, float]],
    threshold: float = TOP_CONCERN_THRESHOLD,
) -> str:
    """Minimal horizontal bar chart: one hue, 0-1 scale, value at the bar tip.

    The hairline marks `threshold`, i.e. the cut-off SkinAnalysis.top_concerns()
    uses to decide which concerns are passed to retrieval.
    """
    if not rows:
        return ""
    parts = [_CHART_CSS, '<div class="sc-chart">']
    thr_pct = max(0.0, min(1.0, threshold)) * 100.0
    for name, score in rows:
        label = html.escape(name.replace("_", " "))
        parts.append(
            '<div class="sc-row">'
            f'<div class="sc-name">{label}</div>'
            '<div class="sc-track">'
            f'<div class="sc-bar" style="width:{score * 100:.1f}%"></div>'
            f'<div class="sc-thr" style="left:{thr_pct:.1f}%"></div>'
            "</div>"
            f'<div class="sc-val">{score:.2f}</div>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def top_concerns(analysis: dict[str, Any], threshold: float = TOP_CONCERN_THRESHOLD) -> list[str]:
    """Same rule as SkinAnalysis.top_concerns() in app/schemas.py."""
    return [name for name, score in concern_rows(analysis) if score >= threshold]


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
            "Check that the API is running and that the URL in the sidebar is correct, e.g. "
            "`uvicorn app.main:app --port 8000`."
        )


def render_status_line(api: str) -> None:
    """One-line backend status: URL, health, and which mode the backend reports."""
    result = get_health(api)
    if result.ok and isinstance(result.data, dict):
        status = result.data.get("status", "?")
        mock_mode = result.data.get("mock_mode")
        mode = "mock fixtures (USE_MOCKS=1)" if mock_mode else "real pipeline (USE_MOCKS=0)"
        st.caption(
            f"API `{normalize_api_url(api)}` - health: **{status}** - backend mode: **{mode}**"
        )
    else:
        st.caption(f"API `{normalize_api_url(api)}` - health: **unreachable**")
        st.warning(f"Backend health check failed. {result.error}")


def render_analysis(analysis: dict[str, Any], source: str) -> None:
    """Claim 1: skin type + confidence and the six concern scores."""
    st.subheader("1. Skin analysis")
    try:
        confidence = float(analysis.get("skin_type_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    left, right = st.columns([1, 1])
    with left:
        st.metric("Skin type", str(analysis.get("skin_type", "?")).capitalize())
    with right:
        st.metric("Confidence", f"{confidence:.0%}")
    st.caption(f"Source: {source} - CNN version `{analysis.get('model_version', 'unknown')}`")

    rows = concern_rows(analysis)
    if not rows:
        st.info("This analysis contains no concern scores.")
        return

    st.markdown("**Concern scores** (0 to 1)")
    st.markdown(concern_chart_html(rows), unsafe_allow_html=True)
    hits = top_concerns(analysis)
    st.caption(
        f"Hairline marks the {TOP_CONCERN_THRESHOLD:.2f} threshold. Concerns at or above it are "
        "passed to retrieval as the query concerns: "
        + (", ".join(h.replace("_", " ") for h in hits) if hits else "none")
    )
    with st.expander("Concern scores as a table"):
        st.table({"concern": [r[0] for r in rows], "score": [round(r[1], 2) for r in rows]})


def render_generator_badge(generator: str) -> None:
    """Claim 4: which model produced this answer."""
    headline, explanation = describe_generator(generator)
    with st.container(border=True):
        st.markdown(f"**Answering model: {headline}**")
        st.caption(f"`AdvisorResponse.generator = \"{generator}\"` - {explanation}")


def render_safety(response: dict[str, Any], profile: dict[str, Any] | None = None) -> None:
    """Claim 3: safety flags prominent, blocked/empty answers explained."""
    flags = response.get("safety_flags") or []
    recommendations = response.get("recommendations") or []

    if flags:
        st.markdown(f"**Safety guard: {len(flags)} flag(s) raised**")
        for flag in flags:
            st.warning(str(flag))
    else:
        st.success("Safety guard ran and raised no flags for this profile.")

    if recommendations:
        return

    # An empty list is never left unexplained: either the guard removed everything,
    # or the retrieval layer's hard filters left no candidate at all.
    if flags:
        st.error(
            "No products are being recommended for this request. This is a deliberate "
            "guard decision, not an empty response: the flags above say why."
        )
    else:
        budget = (profile or {}).get("budget_usd")
        hint = (
            f"No catalog product matched this profile under the ${float(budget):.0f} budget. "
            "Budget is a hard filter applied during retrieval, not a hint to the model, so "
            "nothing over that price can be recommended - try raising the budget."
            if budget
            else "Retrieval returned no product matching the detected concerns for this profile. "
            "Try widening the profile, or lowering the number of preferences."
        )
        st.error(f"No products found. {hint}")

    note = response.get("routine_note")
    if note:
        st.info(str(note))


def render_recommendation(
    index: int,
    rec: dict[str, Any],
    evidence_texts: dict[str, str],
) -> None:
    """Claim 2: one recommendation with its expandable citation list."""
    name = rec.get("name", "Unnamed product")
    brand = rec.get("brand", "")
    cited = [str(e) for e in (rec.get("cited_evidence") or [])]
    reason = str(rec.get("reason", ""))

    with st.container(border=True):
        header = f"{index}. {name}" + (f" - {brand}" if brand else "")
        st.markdown(f"**{header}**")

        bits = []
        price = rec.get("price_usd")
        if isinstance(price, (int, float)):
            bits.append(f"${float(price):.2f}")
        if rec.get("product_id"):
            bits.append(f"product `{rec['product_id']}`")
        matched = rec.get("matched_concerns") or []
        if matched:
            bits.append("matches: " + ", ".join(str(m).replace("_", " ") for m in matched))
        if bits:
            st.caption(" - ".join(bits))

        st.write(reason)
        if rec.get("key_ingredients"):
            st.caption("Key ingredients: " + ", ".join(str(i) for i in rec["key_ingredients"]))

        if not cited:
            st.warning(
                "This recommendation cites no evidence ids - it is not grounded in retrieval."
            )
            return

        # Always visible on the card, then expandable for the full snippets.
        st.caption("Evidence: " + ", ".join(f"`{e}`" for e in cited))

        with st.expander(f"Grounding: {len(cited)} cited evidence id(s)"):
            for evidence_id in cited:
                meta = parse_evidence_id(evidence_id)
                inline = (
                    "quoted in the explanation" if evidence_id in reason else "not quoted inline"
                )
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


def render_response(
    response: dict[str, Any],
    evidence_texts: dict[str, str],
    profile: dict[str, Any] | None = None,
) -> None:
    """Full AdvisorResponse: provenance, safety, recommendations, grounding, disclaimer."""
    recommendations = response.get("recommendations") or []

    st.subheader("2. Recommendations")
    render_generator_badge(str(response.get("generator", "unknown")))

    cited, total, citations = grounding_stats(recommendations)
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommendations", total)
    col2.metric("Grounded in evidence", f"{cited}/{total}" if total else "0/0")
    col3.metric("Citations total", citations)
    if total and cited < total:
        st.warning(
            f"{total - cited} of {total} recommendations cite no evidence. "
            "Ungrounded output is exactly what the reward functions penalise."
        )

    render_safety(response, profile)

    for i, rec in enumerate(recommendations, start=1):
        render_recommendation(i, rec, evidence_texts)

    note = response.get("routine_note")
    if note and recommendations:
        st.info(f"Routine note: {note}")

    with st.expander("Raw AdvisorResponse JSON"):
        st.json(response)

    disclaimer = str(response.get("disclaimer") or "").strip()
    with st.container(border=True):
        st.markdown("**Disclaimer**")
        st.caption(disclaimer or f"{SHORT_DISCLAIMER} (the API returned an empty disclaimer field)")


# --------------------------------------------------------------------------- #
# Session state / sidebar
# --------------------------------------------------------------------------- #

def _apply_preset(presets: list[dict[str, Any]]) -> None:
    """Fill every profile widget from the selected preset (runs before the rerun)."""
    label = st.session_state.get("preset_label")
    preset = next((p for p in presets if p["label"] == label), None)
    if preset is None:
        return
    st.session_state["query"] = preset["query"]
    st.session_state["budget"] = float(preset["budget_usd"] or 0.0)
    st.session_state["pregnant"] = preset["pregnant"]
    st.session_state["prefs"] = list(preset["preferences"])
    st.session_state["avoid"] = ", ".join(preset["avoid_ingredients"])
    st.session_state["analysis"] = preset["analysis"]
    st.session_state["analysis_source"] = (
        f"demo preset '{preset['label']}'" if preset["analysis"] else ""
    )
    st.session_state["response"] = None


def _init_state(presets: list[dict[str, Any]]) -> None:
    """Seed the widgets from the first preset so the demo opens ready to run."""
    if st.session_state.get("_initialised"):
        return
    first = presets[0]
    st.session_state.setdefault("api_url", DEFAULT_API)
    st.session_state.setdefault("preset_label", first["label"])
    st.session_state.setdefault("query", first["query"])
    st.session_state.setdefault("budget", float(first["budget_usd"] or 0.0))
    st.session_state.setdefault("pregnant", first["pregnant"])
    st.session_state.setdefault("prefs", list(first["preferences"]))
    st.session_state.setdefault("avoid", ", ".join(first["avoid_ingredients"]))
    st.session_state.setdefault("top_k", 3)
    st.session_state.setdefault("analysis", first["analysis"])
    st.session_state.setdefault(
        "analysis_source", f"demo preset '{first['label']}'" if first["analysis"] else ""
    )
    st.session_state.setdefault("response", None)
    st.session_state["_initialised"] = True


def main() -> None:
    st.set_page_config(page_title="Skincare Advisor", page_icon="*", layout="centered")

    presets, presets_source = load_demo_profiles()
    _init_state(presets)
    evidence_texts = load_evidence_texts()

    st.title("AI Skincare Advisor")
    st.caption("CNN skin analysis + retrieval-grounded, post-trained LLM recommendations")
    st.caption(SHORT_DISCLAIMER)

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Demo preset")
        # Picking a preset fills the whole profile immediately -- one click, as the
        # demo script assumes ("select preset -> press Get recommendations").
        st.selectbox(
            "Preset profile",
            [p["label"] for p in presets],
            key="preset_label",
            on_change=_apply_preset,
            args=(presets,),
        )
        st.button(
            "Reset fields to preset",
            use_container_width=True,
            on_click=_apply_preset,
            args=(presets,),
        )
        st.caption(f"Loaded from: {presets_source}")

        st.divider()
        st.header("Profile")
        st.text_area("Describe your skin and goals", key="query", height=110)
        st.number_input(
            "Budget per product (USD, 0 = no limit)",
            min_value=0.0, max_value=1000.0, step=5.0, key="budget",
        )
        st.checkbox("Pregnant or breastfeeding", key="pregnant")
        st.multiselect("Preferences", preference_options(presets), key="prefs")
        st.text_input("Avoid ingredients (comma separated)", key="avoid")
        st.slider("Number of products (top_k)", 1, 5, key="top_k")

        st.divider()
        st.header("Backend")
        st.text_input("API URL", key="api_url")

    api = normalize_api_url(st.session_state["api_url"])
    render_status_line(api)

    profile = build_profile(
        st.session_state["query"],
        st.session_state["budget"],
        st.session_state["pregnant"],
        st.session_state["prefs"],
        parse_ingredient_list(st.session_state["avoid"]),
    )

    # -------------------------------------------------------------- skin photo
    st.subheader("Optional: selfie analysis")
    st.caption(
        "The CNN step is optional. Without a photo the advisor still runs on the text "
        "profile alone; a demo preset can also supply a stored analysis."
    )
    photo = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])
    col_a, col_b = st.columns(2)
    with col_a:
        analyze_clicked = st.button(
            "Analyze skin", disabled=photo is None, use_container_width=True
        )
    with col_b:
        if st.button("Clear analysis", use_container_width=True):
            st.session_state["analysis"] = None
            st.session_state["analysis_source"] = ""

    if analyze_clicked and photo is not None:
        with st.spinner("Calling /analyze-skin ..."):
            result = post_analyze_skin(
                api, photo.name, photo.getvalue(), getattr(photo, "type", None) or "image/jpeg"
            )
        if result.ok and is_valid_analysis(result.data):
            st.session_state["analysis"] = result.data
            st.session_state["analysis_source"] = "POST /analyze-skin (uploaded selfie)"
        elif result.ok:
            st.error("The API answered, but the payload did not match the SkinAnalysis schema.")
            st.json(result.data)
        else:
            render_api_error(result, "Skin analysis failed")

    analysis = st.session_state.get("analysis")
    if is_valid_analysis(analysis):
        render_analysis(analysis, st.session_state.get("analysis_source") or "unknown source")
    else:
        st.info("No skin analysis loaded. /recommend will run in text-only mode.")

    # ------------------------------------------------------------ recommend
    st.divider()
    if st.button("Get recommendations", type="primary", use_container_width=True):
        payload = build_recommend_payload(profile, analysis, st.session_state["top_k"])
        with st.spinner("Calling /recommend ..."):
            result = post_recommend(api, payload)
        if result.ok and isinstance(result.data, dict):
            st.session_state["response"] = result.data
            # Remember the profile that produced this response, so the explanation of an
            # empty result cannot drift when the sidebar is edited afterwards.
            st.session_state["request_profile"] = payload["profile"]
        elif result.ok:
            st.session_state["response"] = None
            st.error("The API answered, but the payload was not an AdvisorResponse object.")
            st.json(result.data)
        else:
            st.session_state["response"] = None
            render_api_error(result, "Recommendation request failed")

    response = st.session_state.get("response")
    if isinstance(response, dict):
        used_profile = st.session_state.get("request_profile") or profile
        render_response(response, evidence_texts, used_profile)
    else:
        st.caption("Pick a preset, then press 'Get recommendations'.")

    with st.expander("Request that will be sent to POST /recommend"):
        st.json(build_recommend_payload(profile, analysis, st.session_state["top_k"]))


if __name__ == "__main__":
    main()
