"""Demo UI. TEAMMATE D owns this -- works today against USE_MOCKS=1."""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Skincare Advisor", page_icon="✨", layout="centered")
st.title("AI Skincare Advisor")
st.caption("CNN skin analysis + retrieval-grounded, post-trained LLM recommendations")

with st.sidebar:
    st.header("Your profile")
    query = st.text_area("Describe your skin and goals",
                         "Combination skin, breakouts on my chin and some dark spots.")
    budget = st.slider("Budget per product (USD)", 5, 150, 40)
    pregnant = st.checkbox("Pregnant or breastfeeding")
    prefs = st.multiselect("Preferences", ["fragrance-free", "vegan", "gentle", "lightweight"])

photo = st.file_uploader("Optional: upload a selfie for skin analysis", type=["jpg", "jpeg", "png"])

analysis = None
if photo and st.button("Analyze skin"):
    r = requests.post(f"{API}/analyze-skin", files={"image": photo.getvalue()}, timeout=60)
    if r.ok:
        analysis = r.json()
        st.session_state["analysis"] = analysis
        st.success(f"Skin type: **{analysis['skin_type']}** "
                   f"({analysis['skin_type_confidence']:.0%} confidence)")
        st.bar_chart({c["concern"]: c["score"] for c in analysis["concerns"]})
    else:
        st.error(r.text)

if st.button("Get recommendations", type="primary"):
    payload = {
        "profile": {"query": query, "budget_usd": budget,
                    "preferences": prefs, "pregnant": pregnant},
        "analysis": st.session_state.get("analysis"),
        "top_k": 3,
    }
    r = requests.post(f"{API}/recommend", json=payload, timeout=120)
    if r.ok:
        data = r.json()
        for rec in data["recommendations"]:
            with st.container(border=True):
                st.subheader(f"{rec['name']} — {rec['brand']}")
                if rec.get("price_usd"):
                    st.caption(f"${rec['price_usd']:.2f}")
                st.write(rec["reason"])
                st.caption("Key ingredients: " + ", ".join(rec["key_ingredients"]))
                st.caption("Evidence: " + ", ".join(rec["cited_evidence"]))
        if data.get("routine_note"):
            st.info(data["routine_note"])
        for f in data.get("safety_flags", []):
            st.warning(f)
        st.caption(data.get("disclaimer", ""))
    else:
        st.error(r.text)
