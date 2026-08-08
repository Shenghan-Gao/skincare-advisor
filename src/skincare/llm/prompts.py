"""Prompt construction shared by SFT data, RL rollouts and inference.

Keeping ONE prompt builder matters: if training and serving prompts drift,
the post-trained model silently degrades in the demo.
"""
import json

SYSTEM = (
    "You are a careful skincare advisor. You are given a user's skin profile and a list "
    "of retrieved product evidence. Recommend products ONLY from the evidence. "
    "Every recommendation must cite the evidence_ids you used. Never diagnose medical "
    "conditions. Always include a disclaimer.\n"
    "Reply with ONLY a JSON object of the form:\n"
    '{"recommendations":[{"product_id":str,"name":str,"brand":str,"reason":str,'
    '"key_ingredients":[str],"cited_evidence":[str],"matched_concerns":[str]}],'
    '"routine_note":str,"disclaimer":str}'
)


def build_user_prompt(profile: dict, analysis: dict | None, evidence: list[dict]) -> str:
    lines = ["## User", json.dumps(profile, ensure_ascii=False)]
    if analysis:
        lines += ["## Skin analysis (from image model)", json.dumps(analysis, ensure_ascii=False)]
    lines.append("## Retrieved evidence")
    for e in evidence:
        lines.append(f"- [{e['evidence_id']}] (product {e['product_id']}, {e['source']}) {e['text']}")
    lines.append("## Task\nRecommend up to 3 products. Output JSON only.")
    return "\n".join(lines)


def build_messages(profile, analysis, evidence):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(profile, analysis, evidence)},
    ]
