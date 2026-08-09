"""Verify the wiring of GRPO's reward functions — the easiest thing to get wrong and the hardest to notice.

TRL calls reward_func like this:
    reward_func(prompts=[...], completions=[...], <remaining dataset columns>=[one value per row], ...)
If the adapter indexes them wrongly, the rewards silently come back as 0 — training looks like it is
running, but the model learns nothing.
This test uses an answer with a "known perfect score" to prove the context really does get through.
"""
import json

from skincare.llm import rewards as R
from skincare.llm.grpo_train import _make_reward_fn

GOOD = json.dumps({
    "recommendations": [{
        "product_id": "P001", "name": "N", "brand": "B",
        "reason": "helps acne [P001:rev:0]", "key_ingredients": ["niacinamide"],
        "cited_evidence": ["P001:rev:0"], "matched_concerns": ["acne"]}],
    "routine_note": "am",
    "disclaimer": "Cosmetic suggestions only, not medical advice; see a dermatologist."})
BAD = "just buy something nice"


def _trl_style_call(fn, completions, **columns):
    """Emulate TRL's calling convention: every dataset column is a list the same length as completions."""
    return _make_reward_fn(fn)(completions, **columns)


def test_context_actually_reaches_reward_functions():
    out = _trl_style_call(
        R.grounding_reward, [GOOD, BAD],
        evidence_ids=[["P001:rev:0", "P001:desc:0"], ["P001:rev:0"]],
        product_ids=[["P001"], ["P001"]], concerns=[["acne"], ["acne"]])
    assert out[0] == 1.0, f"the context never reached grounding_reward; got {out[0]}"
    assert out[1] == 0.0


def test_per_completion_context_is_indexed_not_broadcast():
    """The two completions have different contexts — if the adapter passes the whole list to each one, this fails."""
    out = _trl_style_call(R.product_validity_reward, [GOOD, GOOD],
                          product_ids=[["P001"], ["P999"]])
    assert out == [1.0, 0.0], f"per-row indexing failed: {out}"


def test_all_five_reward_funcs_survive_trl_extra_kwargs():
    """TRL also injects extra kwargs such as trainer_state / log_metric, which must not break the functions."""
    extras = {"trainer_state": object(), "log_metric": lambda *a, **k: None,
              "prompts": ["p1", "p2"]}
    for fn in [R.format_reward, R.ingredient_match_reward, R.grounding_reward,
               R.product_validity_reward, R.safety_reward]:
        out = _trl_style_call(fn, [GOOD, BAD], concerns=[["acne"], ["acne"]],
                              evidence_ids=[["P001:rev:0"], ["P001:rev:0"]],
                              product_ids=[["P001"], ["P001"]],
                              pregnant=[False, False], avoid=[[], []], **extras)
        assert len(out) == 2 and all(isinstance(x, float) for x in out), f"{fn.__name__} returned {out}"
        assert out[0] >= out[1], f"{fn.__name__} failed to distinguish a good answer from a junk one"


def test_reward_func_names_are_preserved():
    """TRL uses the function name as the logging key (rewards/<name>/mean); lose the name and the report curves become indistinguishable."""
    for fn in [R.format_reward, R.grounding_reward]:
        assert _make_reward_fn(fn).__name__ == fn.__name__
