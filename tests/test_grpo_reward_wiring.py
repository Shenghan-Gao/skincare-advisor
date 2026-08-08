"""验证 GRPO 的奖励函数接线 —— 最容易出错又最难发现的地方。

TRL 调用 reward_func 的方式是:
    reward_func(prompts=[...], completions=[...], <数据集其余列>=[每条一个值], ...)
如果适配器索引错了,奖励会静默返回 0 —— 训练看起来在跑,其实学不到东西。
本测试用"已知满分"的答案证明上下文确实传到了。
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
    """模拟 TRL 的调用约定:每个数据集列都是与 completions 等长的 list。"""
    return _make_reward_fn(fn)(completions, **columns)


def test_context_actually_reaches_reward_functions():
    out = _trl_style_call(
        R.grounding_reward, [GOOD, BAD],
        evidence_ids=[["P001:rev:0", "P001:desc:0"], ["P001:rev:0"]],
        product_ids=[["P001"], ["P001"]], concerns=[["acne"], ["acne"]])
    assert out[0] == 1.0, f"上下文没传到 grounding_reward,得到 {out[0]}"
    assert out[1] == 0.0


def test_per_completion_context_is_indexed_not_broadcast():
    """两条 completion 的上下文不同 —— 若适配器把整个 list 传给每一条,这里会挂。"""
    out = _trl_style_call(R.product_validity_reward, [GOOD, GOOD],
                          product_ids=[["P001"], ["P999"]])
    assert out == [1.0, 0.0], f"按条索引失败: {out}"


def test_all_five_reward_funcs_survive_trl_extra_kwargs():
    """TRL 还会塞 trainer_state / log_metric 等额外 kwarg,不能把函数打挂。"""
    extras = {"trainer_state": object(), "log_metric": lambda *a, **k: None,
              "prompts": ["p1", "p2"]}
    for fn in [R.format_reward, R.ingredient_match_reward, R.grounding_reward,
               R.product_validity_reward, R.safety_reward]:
        out = _trl_style_call(fn, [GOOD, BAD], concerns=[["acne"], ["acne"]],
                              evidence_ids=[["P001:rev:0"], ["P001:rev:0"]],
                              product_ids=[["P001"], ["P001"]],
                              pregnant=[False, False], avoid=[[], []], **extras)
        assert len(out) == 2 and all(isinstance(x, float) for x in out), f"{fn.__name__} 返回 {out}"
        assert out[0] >= out[1], f"{fn.__name__} 没能区分好答案与垃圾答案"


def test_reward_func_names_are_preserved():
    """TRL 用函数名做日志键(rewards/<name>/mean),名字丢了报告曲线就没法区分。"""
    for fn in [R.format_reward, R.grounding_reward]:
        assert _make_reward_fn(fn).__name__ == fn.__name__
