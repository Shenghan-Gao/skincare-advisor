"""LLM-as-judge —— 规则奖励覆盖不到的主观质量维度。组员 C 拥有。

规则奖励(rewards.py)查的是"有没有引用、成分对不对、安不安全"。
judge 查的是"读起来有没有用、解释是否清楚、有没有答非所问"。
两者互补,报告里要分开呈现。
"""
import json
import re

RUBRIC = """你是护肤推荐质量的严格评审。给下面的推荐打分,每项 1-5 分。

helpfulness: 是否真正解决了用户描述的问题
clarity:     解释是否清晰易懂,普通消费者能否看懂
specificity: 是否具体到成分与用法,而不是空泛套话
faithfulness:是否只依据给出的证据,没有编造

只输出 JSON:{"helpfulness":n,"clarity":n,"specificity":n,"faithfulness":n,"comment":"一句话"}
"""

DIMS = ["helpfulness", "clarity", "specificity", "faithfulness"]


def judge_one(prompt: str, completion: str, model: str = "gpt-4o-mini") -> dict:
    """TODO(C): 调 OpenAI,把 RUBRIC + prompt + completion 拼起来,解析返回的 JSON。
    要点:
      1. temperature=0,否则同一条打分会飘
      2. **打分顺序随机化**,避免位置偏见
      3. 解析失败时返回 None 而不是 0 分,免得污染均值
    """
    raise NotImplementedError


def aggregate(scores: list[dict]) -> dict:
    valid = [s for s in scores if s]
    if not valid:
        return {d: None for d in DIMS}
    return {d: sum(s[d] for s in valid) / len(valid) for d in DIMS}
