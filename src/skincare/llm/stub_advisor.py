"""Advisor 的确定性替身 —— 无 API key、无 GPU 也能把真实链路跑通。

为什么要它:generate.Advisor 要么调 OpenAI(需要 key),要么加载本地 Qwen(需要 GPU)。
两者都不可用时,整条「检索 → 生成 → 安全过滤」就没法端到端验证。本文件用纯 Python
拼一个**合法**的 AdvisorResponse,关键约束与真实模型完全相同:

  1. product_id 只能来自本次检索返回的 products(不许凭记忆编产品);
  2. cited_evidence 只能来自本次检索返回的 evidence_id(不许幻觉引用);
  3. key_ingredients 来自目录里的真实成分表 —— 安全护栏靠它判断孕期禁用成分。

因此 rewards.py 的 grounding / product_validity / format 都会给满分,
可以当作训练与评测链路的**上界基线(oracle baseline)**。
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
    """确定性生成器:把检索结果结构化地"复述"成推荐。"""

    def __init__(self, name: str = "stub"):
        self.backend = name
        self.adapter = None

    @staticmethod
    def _matched_concerns(ingredients: list[str], wanted: list[str]) -> list[str]:
        """用 ingredient_rules 把成分反查回它能处理的关注点,只保留用户真正关心的。"""
        rules = load_rules()["concern_to_ingredients"]
        ings = " ".join(ingredients).lower()
        hit = [c for c, needles in rules.items() if any(n.lower() in ings for n in needles)]
        return [c for c in hit if c in wanted] or hit[:1]

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
