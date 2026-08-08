"""Retriever 的替身 —— 用合成目录顶替真实 FAISS 索引。

为什么需要它:Anna 的 SFT/GRPO 链路要等组员 A 交出 products/chunks.parquet 才能跑。
有了它,Anna 第一天就能把「造数据 → 蒸馏 → 过滤 → 训练」整条链路跑通测通,
等真实数据到位后换掉一行即可。**这是把并行原则用在 Anna 自己身上。**

接口与 rag.retrieve.Retriever 完全一致(鸭子类型),可直接互换。
"""
import json
import random

from app.schemas import Evidence, Product, RetrievalResult, SkinAnalysis, UserProfile
from skincare.config import FIXTURES, SEED


class MockRetriever:
    def __init__(self, path=None, seed: int = SEED):
        data = json.loads((path or (FIXTURES / "mock_catalog.json")).read_text()
                          if hasattr(path or FIXTURES, "read_text")
                          else open(path or FIXTURES / "mock_catalog.json").read())
        self._products = data["products"]
        self._chunks = data["chunks"]
        self.rng = random.Random(seed)

    def search(self, profile: UserProfile, analysis: SkinAnalysis | None,
               top_k: int = 3, n_chunks: int = 12) -> RetrievalResult:
        wanted = set(analysis.top_concerns() if analysis else [])

        # 按关注点重合度打分,保证检索结果与画像相关(不是随机的,奖励才有信号)
        scored = []
        for p in self._products:
            overlap = len(wanted & set(p["_concerns"]))
            if profile.budget_usd and p["price_usd"] > profile.budget_usd:
                continue
            scored.append((overlap, overlap + self.rng.random() * 0.1, p))
        scored.sort(key=lambda x: -x[1])
        # 只保留真正有关注点重合的;全都不相关时才退回 top_k(否则奖励信号会被稀释)
        chosen = [p for ov, _, p in scored[:top_k] if ov > 0] or [p for _, _, p in scored[:top_k]]
        ids = {p["product_id"] for p in chosen}

        evidence = [Evidence(evidence_id=c["evidence_id"], product_id=c["product_id"],
                             source=c["source"], text=c["text"], score=1.0)
                    for c in self._chunks if c["product_id"] in ids][:n_chunks]
        products = [Product(**{k: v for k, v in p.items() if not k.startswith("_")})
                    for p in chosen]
        return RetrievalResult(products=products, evidence=evidence)
