# 把训练外包给队友:什么能给、什么不能给、怎么给

> 核心原则:**冲突只发生在两个人改同一个文件,不发生在两个人同时跑代码。**
> 训练本身不改代码 —— 它吃 GPU、产出一个 checkpoint 文件。
> 所以只要队友「跑训练但不碰训练代码」,就完全不冲突。

---

## 1. 什么能外包,什么不能

| 任务 | 能否外包 | 理由 |
|---|---|---|
| **CNN 训练 / 调参扫描** | ✅ **最适合** | GPU 密集、大部分时间在等;目标明确(F1 越高越好);产出是文件 |
| **RAG 建索引** | ✅ **适合** | CPU 跑 embedding、确定性、无需研究判断;本来就是 B 数据工作的延伸 |
| **SFT 数据蒸馏(调 API)** | ⚠️ 可外包但要盯 | 花钱、要控质量;**教师输出必须用 `rewards.total_reward` 过滤**,这步做砸了 SFT 就白训 |
| **LoRA SFT 训练** | ⚠️ 谨慎 | 在通往 GRPO 的关键路径上,出问题排查要上下文 |
| **GRPO / RL 后训练** | ❌ **不要外包** | 项目的差异化所在;最容易需要边训边改代码;也正是你想学的部分 |

**推荐组合:CNN 训练 + RAG 建索引外包出去,SFT 和 GRPO 自己留着。**
这样你省下最多的等待时间,又保住了最有价值、最需要连续上下文的两段。

> 关于学习:模型代码是你写的,学习发生在**设计和读结果**,不在守着进度条。
> 但建议你**亲手完整跑一遍第一次训练**(约 1 小时),把整个流程摸清楚,
> 再把重复的调参扫描交出去 —— 既学到了,又省了时间。

---

## 2. 交接协议(四步,少一步都会出事)

### 第 1 步:冻结并打 tag
交接前把代码定住,队友基于这个 commit 训练:

```bash
git add -A && git commit -m "freeze: vision code for training handoff"
git tag vision-freeze-v1
git push && git push --tags
```
**交接期间你不要改 `src/skincare/vision/**`。** 你改了,他的 checkpoint 就可能加载不回来。
真要改,先通知他,然后重新打 tag。

### 第 2 步:划清文件归属

| 队友**可以**改 | 队友**绝对不碰** |
|---|---|
| `configs/vision_*.yaml` | `src/skincare/vision/**` |
| 自己的 Colab notebook 副本 | `app/schemas.py` |
| `models/` 下自己 run_name 的产物 | `src/skincare/llm/**` |

**超参只能通过 config 文件改。** 发现代码 bug → 报给你,你改,重新打 tag。
这一条是整套协议的核心,守住它就基本不会打架。

### 第 3 步:队友跑训练

```bash
python -m skincare.vision.train --config configs/vision_transfer.yaml
python -m skincare.vision.train --config configs/vision_simple.yaml   # 对照基线,别省
```
每次实验改 `run_name`,产物不会互相覆盖:
`models/vision/<run_name>.pt` + `models/vision/<run_name>_report.json`

建议让他做一轮小扫描(这正是外包最划算的部分):
backbone ∈ {resnet18, resnet50} × lr ∈ {1e-4, 3e-4, 1e-3},选验证集最好的。

### 第 4 步:验收(**跑不过就不算交付**)

```bash
python scripts/verify_handoff.py vision models/vision/transfer_resnet18_lr3e4.pt
python scripts/verify_handoff.py rag
```
这个脚本查的是**契约**不是效果:文件能否被 `SkinClassifier` 加载、
前向输出是否满足 `SkinAnalysis`(6 个 concern 分数)、evidence_id 是否唯一。

**队友交付清单**:`.pt` + `_report.json` + verify 通过的截图/输出。
效果好不好看 `_report.json` 里的指标表,那是另一回事。

---

## 3. checkpoint 契约(改了会连锁崩)

`infer.py` 依赖这个结构,队友的产物必须长这样:

```python
{
  "state_dict": ...,          # 模型权重
  "kind": "transfer"|"simple", # 决定重建哪个架构
  "metrics": {...},            # 验证集指标
  "config": {...},             # 复现用
}
```

---

## 4. 你和队友的时间线怎么错开

```
你:  [骨架+rewards] → [RAG打通] → [SFT数据] → [SFT] → [GRPO] → [评估]
                ↘ 交接 tag                                    ↗ 收回 .pt
队友:            [CNN训练+调参扫描] ──────────────────────┘
                 [RAG建索引]
```

关键点:**CNN 的 checkpoint 只在你做「三段式评估」和最终 demo 时才真正需要**。
在那之前,`USE_MOCKS=1` 让你完全不被他堵住。
所以他晚一两天交,不影响你推进 SFT 和 GRPO —— 这就是真正的并行。

---

## 5. 为什么改了代码会加载不了 checkpoint(原理)

`.pt` 文件里**只有权重,没有架构**。`load_state_dict` 靠「参数名 + 形状」匹配,
而架构是加载时从**当前的 `model.py`** 重新构建的。代码一改,两边就对不上:

| 队友改了什么 | 报错 |
|---|---|
| 层改名(`features` → `backbone`) | `Missing key(s) in state_dict` |
| 通道宽度 `block(32,64)` → `block(32,128)` | `size mismatch ... shape [64,32,3,3] vs [128,32,3,3]` |
| 增删一层 | `Unexpected key(s)` |
| 换 backbone(resnet18 → resnet50) | 键名与 feat_dim 全变 |
| **改 `config.py` 的 `CONCERNS`** | 输出头 6→7,形状不匹配 |

**最后一条最阴险** —— 它不在 `model.py` 里,而在 `config.py`。B 在整理标签时
顺手加一个 concern,就会同时打穿:checkpoint、`schemas.py`、奖励函数的成分映射、
以及你**已经生成好的 SFT/GRPO 训练数据**。

### 已经做的两个防护(`infer.py`)

1. **自描述 checkpoint** —— 架构参数(`kind` / `backbone`)从 checkpoint 自己读,
   不从当前代码猜。**队友换 backbone、调宽度重训,你这边不用改任何代码就能加载。**
   已用 resnet18 / resnet50 / 自建 CNN 实测通过。
2. **标签空间校验** —— 加载前先比对输出头宽度与 `config.py`,不一致时抛
   `CheckpointMismatch` 并说清是谁改了什么,而不是甩一段 shape mismatch。

> 记住一句话:**checkpoint 不是独立文件,它是 (代码, 权重) 的配对。**
> 队友交回权重时,必须说清基于哪个 git tag。

---

## 6. 「MVP 之后交给队友 refine,我同步做 GRPO」—— 可行性评估

**结论:可行,而且比先前的排法更好。但真正的冲突点不在 CNN,在别处。**

### CNN refine ∥ GRPO:**零冲突,可以放心并行**

关键事实:**GRPO 训练根本不依赖 CNN。**
看 `data_build.py` —— RL 训练样本里的 `analysis` 是 `sample_profile()` **合成**出来的,
不是 CNN 推理出来的。CNN 只在**服务/demo 时**才进入链路。

所以队友把 CNN 从 0.71 refine 到 0.78,对你的 GRPO 训练**没有任何影响**,
你也不需要等他。他最后交个 `.pt`,你放进去就行。这是真正意义上的并行。

### RAG refine ∥ GRPO:**内容可以改,格式必须冻**

- ✅ **安全**:加产品、加评论、换 embedding 模型、调 chunk 大小 →
  你的训练数据每一行里 prompt 和 evidence_ids 是**自洽的**,不受影响。
- ❌ **危险**:改 `evidence_id` 的**命名格式**,或改 `prompts.py` 的**模板** →
  训练时和服务时的输入长得不一样,模型会**静默退化**(不报错,但效果变差)。
  这比加载失败更可怕 —— 加载失败你立刻知道,这个你不知道。

### 真正要冻的四样东西(冻这些,其余随便改)

| 冻结项 | 文件 | 动了会怎样 |
|---|---|---|
| 数据契约 | `app/schemas.py` | 全线返工 |
| **标签空间** | `config.py::CONCERNS` / `SKIN_TYPES` | checkpoint + 奖励函数 + 训练数据一起崩 |
| **Prompt 模板** | `llm/prompts.py` | 训练/服务不一致,**静默退化** |
| **evidence_id 格式** | `rag/ingest.py` 的命名约定 | grounding 奖励失真 |

**冻结的是接口,不是实现。** 架构、backbone、超参、索引内容、产品数量 ——
队友随便改,不会碰到你。

### 建议的排法

```
你:  [MVP] → [SFT数据] → [SFT] → [GRPO] → [评估] → [装上最终 .pt]
        │                                              ↑
        └─ 冻结 4 项接口 + 打 tag ──► 队友 refine CNN ─┘
                                      队友 refine RAG 内容
```
在「装上最终 .pt」之前,`USE_MOCKS=1` 让你完全不被堵住。
队友晚一两天交,不影响你 —— 甚至他 refine 失败你也还有 MVP 那版兜底。
