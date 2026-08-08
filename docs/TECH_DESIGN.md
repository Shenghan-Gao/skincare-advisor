# 技术设计文档 v4 · AI 皮肤识别 + 后训练护肤顾问系统

> Group 2 · Gen AI Course Project
> **这是一份工作手册,不是提案。** 每个模块都写清了:谁负责、碰哪些文件、
> 跑什么命令、交什么、怎么验收。

---

## 0. 这份文档怎么读

| 你是 | 必读 | 选读 |
|---|---|---|
| **Anna(主线)** | 全部 | — |
| **组员 A(数据+安全+统稿)** | §1–§3、§5、§6.4、§6.6、§8、§10 | §6.2 |
| **组员 B(模型训练)** | §1–§3、§5、§6.1、§6.4、§8、§11 | §6.7 |
| **组员 C(增强+评估)** | §1–§3、§5、§6.3、§6.7、§8、§9 | §6.2 |

配套文档:`TEAM_TASKS.md`(逐人任务与日程)、`HANDOFF.md`(交接协议与 checkpoint 契约)。

---

## 1. 项目一句话

用户上传自拍 + 用自然语言描述诉求 → **CNN 识别皮肤状况** → **RAG 从真实产品与评论中检索证据**
→ **经过 LoRA 微调 + 强化学习后训练的 LLM** 生成个性化、可解释、带证据引用的推荐
→ 安全模块加注免责与禁忌过滤 → FastAPI 服务 + Streamlit 前端,Docker 交付。

**我们真正动手训练的有三块**(这决定了项目不是"套壳 API"):

| | 支柱 | 内容 | 课程锚点 | 负责人 |
|---|---|---|---|---|
| **一** | 视觉识别 | CNN 皮肤分类(自建 CNN + 迁移学习微调) | M3 / A2 | B |
| **二** | LLM 后训练 | 预训练基座 → **LoRA SFT** → **RL(GRPO)** | M8 / M9–11 / A5 | Anna |
| **三** | 生成式图像 | 扩散 img2img 数据增强 + 消融实验 | M4–M7 / A3–A4 | C |

**粘合层**(课程技术,无需额外训练):RAG 检索(M1–M2)、FastAPI+Docker(M1/A1)、
安全与伦理(L1)、Streamlit 前端、评估体系。

---

## 2. 概念澄清(写报告时术语要用对)

| 说法 | 对不对 | 正确表述 |
|---|---|---|
| "用 LoRA 做预训练" | ❌ 不存在 | **现成预训练基座 → LoRA 监督微调(SFT) → RL 后训练**。我们 adapt,不 pre-train |
| "我们训练了 GPT" | ❌ | 我们**微调**了开源基座(Qwen2.5-1.5B),并用 GPT 做教师蒸馏与 judge |
| "RLHF" | ⚠️ 不准确 | 我们用的是 **GRPO + 可验证奖励**,无需人工偏好标注,更接近 DeepSeek-R1 的做法 |
| "扩散模型生成产品图" | ❌ | 扩散用于**训练数据增强**,目标是提升分类 F1,不是生成展示图 |

---

## 3. 系统架构

```
      用户(自拍 + 自然语言诉求)
                │
        ┌───────┴────────┐
        ▼                ▼
 ┌─────────────┐   ┌──────────────┐
 │ 支柱一 CNN  │   │ UserProfile  │
 │ 皮肤识别    │   │ 预算/偏好/孕期│
 └──────┬──────┘   └──────┬───────┘
        │ SkinAnalysis     │
        └────────┬─────────┘
                 ▼
        ┌────────────────┐      ┌──────────────────┐
        │ RAG 检索        │◄─────│ 产品/成分/评论    │
        │ 融合查询构建     │      │ FAISS 向量库      │
        └────────┬───────┘      └──────────────────┘
                 │ Evidence[] + Product[]
                 ▼
        ┌─────────────────────────┐
        │ 支柱二 后训练 LLM        │
        │ base → LoRA SFT → GRPO  │
        └────────┬────────────────┘
                 │ AdvisorResponse
                 ▼
        ┌────────────────┐
        │ 安全模块 (L1)   │  禁忌过滤 + 免责声明
        └────────┬───────┘
                 ▼
        FastAPI ──► Streamlit 前端

  离线:支柱三 扩散增强 ──► 合成样本 ──► 支柱一重训(消融实验)
```

**关键设计:CNN 不在 GRPO 训练回路里。** RL 训练样本中的皮肤画像是合成的
(见 `llm/data_build.py::sample_profile`),CNN 只在**服务时**进入链路。
这就是为什么 B 优化 CNN 与 Anna 训练 GRPO 可以完全并行。

---

## 4. 代码结构与文件归属

**改别人的文件 = 冲突。动手前先对照这张表。**

```
app/                          【Anna】FastAPI 服务
  schemas.py                  ⚠️ 冻结契约,改动需全组同意
  main.py  deps.py  routers/{health,skin,recommend}.py
src/skincare/
  config.py                   ⚠️ 冻结:CONCERNS / SKIN_TYPES 标签空间
  vision/                     【B 训练 / Anna 维护代码】
    model.py                  SimpleCNN + TransferNet(双头)
    data.py  train.py  infer.py
  rag/                        【A 建库 / B 调优】
    ingest.py                 A:原始 CSV → products/chunks
    index.py  retrieve.py     B:向量化与检索调优
  llm/                        【Anna 独占】
    prompts.py                ⚠️ 冻结:训练与服务共用同一模板
    rewards.py                可验证奖励(项目智力核心)
    data_build.py  sft_lora.py  grpo_train.py  dpo_train.py  generate.py
  augment/                    【C 独占】
    diffusion_aug.py          扩散 img2img 增强
  safety/guard.py             【A】禁忌过滤 + 免责
  eval/                       【C 独占】
    harness.py                模型加载解耦层(读 manifest)
    judge.py  run_eval.py  rag_eval.py  vision_eval.py
configs/vision_*.yaml         【B】超参唯一旋钮
scripts/
  validate_data.py            A 的交付验收
  verify_handoff.py           B 的交付验收
fixtures/                     mock 响应 + 10 个已验证的评估样本
tests/                        契约 / 奖励 / 评估自检(11 个)
data/{raw,processed,knowledge} 数据(不进 git)
models/{vision,llm/manifest.json}
```

---

## 5. 冻结契约(动这四样会连锁崩)

> 冻的是**接口**,不是实现。架构、backbone、超参、索引内容、产品数量随便改。

| # | 冻结项 | 文件 | 动了会怎样 |
|---|---|---|---|
| 1 | **数据契约** | `app/schemas.py` | 全线返工 |
| 2 | **标签空间** | `config.py::CONCERNS`/`SKIN_TYPES` | checkpoint + 奖励函数 + 已生成训练数据一起崩 |
| 3 | **Prompt 模板** | `llm/prompts.py` | 训练/服务输入不一致 → **静默退化**(不报错,但变差) |
| 4 | **evidence_id 格式** | `rag/ingest.py` 命名约定 | grounding 奖励失真 |

### 5.1 核心数据结构(`app/schemas.py`)

```python
SkinAnalysis:   skin_type(oily|dry|combination|normal), skin_type_confidence,
                concerns[6 x ConcernScore], model_version
Evidence:       evidence_id, product_id, source(description|review|ingredient), text, score
Product:        product_id, name, brand, category, price_usd, rating, ingredients[]
UserProfile:    query, budget_usd, preferences[], avoid_ingredients[], pregnant
Recommendation: product_id, name, brand, price_usd, reason,
                key_ingredients[], cited_evidence[], matched_concerns[]
AdvisorResponse:analysis, recommendations[], routine_note, disclaimer,
                safety_flags[], generator
```

### 5.2 标签空间(6 个关注点,**不要加第 7 个**)

```python
SKIN_TYPES = ["oily", "dry", "combination", "normal"]
CONCERNS   = ["acne", "dark_spots", "redness", "large_pores", "wrinkles", "dryness"]
```

### 5.3 evidence_id 格式

```
{product_id}:{desc|rev|ing}:{序号}      例:P1001:rev:3  /  P1001:desc:0
```

### 5.4 Mock 模式(并行开发的基础设施)

三层 mock 基础设施,让四个人第一天就能同时开工:

| 文件 | 顶替什么 | 谁受益 |
|---|---|---|
| `fixtures/mock_skin_analysis.json`、`mock_advisor_response.json` | 整个 API 响应(`USE_MOCKS=1`) | 前端 / 部署 / 安全 |
| `fixtures/mock_catalog.json`(12 产品 / 48 evidence) | 真实 FAISS 索引 | **Anna 的 SFT/GRPO 链路** |
| `fixtures/eval_samples.jsonl`(10 个已知答案) | 被评估的模型 | **组员 C 的评估器** |

**没有任何模型、任何真实数据时,四条线都能推进。**

---

## 6. 模块详细设计

每个模块统一给出:**负责人 / 文件 / 命令 / 契约 / 验收**。

### 6.1 支柱一:CNN 皮肤识别 —— M3 / A2

**负责人** B(训练调优) · **文件** `src/skincare/vision/`、`configs/vision_*.yaml`

两个模型是刻意的,报告要做对比:

- `SimpleCNN` —— 手写 4 层卷积块(Conv→BN→ReLU→MaxPool)+ 全局池化 + 双头。
  这是"从第一性原理理解"的证据,rubric 奖励这一点,**不要删**。
- `TransferNet` —— 预训练 ResNet/EfficientNet 主干 + 相同双头。**微调预训练模型即 L2**。

**双头设计**:肤质类型单标签(CrossEntropy)+ 6 个关注点多标签(BCEWithLogits),
一次前向输出两者(`model.py::multitask_loss`)。

```bash
python -m skincare.vision.train --config configs/vision_transfer.yaml
python -m skincare.vision.train --config configs/vision_simple.yaml   # 对照基线,别省
```

**超参只通过 yaml 改,不改 .py。** 扫描建议:backbone ∈ {resnet18, resnet50} × lr ∈ {1e-4, 3e-4, 1e-3}。

**checkpoint 契约**(`infer.py` 依赖):
```python
{"state_dict": ..., "kind": "transfer"|"simple", "metrics": {...}, "config": {...}}
```
checkpoint **自描述**:架构从 `config.backbone` 读,不从当前代码猜 ——
所以换 backbone 重训,Anna 那边不用改代码就能加载(已实测 resnet18/resnet50/SimpleCNN)。
标签空间不一致会抛 `CheckpointMismatch` 并说清原因。

**验收** `python scripts/verify_handoff.py vision models/vision/<run_name>.pt`
**指标** accuracy、macro-F1、每类混淆矩阵;自建 CNN vs 迁移学习对比表。

---

### 6.2 支柱二:LLM 后训练栈 —— M8 / M9–11 / A5 【项目核心】

**负责人** Anna · **文件** `src/skincare/llm/`

```
Qwen2.5-1.5B-Instruct(现成基座,不自己训)
   │
   ├─ 阶段1  LoRA 监督微调 SFT        ← M8:学会领域的结构化解释风格
   │
   └─ 阶段2  RL 后训练 GRPO           ← M9–11 / A5:用可验证奖励优化质量
```
选 1.5B 是为了**免费 Colab T4 也能跑通**;有更好的 GPU 可换 7–8B。

#### (a) 数据构造 `data_build.py` 【已实现】

合成用户画像 → 检索真实证据 → 教师蒸馏 → **按奖励过滤** → 切分数据集。

```bash
# 第一天:零成本跑通全链路(合成目录 + 离线假教师,不需要 A 的数据、不需要 API key)
python -m skincare.llm.data_build --n 60 --mock-retrieval --dry-teacher

# 真正造 SFT 数据(需要 OPENAI_API_KEY)
python -m skincare.llm.data_build --n 800 --mode sft

# 只造 RL 数据(不需要目标答案,不花钱)
python -m skincare.llm.data_build --n 600 --mode rl
```

**产出三个文件**
| 文件 | 用途 | 谁用 |
|---|---|---|
| `sft.jsonl` | 带教师目标答案,供 LoRA SFT | Anna |
| `rl.jsonl` | 只有 prompt + 奖励上下文,供 GRPO | Anna |
| `rl_test.jsonl` | 留出集(默认 15%) | **组员 C 评估** |

**三个关键设计**
1. **教师过滤是 SFT 有效性的分水岭**:低于 `--threshold`(默认 0.8)的教师输出直接丢弃。
   不过滤等于让模型学噪声。实测会丢掉"成分与关注点不匹配"的答案。
2. **断点续跑**:每条教师结果即时写入 `sft.cache.jsonl`,中途挂掉重跑不会重复烧钱。
3. **`--mock-retrieval` 让 Anna 也能并行**:用 `fixtures/mock_catalog.json`
   (12 个产品 / 48 条 evidence,覆盖全部 6 个关注点)顶替真实索引,
   等 A 交付后去掉这个参数即可。**把并行原则用在自己身上。**

RL 数据只需要 prompt + 奖励上下文(`concerns` / `evidence_ids` / `product_ids` /
`pregnant` / `avoid`),不需要目标答案 —— 这也是 GRPO 相比 SFT 省事的地方。

#### (b) 阶段一:LoRA SFT `sft_lora.py`

PEFT LoRA(r=16, α=32, dropout=0.05)作用于全部注意力与 MLP 投影层,TRL `SFTTrainer`。
显存不足换 QLoRA(4-bit)。

```bash
python -m skincare.llm.sft_lora --epochs 2
```

#### (c) 阶段二:GRPO `grpo_train.py`

**为什么选 GRPO**:①它就是 **M11 精读的 DeepSeek-R1 所用算法**;
②**不需要训练奖励模型**(我们的奖励规则可验证);③比 PPO 轻得多,没有 value network。

机制:每个 prompt 采样一**组**候选 → 奖励打分 → 把概率质量推向组内高于平均的答案。
`beta=0.04` 的 KL 惩罚防止 reward hacking;RL 的学习率要比 SFT 小两个量级(1e-6)。

```bash
python -m skincare.llm.grpo_train --steps 300 --group-size 8
```

#### (d) 可验证奖励 `rewards.py` —— **项目的智力核心**

**不需要 GPU,第一天就能写完并单元测试。**

| 奖励项 | 权重 | 程序化判定 |
|---|---|---|
| `format` | 0.15 | 能否解析为规定 JSON schema |
| `ingredient_match` | 0.30 | 推荐成分是否命中检测到的关注点(查 `ingredient_rules.json`) |
| `grounding` | 0.25 | `cited_evidence` 的 id 是否真实存在于上下文 —— **惩罚幻觉引用** |
| `product_validity` | 0.15 | 推荐产品是否来自检索,而非模型记忆 |
| `safety` | 0.15 | 是否含免责声明;孕期/过敏禁忌成分则重罚 |

规则表由 **A** 维护 —— **A 每扩充一条,RL 训练信号就更准**,这是天然的并行接口。

#### (e) 回退阶梯

```
GRPO 不收敛 → DPO(dpo_train.py,离线偏好对,半天出结果)
DPO 也来不及 → 仅 SFT-LoRA(仍完整覆盖 A5"微调 LLM"要求)
```
用**同一套 `rewards.py`** 给采样答案打分,取最高/最低构造 chosen/rejected ——
切到 DPO **不浪费任何已完成的工作**。

#### (f) 交给 C 的接口

训完往 `models/llm/manifest.json` 填路径即可,评估由 C 跑:
```json
{"base": "Qwen/Qwen2.5-1.5B-Instruct", "sft": "models/llm/sft-lora", "grpo": "models/llm/grpo"}
```

---

### 6.3 支柱三:扩散图像增强 —— M4–M7 / A3–A4

**负责人** C · **文件** `src/skincare/augment/diffusion_aug.py`

**目标不是"生成好看的皮肤图",而是用消融实验证明生成式增强提升分类性能。**
报告里最有力的一句:*"加入 N 张扩散合成样本后,少数类 macro-F1 从 0.61 提升到 0.68"*

```bash
python -m skincare.augment.diffusion_aug --concern acne --n 200
```

**方法**:`StableDiffusionImg2ImgPipeline` 做 img2img 重绘 —— 以真实少数类图片为起点,
保留皮肤纹理语义。**比从零训 GAN 稳得多,一天能出结果**。
prompt 用临床描述性措辞,`strength` 在 0.5–0.7 之间试。

**两条红线**
1. **只增强 train,绝不动 val/test** —— 否则指标失去意义,答辩会被追问。
2. 合成人脸**仅用于训练**,不得在 demo 里冒充真实用户;报告须写明合成数据占比与人工剔除比例。

**回退**:img2img 效果差 → "经典增强 vs 扩散增强"对比 → 最低档只交类别不均衡分析 + 经典增强消融。
**任一档都覆盖 A3/A4。**

**交付** `data/processed/vision_train_aug.csv` → B 用**同样超参**重训,两组结果之差即消融结论。

---

### 6.4 RAG 检索 —— M1–M2

**负责人** A(建库)+ B(调优) · **文件** `src/skincare/rag/`

```bash
python -m skincare.rag.ingest    # A: 原始 CSV → products.parquet / chunks.parquet
python -m skincare.rag.index     # B: 向量化 → FAISS
```

**查询构建是增值的兑现点**(`retrieve.py::build_query`):把
「用户诉求 + CNN 皮肤标签 + 成分规则展开」融合成一个查询向量。
通用 chatbot 看不到照片、也访问不了产品表 —— 这就是"超越开箱即用 LLM"的具体证据。

硬过滤(预算等)在检索层做,不交给 LLM 推理。
**B 可以换 embedding 模型、调 chunk 与 top-k,但不要改 `evidence_id` 格式。**

**验收** `python scripts/verify_handoff.py rag`

---

### 6.5 安全与伦理 —— L1

**负责人** A · **文件** `src/skincare/safety/guard.py`

孕期/过敏/致痘成分过滤、强制医疗免责、越界医疗问题拒答。
它读的就是 A 自己维护的 `ingredient_rules.json`,所以同一人做最顺。

报告伦理章节要覆盖:人脸图像**用后即弃不持久化**、医疗免责边界、
**推荐偏见分析**(品牌/价格/肤色分布)、合成数据的诚实披露、数据集许可与来源。

---

### 6.6 API、前端与部署 —— A1 / L4

**负责人** Anna · **文件** `app/`、`ui/`、`docker/`

| 端点 | 输入 | 输出 |
|---|---|---|
| `GET /health` | — | 状态 + 是否 mock 模式 |
| `POST /analyze-skin` | 图片 | `SkinAnalysis` |
| `POST /recommend` | `RecommendRequest` | `AdvisorResponse` |

```bash
make api          # :8000/docs
make ui           # :8501
make docker-up    # 完整栈
```
**L4 独立得分点**:部署到 AWS/GCP,报告本地 vs 云端的**延迟 p50/p95、吞吐 req/s、单次成本**。
mock 模式下就能先跑通部署链路与压测脚本。

⚠️ 评分是在教授机器上跑仓库 —— **依赖要 pin 死,并在干净环境验证 `make docker-build`**。

---

### 6.7 评估体系

**负责人** C · **文件** `src/skincare/eval/`

**核心设计:评估工具与被评估模型解耦**,让 C 不必等 Anna。

```bash
python -m skincare.eval.run_eval --self-test        # 第一天:无模型自检
python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl \
       --variants base sft grpo                     # 最后:三段式对比
```

`harness.py` 读 `models/llm/manifest.json`,**缺的档位自动跳过**;
`fixtures/eval_samples.jsonl` 有 **10 个已知答案的样本**(完美/伪造引用/成分不符/
孕期不安全/格式错乱/部分引用……),每个标注期望分数区间。
**自检通过 = 评估器本身正确**,已实测 10/10。

`judge.py` 做规则奖励覆盖不到的主观维度(helpfulness / clarity / specificity / faithfulness):
temperature=0、打分顺序随机化避免位置偏见、解析失败返回 None 而非 0 分。

---

## 7. 课程覆盖矩阵

### 7.1 Module 1–12

| 模块 | 主题 | 项目载体 | 谁 |
|---|---|---|---|
| M1–M2 | GenAI 基础、嵌入、FastAPI/Docker | RAG 句嵌入 + API 底座 | A/B/Anna |
| M3 | FCNN / CNN | **支柱一** 皮肤识别 | B |
| M4–M5 | VAE / GAN | **支柱三** 生成式增强的对照与讨论 | C |
| M6 | Normalizing Flows | ⚠️ 本项目未覆盖(报告中说明取舍即可) | — |
| M7 | 能量与扩散模型 | **支柱三** 扩散 img2img 增强 + 消融 | C |
| M8 | Transformer / 微调 GPT | **支柱二** LoRA SFT | Anna |
| M9–M10 | RL 基础与策略优化 | **支柱二** GRPO | Anna |
| M11 | LLM 与推理(DeepSeek-R1) | **可验证奖励设计** | Anna |
| M12 | 多模态 / 部署 SD | 图文多模态输入 + SD 用于支柱三 | C/Anna |

### 7.2 学习目标 L1–L5

| | 目标 | 载体 |
|---|---|---|
| L1 | 局限与伦理 | §6.5 安全模块 + 报告伦理章节 + 合成数据披露 |
| L2 | 领域微调预训练模型 | CNN 迁移学习微调 + LLM LoRA SFT |
| L3 | 用 PyTorch 实现生成模型 | SimpleCNN 自建 + 扩散增强 + Transformer 后训练 |
| L4 | 云部署与基准对比 | FastAPI/Docker → AWS/GCP + benchmark 表 |
| L5 | 团队协作交付 | 四人分工 + 报告 + demo + GitHub |

### 7.3 Assignment 1–5

| 作业 | 复用/延伸 |
|---|---|
| A1 FastAPI/Docker/uv | 项目 API 与部署底座 |
| A2 CNN | 支柱一(升级为多标签 + 迁移学习对比) |
| A3 VAE/GAN | 支柱三对照实验 |
| A4 扩散 | 支柱三主线 |
| A5 Transformer + RL | 支柱二 SFT + GRPO |

---

## 8. 数据流与交接点

```
组员 A ──vision_*.csv──────────────► 组员 B(训 CNN)
       ──vision_*.csv──────────────► 组员 C(挑少数类做增强)
       ──products/chunks.parquet──► 组员 B(建索引) ──► Anna(RAG 打通)
       ──ingredient_rules.json────► Anna(奖励函数)+ A 自己的安全模块

组员 C ──vision_train_aug.csv─────► 组员 B(重训 → 消融结论)
Anna  ──models/llm/manifest.json─► 组员 C(三段式评估)
组员 B ──models/vision/*.pt───────► Anna(最终装机 / demo)
```

| # | 交接 | 交付物 | 何时 | 卡住的绕行 |
|---|---|---|---|---|
| 1 | A → B、C | `vision_*.csv` + `class_distribution.md` | Day 2–3 | B/C 用小样本子集先跑通流程 |
| 2 | A → B、Anna | `products/chunks.parquet` | MVP 之前 | Anna 用 `fixtures/` 继续做 SFT 格式 |
| 3 | C → B | `vision_train_aug.csv` | 消融前 | B 先交基线结果,增强版后补 |
| 4 | Anna → C | 填 `manifest.json` | Day 5 | **C 全程用 fixtures + base 档推进,不阻塞** |
| 5 | B → Anna | `*.pt` + `_report.json` | demo 前 | Anna 用 `USE_MOCKS=1` 顶着 |

**每次交接都有验收命令,跑不过不算交付:**
```bash
python scripts/validate_data.py {vision|products|chunks|rules|all}   # A 的交付
python scripts/verify_handoff.py vision models/vision/<run>.pt       # B 的 CNN
python scripts/verify_handoff.py rag                                 # B 的索引
python -m skincare.eval.run_eval --self-test                         # C 的评估器
make test                                                            # 全组,合并前必跑
```

---

## 9. 评估方案与报告主结果

| 模块 | 指标 | 对比 | 谁 |
|---|---|---|---|
| CNN | accuracy、macro-F1、混淆矩阵 | **自建 CNN vs 迁移学习** | B/C |
| 扩散增强 | 少数类 macro-F1 | **增强前 vs 后(消融)** | C |
| RAG | Precision@3、幻觉率 | embedding 模型 A vs B | C |
| LLM 后训练 | 五个奖励分量 + LLM-judge 四维 | **base vs SFT vs GRPO** | C |
| 部署 | 延迟 p50/p95、吞吐、成本 | 本地 vs 云端 | Anna |

**报告的三张主图**(缺一不可):
1. **三段式提升曲线** base → SFT → GRPO 的各奖励分量(证明后训练有效)
2. **消融柱状图** 加/不加扩散合成数据的少数类 F1(证明生成式建模有价值)
3. **模型对比表** 自建 CNN vs 迁移学习(证明理解底层 + 微调收益)

因为奖励是规则化的,第 1 张图**完全可复现、不需要人工标注** —— 这是最有说服力的部分。

---

## 10. 风险与回退

| 风险 | 触发信号 | 回退 |
|---|---|---|
| GRPO 不收敛 | 奖励曲线震荡/塌陷 | → DPO → 仅 SFT-LoRA(仍覆盖 A5) |
| 显存不足 | OOM | QLoRA 4-bit / 换更小基座 / 减 group-size |
| 扩散增强无效 | F1 无提升甚至下降 | → 经典增强对比 → 只交类别不均衡分析 |
| 数据延期 | A 未按时交付 | 全员 `USE_MOCKS=1` 继续;Anna 用 fixtures 推进 SFT 格式 |
| Docker 在教授机器跑不起来 | 干净环境构建失败 | **提前一天在干净环境验证**,依赖 pin 死 |
| 标签空间被改动 | `CheckpointMismatch` | 立刻统一标签空间并重训,别用 `strict=False` 绕 |

**最高优先级**:前 3 天必须拿到 **MVP(CNN + RAG + API 打通)**,这是保底分。
SFT/GRPO/扩散都是往上叠的加分,不是命门。

---

## 11. 环境与命令速查

```bash
# 一次性
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ui]"          # 基础(mock 模式够用)
uv pip install -e ".[vision,rag,llm]"  # 需要训练时再装
cp .env.example .env                   # 填 OPENAI_API_KEY / HF_TOKEN

# 日常
make api / make ui / make test / make docker-up

# 训练(用 Colab / Kaggle GPU)
python -m skincare.vision.train --config configs/vision_transfer.yaml
python -m skincare.rag.ingest && python -m skincare.rag.index
python -m skincare.llm.data_build --n 800 --with-targets
python -m skincare.llm.sft_lora --epochs 2
python -m skincare.llm.grpo_train --steps 300
python -m skincare.augment.diffusion_aug --concern acne --n 200
python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl
```

**环境要点**:CNN 与扩散、SFT/GRPO 都要 GPU,一律用 Colab/Kaggle;
本机只跑 API 与前端。`USE_MOCKS=1` 是默认值,有真模型后改成 0。

---

## 12. 交付物

- **可运行系统**(Docker,本地/云端可评测):照片 + 文字 → 皮肤识别 → 可解释推荐
- **最终报告**:含 §9 三张主图、方法、伦理与局限
- **现场 demo**
- **公开 GitHub 仓库**:完整代码 + 复现说明 + 数据来源说明

报告章节归属见 `TEAM_TASKS.md` §6(硬性交付,不是"大家一起写")。
