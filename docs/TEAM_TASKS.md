# 分工(v3)

| 角色 | 负责人 | 范围 |
|---|---|---|
| **主线** | **Anna** | LLM 后训练(LoRA SFT + GRPO)+ 前端 + 部署 + 集成 + demo |
| **数据 + 安全 + 统稿** | 组员 A | 数据清洗、成分规则表、安全模块、报告统稿 |
| **模型训练** | 组员 B | CNN 训练调优、RAG 索引与检索优化 |
| **生成式增强 + 评估** | 组员 C | 扩散图像增强与消融、评估体系、报告图表 |

**这版取消了"安全+报告"这个独立角色。** 理由:`guard.py` 只有百来行规则过滤,
撑不起一个人;而报告"大家一起写"必然没人写。所以:

- **安全模块**归组员 A —— 它读的 `ingredient_rules.json` 本来就是 A 建的,同一个人做最顺。
- **报告**每人写自己的模块章节(**这是硬性交付,不是可选**),**统稿**归组员 A ——
  A 的数据工作前 3 天最重、后期最闲,正好接统稿。
- 腾出的第四个人去补**目前最大的课程缺口**:M4–M7(VAE/GAN/扩散)+ A3–A4 现在完全没覆盖,
  而这是 syllabus 里三个模块两次作业的分量。

> 判断能否并行,只看一条:**会不会和别人改同一个文件。**
> `USE_MOCKS=1` 让所有人第一天就能跑起完整系统,谁也不等谁。

---

## 0. 全员第一天

```bash
cd ~/Documents/skincare
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ui]" && cp .env.example .env
make api    # :8000/docs
make ui     # :8501
make test   # 17 passed
```

---

## 1. Anna — 主线

**独占:** `src/skincare/llm/**`、`app/**`、`ui/**`、`docker/**`
**注意:** `src/skincare/eval/**` 归组员 C —— 你只需在训完后把路径填进 `models/llm/manifest.json`,评估由 C 跑。

| # | 任务 | 产出 |
|---|---|---|
| 1 | 补齐 `rewards.py` 单元测试(**不需要 GPU**) | `make test` 全绿 |
| 2 | 前端打磨 + Docker 干净构建 | 可演示 UI、`make docker-up` 通过 |
| 3 | 接真实检索,`USE_MOCKS=0` 打通 | **可演示 MVP** |
| 4 | SFT 数据构造【脚本已就绪】`data_build.py --mock-retrieval --dry-teacher` 可先零成本跑通 | `sft.jsonl` / `rl.jsonl` / `rl_test.jsonl` |
| 5 | LoRA SFT | `models/llm/sft-lora` |
| 6 | GRPO 后训练 | `models/llm/grpo` + 奖励曲线 |
| 7 | 云部署 + benchmark(L4) | 本地 vs 云端对比表 |
| 8 | 填 `models/llm/manifest.json` 交给 C | C 产出三段式对比表 |

**报告章节**:LLM 后训练方法、奖励设计、部署与 benchmark。
> 第 3 步(MVP)**前 3 天必须完成**,这是保底分。

---

## 2. 组员 A — 数据 + 安全 + 报告统稿

**你拥有:** `rag/ingest.py`、`data/knowledge/*`、`data/processed/*`、`safety/guard.py`、报告主文档
**你不碰:** 任何训练脚本、`app/schemas.py`

**节奏**:前 3 天全力做数据(你是所有人的上游),中期做安全模块,后期统稿。

### 阶段一(Day 1–3):数据 —— 四份交付物

**通用规则**:产物放 `data/processed/`(不进 git,用共享盘);
每交一份先跑 `python scripts/validate_data.py <类型>`,**跑不过不算交付**;
清洗代码写成脚本/notebook 提交,报告要写数据处理方法。

#### 交付物 1:视觉标签表(**最紧急**,B 和 C 都等着)

**输入**:建议合并 2–3 个 Kaggle 数据集:
`unidpro/facial-skin-condition-dataset`、`killa92/facial-skin-analysis-and-type-classification`、
`trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes`

**清洗步骤**
1. **统一标签体系**(最花时间):各数据集类名不同,映射到固定标签空间:
   - `skin_type` ∈ `oily | dry | combination | normal`
   - 6 个 0/1 关注点:`acne, dark_spots, redness, large_pores, wrinkles, dryness`
   - 映射不上的类别(如 "eye bags")**直接丢弃**,不要硬塞
   - ⚠️ **不要擅自增加第 7 个关注点** —— 会连锁打穿 checkpoint、奖励函数和已生成的训练数据
2. **删坏文件**:PIL 打不开、尺寸 < 100px、非 RGB
3. **去重**:`imagehash.phash`,汉明距离 ≤ 5 视为重复
4. **人脸裁剪**(推荐):OpenCV/MediaPipe 裁到人脸区域,提升信噪比
5. **划分**:70/15/15,**按 skin_type 分层**
   ⚠️ **同一个人的多张照片必须落在同一个 split**,否则数据泄漏、验证集虚高,答辩会被追问
6. **统计类别分布**(C 要靠它决定增强哪些类)

**输出**
```
data/processed/vision_{train,val,test}.csv
列: filepath, skin_type, acne, dark_spots, redness, large_pores, wrinkles, dryness
    filepath 相对路径 | skin_type 字符串 | 其余 6 列 0/1 整数
data/processed/class_distribution.md
```
**验收**:`python scripts/validate_data.py vision`

#### 交付物 2:产品表

**输入**:`nadyinky/sephora-products-and-skincare-reviews` 的 `product_info.csv`(约 8000 行)

1. 只保留护肤品类(`primary_category` 含 Skincare),丢彩妆/香水/工具
2. **成分串解析**(最脏):`ingredients` 是长文本,混着营销话术和换行。
   按逗号切 → 去首尾空格 → **转小写** → 丢掉 > 60 字符的碎片(那是句子不是成分)→ 去重 → 存 list
3. **成分为空的行直接丢**
4. `price_usd` 转 float,缺失或 ≤ 0 丢弃;`rating` 转 float,缺失填 `None`(**不要填 0**)
5. 按 `product_id` 去重,保留评论数最多那条

**输出**
```
data/processed/products.parquet
product_id(str,唯一) | name | brand | category
price_usd(float) | rating(float|null) | ingredients(list[str],全小写)
```
**验收**:`python scripts/validate_data.py products`

#### 交付物 3:检索片段表

**输入**:`reviews_*.csv`(5 分片,约 100 万行)+ 产品表

1. **只留英文**(langdetect 或 ASCII 比例阈值)
2. **长度过滤**:< 20 字符丢,> 1500 字符截断
3. **去重**:相同 `review_text` 去重
4. **每个产品最多 20 条**,按 helpfulness 排序取前 20(不封顶热门商品会霸占检索)
5. **PII 清理**:正则去邮箱、@handle、电话
6. **切三类片段**:`description`(每产品 1 条)、`ingredient`(成分拼句,每产品 1 条)、`review`(每条 1 条)
7. **`evidence_id` 固定格式**:`{product_id}:{desc|rev|ing}:{序号}`,如 `P1001:rev:3`
   ⚠️ **格式定了不能改** —— Anna 的奖励函数靠它判断引用真假

**输出**
```
data/processed/chunks.parquet
evidence_id(str,全局唯一) | product_id(须存在于products) | source | text
```
**验收**:`python scripts/validate_data.py chunks`

#### 交付物 4:成分规则表(**纯查资料,但对 RL 效果影响最大**)

扩充 `data/knowledge/ingredient_rules.json`:
1. `concern_to_ingredients`:每个关注点补到 **8–12 个**成分(现在只有 4–6)
2. `pregnancy_unsafe`:视黄醇类、高浓度水杨酸、氢醌等
3. `common_irritants` / `comedogenic`
4. `skin_type_to_avoid`

**每条要有出处**,另存 `data/knowledge/sources.md`(报告要引用)。
**Anna 的 `ingredient_match_reward` 直接读这个文件,你每加一条,RL 训练信号就更准。**
**验收**:`python scripts/validate_data.py rules`

### 阶段二(中期):安全模块

完善 `src/skincare/safety/guard.py` —— 它读的就是你自己建的规则表:
孕期/过敏/致痘成分过滤、免责声明、越界医疗问题拒答。
补单元测试(参考 `tests/test_rewards.py`)。

### 阶段三(后期):报告统稿

**你不是一个人写报告**,你是把大家的章节缝成一份:搭骨架 → 催各人交 →
统一术语和图表编号 → 补写数据方法章节和伦理章节(L1:人脸隐私、医疗免责边界、
**推荐偏见分析**、数据集许可)。

---

## 3. 组员 B — CNN / RAG 训练与优化

**你拥有:** `configs/vision_*.yaml`、`rag/index.py` 与 `retrieve.py` 的调优、`models/` 下自己的产物
**你不碰:** `src/skincare/vision/**`、`src/skincare/llm/**`、`app/schemas.py`、`config.py`

> 超参只能通过 `configs/*.yaml` 改。发现代码 bug → 报给 Anna,不要自己改。
> 详见 `docs/HANDOFF.md`(交接协议、checkpoint 契约、为什么改代码会加载不了)。

### B1:CNN 训练与调优
```bash
python -m skincare.vision.train --config configs/vision_transfer.yaml
python -m skincare.vision.train --config configs/vision_simple.yaml   # 对照基线,别省
```
- 扫描:backbone ∈ {resnet18, resnet50} × lr ∈ {1e-4, 3e-4, 1e-3},每次换 `run_name`
- **报告要的对比**:自建 CNN vs 迁移学习(说明微调预训练模型的收益)
- **和 C 配合**:C 交来 `vision_train_aug.csv` 后,用**同样超参**重训一遍,
  两组结果的差值就是消融实验结论

**交付**:`{run_name}.pt` + `{run_name}_report.json` + 混淆矩阵图
**验收**:`python scripts/verify_handoff.py vision models/vision/<run_name>.pt`

### B2:RAG 索引与检索优化
```bash
python -m skincare.rag.index
```
- 对比 2 个 embedding 模型(`all-MiniLM-L6-v2` vs `all-mpnet-base-v2`)
- 调 chunk 粒度、top-k、是否加 rerank
- ⚠️ 可以改索引内容和 embedding 模型,**但不要改 `evidence_id` 格式**

**交付**:`data/processed/index/` + 检索质量对比表
**验收**:`python scripts/verify_handoff.py rag`
**报告章节**:视觉模型对比、检索优化。

---

## 4. 组员 C — 生成式图像增强 + 评估体系

**你拥有:** `src/skincare/augment/**`、`src/skincare/eval/**`、`fixtures/eval_samples.jsonl`、`reports/**`
**你不碰:** `vision/model.py`、`llm/**`(除了**读** `rewards.py`)、`app/schemas.py`、`config.py`

> 这个角色补的是**课程最大缺口**:M4–M7(VAE/GAN/扩散)+ A3–A4。
> 而且它**自带回退** —— 即使扩散部分失败,评估体系是项目必需品,你的工作不会白做。

---

### 🔑 与 Anna 并行的保证机制(先读这段)

你的评估工作天然要用 Anna 的模型输出。如果不处理,你会在最后一天卡住等她 —— 那就不叫并行了。
解法是把**评估工具**和**被评估的模型**彻底解耦:

**1. 唯一的交接接口是一个 JSON 文件**

```
models/llm/manifest.json
{"base": "Qwen/Qwen2.5-1.5B-Instruct", "sft": null, "grpo": null, "gpt": "gpt-4o-mini"}
```
Anna 每训完一档就往里填一个路径。**你的脚本读它,缺的 key 自动跳过。**
所以 Anna 一档没训完,你照样能跑完整流程 —— 只是表里少两列。

**2. 你第一天就能验证评估器正确(不需要模型 / GPU / API key)**

```bash
python -m skincare.eval.run_eval --self-test
pytest tests/test_eval_harness.py -v
```
`fixtures/eval_samples.jsonl` 里有 **10 个已知答案的样本**:完美答案、伪造引用、
成分不匹配、孕期不安全、格式错乱、部分引用……每个都标了期望分数区间。
**自检通过 = 你的评估器是对的**,之后接任何模型都可信。已实测 10/10 通过。

**3. Anna 交付时你只多跑一条命令**

```bash
python -m skincare.eval.run_eval --split data/processed/rl_test.jsonl --variants base sft grpo
```
从"等 Anna"变成"多跑十分钟",这就是并行的实质。

**4. 万一 Anna 延期,你的交付依然完整**

不依赖 Anna 任何产出的部分:扩散消融实验、RAG Precision@3、
视觉混淆矩阵、偏见分析、base/GPT 两档的评估基线。**这些已经够写两个报告章节。**

---

### C1:扩散图像增强 + 消融实验(**主攻,课程加分核心**)

**目标不是"生成好看的皮肤图",而是用消融实验证明生成式增强能提升分类性能。**
报告里最有说服力的一句是:*"加入 N 张扩散合成样本后,少数类 macro-F1 从 0.61 提升到 0.68"*

```bash
python -m skincare.augment.diffusion_aug --concern acne --n 200
```

**步骤**
1. 看 A 的 `class_distribution.md`,挑**样本最少的 2–3 个关注点**
   (等不及就先自己扫一眼原始数据集的类别分布,不必等 A 交付)
2. 用 `diffusers` 的 `StableDiffusionImg2ImgPipeline` 做 **img2img 重绘** ——
   以真实少数类图片为起点,保留皮肤纹理语义。**比从零训 GAN 稳得多,一天能出结果**
3. prompt 用**临床描述性措辞**(如 "clinical photo of facial skin with mild acne"),
   不要艺术化措辞;`strength` 在 0.5–0.7 之间试
4. **人工抽查剔除明显失真的样本**,记录剔除比例(报告要写)
5. 生成 `data/processed/vision_train_aug.csv` 交给 B 重训
   ⚠️ **只增强 train,绝不动 val/test** —— 否则指标失去意义,这是评分会被追问的点

**回退阶梯**:img2img 效果差 → 退到"经典增强 vs 扩散增强"对比;
再不行 → 只交类别不均衡分析 + 经典增强消融。**任一档都覆盖 A3/A4。**

**伦理红线**:合成人脸**仅用于训练**,不得出现在 demo 里冒充真实用户;
报告须说明合成数据占比。

### C2:评估体系(和 C1 并行,项目必需)

| 评什么 | 怎么评 | 依赖 Anna? |
|---|---|---|
| 规则化质量 | `run_eval.py` 五个奖励分量 | ❌ base/gpt 档随时可跑 |
| 主观质量 | `judge.py` LLM-as-judge(helpfulness/clarity/specificity/faithfulness) | ❌ 同上 |
| 检索质量 | 人工标 ~40 条 (query, product) → Precision@3 | ❌ 完全独立 |
| 视觉 | 两个模型的混淆矩阵、每类 F1 | ❌ 只依赖 B |
| 偏见分析 | 推荐结果在品牌/价格区间的分布 | ❌ 完全独立 |
| **三段式对比** | base vs SFT vs GRPO | ✅ **只在最后多跑一次** |

`judge.py` 里的 `judge_one()` 是留给你实现的,三个要点已写在 docstring:
temperature=0、打分顺序随机化避免位置偏见、解析失败返回 None 而不是 0 分。

### C3:报告图表

统一风格产出到 `reports/`:奖励曲线、三段式对比表、**消融柱状图**、混淆矩阵、benchmark 表。
**报告章节**:生成式增强与消融、评估方法与结果。

---

### C 的七天节奏(标注了每天是否需要等人)

| Day | 做什么 | 等人吗 |
|---|---|---|
| 1 | 跑 `--self-test` 吃透评估器;下载原始图像数据集 | **不等** |
| 2 | img2img 管线跑通,出第一批合成样本 | **不等** |
| 3 | 生成增强训练集交给 B;跑 base/gpt 档评估基线 | 不等(B 是下游) |
| 4 | 拿 B 的消融结果;人工标注检索相关性 | 等 B(可先做标注) |
| 5 | Anna 填 manifest → 重跑得到三段式表;偏见分析 | **仅此一处等 Anna** |
| 6 | 出全部图表,写两个报告章节 | 不等 |
| 7 | demo 支持、答辩问题准备 | — |

---

## 5. 硬依赖与冻结项

| 谁 → 谁 | 交付物 | 何时 | 卡住的绕行 |
|---|---|---|---|
| A → B、C | `vision_*.csv` + `class_distribution.md` | Day 2–3 | B/C 先用小样本子集跑通流程 |
| A → Anna、B | `products/chunks.parquet` | MVP 之前 | Anna 用 `fixtures/` 继续做 SFT 格式 |
| C → B | `vision_train_aug.csv` | 消融实验前 | B 先交基线结果,增强版后补 |
| Anna → C | 往 `models/llm/manifest.json` 填 adapter 路径 | Day 5 | **C 全程用 fixtures + base 档推进,不阻塞** |

**四样冻结的东西(冻接口,不冻实现):**

| 冻结项 | 文件 | 动了会怎样 |
|---|---|---|
| 数据契约 | `app/schemas.py` | 全线返工 |
| 标签空间 | `config.py::CONCERNS`/`SKIN_TYPES` | checkpoint + 奖励函数 + 训练数据一起崩 |
| Prompt 模板 | `llm/prompts.py` | 训练/服务不一致,**静默退化** |
| evidence_id 格式 | `rag/ingest.py` 命名约定 | grounding 奖励失真 |

架构、backbone、超参、索引内容、产品数量 —— **随便改,不会碰到别人。**

---

## 6. 报告章节归属(硬性交付)

| 章节 | 作者 |
|---|---|
| 引言、问题定义、系统架构 | Anna |
| 数据来源与处理方法 | A |
| 视觉模型与检索优化 | B |
| 生成式增强与消融实验 | C |
| LLM 后训练:LoRA SFT + GRPO 与奖励设计 | Anna |
| 评估方法与结果 | C |
| 部署与 benchmark(L4) | Anna |
| 伦理与局限(L1) | A |
| 统稿、术语与图表编号统一 | A |

---

## 7. 协作纪律

1. **每天 15 分钟同步**:昨天做完什么、今天做什么、被谁卡住。
2. **动上面四个冻结项之前,先在群里说。**
3. **合并前跑 `make test`**,挂了说明破坏了别人的假设。
4. **数据文件不进 git**,用共享盘;**代码进 git**。
