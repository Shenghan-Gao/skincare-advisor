# 团队协作与交接规范

> 每个人开工前**先读完这一页**(约 5 分钟)。
> 详细任务在 `docs/TEAM_TASKS.md`,技术细节在 `docs/TECH_DESIGN.md`。

---

## 1. 五分钟上手

```bash
git clone https://github.com/Shenghan-Gao/skincare-advisor.git
cd skincare-advisor

uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ui]"
cp .env.example .env

make test        # 应该看到 27 passed
make api         # http://localhost:8000/docs
make ui          # http://localhost:8501
```

`make test` 不是 27 passed 就先别往下走,在群里说一声。

**为什么没有模型也能跑起来**:`.env` 里 `USE_MOCKS=1`,API 会从 `fixtures/` 返回真实形状的假数据。
这就是四个人第一天能同时开工的原因 —— 你不用等任何人。

---

## 2. 你负责什么

| 角色 | 谁 | 独占的文件 |
|---|---|---|
| 主线:LLM 后训练 + 前端 + 部署 | Anna | `src/skincare/llm/**`、`app/**`、`ui/**`、`docker/**` |
| 数据 + 安全 + 报告统稿 | A | `src/skincare/rag/ingest.py`、`data/knowledge/**`、`src/skincare/safety/**` |
| CNN / RAG 训练调优 | B | `configs/vision_*.yaml`、`src/skincare/rag/index.py`、`retrieve.py` |
| 扩散增强 + 评估 | C | `src/skincare/augment/**`、`src/skincare/eval/**`、`reports/**` |

**只改自己那一栏的文件。** 需要改别人的 → 在群里说,让他改。

完整任务清单(细到命令级)在 `docs/TEAM_TASKS.md`,先看你自己那一节。

---

## 3. ⚠️ 四条冻结契约 —— 动之前必须先在群里说

这四样东西一改,别人的东西会**连锁崩掉**,而且有些是**静默失效**(不报错,但结果变差),
比崩溃更难发现:

| 冻结项 | 文件 | 动了会怎样 |
|---|---|---|
| **数据契约** | `app/schemas.py` | 全线返工 |
| **标签空间** | `src/skincare/config.py` 的 `CONCERNS` / `SKIN_TYPES` | 模型 checkpoint、奖励函数、已生成的训练数据一起失效 |
| **Prompt 模板** | `src/skincare/llm/prompts.py` | 训练与服务输入不一致 → 模型**静默退化** |
| **evidence_id 格式** | `src/skincare/rag/ingest.py` 的命名约定(`P1001:rev:3`) | grounding 奖励失真,RL 学歪 |

**特别提醒组员 A**:整理标签时**不要擅自增加第 7 个关注点**。
现在固定是这 6 个:`acne, dark_spots, redness, large_pores, wrinkles, dryness`。
真要加,先说,大家一起改。

**冻的是接口,不是实现。** 模型架构、backbone、超参、索引内容、产品数量 —— 随便改,不会碰到别人。

---

## 4. 交接协议:**验收命令跑不过,不算交付**

每次交东西之前,自己先跑对应命令。跑不过就别交,省得来回折腾。

| 谁 | 交什么 | 验收命令 |
|---|---|---|
| A | 视觉标签表 | `python scripts/validate_data.py vision` |
| A | 产品表 | `python scripts/validate_data.py products` |
| A | 检索片段表 | `python scripts/validate_data.py chunks` |
| A | 成分规则表 | `python scripts/validate_data.py rules` |
| B | CNN checkpoint | `python scripts/verify_handoff.py vision models/vision/<run>.pt` |
| B | 向量索引 | `python scripts/verify_handoff.py rag` |
| C | 评估器 | `python -m skincare.eval.run_eval --self-test` |
| Anna | 训练好的模型 | 填 `models/llm/manifest.json` 交给 C |
| 全员 | 任何代码改动 | `make test` |

一次跑全部数据检查:`python scripts/validate_data.py all`

**这些脚本查的是"契约"**(列名、类型、唯一性、引用完整性),不是"数据好不好"。
契约过了只说明能对接上,质量另说。

### 给 B 的额外提醒

模型 checkpoint **不是独立文件,它是 (代码, 权重) 的配对**。
你改了 `model.py` 再训出来的权重,别人用旧代码加载会直接报错。
所以交回权重时**必须说清基于哪个 commit**。详见 `docs/HANDOFF.md`。

超参只通过 `configs/*.yaml` 改,**不要改 `src/skincare/vision/*.py`**。
发现代码有 bug → 报给 Anna,别自己动手。

---

## 5. Git 纪律

**核心只有一条:`main` 分支必须永远是能跑的。**

不是洁癖 —— Colab 从 main 拉代码,教授也是 clone main 来跑。main 坏了,四个人一起停摆。

```bash
git checkout -b feat/<你的名字>-<模块>     # 例:feat/b-vision
# ... 改代码 ...
make test                                  # 必须全绿
git add -A && git commit -m "feat: 说清楚改了什么"
git push -u origin feat/b-vision
```

然后在 GitHub 发 PR,或者直接合:

```bash
git checkout main && git pull && git merge feat/b-vision && git push
```

**判断要不要开分支**:这次改动有没有可能让 `make test` 挂?
会 → 开分支。只改自己独占的文件、测试还全绿 → 直接推 main 也行。

**不要提交的东西**(`.gitignore` 已覆盖,但心里要有数):

- `.env`(含密钥)
- `data/raw/`、`data/processed/` 下的数据文件 —— **用共享盘传**
- `models/` 下的权重文件 —— **用 Google Drive 传**
- 任何 `.zip`、`__pycache__/`、`.venv/`

---

## 6. 🔐 密钥纪律

**仓库是公开的。** 密钥一旦提交,几分钟内就会被爬虫扫走盗刷。

**谁需要 OpenAI key:**

| 谁 | 用途 | 需要吗 |
|---|---|---|
| Anna | 教师蒸馏 | ✅ |
| C | LLM-as-judge、评估表的 gpt 对照列 | ⚠️ 可选,建议自己充 $5 |
| A | — | ❌ 不需要 |
| B | — | ❌ 完全不需要,CNN 与嵌入模型都是本地跑的 |

**C 注意**:报告的头号结果(base vs SFT vs GRPO 三段式对比)**不需要 key**,
那三档都是本地模型。key 只用于可选的 judge 打分和 GPT 对照列。
所以可以**先把规则化评估跑通,确认真需要再充值**。

**三条铁律:**

1. key 只存在 `.env`(已 gitignore)或 Colab Secrets 里,**绝不写进代码或 notebook**
2. 要分享只走私聊,**不要发群、不要进 git**
3. 怀疑泄露 → **立刻去 platform.openai.com 删掉那把 key 再建新的**。这是唯一有效的补救

---

## 7. 遇到问题

**先自查这三样**,能解决大部分情况:

```bash
make test                                    # 代码是不是坏了
git pull                                     # 是不是拿的旧代码
python scripts/validate_data.py all          # 数据契约对不对
```

还不行就在群里说,**带上完整报错信息**,不要只说"跑不起来"。

**每天 15 分钟同步**:昨天做完什么、今天做什么、被谁卡住。
被卡住当天就说,别憋到第二天 —— 一周的项目经不起等。
