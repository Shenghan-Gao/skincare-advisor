# 部署与基准测试(L4)

> 学习目标 L4:**在 AWS/Azure/GCP 上部署训练好的模型,并与本地做基准对比。**
> 这是一个独立得分点,而且**不依赖任何人**—— mock 模式下现在就能全流程跑通。
>
> 负责人:Anna

---

## 0. 先跑通本地(10 分钟)

```bash
make docker-build
make docker-up
```

打开 http://localhost:8000/docs 和 http://localhost:8501。

**这一步是评分硬指标** —— 教授会 clone 仓库在他自己机器上跑。
所以务必在**干净环境**验证(最好用另一台机器,或 `docker system prune -af` 后重来)。

> 已知坑(已修复,记录备查):
> - `pip install -e .` 必须在 `src/` 存在之后执行,否则报
>   `Getting requirements to build editable did not run successfully`
> - `docker-compose.yml` 不能硬依赖 `.env`(它被 gitignore,全新 clone 没有),
>   现在所有变量都带默认值

---

## 1. 本地基准

```bash
make api
make bench
```

或手动指定参数:

```bash
python scripts/benchmark.py --url http://localhost:8000 --label local \
       --n 50 --concurrency 4
```

产出 `reports/bench_local.json`,包含:冷启动、延迟 p50/p95/p99、吞吐、失败数。

---

## 2. 云端部署

三家云都行,下面以 **AWS EC2** 为例(GCP Compute Engine 步骤等价)。

### 2.1 起实例

- **只跑 API/检索**(轻量):`t3.medium` 之类的 CPU 实例就够
- **要跑本地 LLM 推理**:需要 GPU 实例(如 `g4dn.xlarge`,T4 卡)
- 安全组放行 **8000**(API)和 **8501**(前端)端口
- 系统选 Ubuntu 22.04+

### 2.2 装 Docker 并起服务

```bash
ssh -i <你的key>.pem ubuntu@<公网IP>

sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/Shenghan-Gao/skincare-advisor.git
cd skincare-advisor

docker compose --project-directory . -f docker/docker-compose.yml up -d --build
curl localhost:8000/health
```

### 2.3 把训练好的模型带上去

模型权重不在 git 里(体积大),从 Google Drive 下载后放进 `models/`:

```
models/
  llm/sft-lora/      # Anna 训练产出
  llm/grpo/
  vision/best.pt     # 组员 B 训练产出
```

然后把 `.env` 里的 `USE_MOCKS` 设为 `0` 并重启容器。

---

## 3. 云端基准 + 对比

在**你本机**上打云端的地址(这样测到的是真实用户视角的端到端延迟):

```bash
python scripts/benchmark.py --url http://<公网IP>:8000 --label cloud \
       --n 50 --concurrency 4 --hourly-cost 0.526
```

`--hourly-cost` 填该实例的每小时价格,脚本会折算出**每千次请求成本** ——
报告里要的是这个数,不是原始时长。

生成对比表:

```bash
python scripts/benchmark.py --compare reports/bench_local.json reports/bench_cloud.json
```

产出 `reports/bench_comparison.md`,可以直接贴进报告。

---

## 4. 报告里要放的两张表

### 4.1 推理服务基准(脚本自动生成)

| 指标 | local | cloud |
|---|---|---|
| 冷启动 (ms) | | |
| 延迟 p50 / p95 / p99 (ms) | | |
| 吞吐 (req/s) | | |
| 每千次成本 (USD) | | |

### 4.2 训练耗时对比(手工记录)

L4 原文提到"与本地训练做基准对比",所以除了推理还要记训练:

| 阶段 | 本地(Mac CPU/MPS) | Colab T4 | 加速比 |
|---|---|---|---|
| LoRA SFT(2 epoch) | | | |
| GRPO(300 步) | | | |

**数据从哪来**:训练脚本会打印 `train_runtime`,直接抄。
本地那一列不必真跑完 —— 跑 10 步测出单步耗时再线性外推即可,
在报告里注明是外推值就行(实测几小时不值当)。

---

## 5. 讨论要点(报告加分项)

光贴数字不够,rubric 看的是**权衡分析**。建议讨论:

- **冷启动 vs 稳态延迟**:第一个请求要加载检索索引/模型,明显更慢。
  生产上怎么办(预热请求、常驻进程)?
- **CPU vs GPU 的性价比**:检索和 API 在 CPU 上就够快,只有 LLM 推理需要 GPU。
  是否值得为此付 GPU 实例的钱?能否拆成两个服务?
- **本地训练不可行的原因**:显存、时间、精度支持(Mac 的 MPS 不支持 bf16)。
- **成本估算**:按每千次请求成本,推算服务 1000 个用户一个月要多少钱。
