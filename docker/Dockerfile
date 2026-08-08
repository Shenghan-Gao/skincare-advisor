FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl && rm -rf /var/lib/apt/lists/*

# uv —— 课程要求的包管理器(Assignment 1)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 先只装依赖,利用 Docker 层缓存。
# 注意:pyproject 用 setuptools 的 src 布局,做 editable 安装时 src/ 必须存在,
# 否则会报 "Getting requirements to build editable did not run successfully"。
# 这里先放一个占位包骗过构建,装完再删,真正的源码在下面拷进来。
COPY pyproject.toml README.md ./
RUN mkdir -p src/skincare && touch src/skincare/__init__.py \
    && uv pip install --system -e ".[ui]" \
    && rm -rf src

COPY app ./app
COPY src ./src
COPY ui ./ui
COPY fixtures ./fixtures
COPY data/knowledge ./data/knowledge
RUN uv pip install --system -e .

ENV USE_MOCKS=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
