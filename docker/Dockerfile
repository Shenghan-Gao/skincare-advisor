FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl && rm -rf /var/lib/apt/lists/*

# uv -- the package manager the course requires (Assignment 1)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
RUN uv pip install --system -e ".[ui]"

COPY app ./app
COPY src ./src
COPY fixtures ./fixtures
COPY data/knowledge ./data/knowledge
RUN uv pip install --system -e .

ENV USE_MOCKS=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
