FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl && rm -rf /var/lib/apt/lists/*

# uv -- the package manager required by the course (Assignment 1)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first so this layer stays cached across source changes.
# Note: the project uses a setuptools src layout, so an editable install fails with
# "Getting requirements to build editable did not run successfully" unless src/
# already exists. A placeholder package satisfies the build and is removed right
# after; the real sources are copied in below.
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
