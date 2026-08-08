.PHONY: install api ui test lint docker-build docker-up

install:            ## base env (API in mock mode)
	uv venv && uv pip install -e ".[dev,ui]"

install-full:       ## everything incl. torch / llm
	uv pip install -e ".[vision,rag,llm,ui,dev]"

api:                ## run FastAPI at :8000
	uv run uvicorn app.main:app --reload --port 8000

ui:                 ## run Streamlit at :8501
	uv run streamlit run ui/streamlit_app.py

test:
	uv run pytest -q

lint:
	uv run ruff check .

docker-build:
	docker build -f docker/Dockerfile -t skincare-advisor .

docker-up:
	docker compose --project-directory . -f docker/docker-compose.yml up --build

bench:              ## 本地基准测试(需先 make api)
	uv run python scripts/benchmark.py --url http://localhost:8000 --label local
