.PHONY: install install-full api ui test lint bench docker-build docker-up verify

install:            ## base env + dev/ui extras (enough for tests and the UI)
	uv venv && uv pip install -e ".[dev,ui]"

install-full:       ## everything incl. torch / rag / llm (needed for training)
	uv pip install -e ".[vision,rag,llm,ui,dev]"

api:                ## run FastAPI at :8000
	uv run uvicorn app.main:app --reload --port 8000

ui:                 ## run Streamlit at :8501
	uv run --extra ui streamlit run ui/streamlit_app.py

test:               ## run the test suite
	uv run --extra dev pytest -q

lint:
	uv run --extra dev ruff check .

bench:              ## benchmark a running API (start `make api` first)
	uv run --extra ui python scripts/benchmark.py --url http://localhost:8000 --label local

verify:             ## what the grader does: clean install from pyproject, then test
	rm -rf .venv && uv venv && uv pip install -e ".[dev,ui]" && uv run --extra dev pytest -q

docker-build:
	docker build -f docker/Dockerfile -t skincare-advisor .

docker-up:
	docker compose --project-directory . -f docker/docker-compose.yml up --build
