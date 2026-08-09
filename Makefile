.PHONY: install install-full api ui test lint validate bench check-secrets hooks set-key docker-build docker-up verify

install:            ## base env + dev/ui extras (enough for tests and the UI)
	uv venv && uv pip install -e ".[dev,ui]"

install-full:       ## everything incl. torch / rag / llm (needed for training)
	uv pip install -e ".[vision,rag,llm,ui,dev]"

api:                ## run FastAPI at :8000
	uv run uvicorn app.main:app --reload --port 8000

ui:                 ## run Streamlit at :8501
	uv run --extra ui streamlit run ui/streamlit_app.py

set-key:            ## store the OpenAI key outside this repo (Keychain, or ~/.config)
	@printf 'OpenAI API key (input hidden): ' && read -rs K && echo && \
	  if command -v security >/dev/null 2>&1; then \
	    security add-generic-password -U -a "$$USER" -s skincare-openai -w "$$K" && \
	    echo "stored in the macOS Keychain (not in this repository)"; \
	  else \
	    mkdir -p $$HOME/.config/skincare-advisor && \
	    printf '%s' "$$K" > $$HOME/.config/skincare-advisor/openai_key && \
	    chmod 600 $$HOME/.config/skincare-advisor/openai_key && \
	    echo "stored in ~/.config/skincare-advisor/ (not in this repository)"; \
	  fi

check-secrets:      ## scan staged changes for API keys and tokens
	./scripts/check_secrets.sh --staged

hooks:              ## install the pre-commit secret scan (run once per clone)
	git config core.hooksPath .githooks && echo "pre-commit hook installed"

test:               ## run the test suite (secret scan first)
	./scripts/check_secrets.sh --worktree
	uv run --extra dev pytest -q

lint:
	uv run --extra dev ruff check .

validate:           ## check Member A data deliverables against the contract
	uv run python scripts/validate_data.py all

bench:              ## benchmark a running API (start `make api` first)
	uv run --extra ui python scripts/benchmark.py --url http://localhost:8000 --label local

verify:             ## what the grader does: clean install from pyproject, then test
	rm -rf .venv && uv venv && uv pip install -e ".[dev,ui]" && uv run --extra dev pytest -q

docker-build:
	docker build -f docker/Dockerfile -t skincare-advisor .

docker-up:
	docker compose --project-directory . -f docker/docker-compose.yml up --build
