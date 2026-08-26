# VisioVox — entry points. See docs/23-runbook.md.
COMPOSE := docker compose -f infra/docker/compose.yaml
UV      := uv
PNPM    := pnpm

.DEFAULT_GOAL := help
.PHONY: help dev dev-down dev-logs lint fmt typecheck test test-gpu \
        eval-quick smoke check install clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install all Python and Node dependencies
	$(UV) sync --all-packages --extra ml
	$(PNPM) install
	$(UV) run pre-commit install --install-hooks

dev:  ## Bring up the local stack (Postgres, Redis, MinIO, mail)
	$(COMPOSE) up -d
	@$(COMPOSE) ps

dev-down:  ## Stop the local stack
	$(COMPOSE) down

dev-logs:  ## Tail stack logs
	$(COMPOSE) logs -f

lint:  ## Lint Python and TypeScript
	$(UV) run ruff check .
	$(PNPM) exec eslint .

fmt:  ## Format Python and TypeScript
	$(UV) run ruff format .
	$(PNPM) exec prettier --write .

typecheck:  ## mypy --strict and tsc
	$(UV) run mypy .
	$(PNPM) exec tsc --build

test:  ## Run the test suite (skips GPU-marked tests)
	$(UV) run pytest -m "not gpu"

test-gpu:  ## Run GPU-marked tests (workstation only)
	$(UV) run pytest -m gpu

smoke:  ## Phase 0 pretrained-model smoke test (requires CUDA)
	@set -a; [ -f .env.local ] && . ./.env.local; set +a; $(UV) run python scripts/smoke_pretrained.py

contracts:  ## Regenerate the OpenAPI spec and the TS client
	$(UV) run python scripts/export_openapi.py
	$(PNPM) exec openapi-typescript packages/contracts/openapi.json -o packages/ts-client/src/generated/api.ts
	$(PNPM) exec prettier --write packages/ts-client/src/generated/api.ts packages/contracts/openapi.json

contracts-check:  ## Fail if the committed contract is stale
	@$(MAKE) --no-print-directory contracts
	@git diff --exit-code -- packages/contracts/openapi.json packages/ts-client/src/generated/api.ts 	  || (echo "contract drift: run 'make contracts' and commit the result"; exit 1)

eval-quick:  ## 30-item ML eval; gates on regression
	$(UV) run python -m eval.quick

check: lint typecheck test contracts-check  ## Everything CI runs

clean:  ## Remove build and cache artefacts
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
