.PHONY: setup precompute build test dev start

# One-time setup: Python env, JS deps, star map, production UI build.
setup:
	cd backend && uv sync
	cd frontend && pnpm install
	$(MAKE) precompute
	$(MAKE) build

# Star positions (UMAP of real neuron weights) and brightness scales. ~2 minutes.
precompute:
	cd backend && uv run python -m galaxy.precompute

build:
	cd frontend && pnpm build

test:
	cd backend && uv run pytest -q
	cd frontend && pnpm typecheck

# Run the app: http://127.0.0.1:8000
start:
	cd backend && uv run python -m galaxy.server

# Frontend hot reload on http://localhost:5173 (needs `make start` in another terminal).
dev:
	cd frontend && pnpm dev
