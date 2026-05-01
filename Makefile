.PHONY: setup backend frontend dev test refresh clean docker-up docker-down docker-build docker-logs docker-pipeline

# Setup everything
setup: setup-backend setup-frontend download-centroids

setup-backend:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

setup-frontend:
	cd frontend && pnpm install

download-centroids:
	bash backend/scripts/download_centroids.sh

# Development servers
backend:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && pnpm dev

# Run both (requires two terminals, or use this with &)
dev:
	@echo "Start in two terminals:"
	@echo "  make backend"
	@echo "  make frontend"

# Test backend health
test:
	@echo "Testing backend..."
	@curl -sf http://localhost:8000/api/v1/health | python3 -m json.tool
	@echo ""
	@echo "Testing area summary (Austin, TX)..."
	@curl -sf "http://localhost:8000/api/v1/area/summary?west=-97.85&south=30.15&east=-97.60&north=30.45" | python3 -m json.tool

# Data refresh
refresh:
	bash backend/scripts/refresh_data.sh

# Build frontend for production
build-frontend:
	cd frontend && pnpm build

# Docker commands
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-pipeline:
	docker compose -f docker-compose.pipeline.yml up

# Clean generated files
clean:
	rm -rf backend/.venv backend/app/__pycache__ backend/app/routers/__pycache__ backend/app/services/__pycache__
	rm -rf frontend/node_modules frontend/dist
