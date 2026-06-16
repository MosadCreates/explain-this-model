.PHONY: backend frontend worker test lint docker docker-build docker-down clean setup install-dev

BACKEND_DIR = .
DASHBOARD_DIR = dashboard

install-dev:
	pip install -r requirements.txt
	cd $(DASHBOARD_DIR) && npm install

backend:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd $(DASHBOARD_DIR) && npm run dev

worker:
	celery -A src.tasks.celery_app worker --loglevel=info --concurrency=2

test:
	pytest tests/ -v --run-slow

test-fast:
	pytest tests/ -v --no-header -x -q

lint:
	cd $(DASHBOARD_DIR) && npm run lint
	python -m flake8 src/ tests/ --max-line-length=100 2>/dev/null || echo "flake8 not installed, skipping"

typecheck:
	cd $(DASHBOARD_DIR) && npm run typecheck

docker:
	docker compose up --build

docker-build:
	docker compose build

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f

setup:
	./setup.sh

clean:
	rm -rf __pycache__ .pytest_cache *.db
	rm -rf $(DASHBOARD_DIR)/.next $(DASHBOARD_DIR)/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
