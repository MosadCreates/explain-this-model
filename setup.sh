#!/usr/bin/env bash
set -euo pipefail

echo "=== Explain This Model — Setup ==="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] Creating Python virtual environment..."
python3 -m venv venv 2>/dev/null || python -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/6] Installing Node.js dependencies..."
cd dashboard
npm install
cd ..

echo "[4/6] Starting Redis via Docker..."
docker rm -f explain-redis 2>/dev/null || true
docker run -d --name explain-redis -p 6379:6379 redis:7-alpine 2>/dev/null || \
  echo "Redis container already running or Docker unavailable"

echo "[5/6] Creating data directories..."
mkdir -p data

echo "[6/6] Checking API keys..."
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "${GOOGLE_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "⚠  WARNING: No API keys set."
  echo "   Set GOOGLE_API_KEY in .env for Gemini-powered explanations."
  echo "   Set ANTHROPIC_API_KEY in .env for Claude-powered explanations."
  echo "   Without an API key, explanations will show as 'unavailable'."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "  make backend   — Start FastAPI server (port 8000)"
echo "  make frontend  — Start Next.js dev server (port 3000)"
echo "  make worker    — Start Celery worker"
echo "  make docker    — Start everything with docker-compose"
echo "  make test      — Run tests"
echo ""
