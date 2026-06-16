#!/usr/bin/env bash
set -euo pipefail

echo "=== Running demo: GPT-2 + sample prompt ==="

MODEL="gpt2"
PROMPT="The future of AI interpretability depends on"

echo "Model: $MODEL"
echo "Prompt: \"$PROMPT\""
echo ""

echo "[1/3] Validating model..."
curl -s "http://localhost:8000/api/models/validate?model_name=$MODEL" | python -m json.tool || echo "  (server might not be running)"
echo ""

echo "[2/3] Submitting analysis job..."
JOB_RESPONSE=$(curl -s -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"model_name\": \"$MODEL\", \"prompt\": \"$PROMPT\"}")
echo "$JOB_RESPONSE" | python -m json.tool || echo "  (failed to submit)"
JOB_ID=$(echo "$JOB_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")
echo "Job ID: $JOB_ID"
echo ""

if [ -z "$JOB_ID" ]; then
  echo "No job ID received. Is the server running?"
  exit 1
fi

echo "[3/3] Polling for results..."
for i in $(seq 1 60); do
  STATUS_RESPONSE=$(curl -s "http://localhost:8000/api/jobs/$JOB_ID/status")
  STATUS=$(echo "$STATUS_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
  PROGRESS=$(echo "$STATUS_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('progress', 0))" 2>/dev/null || echo "0")
  STAGE=$(echo "$STATUS_RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin).get('stage', ''))" 2>/dev/null || echo "")
  echo "  Status: $STATUS | Progress: $PROGRESS | Stage: $STAGE"

  if [ "$STATUS" = "complete" ]; then
    echo ""
    echo "=== Analysis complete! ==="
    curl -s "http://localhost:8000/api/jobs/$JOB_ID/results" | python -c "
import sys, json
result = json.load(sys.stdin)
print(f'Model: {result[\"model_name\"]}')
print(f'Prompt: {result[\"prompt\"]}')
print(f'Top neurons: {result.get(\"neuron_count\", 0)}')
print(f'Top heads: {result.get(\"head_count\", 0)}')
if result.get('top_neuron_explanation'):
    print(f'Top neuron explanation: {result[\"top_neuron_explanation\"][:200]}...')
"
    exit 0
  fi

  if [ "$STATUS" = "failed" ]; then
    echo "ERROR: Job failed"
    curl -s "http://localhost:8000/api/jobs/$JOB_ID/status"
    exit 1
  fi

  sleep 2
done

echo "TIMEOUT: Job did not complete within 120 seconds"
exit 1
