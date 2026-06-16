#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-warming cache with GPT-2 + example prompts ==="

MODEL="gpt2"
PROMPTS=(
  "The meaning of life is"
  "In the beginning, there was"
  "The key to understanding neural networks is"
  "Once upon a time in a land far away"
  "The most important discovery in science was"
)

for prompt in "${PROMPTS[@]}"; do
  echo ""
  echo "Submitting: \"$prompt\""
  RESPONSE=$(curl -s -X POST http://localhost:8000/api/analyze \
    -H "Content-Type: application/json" \
    -d "{\"model_name\": \"$MODEL\", \"prompt\": \"$prompt\"}")
  JOB_ID=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "failed")
  echo "  Job ID: $JOB_ID"

  if [ "$JOB_ID" != "failed" ]; then
    sleep 5
    STATUS=$(curl -s "http://localhost:8000/api/jobs/$JOB_ID/status" | \
      python -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
    echo "  Status: $STATUS"
  fi
done

echo ""
echo "=== Cache seeding complete ==="
echo "Model '$MODEL' is now cached."
echo "The 5 example prompts have been analysed and cached."
