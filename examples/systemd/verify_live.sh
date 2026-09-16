#!/usr/bin/env bash
# Example live-verification gate (ExecStartPost). Fails the unit unless the
# server answers /health, serves the EXPECTED model id at the EXPECTED
# max_model_len, and completes one generation. No secrets here: read the
# API key from wherever your deployment keeps it.
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8090}"
KEY="${VLLM_API_KEY:-$(cat /run/secrets/llm-api-key 2>/dev/null || true)}"
EXPECT_MODEL="${EXPECT_MODEL:-mymodel}"
EXPECT_MAXLEN="${EXPECT_MAXLEN:-114688}"
AUTH=(-H "Authorization: Bearer ${KEY}" -H "Content-Type: application/json")

for i in $(seq 1 120); do curl -sf "$BASE/health" >/dev/null && break; sleep 2; done
curl -sf "$BASE/health" >/dev/null || { echo "verify_live: /health never answered"; exit 1; }

models=$(curl -sf "${AUTH[@]}" "$BASE/v1/models")
echo "$models" | grep -q "\"id\": *\"$EXPECT_MODEL\"" || { echo "verify_live: wrong served model: $models"; exit 1; }
echo "$models" | grep -q "$EXPECT_MAXLEN" || { echo "verify_live: max_modellen drift (want $EXPECT_MAXLEN): $models"; exit 1; }

out=$(curl -sf "${AUTH[@]}" "$BASE/v1/chat/completions" \
  -d '{"model":"'"$EXPECT_MODEL"'","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":8,"temperature":0}')
echo "$models" | grep -q "$EXPECT_MAXLEN" || { echo "verify_live: max_model_len drift (want $EXPECT_MAXLEN): $models"; exit 1; }
echo "verify_live: OK"
