#!/usr/bin/env bash
# Test request for the serp search server.
# Start the server first: ./serp_search_launch.sh
# Default: http://127.0.0.1:8000

BASE_URL="${1:-http://127.0.0.1:8000}"

response=$(curl -s -X POST "${BASE_URL}/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is the capital of France?"]}')

if command -v jq &>/dev/null; then
  echo "$response" | jq .
else
  echo "$response"
fi
