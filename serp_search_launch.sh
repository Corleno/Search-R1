#!/usr/bin/env bash
# Requires env SERPAPI_API_KEY (never commit the key here). Examples:
#   - add: export SERPAPI_API_KEY='...'  to ~/.bashrc  then: source ~/.bashrc
#   - or one-shot: SERPAPI_API_KEY='...' bash serp_search_launch.sh

search_url=https://serpapi.com/search
serp_api_key="${SERPAPI_API_KEY:?}"

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
