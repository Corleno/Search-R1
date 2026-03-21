#!/usr/bin/env bash
# Usage: export SERPAPI_API_KEY='your_key'   then:  bash serp_search_launch.sh

search_url=https://serpapi.com/search
serp_api_key="${SERPAPI_API_KEY:?Please export SERPAPI_API_KEY (https://serpapi.com/)}"

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
